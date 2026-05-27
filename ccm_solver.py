"""
CCM Tunnel Solver — Python port of "GRC (2000) framework copy one sheet.xlsx".

Convergence-Confinement Method (CCM):
    1. Build the Ground Reaction Curve (GRC) from Hoek-Brown rock-mass parameters
       (Carranza-Torres / Hoek closed-form).
    2. Build the Support Characteristic Curve (SCC) for one of seven support types,
       offset by the pre-support deformation u_r(L).
    3. Find the GRC & SCC intersection -> equilibrium (u_r*, p_i*).
    4. Back-calculate the in-situ GSI that makes u_r* match a measured convergence,
       and report the Factor of Safety FoS = p_max / p_i*.

Run:
    python ccm_solver.py             # solve the Excel reference case
    python ccm_solver.py --no-plot   # skip the chart
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from enum import Enum
from math import exp, log, pi, radians, sin, sqrt

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, minimize_scalar


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

class SupportType(str, Enum):
    SHOTCRETE = "Shotcrete"
    BOLTS = "Bolts"
    RRS_EQUIVALENT = "RRS equivalent"
    RRS_PARALLEL = "RRS parallel"
    SHOTCRETE_BOLTS = "Shotcrete+Bolts"
    RRS_EQUIVALENT_BOLTS = "RRS equivalent+Bolts"
    RRS_PARALLEL_BOLTS = "RRS parallel+Bolts"


@dataclass
class CCMInputs:
    # Geometry & loading
    R: float = 4.5            # tunnel radius [m]
    sigma_0: float = 3.3      # in-situ stress [MPa]
    L: float = 0.5            # distance from face when support is installed [m]

    # Rock mass (Hoek-Brown)
    GSI: float = 26.42        # Geological Strength Index (initial guess)
    sigma_ci: float = 35.0    # UCS of intact rock [MPa]
    m_i: float = 20.0         # Hoek-Brown m_i
    nu: float = 0.30          # Poisson ratio
    psi: float = 0.0          # dilation angle [deg]
    gamma: float = 0.026      # unit weight [MN/m^3]

    # Field measurement
    u_r_measured: float = 19.5  # measured convergence at the wall [mm]

    # Support choice + parameters
    support: SupportType = SupportType.RRS_PARALLEL_BOLTS

    # Shotcrete (Excel B25:B28)
    sigma_cc: float = 35.0    # compressive strength [MPa]
    E_c: float = 30.0         # Young's modulus [GPa]
    nu_c: float = 0.25
    t_c: float = 150.0        # thickness [mm]

    # Bolts (Excel B31:B37)
    d_b: float = 20.0         # bolt diameter [mm]
    l_b: float = 3.0          # bolt length [m]
    T_bf: float = 0.188       # ultimate load [MN]
    Q_b: float = 0.03         # compliance [m/MN]
    E_b: float = 200.0        # bolt modulus [GPa]
    n_bolt: int = 21          # bolts per ring
    s_t: float = 1.3          # ring spacing along tunnel [m]

    # RRS equivalent (Excel B40:B49)
    rrs_eq_tc: float = 300.0          # equivalent shotcrete thickness [mm]
    rrs_eq_Ec: float = 30_000.0       # concrete modulus [MPa]
    rrs_eq_nu_c: float = 0.25
    rrs_eq_sigma_cc: float = 35.0     # concrete strength [MPa]
    rrs_eq_nbars: int = 1             # bars per rib
    rrs_eq_dbar: float = 20.0         # bar diameter [mm]
    rrs_eq_Es: float = 200_000.0      # steel modulus [MPa]
    rrs_eq_nu_s: float = 0.30
    rrs_eq_sigma_ys: float = 400.0    # steel yield [MPa]
    rrs_eq_Sl: float = 0.25           # rib spacing [m]

    # RRS parallel (Excel B52:B61) — same fields, separate values
    rrs_par_tc: float = 300.0
    rrs_par_Ec: float = 30_000.0
    rrs_par_nu_c: float = 0.25
    rrs_par_sigma_cc: float = 35.0
    rrs_par_nbars: int = 1
    rrs_par_dbar: float = 20.0
    rrs_par_Es: float = 200_000.0
    rrs_par_nu_s: float = 0.30
    rrs_par_sigma_ys: float = 400.0
    rrs_par_Sl: float = 0.25


# ---------------------------------------------------------------------------
# Ground Reaction Curve
# ---------------------------------------------------------------------------

@dataclass
class GRC:
    u_r: np.ndarray   # radial displacement at wall [mm]
    p_i: np.ndarray   # internal pressure [MPa]
    picr: float       # critical pressure (elastic/plastic transition) [MPa]
    u_max: float      # u_r at p_i = 0 [mm] — used as LDP u_max
    G_rm_MPa: float   # rock-mass shear modulus [MPa]
    K_psi: float
    Picr: float
    S0: float
    mb: float
    s: float


def compute_grc(inp: CCMInputs, GSI: float | None = None, n_per_branch: int = 100) -> GRC:
    """Build the GRC for a given GSI. Math matches Python_Spec sections 1-6."""
    GSI = inp.GSI if GSI is None else GSI

    # Hoek-Brown
    mb = inp.m_i * exp((GSI - 100.0) / 28.0)
    s = exp((GSI - 100.0) / 9.0) if GSI >= 25.0 else 0.0

    # Rock-mass moduli — Hoek & Brown (1997)
    #     E_rm = sqrt(sigma_ci / 100) · 10^((GSI − 10) / 40)   [GPa]
    # Hoek, E. & Brown, E.T. (1997). Practical estimates of rock mass strength.
    # Int. J. Rock Mech. Min. Sci., 34(8), 1165–1186.
    E_rm = sqrt(inp.sigma_ci / 100.0) * 10.0 ** ((GSI - 10.0) / 40.0)  # GPa
    G_rm = E_rm / (2.0 * (1.0 + inp.nu))                                # GPa
    G_rm_MPa = G_rm * 1000.0

    # Dilation factor
    K_psi = (1.0 + sin(radians(inp.psi))) / (1.0 - sin(radians(inp.psi)))

    # Critical (elastic-plastic transition) pressure
    S0 = inp.sigma_0 / (mb * inp.sigma_ci) + s / mb**2
    Picr = (1.0 - sqrt(1.0 + 16.0 * S0)) ** 2 / 16.0
    picr = (Picr - s / mb**2) * mb * inp.sigma_ci

    # --- Elastic branch: p_i from sigma_0 down to picr ---
    p_elastic = np.linspace(inp.sigma_0, max(picr, 0.0), n_per_branch)
    u_elastic = (inp.sigma_0 - p_elastic) / (2.0 * G_rm_MPa) * inp.R * 1000.0  # [mm]

    if picr <= 0.0:
        # Fully elastic: extend the elastic branch all the way to p_i = 0
        p_full = np.linspace(inp.sigma_0, 0.0, 2 * n_per_branch)
        u_full = (inp.sigma_0 - p_full) / (2.0 * G_rm_MPa) * inp.R * 1000.0
        return GRC(
            u_r=u_full, p_i=p_full, picr=picr, u_max=float(u_full[-1]),
            G_rm_MPa=G_rm_MPa, K_psi=K_psi, Picr=Picr, S0=S0, mb=mb, s=s,
        )

    # --- Plastic branch: p_i from picr down to a tiny positive value ---
    # Fully vectorised with NumPy — ~50× faster than the equivalent Python loop.
    p_plastic = np.linspace(picr, 1e-9, n_per_branch)
    coef = inp.R * (inp.sigma_0 - picr) / (2.0 * G_rm_MPa)

    P = p_plastic / (mb * inp.sigma_ci) + s / mb ** 2
    Rpl_R = np.exp(2.0 * (sqrt(Picr) - np.sqrt(P)))
    ln_Rpl = np.log(Rpl_R)

    term1 = (K_psi - 1.0) / (K_psi + 1.0)
    term2 = 2.0 / (K_psi + 1.0) * Rpl_R ** (K_psi + 1.0)
    term3 = (1.0 - 2.0 * inp.nu) / (4.0 * (S0 - Picr)) * ln_Rpl ** 2
    bracket = (
        (1.0 - 2.0 * inp.nu) / (K_psi + 1.0) * sqrt(Picr) / (S0 - Picr)
        + (1.0 - inp.nu) / 2.0 * (K_psi - 1.0) / (K_psi + 1.0) ** 2 / (S0 - Picr)
    )
    term4 = -bracket * ((K_psi + 1.0) * ln_Rpl - Rpl_R ** (K_psi + 1.0) + 1.0)
    u_plastic = coef * (term1 + term2 + term3 + term4) * 1000.0  # [mm]

    # Stitch elastic + plastic, sorted by u_r ascending
    u_r = np.concatenate([u_elastic, u_plastic])
    p_i = np.concatenate([p_elastic, p_plastic])
    order = np.argsort(u_r)
    u_r, p_i = u_r[order], p_i[order]

    return GRC(
        u_r=u_r, p_i=p_i, picr=picr, u_max=float(u_plastic[-1]),
        G_rm_MPa=G_rm_MPa, K_psi=K_psi, Picr=Picr, S0=S0, mb=mb, s=s,
    )


# ---------------------------------------------------------------------------
# Support Characteristic Curves
# ---------------------------------------------------------------------------

@dataclass
class SCC:
    """Piecewise-linear SCC anchored at u_r_L."""
    u_r_L: float       # pre-support displacement [mm]
    p_max: float       # support capacity [MPa]
    u_max_supp: float  # support deformation at p_max [mm]
    K: float           # stiffness [MPa/m]
    label: str = ""

    def pressure(self, u_r):
        """SCC pressure at total wall displacement u_r [mm].
        Vectorised: scalars in -> 0-d arrays out, arrays in -> arrays out."""
        u = np.asarray(u_r, dtype=float)
        delta = u - self.u_r_L
        return np.where(
            delta < 0.0, 0.0,
            np.where(delta < self.u_max_supp,
                     self.K * delta / 1000.0,    # K is MPa/m, delta is mm
                     self.p_max),
        )

    def yield_point(self) -> tuple[float, float]:
        return self.u_r_L + self.u_max_supp, self.p_max


def _shotcrete(R: float, sigma_cc: float, E_c_GPa: float, nu_c: float, t_mm: float):
    """Excel cells F80:F82."""
    t = t_mm / 1000.0
    p_max = sigma_cc / 2.0 * (1.0 - (R - t) ** 2 / R ** 2)
    K = (E_c_GPa * 1000.0 / (1.0 + nu_c)) * (
        (R ** 2 - (R - t) ** 2) / ((1.0 - 2.0 * nu_c) * R ** 2 + (R - t) ** 2)
    ) / R
    u_max = p_max / K * 1000.0  # [mm]
    return p_max, K, u_max


def _bolts(R: float, n_bolt: int, T_bf: float, s_t: float,
           d_b_mm: float, l_b: float, Q: float, E_b_GPa: float):
    """Excel cells F86:F89."""
    s_c = 2.0 * pi * R / n_bolt
    p_b = T_bf / (s_c * s_t)
    d_b = d_b_mm / 1000.0
    K = (1.0 / (s_c * s_t)) * (
        pi * d_b ** 2 * E_b_GPa * 1000.0
        / (4.0 * l_b + Q * pi * d_b ** 2 * E_b_GPa * 1000.0)
    )
    u_max = p_b / K * 1000.0
    return p_b, K, u_max


def _rrs_equivalent(R: float, *, tc_mm: float, Ec_MPa: float, nu_c: float,
                    sigma_cc: float, nbars: int, dbar_mm: float,
                    Es_MPa: float, Sl: float):
    """Equivalent shotcrete shell with embedded ribs.

    The reinforcement is accounted for ONCE: it inflates the equivalent
    shell thickness via the transformed-section concept
        A_tr = t + (n - 1) · A_s,l       with  n = E_s / E_c.
    The resulting cylinder is then treated as plain shotcrete with the
    original concrete modulus E_c and compressive strength σ_cc. Mixing
    material properties on top of the transformed thickness (as the
    earlier Excel version did) is a double-counting of the reinforcement
    and is therefore avoided here.
    """
    t = tc_mm / 1000.0
    A_s = nbars * pi * (dbar_mm / 1000.0) ** 2 / 4.0       # bar area [m^2]
    A_sl = A_s / Sl                                         # steel area per metre of tunnel [m^2/m]
    n = Es_MPa / Ec_MPa                                     # modular ratio
    A_tr = t + (n - 1.0) * A_sl                             # transformed thickness [m]
    R_i_eq = R - A_tr                                       # equivalent inner radius

    # Cylinder behaves as pure shotcrete — no further material averaging.
    E_eq = Ec_MPa
    sigma_eq = sigma_cc

    p_max = sigma_eq / 2.0 * (1.0 - R_i_eq ** 2 / R ** 2)
    K = (E_eq / ((1.0 + nu_c) * R)) * (
        (R ** 2 - R_i_eq ** 2) / ((1.0 - 2.0 * nu_c) * R ** 2 + R_i_eq ** 2)
    )
    u_max = p_max / K * 1000.0
    return p_max, K, u_max


def _rrs_parallel(R: float, *, tc_mm: float, Ec_MPa: float, nu_c: float,
                  sigma_cc: float, nbars: int, dbar_mm: float,
                  Es_MPa: float, nu_s: float, sigma_ys: float, Sl: float):
    """Excel block F127:F128 + F99:F102 + I137/L137 area.

    Models a thin shotcrete shell + a thin steel cylinder in parallel:
    stiffness adds, capacity = K_total * min(u_c_max, u_s_max).
    """
    # Shotcrete shell (Excel F81/F80/F82 evaluated with rrs_par values)
    p_c_max, K_c, u_c_max = _shotcrete(R, sigma_cc, Ec_MPa / 1000.0, nu_c, tc_mm)

    # Steel-rib equivalent thin cylinder
    A_s = nbars * pi * (dbar_mm / 1000.0) ** 2 / 4.0       # F127
    t_s = A_s / Sl                                          # equivalent thin-shell thickness [m]
    R_i_s = R - t_s
    p_s_max = sigma_ys / 2.0 * (1.0 - R_i_s ** 2 / R ** 2)
    K_s = (Es_MPa / ((1.0 + nu_s) * R)) * (
        (R ** 2 - R_i_s ** 2) / ((1.0 - 2.0 * nu_s) * R ** 2 + R_i_s ** 2)
    )
    u_s_max = p_s_max / K_s * 1000.0

    K_total = K_c + K_s
    u_max = min(u_c_max, u_s_max)
    p_max = K_total * u_max / 1000.0
    return p_max, K_total, u_max


def build_scc(inp: CCMInputs, u_r_L: float) -> SCC:
    """Dispatch to the right SCC formula and wrap it as a piecewise-linear curve."""
    R = inp.R
    if inp.support == SupportType.SHOTCRETE:
        p_max, K, u_max = _shotcrete(R, inp.sigma_cc, inp.E_c, inp.nu_c, inp.t_c)
    elif inp.support == SupportType.BOLTS:
        p_max, K, u_max = _bolts(
            R, inp.n_bolt, inp.T_bf, inp.s_t, inp.d_b, inp.l_b, inp.Q_b, inp.E_b
        )
    elif inp.support == SupportType.RRS_EQUIVALENT:
        p_max, K, u_max = _rrs_equivalent(
            R, tc_mm=inp.rrs_eq_tc, Ec_MPa=inp.rrs_eq_Ec, nu_c=inp.rrs_eq_nu_c,
            sigma_cc=inp.rrs_eq_sigma_cc, nbars=inp.rrs_eq_nbars,
            dbar_mm=inp.rrs_eq_dbar, Es_MPa=inp.rrs_eq_Es, Sl=inp.rrs_eq_Sl,
        )
    elif inp.support == SupportType.RRS_PARALLEL:
        p_max, K, u_max = _rrs_parallel(
            R, tc_mm=inp.rrs_par_tc, Ec_MPa=inp.rrs_par_Ec, nu_c=inp.rrs_par_nu_c,
            sigma_cc=inp.rrs_par_sigma_cc, nbars=inp.rrs_par_nbars,
            dbar_mm=inp.rrs_par_dbar, Es_MPa=inp.rrs_par_Es, nu_s=inp.rrs_par_nu_s,
            sigma_ys=inp.rrs_par_sigma_ys, Sl=inp.rrs_par_Sl,
        )
    elif inp.support == SupportType.SHOTCRETE_BOLTS:
        _, K_c, u_c_max = _shotcrete(R, inp.sigma_cc, inp.E_c, inp.nu_c, inp.t_c)
        _, K_b, u_b_max = _bolts(
            R, inp.n_bolt, inp.T_bf, inp.s_t, inp.d_b, inp.l_b, inp.Q_b, inp.E_b
        )
        K = K_c + K_b
        u_max = min(u_c_max, u_b_max)
        p_max = K * u_max / 1000.0
    elif inp.support == SupportType.RRS_EQUIVALENT_BOLTS:
        _, K_r, u_r_max = _rrs_equivalent(
            R, tc_mm=inp.rrs_eq_tc, Ec_MPa=inp.rrs_eq_Ec, nu_c=inp.rrs_eq_nu_c,
            sigma_cc=inp.rrs_eq_sigma_cc, nbars=inp.rrs_eq_nbars,
            dbar_mm=inp.rrs_eq_dbar, Es_MPa=inp.rrs_eq_Es, Sl=inp.rrs_eq_Sl,
        )
        _, K_b, u_b_max = _bolts(
            R, inp.n_bolt, inp.T_bf, inp.s_t, inp.d_b, inp.l_b, inp.Q_b, inp.E_b
        )
        K = K_r + K_b
        u_max = min(u_r_max, u_b_max)
        p_max = K * u_max / 1000.0
    elif inp.support == SupportType.RRS_PARALLEL_BOLTS:
        _, K_r, u_r_max = _rrs_parallel(
            R, tc_mm=inp.rrs_par_tc, Ec_MPa=inp.rrs_par_Ec, nu_c=inp.rrs_par_nu_c,
            sigma_cc=inp.rrs_par_sigma_cc, nbars=inp.rrs_par_nbars,
            dbar_mm=inp.rrs_par_dbar, Es_MPa=inp.rrs_par_Es, nu_s=inp.rrs_par_nu_s,
            sigma_ys=inp.rrs_par_sigma_ys, Sl=inp.rrs_par_Sl,
        )
        _, K_b, u_b_max = _bolts(
            R, inp.n_bolt, inp.T_bf, inp.s_t, inp.d_b, inp.l_b, inp.Q_b, inp.E_b
        )
        K = K_r + K_b
        u_max = min(u_r_max, u_b_max)
        p_max = K * u_max / 1000.0
    else:
        raise ValueError(f"Unknown support type: {inp.support}")

    return SCC(u_r_L=u_r_L, p_max=p_max, u_max_supp=u_max, K=K, label=inp.support.value)


# ---------------------------------------------------------------------------
# Longitudinal Displacement Profile (Hoek 1999)
#
# Empirical best-fit relationship between radial displacement of the tunnel
# and distance x from the advancing face:
#     u_r(x) = u_max · [1 + exp(−x / (1.1 · R))]^(−1.7)
# Hoek, E. (1999). Support for very weak rock associated with faults and
# shear zones. Distinguished lecture, ISRSRP Mining, Kalgoorlie, Australia.
# ---------------------------------------------------------------------------

def ldp_u_r_at_L(u_max: float, L: float, R: float) -> float:
    """Pre-support displacement at distance L from the face [mm]."""
    return u_max * (1.0 + exp(-L / 1.1 / R)) ** (-1.7)


# ---------------------------------------------------------------------------
# GRC ∩ SCC intersection
# ---------------------------------------------------------------------------

def grc_scc_intersection(grc: GRC, scc: SCC) -> tuple[float, float]:
    """Return (u_star [mm], p_star [MPa]) — first crossing of GRC and SCC.
    Vectorised: the bracketing scan is one NumPy call instead of 500 Python calls."""
    # Search window: between u_r_L and u at p=0 (or GRC max), avoid the flat zero region.
    u_lo = max(scc.u_r_L * 1.0001, float(grc.u_r[0]))
    u_hi = min(float(grc.u_r[-1]), scc.u_r_L + scc.u_max_supp)
    if u_hi <= u_lo:
        u_hi = float(grc.u_r[-1])

    # Bracket: vectorised diff over the whole grid in one shot.
    grid = np.linspace(u_lo, u_hi, 200)
    diffs = np.interp(grid, grc.u_r, grc.p_i) - scc.pressure(grid)
    crossings = np.where(np.diff(np.sign(diffs)) != 0)[0]
    if crossings.size == 0:
        return scc.u_r_L + scc.u_max_supp, scc.p_max

    # Refine with Brent's method — needs a scalar diff function.
    def diff_scalar(u: float) -> float:
        return float(np.interp(u, grc.u_r, grc.p_i)) - float(scc.pressure(u))

    i = crossings[0]
    u_star = brentq(diff_scalar, grid[i], grid[i + 1], xtol=1e-9)
    p_star = float(scc.pressure(u_star))
    return float(u_star), p_star


# ---------------------------------------------------------------------------
# Full forward model + back-calculation of GSI
# ---------------------------------------------------------------------------

@dataclass
class CCMResult:
    GSI: float
    u_star: float       # mm
    p_star: float       # MPa
    p_max: float        # MPa
    FoS: float
    u_r_L: float        # mm
    u_max: float        # mm at p_i = 0
    grc: GRC
    scc: SCC
    converged: bool
    residual: float     # u_star - u_measured  [mm]


def solve_forward(inp: CCMInputs, GSI: float) -> CCMResult:
    grc = compute_grc(inp, GSI=GSI)
    u_r_L = ldp_u_r_at_L(grc.u_max, inp.L, inp.R)
    scc = build_scc(inp, u_r_L=u_r_L)
    u_star, p_star = grc_scc_intersection(grc, scc)
    FoS = scc.p_max / p_star if p_star > 0 else float("inf")
    return CCMResult(
        GSI=GSI, u_star=u_star, p_star=p_star, p_max=scc.p_max, FoS=FoS,
        u_r_L=u_r_L, u_max=grc.u_max, grc=grc, scc=scc,
        converged=True, residual=u_star - inp.u_r_measured,
    )


def solve_back_calculation(inp: CCMInputs, gsi_bounds: tuple[float, float] = (5.0, 100.0)) -> CCMResult:
    """Find GSI such that u_star(GSI) == u_r_measured. Falls back to bounded minimisation."""
    def residual(GSI: float) -> float:
        return solve_forward(inp, GSI).u_star - inp.u_r_measured

    GSI_lo, GSI_hi = gsi_bounds
    try:
        r_lo = residual(GSI_lo)
        r_hi = residual(GSI_hi)
        if r_lo * r_hi < 0:
            GSI_sol = brentq(residual, GSI_lo, GSI_hi, xtol=1e-6)
            res = solve_forward(inp, GSI_sol)
            res.converged = True
            return res
    except (ValueError, FloatingPointError):
        pass

    # Fallback: minimise squared residual (Excel Solver style)
    out = minimize_scalar(
        lambda G: residual(G) ** 2,
        bounds=gsi_bounds, method="bounded",
        options={"xatol": 1e-6},
    )
    res = solve_forward(inp, float(out.x))
    res.converged = bool(out.success and abs(res.residual) < 1e-3)
    return res


# ---------------------------------------------------------------------------
# Multi-section back-calculation
# ---------------------------------------------------------------------------

@dataclass
class SharedRockMass:
    """Hoek-Brown / elastic parameters shared by every section in the
    same tunnel reach (rock-mass properties of the same geological domain)."""
    sigma_ci: float = 35.0
    m_i: float = 20.0
    nu: float = 0.30
    psi: float = 0.0
    gamma: float = 0.026


@dataclass
class SectionInputs:
    """Per-section inputs that vary along the tunnel (geometry, loading,
    measurement, support)."""
    name: str = "Section 1"
    R: float = 4.5
    sigma_0: float = 3.3
    L: float = 0.5
    u_r_measured: float = 19.5
    support: SupportType = SupportType.RRS_PARALLEL_BOLTS
    # Only the support parameters relevant to the chosen support type.
    support_params: dict = field(default_factory=dict)


@dataclass
class MultiCCMInputs:
    """Inputs for multi-section back-calculation. One shared GSI is fitted
    across all sections."""
    rock: SharedRockMass = field(default_factory=SharedRockMass)
    sections: list = field(default_factory=lambda: [SectionInputs()])
    GSI_initial: float = 30.0  # purely a hint for the optimiser


@dataclass
class MultiCCMResult:
    GSI: float
    converged: bool
    total_squared_error: float    # Σ (u_r* − u_meas)²   [mm²]
    rms_residual: float           # √(SSE / N)            [mm]
    section_results: list         # list[CCMResult]
    section_inputs: list          # list[SectionInputs]


def _build_ccm_inputs(rock: SharedRockMass, section: SectionInputs,
                       GSI: float) -> CCMInputs:
    """Combine shared rock-mass + section-specific values into a CCMInputs."""
    valid_field_names = {f.name for f in fields(CCMInputs)}
    kwargs = {
        "R": section.R, "sigma_0": section.sigma_0, "L": section.L,
        "u_r_measured": section.u_r_measured,
        "GSI": GSI,
        "sigma_ci": rock.sigma_ci, "m_i": rock.m_i,
        "nu": rock.nu, "psi": rock.psi, "gamma": rock.gamma,
        "support": section.support,
    }
    for k, v in section.support_params.items():
        if k in valid_field_names:
            kwargs[k] = v
    return CCMInputs(**kwargs)


def solve_back_calculation_multi(
    multi_inp: MultiCCMInputs,
    gsi_bounds: tuple[float, float] = (5.0, 100.0),
) -> MultiCCMResult:
    """Optimise ONE shared GSI such that the sum of squared residuals
    Σ (u_r*(GSI) − u_meas)² across all sections is minimised.

    This generalises the single-section back-calc to a longer tunnel reach:
    the rock-mass parameters (Hoek-Brown + elastic) are assumed shared, while
    each section can have its own geometry, loading, and support.
    """
    if not multi_inp.sections:
        raise ValueError("MultiCCMInputs.sections must not be empty.")

    rock = multi_inp.rock
    secs = multi_inp.sections

    def total_sse(GSI: float) -> float:
        sse = 0.0
        for sec in secs:
            ccm_inp = _build_ccm_inputs(rock, sec, GSI)
            res = solve_forward(ccm_inp, GSI)
            sse += (res.u_star - sec.u_r_measured) ** 2
        return sse

    out = minimize_scalar(
        total_sse,
        bounds=gsi_bounds, method="bounded",
        options={"xatol": 1e-6},
    )
    GSI_opt = float(out.x)

    # Build full per-section result with the optimum GSI.
    section_results = []
    for sec in secs:
        ccm_inp = _build_ccm_inputs(rock, sec, GSI_opt)
        res = solve_forward(ccm_inp, GSI_opt)
        res.residual = res.u_star - sec.u_r_measured
        res.converged = bool(out.success)
        section_results.append(res)

    sse = total_sse(GSI_opt)
    rms = (sse / len(secs)) ** 0.5

    return MultiCCMResult(
        GSI=GSI_opt,
        converged=bool(out.success),
        total_squared_error=sse,
        rms_residual=rms,
        section_results=section_results,
        section_inputs=list(secs),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_result(inp: CCMInputs, result: CCMResult, savepath: str | None = None) -> None:
    grc = result.grc
    scc = result.scc

    # Split GRC at picr for colouring
    if grc.picr > 0:
        elastic_mask = grc.p_i >= grc.picr
        plastic_mask = ~elastic_mask
    else:
        elastic_mask = np.ones_like(grc.p_i, dtype=bool)
        plastic_mask = np.zeros_like(grc.p_i, dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(grc.u_r[elastic_mask], grc.p_i[elastic_mask],
            color="tab:blue", lw=2, label="GRC — elastic")
    if plastic_mask.any():
        ax.plot(grc.u_r[plastic_mask], grc.p_i[plastic_mask],
                color="indigo", lw=2, label="GRC — plastic")

    # SCC: three points (u_r_L, 0) -> yield -> plateau out to right edge
    u_yield, p_yield = scc.yield_point()
    u_right = max(grc.u_r[-1], u_yield + 5.0)
    ax.plot([scc.u_r_L, u_yield, u_right],
            [0.0, p_yield, p_yield],
            color="tab:orange", lw=2, label=f"SCC — {scc.label}")

    # Equilibrium + measured u_r
    ax.plot(result.u_star, result.p_star, "o", color="red", markersize=10,
            label=f"Equilibrium  (u={result.u_star:.2f} mm, p={result.p_star:.3f} MPa)")
    ax.axvline(inp.u_r_measured, color="grey", linestyle="--", alpha=0.6,
               label=f"Measured u_r = {inp.u_r_measured} mm")

    ax.set_xlabel("Radial displacement at wall  u_r  [mm]")
    ax.set_ylabel("Internal pressure  p_i  [MPa]")
    ax.set_title(f"GRC ∩ SCC  —  back-calc GSI = {result.GSI:.2f},  FoS = {result.FoS:.2f}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
        print(f"Saved plot -> {savepath}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CCM tunnel solver — back-calculates in-situ GSI.")
    parser.add_argument("--no-plot", action="store_true", help="skip the chart")
    parser.add_argument("--save-plot", type=str, default=None,
                        help="save chart to this path instead of showing it")
    args = parser.parse_args()

    inp = CCMInputs()  # Excel reference case
    result = solve_back_calculation(inp)

    print("=" * 60)
    print("  CCM TUNNEL SOLVER")
    print("=" * 60)
    print(f"  Support type             : {inp.support.value}")
    print(f"  Measured u_r             : {inp.u_r_measured:.3f} mm")
    print("-" * 60)
    print(f"  Pre-support u_r(L)       : {result.u_r_L:.3f} mm")
    print(f"  Critical pressure p_cr   : {result.grc.picr:.4f} MPa")
    print(f"  Max deformation (no supp): {result.u_max:.3f} mm")
    print("-" * 60)
    print(f"  Total support stiffness K: {result.scc.K:.4f} MPa/m")
    print(f"  Total support capacity   : {result.p_max:.4f} MPa")
    print(f"  Support stroke u_max,supp: {result.scc.u_max_supp:.4f} mm")
    print("-" * 60)
    print(f"  Equilibrium u_r*         : {result.u_star:.4f} mm")
    print(f"  Equilibrium p_i*         : {result.p_star:.4f} MPa")
    print(f"  Residual (u* − u_meas)   : {result.residual:.2e} mm")
    print(f"  Back-calculated GSI      : {result.GSI:.4f}")
    print(f"  Factor of Safety         : {result.FoS:.3f}")
    print(f"  Converged                : {result.converged}")
    print("=" * 60)

    if not args.no_plot:
        plot_result(inp, result, savepath=args.save_plot)


if __name__ == "__main__":
    main()
