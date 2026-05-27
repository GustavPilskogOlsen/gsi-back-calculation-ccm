"""
CCM Tunnel Solver — back-calculation of in-situ GSI.

Optimises ONE shared GSI from convergence measurements at one or more
tunnel cross-sections. Each section can have its own geometry, loading,
support type and measured u_r; the rock-mass parameters
(σ_ci, m_i, ν, ψ, γ, GSI) are shared across all sections. With a single
section the app reduces to the standard single-point back-calculation.

Run:
    streamlit run "/Users/gusta/Claude programeringsfil/ccm_app.py"
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ccm_solver import (
    CCMInputs,
    MultiCCMInputs,
    SCC,
    SectionInputs,
    SharedRockMass,
    SupportType,
    build_scc,
    compute_grc,
    grc_scc_intersection,
    ldp_u_r_at_L,
    solve_back_calculation_multi,
)


st.set_page_config(
    page_title="CCM Tunnel Solver",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("CCM Tunnel Solver")
st.caption(
    "Back-calculate the in-situ GSI from tunnel convergence measurements. "
    "Use one or more cross-sections along the same rock-mass reach — each "
    "section can have its own geometry and support type; the GSI is shared."
)

inp_default = CCMInputs()
rock_default = SharedRockMass()


# ---------------------------------------------------------------------------
# Sidebar — shared rock mass + number of sections
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Shared rock mass (Hoek-Brown)")
    st.caption("These apply to every section.")
    sigma_ci = st.number_input("σ_ci — UCS intact rock [MPa]",
                                min_value=1.0, value=rock_default.sigma_ci,
                                step=1.0, format="%.4f")
    m_i = st.number_input("m_i [-]",
                            min_value=0.1, value=rock_default.m_i,
                            step=0.5, format="%.4f")
    nu = st.number_input("ν — Poisson ratio [-]",
                            min_value=0.0, max_value=0.5,
                            value=rock_default.nu, step=0.01,
                            format="%.4f")
    psi = st.number_input("ψ — dilation angle [°]",
                            min_value=0.0, max_value=60.0,
                            value=rock_default.psi, step=1.0,
                            format="%.4f")
    gamma = st.number_input("γ — unit weight [MN/m³]",
                              min_value=0.001, value=rock_default.gamma,
                              step=0.001, format="%.5f")

    st.divider()

    st.header("Tunnel sections")
    n_sections = st.number_input(
        "Number of sections",
        min_value=1, max_value=20, value=1, step=1,
        help="Each section has its own geometry, σ₀, L, measurement, "
              "and support type. The GSI is shared across them.",
    )


# ---------------------------------------------------------------------------
# Helpers — render per-section inputs
# ---------------------------------------------------------------------------

def render_support_params(idx: int, support: SupportType) -> dict:
    """Render the support parameters for one section, return a dict."""
    params: dict = {}
    needs_shotcrete = support in {
        SupportType.SHOTCRETE, SupportType.SHOTCRETE_BOLTS,
    }
    needs_bolts = support in {
        SupportType.BOLTS, SupportType.SHOTCRETE_BOLTS,
        SupportType.RRS_EQUIVALENT_BOLTS, SupportType.RRS_PARALLEL_BOLTS,
    }
    needs_rrs_eq = support in {
        SupportType.RRS_EQUIVALENT, SupportType.RRS_EQUIVALENT_BOLTS,
    }
    needs_rrs_par = support in {
        SupportType.RRS_PARALLEL, SupportType.RRS_PARALLEL_BOLTS,
    }

    K = f"s{idx}"  # key prefix per section to keep widget keys unique

    if needs_shotcrete:
        with st.expander("Shotcrete parameters", expanded=False):
            params["sigma_cc"] = st.number_input(
                "σ_cc [MPa]", value=inp_default.sigma_cc,
                step=1.0, format="%.4f", key=f"{K}_sc_sigma_cc")
            params["E_c"] = st.number_input(
                "E_c [GPa]", value=inp_default.E_c,
                step=1.0, format="%.4f", key=f"{K}_sc_Ec")
            params["nu_c"] = st.number_input(
                "ν_c", value=inp_default.nu_c,
                step=0.01, format="%.4f", key=f"{K}_sc_nuc")
            params["t_c"] = st.number_input(
                "t_c [mm]", value=inp_default.t_c,
                step=10.0, format="%.4f", key=f"{K}_sc_tc")

    if needs_bolts:
        with st.expander("Bolt parameters", expanded=False):
            params["d_b"] = st.number_input(
                "d_b [mm]", value=inp_default.d_b,
                step=1.0, format="%.4f", key=f"{K}_b_db")
            params["l_b"] = st.number_input(
                "l_b [m]", value=inp_default.l_b,
                step=0.1, format="%.4f", key=f"{K}_b_lb")
            params["T_bf"] = st.number_input(
                "T_bf [MN]", value=inp_default.T_bf,
                step=0.01, format="%.5f", key=f"{K}_b_Tbf")
            params["Q_b"] = st.number_input(
                "Q [m/MN]", value=inp_default.Q_b,
                step=0.01, format="%.5f", key=f"{K}_b_Q")
            params["E_b"] = st.number_input(
                "E_b [GPa]", value=inp_default.E_b,
                step=10.0, format="%.4f", key=f"{K}_b_Eb")
            params["n_bolt"] = int(st.number_input(
                "n_bolt", value=inp_default.n_bolt,
                step=1, key=f"{K}_b_n"))
            params["s_t"] = st.number_input(
                "s_t [m]", value=inp_default.s_t,
                step=0.1, format="%.4f", key=f"{K}_b_st")

    if needs_rrs_eq:
        with st.expander("RRS equivalent parameters", expanded=False):
            params["rrs_eq_tc"] = st.number_input(
                "t_c [mm]", value=inp_default.rrs_eq_tc,
                step=10.0, format="%.4f", key=f"{K}_re_tc")
            params["rrs_eq_Ec"] = st.number_input(
                "E_c [MPa]", value=inp_default.rrs_eq_Ec,
                step=1000.0, format="%.4f", key=f"{K}_re_Ec")
            params["rrs_eq_nu_c"] = st.number_input(
                "ν_c", value=inp_default.rrs_eq_nu_c,
                step=0.01, format="%.4f", key=f"{K}_re_nuc")
            params["rrs_eq_sigma_cc"] = st.number_input(
                "σ_cc [MPa]", value=inp_default.rrs_eq_sigma_cc,
                step=1.0, format="%.4f", key=f"{K}_re_sc")
            params["rrs_eq_nbars"] = int(st.number_input(
                "nbars", value=inp_default.rrs_eq_nbars,
                step=1, key=f"{K}_re_nb"))
            params["rrs_eq_dbar"] = st.number_input(
                "d_bar [mm]", value=inp_default.rrs_eq_dbar,
                step=1.0, format="%.4f", key=f"{K}_re_db")
            params["rrs_eq_Es"] = st.number_input(
                "E_s [MPa]", value=inp_default.rrs_eq_Es,
                step=1000.0, format="%.4f", key=f"{K}_re_Es")
            params["rrs_eq_nu_s"] = st.number_input(
                "ν_s", value=inp_default.rrs_eq_nu_s,
                step=0.01, format="%.4f", key=f"{K}_re_nus")
            params["rrs_eq_sigma_ys"] = st.number_input(
                "σ_ys [MPa]", value=inp_default.rrs_eq_sigma_ys,
                step=10.0, format="%.4f", key=f"{K}_re_sy")
            params["rrs_eq_Sl"] = st.number_input(
                "S_l [m]", value=inp_default.rrs_eq_Sl,
                step=0.1, format="%.4f", key=f"{K}_re_sl")

    if needs_rrs_par:
        with st.expander("RRS parallel parameters", expanded=False):
            params["rrs_par_tc"] = st.number_input(
                "t_c [mm]", value=inp_default.rrs_par_tc,
                step=10.0, format="%.4f", key=f"{K}_rp_tc")
            params["rrs_par_Ec"] = st.number_input(
                "E_c [MPa]", value=inp_default.rrs_par_Ec,
                step=1000.0, format="%.4f", key=f"{K}_rp_Ec")
            params["rrs_par_nu_c"] = st.number_input(
                "ν_c", value=inp_default.rrs_par_nu_c,
                step=0.01, format="%.4f", key=f"{K}_rp_nuc")
            params["rrs_par_sigma_cc"] = st.number_input(
                "σ_cc [MPa]", value=inp_default.rrs_par_sigma_cc,
                step=1.0, format="%.4f", key=f"{K}_rp_sc")
            params["rrs_par_nbars"] = int(st.number_input(
                "nbars", value=inp_default.rrs_par_nbars,
                step=1, key=f"{K}_rp_nb"))
            params["rrs_par_dbar"] = st.number_input(
                "d_bar [mm]", value=inp_default.rrs_par_dbar,
                step=1.0, format="%.4f", key=f"{K}_rp_db")
            params["rrs_par_Es"] = st.number_input(
                "E_s [MPa]", value=inp_default.rrs_par_Es,
                step=1000.0, format="%.4f", key=f"{K}_rp_Es")
            params["rrs_par_nu_s"] = st.number_input(
                "ν_s", value=inp_default.rrs_par_nu_s,
                step=0.01, format="%.4f", key=f"{K}_rp_nus")
            params["rrs_par_sigma_ys"] = st.number_input(
                "σ_ys [MPa]", value=inp_default.rrs_par_sigma_ys,
                step=10.0, format="%.4f", key=f"{K}_rp_sy")
            params["rrs_par_Sl"] = st.number_input(
                "S_l [m]", value=inp_default.rrs_par_Sl,
                step=0.1, format="%.4f", key=f"{K}_rp_sl")

    return params


def render_section(idx: int) -> SectionInputs:
    """Render one section's inputs and return a SectionInputs."""
    K = f"s{idx}"
    name = st.text_input(
        "Section name", value=f"Section {idx + 1}",
        key=f"{K}_name",
    )

    c1, c2 = st.columns(2)
    with c1:
        R = st.number_input(
            "R — radius [m]", min_value=0.1,
            value=inp_default.R, step=0.1, format="%.4f",
            key=f"{K}_R")
        L = st.number_input(
            "L — distance from face [m]", min_value=0.0,
            value=inp_default.L, step=0.1, format="%.4f",
            key=f"{K}_L")
    with c2:
        sigma_0 = st.number_input(
            "σ₀ — in-situ stress [MPa]", min_value=0.01,
            value=inp_default.sigma_0, step=0.1, format="%.4f",
            key=f"{K}_sigma0")
        u_r_measured = st.number_input(
            "Measured u_r [mm]", min_value=0.001,
            value=inp_default.u_r_measured, step=0.1, format="%.4f",
            key=f"{K}_ur")

    support_label = st.selectbox(
        "Support type",
        options=[s.value for s in SupportType],
        index=6,  # RRS parallel + Bolts
        key=f"{K}_support",
    )
    support = SupportType(support_label)

    support_params = render_support_params(idx, support)

    return SectionInputs(
        name=name, R=R, sigma_0=sigma_0, L=L,
        u_r_measured=u_r_measured, support=support,
        support_params=support_params,
    )


# ---------------------------------------------------------------------------
# Main area — section input tabs
# ---------------------------------------------------------------------------

tab_titles = [f"📍 Section {i + 1}" for i in range(int(n_sections))]
tabs = st.tabs(tab_titles)

sections: list[SectionInputs] = []
for i, tab in enumerate(tabs):
    with tab:
        sec = render_section(i)
        sections.append(sec)

st.divider()
solve_clicked = st.button(
    "▶  Solve  —  back-calculate shared GSI from all sections",
    type="primary", use_container_width=True,
)


# ---------------------------------------------------------------------------
# Stale-input detection
# ---------------------------------------------------------------------------

def fingerprint(rock: SharedRockMass, secs: list[SectionInputs]) -> tuple:
    return (
        rock.sigma_ci, rock.m_i, rock.nu, rock.psi, rock.gamma,
        tuple(
            (s.name, s.R, s.sigma_0, s.L, s.u_r_measured, s.support.value,
              tuple(sorted(s.support_params.items())))
            for s in secs
        ),
    )


rock = SharedRockMass(sigma_ci=sigma_ci, m_i=m_i, nu=nu, psi=psi, gamma=gamma)
current_fp = fingerprint(rock, sections)


# ---------------------------------------------------------------------------
# Run solver — only on click or first load
# ---------------------------------------------------------------------------

if solve_clicked or "multi_result" not in st.session_state:
    multi_inp = MultiCCMInputs(rock=rock, sections=sections)
    try:
        with st.spinner("Solving — minimising sum of squared residuals…"):
            result = solve_back_calculation_multi(multi_inp)
    except Exception as e:
        st.error(f"Solver failed: {type(e).__name__}: {e}")
        st.stop()

    st.session_state["multi_result"] = result
    st.session_state["multi_input"] = multi_inp
    st.session_state["multi_fp"] = current_fp

result = st.session_state["multi_result"]
multi_inp = st.session_state["multi_input"]

stale = st.session_state.get("multi_fp") != current_fp
if stale:
    st.info(
        "Inputs have been changed but not yet solved — the result below "
        "is from the previous run. Click **▶ Solve** to refresh.",
        icon="🔁",
    )


# ---------------------------------------------------------------------------
# Combined results
# ---------------------------------------------------------------------------

st.divider()
st.header("Combined back-calculation result")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Optimised GSI (shared)", f"{result.GSI:.4f}")
with c2:
    st.metric("Sections used", f"{len(result.section_inputs)}")
with c3:
    st.metric("Σ squared error", f"{result.total_squared_error:.4e} mm²")
with c4:
    st.metric("RMS residual", f"{result.rms_residual:.4f} mm")

if result.converged:
    st.success("✓ Optimiser converged")
else:
    st.warning("⚠ Optimiser did not fully converge — review inputs.")


# ---------------------------------------------------------------------------
# Per-section breakdown
# ---------------------------------------------------------------------------

st.subheader("Per-section breakdown")

table_rows = []
for sec, res in zip(result.section_inputs, result.section_results):
    table_rows.append({
        "Section":     sec.name,
        "R [m]":       sec.R,
        "σ₀ [MPa]":    sec.sigma_0,
        "L [m]":       sec.L,
        "Support":     sec.support.value,
        "u_meas [mm]": sec.u_r_measured,
        "u_r* [mm]":   res.u_star,
        "Residual [mm]": res.residual,
        "p_i* [MPa]":  res.p_star,
        "p_cr [MPa]":  res.grc.picr,
        "u_max no-supp [mm]": res.u_max,
        "K [MPa/m]":   res.scc.K,
        "p_max [MPa]": res.p_max,
        "u_max,supp [mm]": res.scc.u_max_supp,
        "FoS":         res.FoS,
    })
breakdown = pd.DataFrame(table_rows)
st.dataframe(breakdown, hide_index=True, use_container_width=True)

csv_bytes = breakdown.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️  Download per-section results (CSV)",
    data=csv_bytes,
    file_name="ccm_multi_results.csv",
    mime="text/csv",
)


# ---------------------------------------------------------------------------
# Per-section plots
# ---------------------------------------------------------------------------

st.subheader("Per-section GRC ∩ SCC charts")

plot_tabs = st.tabs([f"📈 {s.name}" for s in result.section_inputs])
for tab, sec, res in zip(plot_tabs, result.section_inputs, result.section_results):
    with tab:
        c1, c2 = st.columns([3, 1])

        with c2:
            st.metric("u_r meas",  f"{sec.u_r_measured:.3f} mm")
            st.metric("u_r*",      f"{res.u_star:.4f} mm")
            st.metric("Residual",  f"{res.residual:+.4f} mm")
            st.metric("p_i*",      f"{res.p_star:.4f} MPa")
            st.metric("Critical pressure p_cr", f"{res.grc.picr:.4f} MPa")
            st.metric("Max deformation (no support)", f"{res.u_max:.3f} mm")
            st.metric("Support stiffness K", f"{res.scc.K:.2f} MPa/m")
            st.metric("Support capacity p_max", f"{res.p_max:.4f} MPa")
            st.metric("FoS",       f"{res.FoS:.3f}")
            st.metric("u_r(L)",    f"{res.u_r_L:.3f} mm")
            st.caption(f"Support: {sec.support.value}")

        with c1:
            fig, ax = plt.subplots(figsize=(9, 5.5))
            grc, scc = res.grc, res.scc
            if grc.picr > 0:
                elastic = grc.p_i >= grc.picr
            else:
                elastic = np.ones_like(grc.p_i, dtype=bool)
            plastic = ~elastic

            ax.plot(grc.u_r[elastic], grc.p_i[elastic],
                    color="tab:blue", lw=2, label="GRC — elastic")
            if plastic.any():
                ax.plot(grc.u_r[plastic], grc.p_i[plastic],
                        color="indigo", lw=2, label="GRC — plastic")

            u_yield, p_yield = scc.yield_point()
            u_right = max(grc.u_r[-1], u_yield + 5.0)
            ax.plot([scc.u_r_L, u_yield, u_right],
                    [0.0, p_yield, p_yield],
                    color="tab:orange", lw=2,
                    label=f"SCC — {scc.label}")
            ax.plot(res.u_star, res.p_star, "o",
                    color="red", markersize=10, label="Equilibrium")
            ax.axvline(sec.u_r_measured, color="grey", linestyle="--",
                       alpha=0.6, label=f"Measured = {sec.u_r_measured:.2f} mm")

            ax.set_xlabel("u_r [mm]")
            ax.set_ylabel("p_i [MPa]")
            ax.set_title(
                f"{sec.name}  —  R={sec.R:.2f} m, "
                f"σ₀={sec.sigma_0:.2f} MPa, L={sec.L:.2f} m"
            )
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
            st.pyplot(fig)


# ---------------------------------------------------------------------------
# Combined plot — all GRCs + measured points
# ---------------------------------------------------------------------------

st.subheader("Combined view  —  all sections")

fig_all, ax_all = plt.subplots(figsize=(11, 6))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(result.section_inputs)))

for sec, res, color in zip(result.section_inputs,
                             result.section_results, colors):
    grc = res.grc
    ax_all.plot(grc.u_r, grc.p_i, color=color, lw=1.6,
                label=f"GRC — {sec.name}")
    ax_all.plot(res.u_star, res.p_star, "o",
                color=color, markersize=8, markeredgecolor="black")
    ax_all.axvline(sec.u_r_measured, color=color, linestyle="--",
                   alpha=0.4)

ax_all.set_xlabel("u_r [mm]")
ax_all.set_ylabel("p_i [MPa]")
ax_all.set_title(
    f"All sections — shared back-calculated GSI = {result.GSI:.3f}"
)
ax_all.set_xlim(left=0)
ax_all.set_ylim(bottom=0)
ax_all.grid(True, alpha=0.3)
ax_all.legend(loc="upper right", fontsize=8)
st.pyplot(fig_all)

st.caption(
    "Dashed vertical lines = measured u_r at each section. "
    "Coloured dots = analytical equilibrium points at the optimised GSI."
)


# ---------------------------------------------------------------------------
# Representative GRC + SCC plot — single curve summary
# ---------------------------------------------------------------------------
st.subheader("Representative GRC and SCC  —  single-curve summary")

# Mean geometry across sections (to construct ONE representative GRC).
R_mean       = float(np.mean([s.R       for s in result.section_inputs]))
sigma_0_mean = float(np.mean([s.sigma_0 for s in result.section_inputs]))
L_mean       = float(np.mean([s.L       for s in result.section_inputs]))

# Detect whether all sections share identical geometry — useful caption hint.
all_same_geom = all(
    np.isclose(s.R, R_mean)
    and np.isclose(s.sigma_0, sigma_0_mean)
    and np.isclose(s.L, L_mean)
    for s in result.section_inputs
)

# Build one representative CCM input from the first section's support
# parameters but with the mean geometry, then compute its GRC + SCC at the
# optimised GSI.
template = result.section_inputs[0]
rep_ccm_inp = CCMInputs(
    R=R_mean, sigma_0=sigma_0_mean, L=L_mean,
    sigma_ci=rock.sigma_ci, m_i=rock.m_i, nu=rock.nu,
    psi=rock.psi, gamma=rock.gamma,
    GSI=result.GSI,
    u_r_measured=float(np.mean([s.u_r_measured for s in result.section_inputs])),
    support=template.support,
    **template.support_params,
)
rep_grc   = compute_grc(rep_ccm_inp, GSI=result.GSI)
rep_u_r_L = ldp_u_r_at_L(rep_grc.u_max, L_mean, R_mean)

# Mean SCC parameters across sections (each section has its own).
u_r_L_mean      = float(np.mean([r.u_r_L          for r in result.section_results]))
p_max_mean      = float(np.mean([r.p_max          for r in result.section_results]))
u_max_supp_mean = float(np.mean([r.scc.u_max_supp for r in result.section_results]))
K_mean          = float(np.mean([r.scc.K          for r in result.section_results]))

# Build a synthetic "mean" SCC from the averaged parameters.
all_same_support = all(
    s.support == template.support
    and s.support_params == template.support_params
    for s in result.section_inputs
)
mean_scc_label = (
    f"Mean SCC ({template.support.value})"
    if all_same_support
    else "Mean SCC (averaged across mixed supports)"
)
mean_scc = SCC(u_r_L=u_r_L_mean, p_max=p_max_mean,
                u_max_supp=u_max_supp_mean, K=K_mean,
                label=mean_scc_label)

# Equilibrium point of the representative GRC and the mean SCC.
rep_u_star, rep_p_star = grc_scc_intersection(rep_grc, mean_scc)

# Per-section equilibrium and measurement statistics.
u_meas_mean = float(np.mean([s.u_r_measured for s in result.section_inputs]))
u_star_mean = float(np.mean([r.u_star       for r in result.section_results]))
p_star_mean = float(np.mean([r.p_star       for r in result.section_results]))

# Mean total support stiffness and capacity across all sections.
mc1, mc2, mc3 = st.columns(3)
mc1.metric("Mean total support stiffness K", f"{K_mean:.2f} MPa/m")
mc2.metric("Mean total support capacity p_max", f"{p_max_mean:.4f} MPa")
mc3.metric("Mean support stroke u_max,supp", f"{u_max_supp_mean:.4f} mm")


fig_rep, ax_rep = plt.subplots(figsize=(11, 6.2))

# Background — individual section curves, faint for context.
for sec, res in zip(result.section_inputs, result.section_results):
    ax_rep.plot(res.grc.u_r, res.grc.p_i,
                color="lightsteelblue", lw=0.9, alpha=0.7, zorder=1)
    u_y, p_y = res.scc.yield_point()
    u_right = max(res.grc.u_r[-1], u_y + 5.0)
    ax_rep.plot([res.scc.u_r_L, u_y, u_right],
                 [0.0, p_y, p_y],
                 color="moccasin", lw=0.9, alpha=0.7, zorder=1)

# Foreground — representative GRC.
if rep_grc.picr > 0:
    el = rep_grc.p_i >= rep_grc.picr
else:
    el = np.ones_like(rep_grc.p_i, dtype=bool)
pl = ~el
ax_rep.plot(rep_grc.u_r[el], rep_grc.p_i[el],
             color="tab:blue", lw=2.6, zorder=3,
             label=("GRC (shared)" if all_same_geom
                    else f"Representative GRC  (R̄={R_mean:.2f} m, σ̄₀={sigma_0_mean:.2f} MPa)"))
if pl.any():
    ax_rep.plot(rep_grc.u_r[pl], rep_grc.p_i[pl],
                 color="indigo", lw=2.6, zorder=3, label="GRC — plastic")

# Foreground — mean SCC.
u_y_m, p_y_m = mean_scc.yield_point()
u_right_m = max(rep_grc.u_r[-1], u_y_m + 5.0)
ax_rep.plot([mean_scc.u_r_L, u_y_m, u_right_m],
             [0.0, p_y_m, p_y_m],
             color="tab:orange", lw=2.6, zorder=3,
             label=mean_scc_label)

# Per-section equilibrium points (small dark dots).
for res in result.section_results:
    ax_rep.plot(res.u_star, res.p_star, "o",
                 color="darkred", markersize=6, alpha=0.65, zorder=4)

# Per-section measured u_r (faint vertical lines).
for sec in result.section_inputs:
    ax_rep.axvline(sec.u_r_measured, color="grey",
                    linestyle=":", lw=0.8, alpha=0.4, zorder=2)

# Mean equilibrium of the representative model (red star).
ax_rep.plot(rep_u_star, rep_p_star, "*",
             color="red", markersize=20, markeredgecolor="black",
             markeredgewidth=1.2, zorder=5,
             label=f"Equilibrium of mean curves  ({rep_u_star:.2f} mm, "
                   f"{rep_p_star:.3f} MPa)")

# Mean of the per-section equilibria (gold diamond).
ax_rep.plot(u_star_mean, p_star_mean, "D",
             color="gold", markersize=11, markeredgecolor="black",
             markeredgewidth=1.0, zorder=5,
             label=f"Mean of per-section equilibria  ({u_star_mean:.2f} mm, "
                   f"{p_star_mean:.3f} MPa)")

# Mean measured u_r (bold black dashed line).
ax_rep.axvline(u_meas_mean, color="black", linestyle="--", lw=1.5, zorder=4,
                label=f"Mean measured u_r = {u_meas_mean:.2f} mm")

ax_rep.set_xlabel("Radial displacement at wall  u_r  [mm]")
ax_rep.set_ylabel("Internal pressure  p_i  [MPa]")
ax_rep.set_title(
    f"Representative GRC and SCC  —  shared GSI = {result.GSI:.3f}",
    fontsize=12,
)
ax_rep.set_xlim(left=0)
ax_rep.set_ylim(bottom=0)
ax_rep.grid(True, alpha=0.3)
ax_rep.legend(loc="upper right", fontsize=8)
st.pyplot(fig_rep)

st.caption(
    "**Bold blue** = Ground Reaction Curve at the optimised shared GSI, "
    "evaluated for the mean tunnel geometry "
    f"(R̄={R_mean:.2f} m, σ̄₀={sigma_0_mean:.2f} MPa).  "
    "**Bold orange** = Support Characteristic Curve built from the mean "
    "u_r(L), p_max, K and u_max,supp across all sections.  "
    "Faint blue/orange = the individual section curves shown for context.  "
    "Dark-red dots = per-section analytical equilibria; "
    "**gold diamond** = arithmetic mean of those equilibria; "
    "**red star** = analytical equilibrium of the mean GRC and mean SCC.  "
    "**Black dashed line** = mean measured u_r across all sections."
)
