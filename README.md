# Optimization program for Back-Calculating GSI (using CCM)

A Python implementation of the Convergence-Confinement Method (CCM) for back-calculating the in-situ Geological Strength Index (GSI) from measured tunnel convergence. The program is provided as a Streamlit web app that opens in your browser.

The app supports both single-section and multi-section back-calculation. With one section it solves the classic single-point inverse problem; with several sections it fits one shared GSI to all convergence measurements simultaneously by minimising the sum of squared residuals.

The two Python files `ccm_solver.py` and `ccm_app.py` must sit in the same folder.

---

## What you need

1. Python 3.10 or newer — install from <https://www.python.org/downloads/>
   - macOS: download the `.pkg` installer and run it
   - Windows: download the `.exe`, and tick "Add python.exe to PATH" on the first installer screen
2. VS Code (recommended editor) — install from <https://code.visualstudio.com/download>. Open the folder containing the two Python files with **File → Open Folder…**, then open a terminal inside the folder with **Terminal → New Terminal**.

---

## Install the required packages (one time)

Run these commands in the terminal:

**macOS**
```bash
/usr/local/bin/python3 -m pip install --user streamlit pandas numpy scipy matplotlib
```

**Windows**
```cmd
python -m pip install --user streamlit pandas numpy scipy matplotlib
```

------------------


## Run the app

In the terminal (still inside the folder with the Python files):

**macOS**
```bash
/usr/local/bin/python3 -m streamlit run ccm_app.py
```

**Windows**
```cmd
python -m streamlit run ccm_app.py
```

After a couple of seconds a browser tab opens automatically at <http://localhost:8501> with the app. Edit the inputs in the sidebar and click ▶ **Solve**.

To stop the app, click back into the terminal and press **Ctrl + C**.
