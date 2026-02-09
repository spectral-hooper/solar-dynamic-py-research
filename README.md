# solar-dynamic-py-research

**An Integrated Pipeline for Solar Magnetic Field Analysis and CME Energy Estimation**

This repository contains a professional-grade software suite developed for analyzing solar active regions (AR) by fusing data from two major space observatories: **Hinode (SOT/SP)** and **SDO (HMI)**.

##  Project Overview
The goal of this project is to predict Coronal Mass Ejections (CMEs) and solar flares by calculating the **Magnetic Free Energy ($E_{free}$)**.

##  Key Features
- **Hinode Spectropolarimetry Analysis:** Implements the "Sigma-V" method to estimate longitudinal magnetic field strength using the Zeeman effect.
- **NLFFF Extrapolation:** A 3D Non-Linear Force-Free Field modeling tool that reconstructs coronal magnetic loops from photospheric magnetograms.
- **Energy Budgeting:** Automated calculation of Potential and NLFFF energies to derive the free energy available for solar eruptions.
- **Numerical Stability:** Features advanced divergence cleaning ($\nabla \cdot \mathbf{B} = 0$), weighted flux balancing, and high-order finite difference schemes.

##  Scientific Methodology
1. **Calibration:** We use high-resolution Hinode data as a "ground truth" to anchor and scale global SDO/HMI magnetograms.
2. **Modeling:** The NLFFF relaxation algorithm evolves the magnetic field into a state of minimum Lorentz force.

##  Technical Stack
- **Language:** Python 3.10+
- **Libraries:** `NumPy` (high-performance computing), `Astropy` (FITS I/O and WCS), `SciPy` (signal processing), `Matplotlib` (visualization).
- **Data Sources:** Hinode SP Level 2, SDO/HMI SHARP series.

##  Repository Structure
- `analyzer_v1.0.1.py`: Processing Hinode spectropolarimetry data.
- `extrapolation_v1.0.1.py`: 3D Magnetic field reconstruction and energy analysis.
- `requirements.txt`: List of necessary Python dependencies.
- `examples/`: Visualizations of reconstructed magnetic loops.

---
**Developed by:** spectral-hooper
