
# Bi-Lipschitz Contracting Dynamics

Code for the paper **"Bi-Lipschitz Contracting Dynamics: A Stable, Invertible, and Re-shapeable Learned System in Continuous Time"**  
Jonas Ulmen · Jonas Landeck · Ganesh Sundaram · Ernesto Dickel Saraiva · Daniel Görges

---

## Overview

A learned continuous-time dynamical system whose symmetric-Jacobian eigenvalues are held inside a two-sided band by construction. The upper band edge makes the system contracting (stable convergence); the lower edge keeps backward expansion bounded, making exact time inversion numerically safe. A free skew-symmetric term adds rotation without disturbing the contraction rate.

---

## Repository structure

```
train/          Training scripts
  run_ct_noenc.py     CT model without encoder     → ct_noenc_<Shape>_seed<s>.json/.pth
  run_ct_enc.py       CT model + RQ-spline encoder → ct_enc_<Shape>_seed<s>.json/.pth
  run_ct_bilip.py     CT BiLip-MLP (std + bilip, fixed seed) → results_bilip_dtwd/

scripts/        Figure generation (pgfplots / tikz output)
  make_e1_enc_tikz.py     E1 — convergence
  make_e2b_tikz.py        E2 — bound shaping
  make_e3_tikz.py         E3 — time inversion
  make_e4_tikz.py         E4 — skew study
  make_e5_tikz.py         E5 — state-variation ablation (Table III / Fig. 5)
  make_teaser_tikz.py     Teaser figure
  make_appendix_tikz.py   Appendix figures

src/            Model and utilities
  bilip_elcd.py     ELCDOrthogonalCTGatedK — main model class
  bilip_layers.py   Bi-Lipschitz MLP layers
  data.py           LASA data loading and normalisation
  utils.py          ELCD_Transform (encoder wrapper), generate_mf_transform, helpers
  acfr/             Bi-Lipschitz network implementation (ACFR)
  manifold_flow/    RQ-spline normalising flow (required by run_ct_enc.py)

docs/           Project page (GitHub Pages)
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

All scripts write results to the same directory they are run from.

```bash
# CT without encoder — 10 seeds × 30 LASA shapes
python train/run_ct_noenc.py --workers 4

# CT with RQ-spline encoder
python train/run_ct_enc.py --workers 4

# CT BiLip-MLP (std MLP vs BiLip-MLP, single seed, all 30 shapes)
python train/run_ct_bilip.py

# Subset
python train/run_ct_noenc.py --shapes Sshape GShape --seeds 0 1 2
```

---

## Reproducing figures

```bash
python scripts/make_e1_enc_tikz.py
python scripts/make_e5_tikz.py
# etc.
```

Output is written to `figures/tikz/`.
