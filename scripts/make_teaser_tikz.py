#!/usr/bin/env python3
"""
Teaser figure as pgfplots tikz — Sine, CT+encoder, seed 1.
Quiver field + demo (orange) + forward rollout (blue dashed) + backward (red dashed).
"""
import os, sys
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, '..', 'src'))
from bilip_elcd      import ELCDOrthogonalCTGatedK
from utils      import generate_mf_transform, ELCD_Transform

from data            import lasa_to_torch, DiscreteNormalisedTrajData

SHAPE      = 'Sine'
SEED       = 1
FLOW_STEPS = 2
HIDDEN_DIM = 64;  N_HIDDEN = 2
ALPHA      = 1.0
LAMBDA_MIN = 0.5;  LAMBDA_MAX = 2.0;  OMEGA_MAX = 0.3
NUM_DEMOS  = 3
DT         = 0.01
FWD_STEPS  = 2000
CONV_TOL   = 1e-3
QUIVER_N   = 20
DEVICE     = torch.device('cpu')

MODEL_PATH = os.path.abspath(os.path.join(THIS_DIR, '..', 'train', f'ct_enc_{SHAPE}_seed{SEED}.pth'))
OUT_DIR    = os.path.join(THIS_DIR, 'figures', 'tikz')
os.makedirs(OUT_DIR, exist_ok=True)


def build_model(dim, x_eq):
    kern = ELCDOrthogonalCTGatedK(
        dim=dim, action_dim=1, alpha=ALPHA,
        lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX, omega_max=OMEGA_MAX,
        hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN,
    )
    flow = generate_mf_transform(flow_steps=FLOW_STEPS, input_dim=dim, latent_dim=dim)
    return ELCD_Transform(x_eq=x_eq.cpu(), transform=flow, model=kern).to(DEVICE)


def field_batch(pts):
    x = torch.tensor(pts, dtype=torch.float32)
    with torch.enable_grad():
        dx = model(x, spec_jacobian=True)
    return dx.detach().numpy()


def field_single(x_np):
    x = torch.tensor(x_np[None], dtype=torch.float32)
    with torch.enable_grad():
        dx = model(x, spec_jacobian=True)
    return dx[0].detach().numpy()


def rollout_rk4(x0_np, x_eq_np, sign=1, steps=FWD_STEPS):
    x    = x0_np.copy().astype(np.float64)
    traj = [x.copy()]
    for _ in range(steps - 1):
        k1 = sign * field_single(x)
        k2 = sign * field_single(x + 0.5*DT*k1)
        k3 = sign * field_single(x + 0.5*DT*k2)
        k4 = sign * field_single(x + DT*k3)
        x  = x + (DT / 6) * (k1 + 2*k2 + 2*k3 + k4)
        if not np.isfinite(x).all(): break
        traj.append(x.copy())
        if np.linalg.norm(x - x_eq_np) < CONV_TOL: break
    return np.array(traj)


def table(arr):
    return '\n'.join(f'{r[0]:.6f} {r[1]:.6f}' for r in arr)


# ── Load ──────────────────────────────────────────────────────────────────────
torch.manual_seed(SEED); np.random.seed(SEED)
pos, vel = lasa_to_torch(SHAPE)
ds       = DiscreteNormalisedTrajData(pos, vel, num_demos=NUM_DEMOS)
x_eq     = ds.pos_eq[0:1].float()
x_eq_np  = x_eq[0].numpy()
pos_norm = (pos.float() / ds.pos_std.unsqueeze(-1))
ref_np   = pos_norm[0].T.numpy()

model = build_model(pos.shape[1], x_eq)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
model.eval()

# ── Quiver field ───────────────────────────────────────────────────────────────
print('Vector field...', flush=True)
gx = np.linspace(-4.5, 4.5, QUIVER_N)
gy = np.linspace(-4.5, 4.5, QUIVER_N)
GX, GY   = np.meshgrid(gx, gy)
quiv_pts = np.stack([GX.ravel(), GY.ravel()], 1).astype(np.float32)
fv       = field_batch(quiv_pts)
nrm      = np.linalg.norm(fv, axis=-1, keepdims=True)
fvn      = fv / np.maximum(nrm, 1e-8)   # normalized, uniform arrow length

ARROW_SCALE = 0.35   # arrow length in data units

# ── Rollouts ───────────────────────────────────────────────────────────────────
print('Forward + backward rollout...', flush=True)
fwd = rollout_rk4(ref_np[0], x_eq_np, sign=+1)
bwd = rollout_rk4(fwd[-1],   x_eq_np, sign=-1, steps=len(fwd))
print(f'  fwd={len(fwd)} steps, bwd={len(bwd)} steps')
print(f'  retrace_err={np.linalg.norm(bwd[-1] - ref_np[0]):.2e}')

# ── Build tikz ────────────────────────────────────────────────────────────────
quiver_rows = '\n'.join(
    f'{quiv_pts[i,0]:.5f} {quiv_pts[i,1]:.5f} {fvn[i,0]*ARROW_SCALE:.5f} {fvn[i,1]*ARROW_SCALE:.5f}'
    for i in range(len(quiv_pts))
)

tikz = r"""\begin{tikzpicture}
\begin{axis}[
  width=7cm, height=7cm,
  xmin=-4.5, xmax=4.5,
  ymin=-4.5, ymax=4.5,
  xtick={-4,-2,0,2,4},
  ytick={-4,-2,0,2,4},
  tick label style={font=\footnotesize},
  label style={font=\footnotesize},
  legend style={font=\footnotesize, at={(0.98,0.98)}, anchor=north east},
  legend cell align={left},
  axis equal image,
  grid=major,
  grid style={gray!20},
]

%% quiver field
\addplot[
  gray!55,
  quiver={u=\thisrow{u}, v=\thisrow{v}, scale arrows=1},
  -stealth,
  line width=0.3pt,
  every arrow/.append style={line width=0.3pt},
] table [col sep=space] {
x y u v
""" + quiver_rows + r"""
};

%% demonstration
\addplot[thick, color={rgb,1:red,1;green,0.435;blue,0}, line width=1.8pt]
table [col sep=space] {
x y
""" + table(ref_np) + r"""
};
\addlegendentry{demonstration}

%% forward rollout
\addplot[dashed, color={rgb,1:red,0.082;green,0.396;blue,0.753}, line width=1.4pt]
table [col sep=space] {
x y
""" + table(fwd) + r"""
};
\addlegendentry{rollout}

%% backward retrace
\addplot[dashed, color={rgb,1:red,0.776;green,0.157;blue,0.157}, line width=1.4pt]
table [col sep=space] {
x y
""" + table(bwd) + r"""
};
\addlegendentry{backward}

%% equilibrium
\addplot[only marks, mark=asterisk, mark size=4pt, black]
coordinates {(""" + f'{x_eq_np[0]:.4f},{x_eq_np[1]:.4f}' + r""")};

\end{axis}
\end{tikzpicture}
"""

out = os.path.join(OUT_DIR, f'teaser_{SHAPE}_seed{SEED}.tikz')
with open(out, 'w') as f:
    f.write(tikz)
print(f'→ {out}')
