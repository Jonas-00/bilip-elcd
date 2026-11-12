#!/usr/bin/env python3
"""
E3 — Time Inversion, CT+encoder, Sshape seed 1.
Generates two tikz panels (forward, backward) for the paper.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tikzplotlib

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, '..', 'src'))
from bilip_elcd      import ELCDOrthogonalCTGatedK
from utils      import generate_mf_transform, ELCD_Transform

from data            import lasa_to_torch, DiscreteNormalisedTrajData

SHAPE      = 'Sshape'
SEED       = 1
FLOW_STEPS = 2
HIDDEN_DIM = 64;  N_HIDDEN = 2
ALPHA      = 1.0
LAMBDA_MIN = 0.5;  LAMBDA_MAX = 2.0;  OMEGA_MAX = 0.3
NUM_DEMOS  = 3
DT         = 0.01
N          = 150
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


def field(x_np, sign=1):
    x = torch.tensor(x_np[None], dtype=torch.float32)
    with torch.enable_grad():
        dx = model(x, spec_jacobian=True)
    return sign * dx[0].detach().numpy()


def rk4(x0, sign, steps):
    x = x0.copy().astype(np.float64)
    traj = [x.copy()]
    for _ in range(steps - 1):
        k1 = field(x, sign);  k2 = field(x + 0.5*DT*k1, sign)
        k3 = field(x + 0.5*DT*k2, sign);  k4 = field(x + DT*k3, sign)
        x  = x + (DT / 6) * (k1 + 2*k2 + 2*k3 + k4)
        traj.append(x.copy())
    return np.array(traj)


def save_tikz(fig, path):
    tikzplotlib.save(path, figure=fig,
        axis_width='7cm', axis_height='7cm',
        extra_axis_parameters=[
            'tick label style={font=\\footnotesize}',
            'label style={font=\\footnotesize}',
            'legend style={font=\\footnotesize}',
        ])
    print(f'  → {path}')


# ── Load ──────────────────────────────────────────────────────────────────────
torch.manual_seed(SEED); np.random.seed(SEED)
pos, vel = lasa_to_torch(SHAPE)
ds       = DiscreteNormalisedTrajData(pos, vel, num_demos=NUM_DEMOS)
x_eq     = ds.pos_eq[0:1].float()
pn       = pos.float() / ds.pos_std.unsqueeze(-1)
ref_np   = pn[0].T.numpy()   # full demo (T, 2)
x0_np    = pn[0, :, 0].numpy()

model = build_model(pos.shape[1], x_eq)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
model.eval()

# ── Rollouts ──────────────────────────────────────────────────────────────────
print(f'Forward rollout N={N}...', flush=True)
fwd = rk4(x0_np, +1, N)

print('Backward retrace...', flush=True)
bwd = rk4(fwd[-1], -1, N)
err = float(np.linalg.norm(bwd[-1] - x0_np))
print(f'  retrace_err = {err:.2e}')

# axis limits from demo
xl = (ref_np[:,0].min() - 0.3, ref_np[:,0].max() + 0.3)
yl = (ref_np[:,1].min() - 0.3, ref_np[:,1].max() + 0.3)

# ── Panel (a): demo + forward ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.3, 3.3))
ax.plot(ref_np[:,0], ref_np[:,1], color='#FF6F00', lw=2.0, label='demonstration')
ax.plot(fwd[:,0],   fwd[:,1],    color='#1565C0', lw=1.5, ls='--', label='rollout')
ax.plot(0, 0, 'k*', ms=9, zorder=10)
ax.set_xlim(*xl);  ax.set_ylim(*yl)
ax.set_aspect('equal');  ax.grid(True, alpha=0.2)
ax.legend(loc='upper right')
fig.tight_layout()
save_tikz(fig, os.path.join(OUT_DIR, 'e3_forward_sshape.tikz'))
plt.close(fig)

# ── Panel (b): forward + backward ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.3, 3.3))
ax.plot(ref_np[:,0], ref_np[:,1], color='#FF6F00', lw=2.0, label='demonstration')
ax.plot(fwd[:,0],   fwd[:,1],    color='#1565C0', lw=1.5,       label='forward')
ax.plot(bwd[:,0],   bwd[:,1],    color='#C62828', lw=1.5, ls='--', label='retrace')
ax.plot(0, 0, 'k*', ms=9, zorder=10)
ax.set_xlim(*xl);  ax.set_ylim(*yl)
ax.set_aspect('equal');  ax.grid(True, alpha=0.2)
ax.legend(loc='upper right')
fig.tight_layout()
save_tikz(fig, os.path.join(OUT_DIR, 'e3_backward_sshape.tikz'))
plt.close(fig)

print(f'\nretrace_err = {err:.2e}')
print('Done.')
