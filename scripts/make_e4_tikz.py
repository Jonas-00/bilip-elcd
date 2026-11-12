#!/usr/bin/env python3
"""
E4 — Skew Study as pgfplots tikz.
CT+encoder, Sshape seed 1.
Two panels: (a) rollouts vs skew gain, (b) certified rate & DTWD.
Output: figures/tikz/e4_skew_rollouts.tikz
        figures/tikz/e4_skew_rate_dtwd.tikz
"""
import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, '..', 'src'))
from bilip_elcd      import ELCDOrthogonalCTGatedK
from utils      import generate_mf_transform, ELCD_Transform

from data            import lasa_to_torch, NormalisedTrajData

SHAPE      = 'Sshape'
SEED       = 1
FLOW_STEPS = 2
HIDDEN_DIM = 64;  N_HIDDEN = 2
ALPHA      = 1.0
LAMBDA_MIN = 0.5;  LAMBDA_MAX = 2.0;  OMEGA_MAX = 0.3
NUM_DEMOS  = 3
DT         = 0.01
ROLLOUT_STEPS = 2000
CONV_TOL   = 1e-5
DEVICE     = torch.device('cpu')

PLOT_GAINS  = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
SWEEP_GAINS = np.concatenate([np.linspace(0.0, 1.0, 11),
                               np.linspace(1.0, 5.0, 9)[1:]])

MODEL_PATH = os.path.abspath(os.path.join(THIS_DIR, '..', 'train', f'ct_enc_{SHAPE}_seed{SEED}.pth'))
OUT_DIR    = os.path.join(THIS_DIR, 'figures', 'tikz')
os.makedirs(OUT_DIR, exist_ok=True)

TRAJ_STEP = 3


def build_model(dim, x_eq):
    kern = ELCDOrthogonalCTGatedK(
        dim=dim, action_dim=1, alpha=ALPHA,
        lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX, omega_max=OMEGA_MAX,
        hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN,
    )
    flow = generate_mf_transform(flow_steps=FLOW_STEPS, input_dim=dim, latent_dim=dim)
    return ELCD_Transform(x_eq=x_eq.cpu(), transform=flow, model=kern).to(DEVICE)


@torch.no_grad()
def get_A_scaled(kern, z, a, z_star, k_gain):
    B   = z.shape[0]
    inp = torch.cat([z, a, z_star], dim=-1)
    S_Q = kern.S_Q_net(inp).view(B, kern.dim, kern.dim)
    Q   = kern._cayley_batch(S_Q)
    lam_min = kern._lambda_min_eff()
    lam_hat = kern.lambda_net(inp)
    lambdas = lam_min + (kern.lambda_max - lam_min) * torch.sigmoid(lam_hat)
    Lambda  = torch.diag_embed(lambdas)
    I = torch.eye(kern.dim, dtype=z.dtype).unsqueeze(0)
    S = -torch.bmm(torch.bmm(Q.transpose(-2, -1), Lambda), Q) - kern.alpha * I
    omega_hat = kern.omega_net(inp)
    omegas    = kern.omega_max * torch.tanh(omega_hat)
    K         = kern._build_K_batch(omegas)
    return S + k_gain * K


def rollout(model, x0, k_gain=1.0, sign=1):
    x    = x0.float()
    kern = model.model
    a    = torch.zeros(1, 1)
    with torch.no_grad():
        z_eq = model.encode(model.x_eq)
    x_eq_np = model.x_eq.squeeze(0).numpy()
    traj = [x[0].numpy().copy()]

    def step(xp):
        with torch.enable_grad():
            z     = model.encode(xp)
            A     = get_A_scaled(kern, z, a, z_eq, k_gain)
            z_dot = torch.bmm(A, (z - z_eq).unsqueeze(-1)).squeeze(-1)
            _, inv_jac = model.transform.inverse(z, full_jacobian=True)
        return torch.bmm(inv_jac, z_dot.unsqueeze(-1)).squeeze(-1).detach()

    for _ in range(ROLLOUT_STEPS - 1):
        k1 = sign * step(x)
        k2 = sign * step(x + 0.5 * DT * k1)
        k3 = sign * step(x + 0.5 * DT * k2)
        k4 = sign * step(x + DT * k3)
        x  = x + (DT / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        pt = x[0].numpy()
        if not np.isfinite(pt).all():
            break
        traj.append(pt.copy())
        if sign == 1 and np.linalg.norm(pt - x_eq_np) < CONV_TOL:
            break
    return np.array(traj)


def dtwd(ro, ref):
    r = torch.tensor(ro,  dtype=torch.float32)
    d = torch.tensor(ref, dtype=torch.float32)
    dists = torch.cdist(r, d)
    return (dists.min(0).values.mean() + dists.min(1).values.mean()).item()


def tbl(arr, step=1):
    sub = arr[::step]
    return '\n'.join(f'{r[0]:.5f} {r[1]:.5f}' for r in sub)


def rgba_to_pgf(rgba):
    r, g, b, _ = rgba
    return f'{{rgb,1:red,{r:.3f};green,{g:.3f};blue,{b:.3f}}}'


# ── Load ──────────────────────────────────────────────────────────────────────
torch.manual_seed(SEED); np.random.seed(SEED)
pos, vel = lasa_to_torch(SHAPE)
ds       = NormalisedTrajData(pos, vel, num_demos=NUM_DEMOS)
x_eq     = ds.pos_eq[0:1].float()
x_eq_np  = x_eq[0].numpy()
pos_norm = pos.float() / ds.pos_std.unsqueeze(-1)
ref_np   = pos_norm[0].T.numpy()
x0       = pos_norm[0, :, 0:1].T.float()

model = build_model(pos.shape[1], x_eq)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
model.eval()

rate = model.model.get_contraction_rate()
print(f'Certified contraction rate = {rate:.4f}')

# ── Axis limits (left panel) ──────────────────────────────────────────────────
cx = (ref_np[:,0].min() + ref_np[:,0].max()) / 2
cy = (ref_np[:,1].min() + ref_np[:,1].max()) / 2
r  = max(ref_np[:,0].max()-ref_np[:,0].min(),
         ref_np[:,1].max()-ref_np[:,1].min()) / 2 * 1.4 + 0.15
xlo, xhi = cx-r, cx+r
ylo, yhi = cy-r, cy+r

# ── Rollouts at plot gains ────────────────────────────────────────────────────
cmap   = plt.cm.viridis
colors = [rgba_to_pgf(cmap(t)) for t in np.linspace(0.05, 0.95, len(PLOT_GAINS))]

ro_arrays  = []
ro_tables  = []
bwd_tables = []
for gain in PLOT_GAINS:
    ro = rollout(model, x0, k_gain=gain, sign=1)
    ro_arrays.append(ro)
    ro_tables.append(tbl(ro, TRAJ_STEP))
    # backward from endpoint of forward rollout
    x_end = torch.tensor(ro[-1][None], dtype=torch.float32)
    bwd   = rollout(model, x_end, k_gain=gain, sign=-1)
    bwd_tables.append(tbl(bwd, TRAJ_STEP))
    print(f'  gain={gain:.2g}  fwd={len(ro)}  bwd={len(bwd)}')

# ── DTWD sweep ────────────────────────────────────────────────────────────────
print('DTWD sweep...', flush=True)
dtwd_vals = []
for g in SWEEP_GAINS:
    ro = rollout(model, x0, k_gain=g)
    dtwd_vals.append(dtwd(ro, ref_np) if np.isfinite(ro).all() else float('nan'))
    print(f'  gain={g:.3f}  dtwd={dtwd_vals[-1]:.4f}')
dtwd_arr = np.array(dtwd_vals)

# ── Panel (a): rollouts ───────────────────────────────────────────────────────
legend_entries = '\n'.join(
    f'\\addlegendentry{{gain={g:.2g}}}'
    for g in PLOT_GAINS
)

rollout_plots = '\n'.join(
    f'\\addplot[color={col}, line width=0.9pt, dashed]\n'
    f'table [col sep=space] {{\nx y\n{ro_tbl}\n}};\n'
    + f'\\addlegendentry{{gain={g:.2g}}}'
    for col, ro_tbl, g in zip(colors, ro_tables, PLOT_GAINS)
)

tikz_a = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=6cm, height=6cm,
  xmin={xlo:.3f}, xmax={xhi:.3f},
  ymin={ylo:.3f}, ymax={yhi:.3f},
  axis equal image,
  tick label style={{font=\\footnotesize}},
  label style={{font=\\footnotesize}},
  grid=major, grid style={{gray!20}},
  legend style={{font=\\tiny, at={{(0.98,0.98)}}, anchor=north east, inner sep=1.5pt}},
  legend cell align={{left}},
  unbounded coords=jump,
]
\\addplot[color={{rgb,1:red,1;green,0.435;blue,0}}, line width=1.6pt]
table [col sep=space] {{
x y
{tbl(ref_np, 5)}
}};
\\addlegendentry{{demo}}
{rollout_plots}
\\addplot[only marks, mark=asterisk, mark size=3pt, black]
coordinates {{({x_eq_np[0]:.4f},{x_eq_np[1]:.4f})}};
\\end{{axis}}
\\end{{tikzpicture}}
"""

out_a = os.path.join(OUT_DIR, 'e4_skew_rollouts.tikz')
with open(out_a, 'w') as f:
    f.write(tikz_a)
print(f'→ {out_a}')

# ── Panel backward: backward traces from endpoint ────────────────────────────
bwd_plots = '\n'.join(
    f'\\addplot[color={col}, line width=0.9pt, dashed]\n'
    f'table [col sep=space] {{\nx y\n{bwd_tbl}\n}};\n'
    + f'\\addlegendentry{{gain={g:.2g}}}'
    for col, bwd_tbl, g in zip(colors, bwd_tables, PLOT_GAINS)
)

tikz_bwd = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=6cm, height=6cm,
  xmin={xlo:.3f}, xmax={xhi:.3f},
  ymin={ylo:.3f}, ymax={yhi:.3f},
  axis equal image,
  tick label style={{font=\\footnotesize}},
  label style={{font=\\footnotesize}},
  grid=major, grid style={{gray!20}},
  legend style={{font=\\tiny, at={{(0.98,0.98)}}, anchor=north east, inner sep=1.5pt}},
  legend cell align={{left}},
  unbounded coords=jump,
]
\\addplot[color={{rgb,1:red,1;green,0.435;blue,0}}, line width=1.6pt]
table [col sep=space] {{
x y
{tbl(ref_np, 5)}
}};
\\addlegendentry{{demo}}
{bwd_plots}
\\addplot[only marks, mark=asterisk, mark size=3pt, black]
coordinates {{({x_eq_np[0]:.4f},{x_eq_np[1]:.4f})}};
\\end{{axis}}
\\end{{tikzpicture}}
"""

out_bwd = os.path.join(OUT_DIR, 'e4_skew_backward.tikz')
with open(out_bwd, 'w') as f:
    f.write(tikz_bwd)
print(f'→ {out_bwd}')

# ── Panel (b): certified rate & DTWD (dual y-axis) ───────────────────────────
mask      = np.isfinite(dtwd_arr)
sg_valid  = SWEEP_GAINS[mask]
dw_valid  = dtwd_arr[mask]
dtwd_min  = float(dw_valid.min())
dtwd_max  = float(dw_valid.max())
dtwd_tbl  = '\n'.join(f'{g:.4f} {d:.6f}' for g, d in zip(sg_valid, dw_valid))

rate_ymax = rate * 2.5
dtwd_ymax = dtwd_max * 1.2

# scale DTWD to rate y-axis for overlay
scale = rate_ymax / dtwd_ymax
dtwd_scaled_tbl = '\n'.join(f'{g:.4f} {d*scale:.6f}' for g, d in zip(sg_valid, dw_valid))
ytick_dtwd = np.linspace(0, dtwd_ymax, 5)
ytick_rate = ytick_dtwd * scale
ytick_labels = ' '.join(f'{v:.2f}' for v in ytick_dtwd)
ytick_pos    = ' '.join(f'{v:.4f}' for v in ytick_rate)

tikz_b = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=6cm, height=4.5cm,
  xmin={SWEEP_GAINS[0]-0.05:.2f}, xmax={SWEEP_GAINS[-1]+0.05:.2f},
  ymin=0, ymax={rate_ymax:.4f},
  xlabel={{Skew gain}},
  ylabel={{Contraction rate}},
  ylabel style={{text color={{rgb,1:red,0.082;green,0.396;blue,0.753}}}},
  ytick align=outside,
  ytick style={{draw={{rgb,1:red,0.082;green,0.396;blue,0.753}}}},
  yticklabel style={{text color={{rgb,1:red,0.082;green,0.396;blue,0.753}}}},
  tick label style={{font=\\footnotesize}},
  label style={{font=\\footnotesize}},
  grid=major, grid style={{gray!20}},
  legend style={{font=\\footnotesize, at={{(0.98,0.98)}}, anchor=north east}},
  legend cell align={{left}},
  axis y line*=left,
  axis x line*=bottom,
]
\\addplot[color={{rgb,1:red,0.082;green,0.396;blue,0.753}}, line width=1.4pt]
coordinates {{({SWEEP_GAINS[0]-0.05:.2f},{rate:.4f}) ({SWEEP_GAINS[-1]+0.05:.2f},{rate:.4f})}};
\\addlegendentry{{cert.~rate $= {rate:.3f}$}}
\\end{{axis}}

\\begin{{axis}}[
  width=6cm, height=4.5cm,
  xmin={SWEEP_GAINS[0]-0.05:.2f}, xmax={SWEEP_GAINS[-1]+0.05:.2f},
  ymin=0, ymax={dtwd_ymax:.4f},
  ylabel={{DTWD}},
  ylabel style={{text color={{rgb,1:red,0.776;green,0.157;blue,0.157}}}},
  ytick align=outside,
  ytick style={{draw={{rgb,1:red,0.776;green,0.157;blue,0.157}}}},
  yticklabel style={{text color={{rgb,1:red,0.776;green,0.157;blue,0.157}}}},
  tick label style={{font=\\footnotesize}},
  label style={{font=\\footnotesize}},
  legend style={{font=\\footnotesize, at={{(0.98,0.80)}}, anchor=north east}},
  legend cell align={{left}},
  axis y line*=right,
  axis x line=none,
]
\\addplot[color={{rgb,1:red,0.776;green,0.157;blue,0.157}}, line width=1.4pt,
  mark=*, mark size=1.5pt]
table [col sep=space] {{
x y
{dtwd_tbl}
}};
\\addlegendentry{{DTWD}}
\\end{{axis}}
\\end{{tikzpicture}}
"""

out_b = os.path.join(OUT_DIR, 'e4_skew_rate_dtwd.tikz')
with open(out_b, 'w') as f:
    f.write(tikz_b)
print(f'→ {out_b}')

print('\nDone.')
