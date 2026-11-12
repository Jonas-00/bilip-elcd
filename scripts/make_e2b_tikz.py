#!/usr/bin/env python3
"""
Fig. 5(b) — Lambda scaling study, CT+encoder, Sshape seed 1.
Post-hoc scale Lambda by different factors; same K (rotation).
Markers every MARKER_STEP steps show time spacing (speed).
Output: figures/tikz/e2b_lambda_rollouts.tikz
"""
import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, '..', 'src'))

from bilip_elcd import ELCDOrthogonalCTGatedK
from utils      import generate_mf_transform, ELCD_Transform
from data       import lasa_to_torch, NormalisedTrajData

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

LAMBDA_SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]
TRAJ_STEP     = 3
MARKER_STEP   = 50   # marker every N steps → density shows speed

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


@torch.no_grad()
def get_A_lambda_scaled(kern, z, z_eq, lam_scale):
    """A(z) with Lambda scaled by lam_scale, K unchanged."""
    a   = torch.zeros(z.shape[0], 1)
    inp = torch.cat([z, a, z_eq], dim=-1)
    S_Q = kern.S_Q_net(inp).view(z.shape[0], kern.dim, kern.dim)
    Q   = kern._cayley_batch(S_Q)
    lam_min  = kern._lambda_min_eff()
    lam_hat  = kern.lambda_net(inp)
    lambdas  = lam_min + (kern.lambda_max - lam_min) * torch.sigmoid(lam_hat)
    lambdas  = lambdas * lam_scale          # scale eigenvalues
    Lambda   = torch.diag_embed(lambdas)
    I        = torch.eye(kern.dim, dtype=z.dtype).unsqueeze(0)
    S        = -torch.bmm(torch.bmm(Q.transpose(-2, -1), Lambda), Q) - kern.alpha * I
    omega_hat = kern.omega_net(inp)
    omegas    = kern.omega_max * torch.tanh(omega_hat)
    K         = kern._build_K_batch(omegas)
    return S + K


def step_scaled(model, x, z_eq, lam_scale):
    with torch.enable_grad():
        z = model.encode(x)
        A = get_A_lambda_scaled(model.model, z, z_eq, lam_scale)
        z_dot = torch.bmm(A, (z - z_eq).unsqueeze(-1)).squeeze(-1)
        _, inv_jac = model.transform.inverse(z, full_jacobian=True)
    return torch.bmm(inv_jac, z_dot.unsqueeze(-1)).squeeze(-1).detach()


def rollout(model, x0, z_eq, lam_scale):
    x    = x0.float()
    traj = [x[0].numpy().copy()]
    x_eq_np = model.x_eq.squeeze(0).numpy()
    for _ in range(ROLLOUT_STEPS - 1):
        k1 = step_scaled(model, x, z_eq, lam_scale)
        k2 = step_scaled(model, x + 0.5*DT*k1, z_eq, lam_scale)
        k3 = step_scaled(model, x + 0.5*DT*k2, z_eq, lam_scale)
        k4 = step_scaled(model, x + DT*k3,      z_eq, lam_scale)
        x  = x + (DT/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        pt = x[0].numpy()
        if not np.isfinite(pt).all(): break
        traj.append(pt.copy())
        if np.linalg.norm(pt - x_eq_np) < CONV_TOL: break
    return np.array(traj)


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

with torch.no_grad():
    z_eq = model.encode(x_eq)

# ── Axis limits ───────────────────────────────────────────────────────────────
cx = (ref_np[:,0].min() + ref_np[:,0].max()) / 2
cy = (ref_np[:,1].min() + ref_np[:,1].max()) / 2
r  = max(ref_np[:,0].max()-ref_np[:,0].min(),
         ref_np[:,1].max()-ref_np[:,1].min()) / 2 * 1.4 + 0.15
xlo, xhi = cx-r, cx+r
ylo, yhi = cy-r, cy+r

# ── Rollouts ──────────────────────────────────────────────────────────────────
cmap   = plt.cm.viridis
colors = [rgba_to_pgf(cmap(t)) for t in np.linspace(0.05, 0.95, len(LAMBDA_SCALES))]

ro_data = []
for scale in LAMBDA_SCALES:
    ro = rollout(model, x0, z_eq, scale)
    ro_data.append(ro)
    print(f'  scale={scale:.2f}  steps={len(ro)}')

# ── Build tikz ────────────────────────────────────────────────────────────────
rollout_plots = ''
for col, ro, scale in zip(colors, ro_data, LAMBDA_SCALES):
    line_tbl    = tbl(ro, TRAJ_STEP)
    marker_tbl  = tbl(ro, MARKER_STEP)
    rollout_plots += (
        f'\\addplot[color={col}, line width=0.9pt, dashed]\n'
        f'table [col sep=space] {{\nx y\n{line_tbl}\n}};\n'
        f'\\addlegendentry{{$\\times${scale:.2g}}}\n'
        f'\\addplot[color={col}, only marks, mark=*, mark size=1.2pt]\n'
        f'table [col sep=space] {{\nx y\n{marker_tbl}\n}};\n'
    )

tikz = f"""\\begin{{tikzpicture}}
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

out = os.path.join(OUT_DIR, 'e2b_lambda_rollouts.tikz')
with open(out, 'w') as f:
    f.write(tikz)
print(f'→ {out}')
print('Done.')
