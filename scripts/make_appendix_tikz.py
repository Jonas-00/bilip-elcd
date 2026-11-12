#!/usr/bin/env python3
"""
Appendix — all 30 LASA shapes, CT+encoder seed 1.
One tikz file per shape: demo (orange) + rollout (blue dashed) + equilibrium.
Output: figures/tikz/appendix_<Shape>.tikz
"""
import os, sys, json
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, '..', 'src'))
from bilip_elcd  import ELCDOrthogonalCTGatedK
from utils      import generate_mf_transform, ELCD_Transform

from data        import lasa_to_torch, DiscreteNormalisedTrajData

ALL_SHAPES = [
    'Angle', 'BendedLine', 'CShape', 'DoubleBendedLine', 'GShape',
    'JShape', 'JShape_2', 'Khamesh', 'LShape', 'Leaf_1', 'Leaf_2', 'Line',
    'Multi_Models_1', 'Multi_Models_2', 'Multi_Models_3', 'Multi_Models_4',
    'NShape', 'PShape', 'RShape', 'Saeghe', 'Sharpc', 'Sine', 'Snake',
    'Spoon', 'Sshape', 'Trapezoid', 'WShape', 'Worm', 'Zshape', 'heee',
]

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
TRAJ_STEP  = 10
MODEL_DIR  = os.path.abspath(os.path.join(THIS_DIR, '..', 'train'))
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


def rollout(model, x0_np, x_eq_np):
    x    = x0_np.copy().astype(np.float64)
    traj = [x.copy()]
    for _ in range(ROLLOUT_STEPS - 1):
        with torch.enable_grad():
            xt = torch.tensor(x[None], dtype=torch.float32)
            xd = model(xt, spec_jacobian=True)[0].detach().numpy()
        x = x + DT * xd
        if not np.isfinite(x).all():
            break
        traj.append(x.copy())
        if np.linalg.norm(x - x_eq_np) < CONV_TOL:
            break
    return np.array(traj)


def table(arr, step=1):
    sub = arr[::step]
    return '\n'.join(f'{r[0]:.5f} {r[1]:.5f}' for r in sub)


for shape in ALL_SHAPES:
    pth = os.path.join(MODEL_DIR, f'ct_enc_{shape}_seed{SEED}.pth')
    if not os.path.exists(pth):
        print(f'[SKIP] {shape}'); continue

    with open(os.path.join(MODEL_DIR, f'ct_enc_{shape}_seed{SEED}.json')) as f:
        meta = json.load(f)
    dtwd = meta['dtwd_mean']

    pos, vel = lasa_to_torch(shape)
    ds       = DiscreteNormalisedTrajData(pos, vel, num_demos=NUM_DEMOS)
    x_eq     = ds.pos_eq[0:1].float()
    x_eq_np  = x_eq[0].numpy()
    pos_norm = pos.float() / ds.pos_std.unsqueeze(-1)
    ref_np   = pos_norm[0].T.numpy()
    x0_np    = pos_norm[0, :, 0].numpy()

    model = build_model(pos.shape[1], x_eq)
    model.load_state_dict(torch.load(pth, map_location=DEVICE, weights_only=False))
    model.eval()

    ro = rollout(model, x0_np, x_eq_np)

    # axis limits tight around demo + rollout
    all_pts = np.concatenate([ref_np, ro], axis=0)
    cx = (all_pts[:,0].min() + all_pts[:,0].max()) / 2
    cy = (all_pts[:,1].min() + all_pts[:,1].max()) / 2
    r  = max(all_pts[:,0].max()-all_pts[:,0].min(),
             all_pts[:,1].max()-all_pts[:,1].min()) / 2 * 1.15 + 0.05
    xlo, xhi = cx-r, cx+r
    ylo, yhi = cy-r, cy+r

    tikz = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=3.5cm, height=3.5cm,
  xmin={xlo:.3f}, xmax={xhi:.3f},
  ymin={ylo:.3f}, ymax={yhi:.3f},
  axis equal image,
  tick label style={{font=\\tiny}},
  grid=major, grid style={{gray!15}},
  title={{\\tiny {shape} DTWD={dtwd:.3f}}},
  title style={{font=\\tiny, yshift=-1mm}},
  legend style={{font=\\tiny, at={{(0.98,0.98)}}, anchor=north east, inner sep=1pt, row sep=-2pt}},
  legend cell align={{left}},
]
\\addplot[color={{rgb,1:red,1;green,0.435;blue,0}}, line width=1.2pt]
table [col sep=space] {{
x y
{table(ref_np, TRAJ_STEP)}
}};
\\addlegendentry{{demo}}
\\addplot[color={{rgb,1:red,0.082;green,0.396;blue,0.753}}, line width=1.0pt, dashed]
table [col sep=space] {{
x y
{table(ro, TRAJ_STEP)}
}};
\\addlegendentry{{rollout}}
\\addplot[only marks, mark=asterisk, mark size=2pt, black]
coordinates {{({x_eq_np[0]:.4f},{x_eq_np[1]:.4f})}};
\\end{{axis}}
\\end{{tikzpicture}}
"""

    out = os.path.join(OUT_DIR, f'appendix_{shape}.tikz')
    with open(out, 'w') as f:
        f.write(tikz)
    print(f'  {shape}  steps={len(ro)}  DTWD={dtwd:.3f}  → {out}', flush=True)

print('Done.')
