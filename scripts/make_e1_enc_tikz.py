#!/usr/bin/env python3
"""
E1 — Skill Imitation plots + Table I metrics, CT+encoder, seed 1.
Shapes: Sshape, GShape, Angle.
Output: figures/tikz/e1_enc_<Shape>.tikz   + printed Table I values.
"""
import os, sys, json
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, '..', 'src'))
from bilip_elcd      import ELCDOrthogonalCTGatedK
from utils      import generate_mf_transform, ELCD_Transform

from data            import lasa_to_torch, DiscreteNormalisedTrajData

SHAPES     = ['Sshape', 'GShape', 'Angle']
MODEL_DIR  = os.path.abspath(os.path.join(THIS_DIR, '..', 'train'))
OUT_DIR    = os.path.join(THIS_DIR, 'figures', 'tikz')
os.makedirs(OUT_DIR, exist_ok=True)

HIDDEN_DIM = 64;  N_HIDDEN = 2
ALPHA      = 1.0
LAMBDA_MIN = 0.5;  LAMBDA_MAX = 2.0;  OMEGA_MAX = 0.3
FLOW_STEPS = 2
SEED       = 1
NUM_DEMOS  = 3
DT         = 0.01
ROLLOUT_STEPS = 2000
CONV_TOL   = 1e-5
GRID_N     = 8
QUIVER_N   = 12
ARROW_SCALE = 0.18
RETRACE_STEPS = 50
DEVICE     = torch.device('cpu')
TRAJ_STEP  = 40
DEMO_STEP  = 8


def build_model(dim, x_eq):
    kern = ELCDOrthogonalCTGatedK(
        dim=dim, action_dim=1, alpha=ALPHA,
        lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX, omega_max=OMEGA_MAX,
        hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN,
    )
    flow = generate_mf_transform(flow_steps=FLOW_STEPS, input_dim=dim, latent_dim=dim)
    return ELCD_Transform(x_eq=x_eq.cpu(), transform=flow, model=kern).to(DEVICE)


def field_single(model, x_np):
    x = torch.tensor(x_np[None], dtype=torch.float32)
    with torch.enable_grad():
        dx = model(x, spec_jacobian=True)
    return dx[0].detach().numpy()


def rollout(model, x0_np, x_eq_np, sign=1):
    x    = x0_np.copy().astype(np.float64)
    traj = [x.copy()]
    for _ in range(ROLLOUT_STEPS - 1):
        k1 = sign * field_single(model, x)
        k2 = sign * field_single(model, x + 0.5*DT*k1)
        k3 = sign * field_single(model, x + 0.5*DT*k2)
        k4 = sign * field_single(model, x + DT*k3)
        x  = x + (DT/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        if not np.isfinite(x).all(): break
        traj.append(x.copy())
        if sign == 1 and np.linalg.norm(x - x_eq_np) < CONV_TOL: break
    return np.array(traj)


AXLIM = 2.5  # axis range — points outside get nan (shown as break, resume when back in)

def traj_table(trajs):
    rows = []
    for t in trajs:
        sub = t[::TRAJ_STEP]
        if len(t) % TRAJ_STEP != 1:
            sub = np.concatenate([sub, t[-1:]])
        prev_out = False
        for pt in sub:
            out = np.abs(pt).max() > AXLIM
            if out:
                if not prev_out:
                    rows.append('nan nan')
                prev_out = True
            else:
                if prev_out:
                    rows.append('nan nan')
                rows.append(f'{pt[0]:.4f} {pt[1]:.4f}')
                prev_out = False
        rows.append('nan nan')
    return '\n'.join(rows)


def single_table(arr):
    sub = arr[::DEMO_STEP]
    return '\n'.join(f'{r[0]:.4f} {r[1]:.4f}' for r in sub)


def compute_sea(ro, ref):
    """Mean min-distance from rollout to demo, summed over rollout steps * DT."""
    r = torch.tensor(ro,  dtype=torch.float32)
    d = torch.tensor(ref, dtype=torch.float32)
    dists = torch.cdist(r, d)
    return float(dists.min(1).values.sum().item() * DT)


def compute_retrace(model, x0_np, x_eq_np):
    """RK4 forward RETRACE_STEPS then backward RETRACE_STEPS, measure endpoint error."""
    fwd = rollout(model, x0_np, x_eq_np, sign=1)
    fwd = fwd[:RETRACE_STEPS]
    if not np.isfinite(fwd[-1]).all():
        return float('nan')
    bwd = rollout(model, fwd[-1], x_eq_np, sign=-1)
    bwd = bwd[:RETRACE_STEPS]
    if not np.isfinite(bwd[-1]).all():
        return float('nan')
    return float(np.linalg.norm(bwd[-1] - x0_np))


def compute_conv_rate(ro, x_eq_np):
    """Empirical convergence rate: mean per-step decay of ||x - x_eq||."""
    dists = np.linalg.norm(ro - x_eq_np, axis=1)
    ratios = dists[1:] / np.maximum(dists[:-1], 1e-10)
    ratios = ratios[dists[:-1] > CONV_TOL * 10]
    return float(np.mean(ratios)) if len(ratios) > 0 else float('nan')


print('=== E1 CT+encoder — Table I metrics ===\n')
print(f'{"Shape":<10} {"DTWD":>8} {"SEA":>8} {"retrace":>12} {"conv_rate":>10}')

for shape in SHAPES:
    pth  = os.path.join(MODEL_DIR, f'ct_enc_{shape}_seed{SEED}.pth')
    jpath = os.path.join(MODEL_DIR, f'ct_enc_{shape}_seed{SEED}.json')
    if not os.path.exists(pth):
        print(f'{shape}: model not found'); continue

    pos, vel = lasa_to_torch(shape)
    ds       = DiscreteNormalisedTrajData(pos, vel, num_demos=NUM_DEMOS)
    x_eq     = ds.pos_eq[0:1].float()
    x_eq_np  = x_eq[0].numpy()
    pn       = pos.float() / ds.pos_std.unsqueeze(-1)
    ref_np   = pn[0].T.numpy()

    model = build_model(pos.shape[1], x_eq)
    model.load_state_dict(torch.load(pth, map_location=DEVICE, weights_only=False))
    model.eval()

    # DTWD from JSON
    dtwd = json.load(open(jpath))['dtwd_mean']

    # SEA and conv_rate from forward rollout
    x0_np  = ref_np[0]
    fwd_ro = rollout(model, x0_np, x_eq_np)
    sea    = compute_sea(fwd_ro, ref_np)
    crate  = compute_conv_rate(fwd_ro, x_eq_np)

    # Retrace
    retrace = compute_retrace(model, x0_np, x_eq_np)

    print(f'{shape:<10} {dtwd:>8.3f} {sea:>8.3f} {retrace:>12.2e} {crate:>10.4f}')

    # ── Grid rollouts ──────────────────────────────────────────────────────────
    gx = np.linspace(-2.5, 2.5, GRID_N)
    gy = np.linspace(-2.5, 2.5, GRID_N)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.stack([GX.ravel(), GY.ravel()], 1)
    grid_trajs = [rollout(model, pt, x_eq_np) for pt in grid_pts]
    grid_trajs = [t for t in grid_trajs if len(t) > 5]

    # ── Quiver ─────────────────────────────────────────────────────────────────
    qx = np.linspace(-2.5, 2.5, QUIVER_N)
    qy = np.linspace(-2.5, 2.5, QUIVER_N)
    QX, QY   = np.meshgrid(qx, qy)
    quiv_pts = np.stack([QX.ravel(), QY.ravel()], 1).astype(np.float32)
    fv  = np.array([field_single(model, p) for p in quiv_pts])
    nrm = np.linalg.norm(fv, axis=-1, keepdims=True)
    fvn = fv / np.maximum(nrm, 1e-8)
    quiver_rows = '\n'.join(
        f'{quiv_pts[i,0]:.5f} {quiv_pts[i,1]:.5f} {fvn[i,0]*ARROW_SCALE:.5f} {fvn[i,1]*ARROW_SCALE:.5f}'
        for i in range(len(quiv_pts))
    )

    # ── Tikz ───────────────────────────────────────────────────────────────────
    tikz = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=7cm, height=7cm,
  xmin=-2.5, xmax=2.5,
  ymin=-2.5, ymax=2.5,
  xtick={{-2,-1,0,1,2}},
  ytick={{-2,-1,0,1,2}},
  tick label style={{font=\\footnotesize}},
  label style={{font=\\footnotesize}},
  legend style={{font=\\footnotesize, at={{(0.98,0.98)}}, anchor=north east}},
  legend cell align={{left}},
  axis equal image,
  grid=major,
  grid style={{gray!20}},
  unbounded coords=jump,
]
\\addplot[
  gray!35,
  quiver={{u=\\thisrow{{u}}, v=\\thisrow{{v}}, scale arrows=1}},
  -stealth,
  line width=0.25pt,
  every arrow/.append style={{line width=0.25pt}},
] table [col sep=space] {{
x y u v
{quiver_rows}
}};
\\addplot[color={{rgb,1:red,0.082;green,0.396;blue,0.753}}, line width=0.3pt, opacity=0.25]
table [col sep=space] {{
x y
{traj_table(grid_trajs)}
}};
\\addlegendentry{{rollouts}}
\\addplot[color={{rgb,1:red,1;green,0.435;blue,0}}, line width=1.8pt]
table [col sep=space] {{
x y
{single_table(ref_np)}
}};
\\addlegendentry{{demo}}
\\addplot[color={{rgb,1:red,0.082;green,0.396;blue,0.753}}, line width=1.4pt, dashed]
table [col sep=space] {{
x y
{single_table(fwd_ro)}
}};
\\addlegendentry{{rollout}}
\\addplot[only marks, mark=asterisk, mark size=4pt, black]
coordinates {{({x_eq_np[0]:.4f},{x_eq_np[1]:.4f})}};
\\end{{axis}}
\\end{{tikzpicture}}
"""
    out = os.path.join(OUT_DIR, f'e1_enc_{shape}.tikz')
    with open(out, 'w') as f:
        f.write(tikz)
    print(f'  → {out}')

print('\nDone.')
