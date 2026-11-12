#!/usr/bin/env python3
"""
E5 — State-variation ablation as pgfplots tikz.
Sshape, CT plain model, std MLP vs BiLip-MLP.
Two panels: (a) state-variation norm, (b) injectivity margin.
Output: figures/tikz/e5_statevar_Sshape.tikz (left)
        figures/tikz/e5_injectivity_Sshape.tikz (right)
"""
import os, sys
import numpy as np
import torch
from torch.autograd.functional import jacobian as torch_jac

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, '..', 'src'))
from bilip_elcd import ELCDOrthogonalCTGatedK
from data       import lasa_to_torch, NormalisedTrajData

SHAPE      = 'Sshape'
HIDDEN_DIM = 64;  N_HIDDEN = 2
ALPHA      = 1.0
LAMBDA_MIN = 0.5;  LAMBDA_MAX = 2.0;  OMEGA_MAX = 0.3
NUM_DEMOS  = 3
BILIP_S_MIN = 0.01;  BILIP_S_MAX = 0.30
DEVICE     = torch.device('cpu')

MODEL_DIR  = os.path.abspath(os.path.join(THIS_DIR, '..', '..', 'ELCD_2', 'saved_models_2'))
STD_PATH   = os.path.join(MODEL_DIR, f'ct_noenc_{SHAPE}.pth')
BILIP_PATH = os.path.join(MODEL_DIR, f'ct_bilip_{SHAPE}.pth')
OUT_DIR    = os.path.join(THIS_DIR, 'figures', 'tikz')
os.makedirs(OUT_DIR, exist_ok=True)


def build_std(dim):
    return ELCDOrthogonalCTGatedK(
        dim=dim, action_dim=1, alpha=ALPHA,
        lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX, omega_max=OMEGA_MAX,
        hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN, use_bilip_mlp=False,
    ).to(DEVICE)


def build_bilip(dim):
    return ELCDOrthogonalCTGatedK(
        dim=dim, action_dim=1, alpha=ALPHA,
        lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX, omega_max=OMEGA_MAX,
        hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN,
        use_bilip_mlp=True, bilip_s_min=BILIP_S_MIN, bilip_s_max=BILIP_S_MAX,
    ).to(DEVICE)


def compute_metrics(model, states, x_eq):
    a      = torch.zeros(1, 1)
    x_star = x_eq.to(DEVICE)
    sv_norms, sigma_mins = [], []
    for x_np in states:
        x        = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
        x_single = x.squeeze(0).detach().requires_grad_(True)

        def f_vec(x_in):
            return model(x_in.unsqueeze(0), a,
                         x_star.unsqueeze(0) if x_star.dim() == 1 else x_star
                         ).squeeze(0)
        try:
            Jf = torch_jac(f_vec, x_single, create_graph=False)
        except Exception:
            continue
        with torch.no_grad():
            xs = x_star.unsqueeze(0) if x_star.dim() == 1 else x_star
            A  = model.get_A(x, a, xs).squeeze(0)
        sv_norms.append(torch.linalg.matrix_norm(Jf - A, ord=2).item())
        sigma_mins.append(torch.linalg.svdvals(Jf).min().item())
    return np.array(sv_norms), np.array(sigma_mins)


# ── Load data ─────────────────────────────────────────────────────────────────
pos, vel = lasa_to_torch(SHAPE)
ds       = NormalisedTrajData(pos, vel, num_demos=NUM_DEMOS)
pos_norm = pos.float() / ds.pos_std.unsqueeze(-1)
x_eq     = ds.pos_eq[0:1].float()
x_eq_np  = x_eq.squeeze(0).numpy()
dim      = pos_norm.shape[1]

states = pos_norm[:NUM_DEMOS].permute(0, 2, 1).reshape(-1, dim).numpy()[::5]
R      = float(np.max(np.linalg.norm(states - x_eq_np, axis=1)))
sR     = BILIP_S_MAX * R
lam_eff_approx  = OMEGA_MAX + LAMBDA_MIN
cert_sigma_min  = (lam_eff_approx - OMEGA_MAX + ALPHA) - BILIP_S_MAX * R
print(f'R={R:.3f}  sR={sR:.3f}  cert_sigma_min={cert_sigma_min:.3f}')

# ── Load models ───────────────────────────────────────────────────────────────
model_std = build_std(dim)
model_std.load_state_dict(torch.load(STD_PATH, map_location=DEVICE, weights_only=True))
model_std.eval()
print(f'std loaded: {STD_PATH}')

model_bilip = build_bilip(dim)
model_bilip.load_state_dict(torch.load(BILIP_PATH, map_location=DEVICE, weights_only=True))
model_bilip.eval()
print(f'bilip loaded: {BILIP_PATH}')

# ── Compute metrics ───────────────────────────────────────────────────────────
print('Computing std metrics...', flush=True)
sv_std,   sm_std   = compute_metrics(model_std,   states, x_eq.squeeze(0))
print('Computing bilip metrics...', flush=True)
sv_bilip, sm_bilip = compute_metrics(model_bilip, states, x_eq.squeeze(0))

print(f'std   ||S|| max={np.max(sv_std):.3f}  bilip max={np.max(sv_bilip):.3f}  cert<={sR:.3f}')
print(f'std   σ_min min={np.min(sm_std):.3f}  bilip min={np.min(sm_bilip):.3f}  cert>={cert_sigma_min:.3f}')

# ── Build tikz ────────────────────────────────────────────────────────────────
idx    = np.arange(len(sv_std))
y_clip = max(4 * sR, 1.0)
n_above = int(np.sum(sv_std > y_clip))

def scatter_coords(x_arr, y_arr, clip=None):
    rows = []
    for xi, yi in zip(x_arr, y_arr):
        if clip is not None:
            yi = min(yi, clip)
        rows.append(f'{xi} {yi:.5f}')
    return '\n'.join(rows)

sv_std_tbl   = scatter_coords(idx, sv_std,   clip=y_clip)
sv_bilip_tbl = scatter_coords(idx, sv_bilip, clip=y_clip)
sm_std_tbl   = scatter_coords(idx, sm_std)
sm_bilip_tbl = scatter_coords(idx, sm_bilip)

clip_note = (f'\\node[font=\\tiny, text={{rgb,1:red,0.082;green,0.396;blue,0.753}}, '
             f'anchor=north east] at (axis cs:{len(idx)-1},{y_clip*0.98:.3f}) '
             f'{{{n_above} std pts clipped}};') if n_above else ''

# Panel (a): state-variation norm
tikz_a = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=6cm, height=4.5cm,
  xmin=0, xmax={len(idx)},
  ymin=0, ymax={y_clip*1.05:.3f},
  xlabel={{state index}},
  ylabel={{$\\|(D_x A)(x-x^*)\\|_2$}},
  tick label style={{font=\\footnotesize}},
  label style={{font=\\footnotesize}},
  grid=major, grid style={{gray!20}},
  legend style={{font=\\footnotesize, at={{(0.02,0.98)}}, anchor=north west}},
  legend cell align={{left}},
]
\\addplot[only marks, mark=*, mark size=0.8pt,
  color={{rgb,1:red,0.082;green,0.396;blue,0.753}}, opacity=0.4]
table [col sep=space] {{
x y
{sv_std_tbl}
}};
\\addlegendentry{{std MLP}}
\\addplot[only marks, mark=*, mark size=1.2pt,
  color={{rgb,1:red,0.776;green,0.157;blue,0.157}}]
table [col sep=space] {{
x y
{sv_bilip_tbl}
}};
\\addlegendentry{{BiLip-MLP}}
\\addplot[color={{rgb,1:red,0.776;green,0.157;blue,0.157}}, line width=1.2pt, dashed]
coordinates {{(0,{sR:.4f}) ({len(idx)},{sR:.4f})}};
\\addlegendentry{{cert. bound $sR={sR:.2f}$}}
{clip_note}
\\end{{axis}}
\\end{{tikzpicture}}
"""

# Panel (b): injectivity margin
sm_ymin = min(np.min(sm_std), 0) * 1.1 - 0.05
sm_ymax = max(np.max(sm_std), np.max(sm_bilip)) * 1.1

cert_line = (f'\\addplot[color={{rgb,1:red,0.776;green,0.157;blue,0.157}}, line width=1.2pt, dashed]\n'
             f'coordinates {{(0,{cert_sigma_min:.4f}) ({len(idx)},{cert_sigma_min:.4f})}};\n'
             f'\\addlegendentry{{cert. floor ${cert_sigma_min:.2f}$}}') if cert_sigma_min > 0 else ''

tikz_b = f"""\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=6cm, height=4.5cm,
  xmin=0, xmax={len(idx)},
  ymin={sm_ymin:.3f}, ymax={sm_ymax:.3f},
  xlabel={{state index}},
  ylabel={{$\\sigma_{{\\min}}(J_f)$}},
  tick label style={{font=\\footnotesize}},
  label style={{font=\\footnotesize}},
  grid=major, grid style={{gray!20}},
  legend style={{font=\\footnotesize, at={{(0.98,0.98)}}, anchor=north east}},
  legend cell align={{left}},
]
\\addplot[only marks, mark=*, mark size=0.8pt,
  color={{rgb,1:red,0.082;green,0.396;blue,0.753}}, opacity=0.4]
table [col sep=space] {{
x y
{sm_std_tbl}
}};
\\addlegendentry{{std MLP}}
\\addplot[only marks, mark=*, mark size=1.2pt,
  color={{rgb,1:red,0.776;green,0.157;blue,0.157}}]
table [col sep=space] {{
x y
{sm_bilip_tbl}
}};
\\addlegendentry{{BiLip-MLP}}
{cert_line}
\\addplot[black, line width=0.6pt, dotted]
coordinates {{(0,0) ({len(idx)},0)}};
\\end{{axis}}
\\end{{tikzpicture}}
"""

out_a = os.path.join(OUT_DIR, f'e5_statevar_{SHAPE}.tikz')
out_b = os.path.join(OUT_DIR, f'e5_injectivity_{SHAPE}.tikz')
with open(out_a, 'w') as f: f.write(tikz_a)
with open(out_b, 'w') as f: f.write(tikz_b)
print(f'→ {out_a}')
print(f'→ {out_b}')
print('Done.')
