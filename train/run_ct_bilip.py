#!/usr/bin/env python3
"""
run_ct_bilip_dtwd.py — DTWD für std MLP vs BiLip-MLP über alle 30 LASA-Shapes.

Lädt vorhandene Modelle aus saved_models_2/, trainiert fehlende BiLip-Modelle,
berechnet DTWD und speichert Ergebnisse als JSON.

Aufruf:
  python run_ct_bilip_dtwd.py                    # alle 30 Shapes
  python run_ct_bilip_dtwd.py --shapes Sshape    # einzeln testen
"""
import os, sys, argparse, time, json
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THIS_DIR   = SCRIPT_DIR
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'src'))

from bilip_elcd import ELCDOrthogonalCTGatedK
from data import lasa_to_torch, NormalisedTrajData, data_set_names

# ── Hyperparameter (identisch mit run_ct_bilip_statevar.py) ──────────────────
EPOCHS        = 15000
LR            = 3e-4
HIDDEN_DIM    = 64
N_HIDDEN      = 2
ALPHA         = 1.0
LAMBDA_MIN    = 0.5
LAMBDA_MAX    = 2.0
OMEGA_MAX     = 0.3
BILIP_S_MIN   = 0.01
BILIP_S_MAX   = 0.30
HORIZON       = 20
NUM_DEMOS     = 3
DT_EVAL       = 0.01
ROLLOUT_STEPS = 2000
CONV_TOL      = 1e-5
SEED          = 42
DEVICE        = torch.device('cpu')

MODEL_DIR = os.path.join(SCRIPT_DIR, 'saved_models_2')
OUT_DIR   = os.path.join(SCRIPT_DIR, 'results_bilip_dtwd')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────

class HorizonDataset(torch.utils.data.Dataset):
    def __init__(self, pos_norm, horizon, num_demos):
        T    = pos_norm.shape[2]
        step = T // horizon
        idx  = torch.arange(0, T, step)[:horizon]
        traj = pos_norm[:num_demos, :, :][:, :, idx].permute(0, 2, 1).float()
        self.traj = traj
        self.x0   = traj[:, 0, :]
    def __len__(self):        return self.traj.shape[0]
    def __getitem__(self, i): return self.x0[i], self.traj[i]


# ── Model builders ────────────────────────────────────────────────────────────

def build_std(dim):
    return ELCDOrthogonalCTGatedK(
        dim=dim, action_dim=1, alpha=ALPHA,
        lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX, omega_max=OMEGA_MAX,
        hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN,
        use_bilip_mlp=False,
    ).to(DEVICE)


def build_bilip(dim):
    return ELCDOrthogonalCTGatedK(
        dim=dim, action_dim=1, alpha=ALPHA,
        lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX, omega_max=OMEGA_MAX,
        hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN,
        use_bilip_mlp=True,
        bilip_s_min=BILIP_S_MIN,
        bilip_s_max=BILIP_S_MAX,
    ).to(DEVICE)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model, loader, x_eq, label='model'):
    dt  = 1.0 / (HORIZON - 1)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.1, total_iters=EPOCHS)
    best, best_sd = float('inf'), None
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        for x0_b, traj_b in loader:
            x0_b   = x0_b.to(DEVICE)
            traj_b = traj_b.to(DEVICE)
            B      = x0_b.shape[0]
            a      = torch.zeros(B, 1, device=DEVICE)
            x_star = x_eq.expand(B, -1)
            opt.zero_grad()
            x    = x0_b
            pred = []
            for _ in range(HORIZON - 1):
                x_dot = model(x, a, x_star)
                x     = x + dt * x_dot
                pred.append(x)
            loss = F.mse_loss(torch.stack(pred, 1), traj_b[:, 1:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sch.step()
        if total < best:
            best    = total
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 3000 == 0:
            print(f'    [{label}] epoch {epoch:5d}/{EPOCHS}  loss={best:.4e}'
                  f'  t={time.time()-t0:.0f}s', flush=True)
    model.load_state_dict(best_sd)
    return best


# ── Rollout + DTWD ────────────────────────────────────────────────────────────

def rollout(model, x0_np, x_eq_np):
    x    = torch.tensor(x0_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    x_eq = torch.tensor(x_eq_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    a    = torch.zeros(1, 1, device=DEVICE)
    traj = [x0_np.copy()]
    for _ in range(ROLLOUT_STEPS - 1):
        with torch.enable_grad():
            dx = model(x, a, x_eq)
        x  = (x + DT_EVAL * dx).detach()
        pt = x.squeeze(0).cpu().numpy()
        if not np.isfinite(pt).all():
            break
        traj.append(pt.copy())
        if np.linalg.norm(pt - x_eq_np) < CONV_TOL:
            break
    return np.array(traj)


def dtwd(ro_np, ref_np):
    """Symmetric chamfer distance (ELCD DTWD formula)."""
    ro  = torch.tensor(ro_np, dtype=torch.float32)
    ref = torch.tensor(ref_np, dtype=torch.float32)
    # ro: (T1, d),  ref: (T2, d)
    dists = torch.cdist(ro, ref)               # (T1, T2)
    d1 = dists.min(dim=1).values.mean()        # mean min over rollout
    d2 = dists.min(dim=0).values.mean()        # mean min over reference
    return (d1 + d2).item()


def eval_dtwd(model, pos_norm, x_eq_np):
    results = []
    for i in range(pos_norm.shape[0]):
        ref_np = pos_norm[i].T.numpy()   # (T, dim)
        ro_np  = rollout(model, ref_np[0], x_eq_np)
        results.append(dtwd(ro_np, ref_np))
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def run_shape(shape):
    torch.manual_seed(SEED); np.random.seed(SEED)
    t0 = time.time()
    print(f'\n═══ {shape} ═══', flush=True)

    out_path = os.path.join(OUT_DIR, f'bilip_dtwd_{shape}.json')
    if os.path.exists(out_path):
        with open(out_path) as f:
            d = json.load(f)
        print(f'  already done: std={d["std_dtwd_mean"]:.4f}  bilip={d["bilip_dtwd_mean"]:.4f}')
        return d

    pos, vel = lasa_to_torch(shape)
    ds       = NormalisedTrajData(pos, vel, num_demos=NUM_DEMOS)
    pos_norm = (pos.float() / ds.pos_std.unsqueeze(-1))
    x_eq     = ds.pos_eq[0:1].float().to(DEVICE)
    x_eq_np  = x_eq.squeeze(0).cpu().numpy()
    dim      = pos_norm.shape[1]

    loader = torch.utils.data.DataLoader(
        HorizonDataset(pos_norm, HORIZON, NUM_DEMOS),
        batch_size=NUM_DEMOS, shuffle=True,
    )

    # ── Std MLP ───────────────────────────────────────────────────────────────
    std_path = os.path.join(MODEL_DIR, f'ct_noenc_{shape}.pth')
    model_std = build_std(dim)
    if os.path.exists(std_path):
        model_std.load_state_dict(torch.load(std_path, map_location=DEVICE))
        print(f'  std: loaded', flush=True)
    else:
        print(f'  std: training …', flush=True)
        train_model(model_std, loader, x_eq, label='std')
        torch.save(model_std.state_dict(), std_path)
    model_std.eval()

    # ── BiLip MLP ─────────────────────────────────────────────────────────────
    bilip_path = os.path.join(MODEL_DIR, f'ct_bilip_{shape}.pth')
    model_bilip = build_bilip(dim)
    if os.path.exists(bilip_path):
        model_bilip.load_state_dict(torch.load(bilip_path, map_location=DEVICE))
        print(f'  bilip: loaded', flush=True)
    else:
        print(f'  bilip: training …', flush=True)
        train_model(model_bilip, loader, x_eq, label='bilip')
        torch.save(model_bilip.state_dict(), bilip_path)
    model_bilip.eval()

    # ── DTWD ──────────────────────────────────────────────────────────────────
    print('  evaluating DTWD …', flush=True)
    std_list   = eval_dtwd(model_std,   pos_norm, x_eq_np)
    bilip_list = eval_dtwd(model_bilip, pos_norm, x_eq_np)

    std_valid   = [d for d in std_list   if np.isfinite(d)]
    bilip_valid = [d for d in bilip_list if np.isfinite(d)]

    result = {
        'shape':           shape,
        'std_dtwd_mean':   float(np.mean(std_valid))   if std_valid   else float('nan'),
        'std_dtwd_std':    float(np.std(std_valid))    if std_valid   else float('nan'),
        'std_dtwd_demos':  std_list,
        'bilip_dtwd_mean': float(np.mean(bilip_valid)) if bilip_valid else float('nan'),
        'bilip_dtwd_std':  float(np.std(bilip_valid))  if bilip_valid else float('nan'),
        'bilip_dtwd_demos': bilip_list,
        'elapsed_s':       time.time() - t0,
    }
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  std={result["std_dtwd_mean"]:.4f}  bilip={result["bilip_dtwd_mean"]:.4f}'
          f'  t={time.time()-t0:.0f}s', flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shapes', nargs='+', default=data_set_names)
    args = p.parse_args()

    all_results = []
    for shape in args.shapes:
        r = run_shape(shape)
        all_results.append(r)

    # Summary
    std_means   = [r['std_dtwd_mean']   for r in all_results if np.isfinite(r['std_dtwd_mean'])]
    bilip_means = [r['bilip_dtwd_mean'] for r in all_results if np.isfinite(r['bilip_dtwd_mean'])]
    print(f'\n{"="*55}')
    print(f'{"Shape":<25} {"std MLP":>10} {"BiLip-MLP":>10}')
    print(f'{"-"*55}')
    for r in all_results:
        print(f'{r["shape"]:<25} {r["std_dtwd_mean"]:>10.4f} {r["bilip_dtwd_mean"]:>10.4f}')
    print(f'{"-"*55}')
    print(f'{"Mean (all 30)":<25} {np.mean(std_means):>10.4f} {np.mean(bilip_means):>10.4f}')
    print('Fertig.')


if __name__ == '__main__':
    main()
