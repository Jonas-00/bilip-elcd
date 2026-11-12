#!/usr/bin/env python3
"""
CT + RQ-spline encoder — apples-to-apples, 10 seeds × 30 shapes.
ELCD protocol: 3 demos, drop-15, per-dim-std (no mean), symmetric chamfer DTWD.

Results land in this directory: ct_enc_<Shape>_seed<s>.json

Usage:
  python run_ct_enc.py --workers 4
  python run_ct_enc.py --seeds 0 1 2
  python run_ct_enc.py --shapes Sshape GShape
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor, as_completed



THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, '..', 'src'))

from bilip_elcd import ELCDOrthogonalCTGatedK
from utils      import generate_mf_transform, ELCD_Transform
from data       import lasa_to_torch, DiscreteNormalisedTrajData

ALL_SHAPES = [
    'Angle', 'BendedLine', 'CShape', 'DoubleBendedLine', 'GShape',
    'JShape', 'JShape_2', 'Khamesh', 'LShape', 'Leaf_1', 'Leaf_2', 'Line',
    'Multi_Models_1', 'Multi_Models_2', 'Multi_Models_3', 'Multi_Models_4',
    'NShape', 'PShape', 'RShape', 'Saeghe', 'Sharpc', 'Sine', 'Snake',
    'Spoon', 'Sshape', 'Trapezoid', 'WShape', 'Worm', 'Zshape', 'heee',
]

EPOCHS     = 15000;  LR = 3e-4;  BATCH_SIZE = 16;  HORIZON = 20
HIDDEN_DIM = 64;     N_HIDDEN = 2;  NUM_DEMOS = 3;  FLOW_STEPS = 2
ROLLOUT_STEPS = 2000;  CONV_TOL = 1e-5
ALPHA = 1.0;  LAMBDA_MIN = 0.5;  LAMBDA_MAX = 2.0;  OMEGA_MAX = 0.3
DT_TRAIN = 1.0 / (HORIZON - 1);  DT_EVAL = 0.01
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FullHorizonDataset(torch.utils.data.Dataset):
    def __init__(self, pos_norm, horizon):
        T = pos_norm.shape[2]
        idx = torch.arange(0, T, T // horizon)[:horizon]
        traj = pos_norm[:, :, idx].permute(0, 2, 1).float()
        self.traj = traj;  self.x0 = traj[:, 0, :]
    def __len__(self): return self.traj.shape[0]
    def __getitem__(self, i): return self.x0[i], self.traj[i]


def sym_chamfer(ro, ref):
    r = torch.tensor(ro, dtype=torch.float32)
    d = torch.tensor(ref, dtype=torch.float32)
    dist = torch.cdist(r, d)
    return (dist.min(0).values.mean() + dist.min(1).values.mean()).item()


def build_model(dim, x_eq):
    kern = ELCDOrthogonalCTGatedK(
        dim=dim, action_dim=1, alpha=ALPHA,
        lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX, omega_max=OMEGA_MAX,
        hidden_dim=HIDDEN_DIM, n_hidden=N_HIDDEN,
    )
    flow = generate_mf_transform(flow_steps=FLOW_STEPS, input_dim=dim, latent_dim=dim)
    return ELCD_Transform(x_eq=x_eq.cpu(), transform=flow, model=kern).to(DEVICE)


def run_shape(shape, seed):
    out_path = os.path.join(THIS_DIR, f'ct_enc_{shape}_seed{seed}.json')
    if os.path.exists(out_path):
        return shape, seed, 'skip', 0.0

    torch.manual_seed(seed);  np.random.seed(seed)
    t0 = time.time()

    pos, vel = lasa_to_torch(shape)
    ds = DiscreteNormalisedTrajData(pos, vel, num_demos=NUM_DEMOS)
    x_eq        = ds.pos_eq[0:1].float().to(DEVICE)
    pos_norm_tr  = ds.pos.float()
    pos_norm_all = (pos.float() / ds.pos_std.unsqueeze(-1))

    loader = torch.utils.data.DataLoader(
        FullHorizonDataset(pos_norm_tr, HORIZON),
        batch_size=BATCH_SIZE, shuffle=True)

    model = build_model(pos.shape[1], x_eq)
    n_params = sum(p.numel() for p in model.parameters())

    # Training — Euler, model(x) returns x_dot in data space via flow Jacobian
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.1, total_iters=EPOCHS)
    best, best_sd, losses = float('inf'), None, []
    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        for x0_b, traj_b in loader:
            x0_b = x0_b.to(DEVICE);  traj_b = traj_b.to(DEVICE)
            opt.zero_grad()
            x = x0_b;  pred = []
            for _ in range(HORIZON - 1):
                x = x + DT_TRAIN * model(x)
                pred.append(x)
            loss = F.mse_loss(torch.stack(pred, 1), traj_b[:, 1:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step();  total += loss.item()
        sch.step();  losses.append(total)
        if total < best:
            best = total
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_sd);  model.eval()
    torch.save(best_sd, os.path.join(THIS_DIR, f'ct_enc_{shape}_seed{seed}.pth'))

    # Rollout — Euler with detach() between steps (Jacobian requires autograd)
    dtwd_list = []
    for d in range(pos_norm_all.shape[0]):
        ref = pos_norm_all[d].T.numpy()
        x   = pos_norm_all[d, :, 0:1].T.float().to(DEVICE)
        traj = [x[0].detach().cpu().numpy()]
        for _ in range(ROLLOUT_STEPS - 1):
            x = (x + DT_EVAL * model(x)).detach()
            traj.append(x[0].cpu().numpy())
            if np.linalg.norm(traj[-1]) < CONV_TOL:
                break
        ro = np.array(traj)
        dtwd_list.append(float(sym_chamfer(ro, ref)) if np.isfinite(ro).all() else float('nan'))

    elapsed = time.time() - t0
    result = {
        'shape': shape, 'model': 'ct_enc', 'seed': seed,
        'dtwd_mean': float(np.nanmean(dtwd_list)),
        'dtwd_std':  float(np.nanstd(dtwd_list)),
        'dtwd_per_demo': dtwd_list,
        'loss_final': losses[-1], 'n_params': n_params, 'elapsed_s': elapsed,
        'hyperparams': {
            'epochs': EPOCHS, 'lr': LR, 'num_demos_train': NUM_DEMOS,
            'flow_steps': FLOW_STEPS, 'alpha': ALPHA,
            'lambda_min': LAMBDA_MIN, 'lambda_max': LAMBDA_MAX,
            'omega_max': OMEGA_MAX, 'hidden_dim': HIDDEN_DIM,
            'n_hidden': N_HIDDEN, 'seed': seed,
        },
    }
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    return shape, seed, 'ok', elapsed / 60


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seeds',   type=int, nargs='+', default=list(range(10)))
    p.add_argument('--shapes',  nargs='+', default=ALL_SHAPES)
    p.add_argument('--task-id', type=int, default=None,
                   help='SLURM array task ID: maps to one (shape, seed) pair')
    args = p.parse_args()

    # SLURM array mode: one task = one (shape, seed)
    if args.task_id is not None:
        n = len(ALL_SHAPES)
        shape = ALL_SHAPES[args.task_id % n]
        seed  = args.task_id // n
        print(f'  SLURM task {args.task_id} → shape={shape}  seed={seed}  device={DEVICE}',
              flush=True)
        run_shape(shape, seed)
        return

    todo = [(s, seed) for seed in args.seeds for s in args.shapes
            if not os.path.exists(os.path.join(THIS_DIR, f'ct_enc_{s}_seed{seed}.json'))]
    done = len(args.seeds) * len(args.shapes) - len(todo)

    print(f'\n{"="*60}')
    print(f'  CT + encoder — apples-to-apples (10 seeds)  device={DEVICE}')
    print(f'  Total: {len(args.seeds)*len(args.shapes)}  Done: {done}  Todo: {len(todo)}')
    print(f'  Workers: {args.workers}  Output: {THIS_DIR}')
    print(f'{"="*60}')
    if not todo: print('  All done.'); return

    t_start = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(run_shape, s, seed): (s, seed) for s, seed in todo}
        for fut in as_completed(futures):
            shape, seed, status, elapsed = fut.result()
            completed += 1
            eta = (time.time() - t_start) / 60 / completed * (len(todo) - completed)
            print(f'  [{completed:3d}/{len(todo)}] {shape:<25} seed={seed}  {status}'
                  f'  {elapsed:.0f}min  ETA={eta:.0f}min', flush=True)
    print(f'\n  Done in {(time.time()-t_start)/3600:.1f}h')


if __name__ == '__main__':
    main()
