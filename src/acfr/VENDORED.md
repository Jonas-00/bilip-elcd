# Vendored: ACFR `RobustNeuralNetworks` (PyTorch BiLipNet / MonLipNet / RENs)

| | |
|---|---|
| Upstream repo | https://github.com/acfr/RobustNeuralNetworks |
| Commit fetched | `417b70716b795c1db0d8b19037d03afd15ac2a26` (committed 2026-07-29) |
| Fetch date | 2026-08-20 |
| License | MIT (full text in `LICENSE`, copied from the same commit) |
| Authors of the vendored files | Dechuan Liu (Aug 2024) for the BiLipNet/MonLipNet files; Yurui Zhang for `ren_torch.py` — per the file headers |
| Papers | Wang, Dvijotham & Manchester, *Monotone, Bi-Lipschitz, and Polyak-Łojasiewicz Networks*, ICML 2024 (arXiv:2402.01344); Revay, Wang & Manchester, *Recurrent Equilibrium Networks*, IEEE TAC 2023 |

> Using this folder from another worktree?  Read **`HANDOFF.md`** in this
> directory — copy procedure, the JEPA-predictor recipe, the decision history and
> the gotchas.  This file is provenance only.

## Verbatim-status table: every file in this folder

| File | Origin | Status |
|---|---|---|
| `monlipnet_torch.py` | upstream `robustnn/` | **verbatim**, 0 changed lines |
| `bilipnet_torch.py` | upstream `robustnn/` | **verbatim**, 0 changed lines |
| `orthogonal_torch.py` | upstream `robustnn/` | **verbatim**, 0 changed lines |
| `ren_torch.py` | upstream `robustnn/` | **verbatim**, 0 changed lines |
| `solvers.py` | upstream `robustnn/` | **verbatim**, 0 changed lines (JAX module) |
| `utils.py` | upstream `robustnn/` | **verbatim**, 0 changed lines (JAX module) |
| `LICENSE` | upstream root | **verbatim** (MIT) |
| `__init__.py` | **ours** | `robustnn` sys.modules alias + the inverse override + `MonLipNetD` exports |
| `monlip_inverse_torch.py` | **ours** | pure-torch Davis-Yin inverse |
| `monlipnet_d.py` | **ours** | per-dimension-band MonLipNet (`MonLipNetD`, `ConjugatedMonLipNetD`) + ladder-bracketed learnable bounds + per-dim Davis-Yin inverse; see `HANDOFF.md` section (g) |
| `VENDORED.md` | **ours** | this file |
| `HANDOFF.md` | **ours** | handoff doc for downstream worktrees |

There are **no edits at all to upstream source**: every deviation lives in one of
our own files.  Both deviations are described below.

Files copied (from `robustnn/` at that commit):

```
monlipnet_torch.py
bilipnet_torch.py
orthogonal_torch.py
ren_torch.py
solvers.py
utils.py
```

Fetched with, e.g.

```
curl -sS -o monlipnet_torch.py \
  https://raw.githubusercontent.com/acfr/RobustNeuralNetworks/417b70716b795c1db0d8b19037d03afd15ac2a26/robustnn/monlipnet_torch.py
```

The MIT header comment and the author attribution at the top of each file are
kept intact.  The files are byte-for-byte upstream — no reformatting, no
renaming, no "improvements".  Re-check at any time:

```bash
cd "$(mktemp -d)"
for f in monlipnet_torch.py bilipnet_torch.py orthogonal_torch.py \
         ren_torch.py solvers.py utils.py; do
  curl -sS -O "https://raw.githubusercontent.com/acfr/RobustNeuralNetworks/417b70716b795c1db0d8b19037d03afd15ac2a26/robustnn/$f"
  echo "$f: $(diff "$f" <path-to-_acfr>/"$f" | wc -l) changed lines"
done
```

Expected: `0 changed lines` for all six.  Last verified 2026-08-20.

## Why vendored

We previously had our own PyTorch implementations of both pieces; both were
deleted on 2026-08-20 in favour of upstream's.  Everything deleted is recoverable
from git commit `7b22d8afa`.

1. **`src/bilip_jepa/models/bilipnet.py`** (our MonLipNet/BiLipNet) — deleted.
   Upstream's version is the published one, gives the same certified band, and is
   far faster.
2. **`src/bilip_jepa/models/ren_torch.py`** (our REN port, written 2026-06-18) —
   deleted.  Upstream added its own `robustnn/ren_torch.py` on **2026-06-29**,
   i.e. eleven days *after* our port; upstream's is now the code we run.
   `src/bilip_jepa/models/ren_torch.py` still exists but is a thin re-export of
   `_acfr/ren_torch.py` and contains no implementation of ours.

Upstream's `ren_torch.py` also ships something ours did not: **`solve_full_layer`**,
a Douglas–Rachford operator-splitting solver for the equilibrium
`w = act(D11 @ w + b)` when `D11` is FULL (non-triangular).  That is what makes
`RENBase.inverse` work — the inverse REN's `D11` is generally not triangular.
Our deleted port solved the same problem with `torchdeq` (Anderson acceleration +
implicit-function backward); note upstream's `solve_full_layer` is decorated
`@torch.no_grad()`, so it does not backpropagate through the solver iterations.

### What upstream does NOT have (deleted, accepted casualties)

Upstream ships only `RENBase` + `BiLipschitzREN`.  These symbols of ours are gone:

| Deleted symbol | Upstream equivalent |
|---|---|
| `ContractingREN` | none |
| `LipschitzREN` (gamma-Lipschitz IQC) | none |
| `GeneralREN` (arbitrary `Q`, `S`, `R` IQC) | none — `RENBase._direct_to_explicit` raises `NotImplementedError`; only `BiLipschitzREN` overrides it with a hard-coded `Q = -alpha2 I, S = I, R = -alpha1 I` |
| `RENBase.simulate_sequence(x0, u)` -> `(xT, y)` over `(T, B, nu)` | none — upstream is single-step `forward(state, inputs)`; a rollout must be written as an explicit `for t` loop over `explicit_call(x, u[t], e)` |
| `RENBase.inverse_sequence(x0, y)` -> `(xT, u, residual)` | partial — upstream has single-step `inverse(state, outputs)` and `inverse_call(x, u, e_inv)`; there is no sequence wrapper and no returned equilibrium residual |
| our `_selftest()` / `_sequence_gains()` | none — upstream ships no self-test in this file |
| `full_d11=True` constructor flag (was already removed earlier) | superseded by `solve_full_layer`, which upstream uses unconditionally in the inverse |

### Known breakages in our call sites (2026-08-20, NOT patched — recorded only)

Verified by import smoke test plus static inspection; the experiment drivers were
not run (they need data / long runtime).

| File:line | Breaks on | Why | Upstream equivalent |
|---|---|---|---|
| `scripts/bilipren_control_demo.py:43` | `from bilip_jepa.models.ren_torch import BiLipschitzREN, GeneralREN, LipschitzREN, tril_equilibrium_layer` → `ImportError: cannot import name 'GeneralREN'` | upstream has neither `GeneralREN` nor `LipschitzREN` | `tril_equilibrium_layer` and `BiLipschitzREN` DO exist upstream, same names; the other two have none |
| `scripts/bilipren_control_demo.py:70, 428` | `BiLipschitzREN(NU_IO, NX, NV, NU_IO, mu=…, nu=…)` → `TypeError: got multiple values for argument 'mu'` | upstream takes 3 sizes, the 4th positional lands on `mu` | drop `output_size`: `BiLipschitzREN(nu, nx, nv, mu=…, nu=…)` |
| `scripts/bilipren_control_demo.py:74, 149, 433, 439, 451, 465-466, 476` | `P.simulate_sequence(x0, u)` → `AttributeError` | no sequence API upstream | hand-rolled `for t` loop over `explicit_call` |
| `scripts/bilipren_control_demo.py:81, 474, 502, 514` | `P.inverse_sequence(x0, y)` (3-tuple) → `AttributeError` | no sequence API upstream | single-step `inverse(state, outputs)` / `inverse_call`; no residual returned |
| `scripts/bilipren_control_demo.py:340, 343` | `GeneralREN(...)`, `LipschitzREN(..., gamma=1.0)` → `NameError` (already failed at import) | E3 needs an output-strictly-passive REN and a gamma-Lipschitz REN | none — upstream's `RENBase._direct_to_explicit` raises `NotImplementedError`; only the fixed bi-Lipschitz `Q, S, R` is implemented |
| `scripts/bilipren_control_demo.py:93, 259, 263` | `P._direct_to_explicit()`, `ctrl.activation`, `tril_equilibrium_layer(act, ex.D11, b)` | — | **these still work**: upstream has `_direct_to_explicit`, `self.activation`, `ExplicitRENParams.D11`, and an identically-signatured `tril_equilibrium_layer` |
| `scripts/bilipren_fitted.py` | imports OK (`BiLipschitzREN` only), fails at **runtime** | — | — |
| `scripts/bilipren_fitted.py:96` | `BiLipschitzREN(m, nx, nv, m, mu=…, nu=…)` → `TypeError: got multiple values for argument 'mu'` | 4th positional lands on `mu` | drop `output_size` |
| `scripts/bilipren_fitted.py:101, 107, 113, 280-281` | `G.simulate_sequence(...)` → `AttributeError` | no sequence API upstream | hand-rolled loop |
| `scripts/bilipren_fitted.py:165` | `G.inverse_sequence(...)` (3-tuple) → `AttributeError` | no sequence API upstream | `inverse(state, outputs)` / `inverse_call`, no residual |
| `scripts/bilipren_fitted.py:186` | `G._direct_to_explicit()` | — | **still works** |
| `src/bilip_jepa/skills/bilipren_skill.py:28` | import of `BiLipschitzREN` → **OK** | — | — |
| `src/bilip_jepa/skills/bilipren_skill.py:56` | `BiLipschitzREN(input_size=…, state_size=…, features=…, output_size=…, mu=…, nu=…)` → `TypeError: unexpected keyword argument 'output_size'` | upstream forces `output_size = input_size` | drop the `output_size=` kwarg |
| `src/bilip_jepa/skills/bilipren_skill.py:68, 70, 72, 85` | `self.ren.state_size`, `.forward(state, inputs)`, `._explicit_call(s0, z, e)`, `._direct_to_explicit()` | — | **all still work**; `forward(state, inputs) -> (next_state, output)` is upstream's exact signature. This file breaks on line 56 alone |

Summary: one import-time failure (`GeneralREN` / `LipschitzREN` in the demo), and
three distinct runtime failure classes — the `BiLipschitzREN` constructor arity,
the missing `simulate_sequence`, and the missing `inverse_sequence`.

## Deviation 1 (no source edit): our torch inverse is the DEFAULT `.inverse`

`src/bilip_jepa/models/_acfr/monlip_inverse_torch.py` is **ours**, not upstream's.
(It lived at `src/bilip_jepa/models/monlip_inverse_torch.py` until 2026-08-20 and
was moved into this folder so the folder is self-contained and copy-pasteable;
its one relative import changed from `from ._acfr.orthogonal_torch import ...` to
`from .orthogonal_torch import ...`.)

**Why it exists.**  Upstream's inverse runs in numpy/JAX:
`MonLipNet.direct_to_explicit()` ends every field with `.numpy(force=True)`,
`MonLipNet.inverse(y, ...)` documents `y` as a numpy array, and the Davis-Yin step
is `robustnn.solvers.DavisYinSplit`, a JAX function.  Same for
`Unitary.inverse`, which goes through `direct_to_explicit()` and so returns a host
numpy array.  That path works (round trip 2.4e-07) but it (a) forces a
torch → numpy → jax → torch hop off the accelerator and (b) **detaches** — no
gradient reaches the weights or `y`, so the inverse cannot appear in a training
loss.  Our version re-implements the *same* algorithm (transcribed from
`solvers.DavisYinSplit` / `mln_RA`) in torch: on device, in the autograd graph,
and faster on GPU (3.6x at `unit_features=[64,64]`, 51x at `[512,512]`,
re-measured 2026-08-20).

**How it is installed — without touching upstream source.**  `__init__.py` (ours)
monkey-patches at import time:

```python
MonLipNet.inverse_numpy = MonLipNet.inverse   # upstream's original, preserved
BiLipNet.inverse_numpy  = BiLipNet.inverse
Unitary.inverse_numpy   = Unitary.inverse
MonLipNet.inverse = _monlipnet_inverse_dispatch   # ours
BiLipNet.inverse  = _bilipnet_inverse_dispatch    # ours
Unitary.inverse   = _unitary_inverse_dispatch     # ours
```

Every such line carries a `# MODIFIED (ours): ...` comment in the source saying
what and why.  Behaviour:

* `.inverse(y_torch, ...)` → our torch Davis-Yin; torch in, torch out, on device,
  differentiable w.r.t. both `y` and the parameters.
* `.inverse_numpy(y_numpy, ...)` → upstream's, unchanged: numpy in, `jaxlib`
  array out, detached.
* `bilip_jepa.models._acfr.USE_TORCH_INVERSE = False` → global switch routing
  `.inverse` back to upstream's numpy path.
* `.inverse` **also** falls back to upstream automatically when handed something
  that is not a `torch.Tensor`.  That is required, not cosmetic: upstream's own
  `BiLipNet.inverse` body calls back into the (now overridden) `.inverse` of each
  sub-layer, with numpy arrays, so without the type check
  `BiLipNet.inverse_numpy` would break.

`BiLipNet`'s inverse has to undo the **orthogonal layers** as well as the
monotone ones.  Each `Unitary` is `O(x) = P x + q` with `P.T @ P = I`, so
`O^-1(y) = P.T (y - q)` — written `(y - q) @ P` for row-vector batches.  Our
`bilipnet_inverse` walks the chain backwards (`O_K^-1, M_{K-1}^-1, …, M_0^-1,
O_0^-1`) doing exactly that in torch.

Note the earlier version of this document described a *different* deviation 1: a
`try/except ImportError` stub around `from robustnn.solvers import DavisYinSplit`
in `monlipnet_torch.py`, used while `jax`/`flax` were not dependencies.  **That is
gone.**  `solvers.py` and `utils.py` are now vendored verbatim, `jax` and `flax`
are declared dependencies in `requirements.txt` and `pyproject.toml`, and
`monlipnet_torch.py` is byte-for-byte upstream again.

## Deviation 2 (no source edit): the `robustnn` package-name alias

Upstream's imports are absolute — `from robustnn.orthogonal_torch import ...`,
`from robustnn.monlipnet_torch import MonLipNet`.  Rewriting those lines would
break the byte-for-byte guarantee, so instead `__init__.py` of THIS package
registers the package itself under the name `robustnn` in `sys.modules` before
importing the submodules:

```python
_sys.modules["robustnn"] = _sys.modules[__name__]
```

so `robustnn.orthogonal_torch` resolves to
`bilip_jepa.models._acfr.orthogonal_torch`.  Each submodule is then aliased under
both dotted names so that importing by either path yields the same module object
(no duplicate `MonLipNet` classes).  If a real `robustnn` package is already in
`sys.modules`, `__init__.py` raises instead of shadowing it.  No file under
`robustnn/` upstream was modified for this.

## Usage

```python
from bilip_jepa.models._acfr import BiLipNet, MonLipNet, BiLipschitzREN, RENBase

f = BiLipNet(features=8, unit_features=[64, 64],
             mu=0.6, nu=0.95, depth=2,
             is_mu_fixed=True, is_nu_fixed=True, is_tau_fixed=True)

# RENs: also re-exported from bilip_jepa.models.ren_torch
G = BiLipschitzREN(input_size=2, state_size=4, features=8, mu=0.5, nu=2.0)
x0 = G.initialize_carry(batch_size=5)
x1, y = G.forward(x0, u)      # one step
x1, u_rec = G.inverse(x0, y)  # one step of the inverse (uses solve_full_layer)

# Static-map inverses (torch by default -- see Deviation 1):
x_rec = f.inverse(y_torch, alphas=1.0, iterations=200)   # ours, differentiable
x_rec = f.inverse_numpy(y_np, [1.0], [np_relu], [200], [1.0])  # upstream's
```

Note: `RENBase.inverse` is upstream's own and is NOT overridden — only the static
`MonLipNet` / `BiLipNet` / `Unitary` inverses are.

At least two of `mu` (lower Lipschitz bound), `nu` (upper Lipschitz bound),
`tau = nu / mu` (distortion) must be supplied; the third is derived.  The
`is_*_fixed=True` flags keep the bounds FIXED rather than learnable.  Note the
signature differs from our deleted `bilipnet.py`, which took
`BiLipNet(n, hidden_widths, mu, nu, blocks=K)` — `blocks` is now `depth`, and
`hidden_widths` is now `unit_features`.

## Verification (2026-08-20, this environment, float32, CPU)

`BiLipNet(features=8, unit_features=[64,64], mu=0.6, nu=0.95, is_*_fixed=True)`,
random init, 400 random pairs per perturbation scale
`||a-b|| in {1e-3, 1e-2, 1e-1, 1, 10}`, measuring
`||f(a)-f(b)|| / ||a-b||`:

| depth | min ratio | max ratio | inside [0.6, 0.95]? |
|---|---|---|---|
| 1 | 0.6823 | 0.7988 | yes |
| 2 | 0.6742 | 0.7720 | yes |

`get_bounds()` reports `lipmin=0.600000, lipmax=0.950000, tau=1.583333` at both
depths, i.e. the certified band is exactly the requested one and the measured
ratios sit strictly inside it.  Reproduce (2000 pairs per depth, seed 0):

```python
import torch
from bilip_jepa.models._acfr import BiLipNet

torch.manual_seed(0)
for K in (1, 2):
    f = BiLipNet(features=8, unit_features=[64, 64], mu=0.6, nu=0.95, depth=K,
                 is_mu_fixed=True, is_nu_fixed=True, is_tau_fixed=True).eval()
    lo, hi = float("inf"), 0.0
    with torch.no_grad():
        for s in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
            a = torch.randn(400, 8) * 2.0
            d = torch.randn(400, 8)
            b = a + d / d.norm(dim=1, keepdim=True) * s
            g = (f(a) - f(b)).norm(dim=1) / (a - b).norm(dim=1)
            lo, hi = min(lo, g.min().item()), max(hi, g.max().item())
    print(K, f.get_bounds(), lo, hi)
```

### Inverse verification (2026-08-20, float32, CPU, `mu=0.6, nu=0.95`)

`MonLipNet(8, [64,64])`, 16 samples, 200 Davis-Yin iterations, `alpha=1.0`:

| quantity | value |
|---|---|
| `MonLipNet.inverse` (ours, torch) round trip `max\|x_rec - x\|` | 4.77e-07 |
| `MonLipNet.inverse_numpy` (upstream) round trip | 2.98e-07 |
| agreement, ours vs upstream, `max` abs diff | 2.38e-07 |
| `dL/dy` for `L = sum(inverse(y)**2)` | finite, nonzero, `absmax` 7.92 |
| `dL/dFq` (a MonLipNet weight) | finite, nonzero, `absmax` 2.80 |
| `USE_TORCH_INVERSE = False` → `.inverse` returns | `jaxlib.ArrayImpl`, round trip 2.98e-07 |
| `Unitary.inverse` (ours) round trip | 4.77e-07 |

`BiLipNet(features=8, unit_features=[64,64], mu=0.6, nu=0.95)`, 300 iterations:

| depth | `get_bounds()` | measured gain band | ours round trip | upstream round trip | ours vs upstream |
|---|---|---|---|---|---|
| 1 | (0.600, 0.950, 1.583) | [0.6823, 0.7988] | 1.43e-06 | 1.19e-06 | 4.77e-07 |
| 2 | (0.600, 0.950, 1.583) | [0.6803, 0.7689] | 1.07e-06 | 8.94e-07 | 4.77e-07 |

Both `BiLipNet` inverses are differentiable in `y` and in the weights (finite,
nonzero gradients at both depths).
