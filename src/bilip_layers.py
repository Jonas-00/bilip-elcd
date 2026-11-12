"""
bilipmlp.py -- bi-Lipschitz MLPs with hard, certified gain bounds (PyTorch)

Requirements: Python >= 3.9, torch, numpy.  For square BiLipNet layers also the
folder ``acfr/`` next to this file plus jax + flax (upstream's PyTorch
monlipnet_torch.py imports its JAX solver, used by MonLipNet.inverse).
Run ``python bilipmlp.py`` for a self-test.

A map f is (m, L)-bi-Lipschitz if   m ||x - y|| <= ||f(x) - f(y)|| <= L ||x - y||
for all inputs x, y: L caps how much f can stretch distances, m > 0 forbids it from
collapsing distinct inputs (f is then injective with a Lipschitz inverse).

Building blocks
---------------
BiLipLinear(in, out, s_min, s_max)
    y = U diag(sigma) V^T x + b,  U, V orthonormal columns,
    sigma_i = s_min + (s_max - s_min) * sigmoid(theta_i)   (always in [s_min, s_max]).
    Gains in [s_min, s_max] for out >= in; for out < in the lower gain is 0 (null space).

BiLipSpline(in, out, s_min, s_max, n_knots=8, knot_range=3.0)
    y = U h(V^T x) + b,  h_i a learned piecewise-linear function per coordinate with
    slope everywhere in [s_min, s_max]: same gains as BiLipLinear, but nonlinear.
    Jacobian = U diag(h'(V^T x)) V^T;  closed-form inverse via .inverse(y).

BiLipNetLayer(n, s_min, s_max, unit_features=None, depth=1)      [needs acfr/]
    Square n -> n layer: ACFR's BiLipNet (Wang, Dvijotham & Manchester, ICML 2024),
    f(x) = P_d M_d( ... P_1 M_1(P_0 x)), P_k orthogonal + bias, M_k monotone
    MonLipNet blocks each with gains in [s_min^(1/d), s_max^(1/d)]  ->  [s_min, s_max].

BiLipSoftplus(a)
    phi(x) = a x + (1 - a) softplus(x), slope in (a, 1)  ->  (a, 1)-bi-Lipschitz.

BiLipMLP(input_dim, output_dim, hidden_dims, s_min, s_max, ...)
    L layers (each with gains in [s_min_layer, s_max_layer]) and L-1 activations
    (gains in [m_phi, L_phi]).  Composition multiplies the bounds:
        m_f = s_min_layer^L * m_phi^(L-1),     L_f = s_max_layer^L * L_phi^(L-1)
    layer_type="linear": BiLipSoftplus(a) between layers (m_phi = a, L_phi = 1)
    layer_type="spline": no activation (m_phi = L_phi = 1)
    square_layer="bilipnet": every layer with in_dim == out_dim is a BiLipNetLayer,
        dimension changes stay BiLipLinear / BiLipSpline (same per-layer band).
    budget_from_global=True: s_min / s_max are the GLOBAL (m_f, L_f) and the per-layer
        band is solved from the formula above.  model.lipschitz_lower() /
        model.lipschitz_upper() return the certified global bounds (the lower one is an
        RMS average-case bound if any layer reduces the dimension).

Quick start
-----------
    import torch
    from bilipmlp import BiLipMLP

    # global target  0.1 ||x-y|| <= ||f(x)-f(y)|| <= 2 ||x-y||,  6 -> 12 -> 12 -> 24
    f = BiLipMLP(6, 24, hidden_dims=[12, 12], s_min=0.1, s_max=2.0,
                 budget_from_global=True, square_layer="bilipnet")
    # layers: BiLipLinear(6->12), BiLipNetLayer(12->12), BiLipLinear(12->24)
    print(f.lipschitz_lower(), f.lipschitz_upper())    # 0.1, 2.0
    y = f(torch.randn(32, 6))      # train with any optimiser: the bounds hold after every step

Notes
-----
* Bounds are guaranteed by construction (no penalty, no projection step).
* BiLipNet layers: mu, nu are kept FIXED; certified values come from
  BiLipNet.get_bounds() (float32, e.g. 2.00000004 for 2.0).  In eval mode BiLipNet
  caches its orthogonal matrices: after loading new weights call .train() then .eval().
* For float64, build modules under torch.set_default_dtype(torch.float64): casting a
  float32-built module with .to(float64) leaves U, V orthonormal only to ~1e-7.
* Layers start with singular values / slopes at s_min + 0.8 (s_max - s_min); with a
  wide band and a small target scale, train longer or lower s_max.

Background
----------
Lipschitz-controlled linear and MLP modules.

These modules expose explicit singular-value bounds and report the available
strict or RMS-style lower-gain information in ``extra_repr``.

Global-budget BiLip MLP
-----------------------
Pass ``budget_from_global=True`` to ``BiLipMLP`` to treat ``s_min``/``s_max``
as *global* BiLip bounds for the whole network and have per-layer weight-matrix
bounds derived automatically via the product formula:

    m_f = s_min_layer^L * m_phi^(L-1)   [all-square / all-expansion]
    L_f = s_max_layer^L * L_phi^(L-1)

where L = number of linear layers and (m_phi, L_phi) are the activation's
Bi-Lipschitz constants.  The default activation in this mode is
``BiLipSoftplus`` (phi_a(x) = a*x + (1-a)*softplus(x), globally (a, 1)-BiLip).

Compression layers and RMS lower gain
-------------------------------------
A compression layer W: R^m -> R^n (n < m) has a null space of dimension m-n,
so the *worst-case* lower Lipschitz constant is 0 — ||Wx|| can be zero for a
nonzero x in that null space.  However the *root-mean-square* lower gain is
non-zero for isotropic directions.  For x uniform on the unit sphere S^{m-1}:

    E[||V^T x||^2]  =  n / m        (trace formula for a rank-n projector)

so    E[||Wx||^2]  >=  s_min^2 * (n/m) * ||x||^2,

i.e. the dimension-adjusted RMS lower gain is  s_min * sqrt(n/m).

When layer_dims contains compression layers, _budget_per_layer_bounds accounts
for this via the correction factor  r_dims = Π_i sqrt(min(1, out_i/in_i)).
BiLipMLP.extra_repr reports ``aggregate_rms_lower_gain_bound`` alongside the
strict ``aggregate_lower_lipschitz_cert`` (which is unavailable for any MLP
containing a compression layer).

This RMS quantity is exact (not probabilistic): it is a deterministic statement
about the expected squared operator norm over Haar-uniform inputs, backed by
the same geometry as the Johnson-Lindenstrauss lemma.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrizations as P


def _format_float(value: float) -> str:
    return f"{value:.4g}"


def _map_regime(s_min: float, s_max: float) -> str:
    if s_max < 1.0:
        return "contractive"
    if s_min > 1.0:
        return "expansive"
    if s_min == 1.0 and s_max == 1.0:
        return "isometric"
    if s_max == 1.0:
        return "nonexpansive"
    return "mixed"


def _safe_product(values: Iterable[float]) -> float:
    result = 1.0
    for v in values:
        result *= float(v)
    return result


# ---------------------------------------------------------------------------
# Activation Bi-Lipschitz bounds
# ---------------------------------------------------------------------------

def _activation_bilip_bounds(activation: nn.Module) -> Tuple[float, float]:
    """Return (m_phi, L_phi) — lower and upper Lipschitz constants of activation.

    Returns (0.0, inf) for activations whose bounds are not tracked.
    """
    if isinstance(activation, BiLipSoftplus):
        return activation.a, 1.0
    if isinstance(activation, nn.Identity):
        return 1.0, 1.0
    if isinstance(activation, nn.ReLU):
        return 0.0, 1.0
    if isinstance(activation, nn.Tanh):
        return 0.0, 1.0
    if isinstance(activation, nn.Hardtanh):
        return 0.0, 1.0
    if isinstance(activation, nn.ELU):
        alpha = float(getattr(activation, "alpha", 1.0))
        return 0.0, max(1.0, alpha)
    if isinstance(activation, nn.LeakyReLU):
        slope = abs(float(getattr(activation, "negative_slope", 0.01)))
        return min(1.0, slope), max(1.0, slope)
    return 0.0, float("inf")


def _activation_lipschitz_bounds(activation: nn.Module) -> Tuple[bool, Optional[float]]:
    """Legacy helper: return (is_1_lipschitz, lower_bound_or_None)."""
    m, L = _activation_bilip_bounds(activation)
    is_1_lip = (L <= 1.0 + 1e-9)
    lower = m if m > 0.0 else None
    return is_1_lip, lower


# ---------------------------------------------------------------------------
# Budgeting
# ---------------------------------------------------------------------------

def _dim_ratio_factor(layer_dims: List[Tuple[int, int]]) -> float:
    """Product of sqrt(min(1, out/in)) over all layers.

    This is the RMS correction factor r_dims that accounts for compression
    layers in the global lower-Lipschitz budget.  For expansion/square layers
    the factor is 1; for a compression layer R^m -> R^n (n < m) it is
    sqrt(n/m) < 1, because E[||V^T x||^2] = n/m for x uniform on S^{m-1}.
    """
    r = 1.0
    for in_d, out_d in layer_dims:
        if out_d < in_d:
            r *= math.sqrt(out_d / in_d)
    return r


def _budget_per_layer_bounds(
    n_layers: int,
    global_lower: float,
    global_upper: float,
    act_lower: float,
    act_upper: float,
    layer_dims: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[float, float]:
    """Derive uniform per-layer [s_min, s_max] from global MLP lower/upper targets.

    All-square / all-expansion case
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For L linear layers and (L-1) activations with bounds (m_phi, L_phi):

        global_lower = s_min^L * m_phi^(L-1)
        global_upper = s_max^L * L_phi^(L-1)

    Solving for a uniform s_min, s_max gives the per-layer bounds.

    Compression layers (layer_dims provided)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For a compression layer R^m -> R^n (n < m), the worst-case lower Lipschitz
    is 0 (null space), but the isotropic RMS lower gain is s_min * sqrt(n/m).
    The dimension-corrected product formula is:

        global_lower = s_min^L * r_dims * m_phi^(L-1)

    where  r_dims = Π_i sqrt(min(1, out_i / in_i))  is the overall dimension
    ratio correction (= 1 for all-square/expansion architectures).

    Solving:  s_min = (global_lower / (r_dims * m_phi^(L-1))) ^ (1/L)

    This makes the budgeted s_min larger for networks with compression layers,
    compensating so that the aggregate RMS lower-gain target still hits global_lower.
    """
    n_acts = max(n_layers - 1, 0)
    inv_L = 1.0 / max(n_layers, 1)

    r_dims = _dim_ratio_factor(layer_dims) if layer_dims is not None else 1.0

    if act_lower > 0.0:
        s_min = (global_lower / (r_dims * act_lower ** n_acts)) ** inv_L
    else:
        s_min = (global_lower / r_dims) ** inv_L

    if act_upper > 0.0 and not math.isinf(act_upper):
        s_max = (global_upper / (act_upper ** n_acts)) ** inv_L
    else:
        s_max = global_upper ** inv_L

    return float(s_min), float(s_max)


# ---------------------------------------------------------------------------
# BiLipSoftplus activation
# ---------------------------------------------------------------------------

class BiLipSoftplus(nn.Module):
    """phi_a(x) = a*x + (1-a)*softplus(x), globally (a, 1)-Bi-Lipschitz.

    Derivative: phi'_a(x) = a + (1-a)*sigmoid(x).
    Since 0 < sigmoid(x) < 1: a < phi'_a(x) < 1 for a in (0, 1).

    Unlike plain Tanh or ELU, this activation has a *strictly positive* global
    lower Lipschitz constant equal to ``a``, which is required for the full
    Bi-Lipschitz product budget to hold through the MLP.

    Args:
        a: residual weight in (0, 1).  Controls the trade-off between
           expressivity (small a) and lower Lipschitz constant (large a).
           Default 0.5 gives (0.5, 1)-BiLip.
    """

    def __init__(self, a: float = 0.5):
        super().__init__()
        if not (0.0 < a < 1.0):
            raise ValueError(f"BiLipSoftplus requires a in (0, 1), got {a}")
        self.a = float(a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * x + (1.0 - self.a) * F.softplus(x)

    def lipschitz_lower(self) -> float:
        return self.a

    def lipschitz_upper(self) -> float:
        return 1.0

    def extra_repr(self) -> str:
        return f"a={self.a:.4g}, bilip=({self.a:.4g}, 1.0)"


# ---------------------------------------------------------------------------
# BiLipLinear
# ---------------------------------------------------------------------------

class BiLipLinear(nn.Module):
    """
    Linear map with constrained singular values:

        W = U diag(sigma) V^T
        sigma_i in [s_min, s_max]

    Supports two parameterizations:
    - ``thin``: rectangular orthogonal factors (memory efficient)
    - ``full``: full square orthogonal factors, sliced to rank r
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s_min: float = 0.01,
        s_max: float = 0.99,
        bias: bool = True,
        init: str = "random",
        factorization: str = "thin",
        orthogonal_map: str = "cayley",
        factored: bool = True,
        freeze_U: bool = False,
        freeze_V: bool = False,
    ):
        super().__init__()
        if s_min < 0.0:
            raise ValueError(f"Expected s_min >= 0, got {s_min}.")
        if s_max < s_min:
            raise ValueError(f"Expected s_max >= s_min, got s_min={s_min}, s_max={s_max}.")
        if factorization not in {"thin", "full"}:
            raise ValueError(f"Unknown factorization '{factorization}', use 'thin' or 'full'.")
        if init not in {"random", "identity"}:
            raise ValueError(f"Unknown init '{init}', use 'random' or 'identity'.")
        if orthogonal_map not in {"matrix_exp", "cayley", "householder"}:
            raise ValueError(f"Unknown orthogonal_map '{orthogonal_map}'.")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = min(self.in_features, self.out_features)
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.init = init
        self.factorization = factorization
        self.orthogonal_map = orthogonal_map
        # Factored forward y = U (sigma * (V^T x)) avoids materialising the full
        # weight W = U diag(sigma) V^T (saves the O(out*in*rank) assembly and the
        # dense x @ W matmul, replacing them with two thin matmuls + a scale).
        self.factored = bool(factored)

        if self.factorization == "full":
            self.U = P.orthogonal(nn.Linear(self.out_features, self.out_features, bias=False), orthogonal_map=orthogonal_map)
            self.V = P.orthogonal(nn.Linear(self.in_features, self.in_features, bias=False), orthogonal_map=orthogonal_map)
        else:
            self.U = P.orthogonal(nn.Linear(self.rank, self.out_features, bias=False), orthogonal_map=orthogonal_map)
            self.V = P.orthogonal(nn.Linear(self.rank, self.in_features, bias=False), orthogonal_map=orthogonal_map)

        self.raw_sigma = nn.Parameter(torch.zeros(self.rank))
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None

        self.reset_parameters()

        # Optionally freeze the left factor U: materialise it once as a constant
        # buffer and skip its (matrix-exp/Cayley) orthogonalisation at every
        # forward.  The bi-Lipschitz guarantee is unaffected (U stays orthogonal,
        # singular values still in [s_min, s_max]); only the trainable left
        # singular subspace is given up.
        self.freeze_U = bool(freeze_U)
        self.freeze_V = bool(freeze_V)
        if self.freeze_U:
            with torch.no_grad():
                u_mat = self.U.weight.detach().clone()
            for p in self.U.parameters():
                p.requires_grad_(False)
            self.register_buffer("U_frozen", u_mat)
        else:
            self.U_frozen = None
        if self.freeze_V:
            with torch.no_grad():
                v_mat = self.V.weight.detach().clone()
            for p in self.V.parameters():
                p.requires_grad_(False)
            self.register_buffer("V_frozen", v_mat)
        else:
            self.V_frozen = None

        if self.s_min == 0.0:
            warnings.warn(
                "BiLipLinear configured with s_min=0: no positive lower-Lipschitz certificate, "
                "rank collapse is possible, and invertibility is not certified even for square maps.",
                UserWarning,
                stacklevel=2,
            )
        if self.s_min == 0.0 and self.s_max == 0.0:
            warnings.warn(
                "BiLipLinear configured with s_min=s_max=0: the layer can collapse to the zero map.",
                UserWarning,
                stacklevel=2,
            )
        if self.s_max > 1.0:
            warnings.warn(
                "BiLipLinear configured with s_max>1: this is an expansive (non-contractive) setting.",
                UserWarning,
                stacklevel=2,
            )

    def _set_raw_sigma_from_target(self, target: float) -> None:
        with torch.no_grad():
            if self.s_max == self.s_min:
                self.raw_sigma.zero_()
                return
            p = (target - self.s_min) / (self.s_max - self.s_min)
            p = float(min(max(p, 1e-6), 1.0 - 1e-6))
            self.raw_sigma.fill_(math.log(p / (1.0 - p)))

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.init == "identity":
                if self.factorization == "full":
                    self.U.weight.copy_(torch.eye(self.out_features))
                    self.V.weight.copy_(torch.eye(self.in_features))
                else:
                    u = torch.zeros(self.out_features, self.rank, device=self.U.weight.device, dtype=self.U.weight.dtype)
                    v = torch.zeros(self.in_features, self.rank, device=self.V.weight.device, dtype=self.V.weight.dtype)
                    diag_u = min(self.out_features, self.rank)
                    diag_v = min(self.in_features, self.rank)
                    u[torch.arange(diag_u), torch.arange(diag_u)] = 1.0
                    v[torch.arange(diag_v), torch.arange(diag_v)] = 1.0
                    self.U.weight.copy_(u)
                    self.V.weight.copy_(v)
            else:
                nn.init.normal_(self.U.weight, mean=0.0, std=0.02)
                nn.init.normal_(self.V.weight, mean=0.0, std=0.02)
            self._set_raw_sigma_from_target(self._sigma_init_target())

            if self.bias is not None:
                self.bias.zero_()

    def _sigma_init_target(self) -> float:
        if self.init == "identity":
            return (self.s_min + self.s_max) * 0.5
        return self.s_min + 0.8 * (self.s_max - self.s_min)

    def sigmas(self) -> torch.Tensor:
        return self.s_min + (self.s_max - self.s_min) * torch.sigmoid(self.raw_sigma)

    def _factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        u_w = self.U_frozen if self.freeze_U else self.U.weight
        v_w = self.V_frozen if self.freeze_V else self.V.weight
        if self.factorization == "full":
            return u_w[:, : self.rank], v_w[:, : self.rank]
        return u_w, v_w

    def weight_matrix(self) -> torch.Tensor:
        sigma = self.sigmas()
        u, v = self._factors()
        return u @ torch.diag(sigma) @ v.transpose(0, 1)

    def lipschitz_upper(self) -> float:
        return self.s_max

    def lipschitz_lower(self) -> float:
        # Compression layers (out < in) have a non-trivial null space: ||Wx|| can be 0
        # for x orthogonal to V's column space, so the worst-case lower Lipschitz is 0.
        # Expansion layers (out >= in) ARE lower Lipschitz with constant s_min.
        if self.out_features < self.in_features:
            return 0.0
        return self.s_min

    def rms_lipschitz_lower(self) -> float:
        """RMS lower gain: s_min * sqrt(min(1, out/in)).

        For compression layers (out < in), the worst-case lower Lipschitz is 0,
        but for x uniform on S^{in-1}: E[||Wx||^2] >= s_min^2 * (out/in) * ||x||^2,
        so the root-mean-square lower gain is s_min * sqrt(out/in).
        For square and expansion layers this equals s_min (the worst-case bound).
        """
        ratio = min(1.0, self.out_features / self.in_features)
        return self.s_min * math.sqrt(ratio)

    def _is_square(self) -> bool:
        return self.in_features == self.out_features

    def _has_bilip_embedding_certificate(self) -> bool:
        return self.out_features >= self.in_features and self.s_min > 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.factored:
            # y = U (sigma * (V^T x)) = ((x @ V) * sigma) @ U^T  (no full W)
            sigma = self.sigmas()
            u, v = self._factors()
            y = ((x @ v) * sigma) @ u.transpose(0, 1)
        else:
            y = x @ self.weight_matrix().transpose(-2, -1)
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        sigma = self.sigmas().detach()
        sv_min = float(sigma.min().item()) if sigma.numel() > 0 else 0.0
        sv_max = float(sigma.max().item()) if sigma.numel() > 0 else 0.0
        regime = _map_regime(self.s_min, self.s_max)
        degenerate = self.s_min == 0.0
        guarantee_type = "bilip_embedding_certified" if self._has_bilip_embedding_certificate() else "upper_bound_only"
        invertible_if_square = self._is_square() and self.s_min > 0.0
        lower_cert = _format_float(self.s_min) if self.s_min > 0.0 else "unavailable"
        upper_cert = _format_float(self.s_max)

        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"factorization={self.factorization}, init={self.init}, "
            f"s_min={_format_float(self.s_min)}, s_max={_format_float(self.s_max)}, "
            f"current_sv_min={_format_float(sv_min)}, current_sv_max={_format_float(sv_max)}, "
            f"regime={regime}, degenerate={str(degenerate).lower()}, "
            f"guarantee_type={guarantee_type}, "
            f"invertible_if_input_eq_output={str(invertible_if_square).lower()}, "
            f"lower_lipschitz_cert={lower_cert}, upper_lipschitz_cert={upper_cert}"
        )


# ---------------------------------------------------------------------------
# BiLipSpline
# ---------------------------------------------------------------------------

class BiLipSpline(BiLipLinear):
    """
    Nonlinear layer with the SVD layer's geometry:

        y = U h(V^T x) + b,     h(z)_i = h_i(z_i),     s_min <= h_i'(t) <= s_max

    The diagonal diag(sigma) of BiLipLinear is replaced by one learnable scalar
    function h_i per singular direction i: a continuous piecewise-linear spline
    on the fixed knots t_0 < ... < t_K (uniform on [-knot_range, knot_range])
    with h_i(0) = 0.  Its K+2 slopes (left tail, K segments, right tail) are

        c_{i,k} = s_min + (s_max - s_min) * sigmoid(theta_{i,k})

    so every slope lies in [s_min, s_max] by construction, and

        h_i(t) = c_{i,0} min(t - t_0, 0)
               + sum_{k=1..K} c_{i,k} clamp(t - t_{k-1}, 0, t_k - t_{k-1})
               + c_{i,K+1} max(t - t_K, 0)   - (same expression at t = 0).

    Guarantees (same as BiLipLinear, no activation factor):
      * s_min |p - q| <= |h_i(p) - h_i(q)| <= s_max |p - q|   (integrate the slope)
      * hence ||f(x) - f(y)|| in [s_min, s_max] ||x - y|| for square/expansion
        (U, V^T isometric), and the upper bound s_max for compression.
      * Jacobian  J_f(x) = U diag(h'(V^T x)) V^T  is an exact SVD: the singular
        vectors are U, V everywhere, the singular values are the local slopes.
      * Equal slopes per coordinate (c_{i,k} = sigma_i) give exactly the linear
        layer U diag(sigma) V^T x + b; at init all slopes equal the BiLipLinear
        init value, so the layer starts linear.
      * Closed-form inverse on the range: x = V h^{-1}(U^T (y - b)).

    ``sigmas()`` returns the (rank, K+2) slope table: every entry is a singular
    value the Jacobian takes somewhere, so min/max over it are the realised band.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s_min: float = 0.01,
        s_max: float = 0.99,
        n_knots: int = 8,
        knot_range: float = 3.0,
        bias: bool = True,
        init: str = "random",
        factorization: str = "thin",
        orthogonal_map: str = "cayley",
        freeze_U: bool = False,
        freeze_V: bool = False,
    ):
        super().__init__(
            in_features, out_features, s_min=s_min, s_max=s_max, bias=bias, init=init,
            factorization=factorization, orthogonal_map=orthogonal_map, factored=True,
            freeze_U=freeze_U, freeze_V=freeze_V,
        )
        if n_knots < 1:
            raise ValueError(f"Expected n_knots >= 1, got {n_knots}.")
        if knot_range <= 0.0:
            raise ValueError(f"Expected knot_range > 0, got {knot_range}.")
        self.n_knots = int(n_knots)
        self.knot_range = float(knot_range)
        # knots t_0 .. t_K  (K = n_knots segments); slopes: [left tail, K segments, right tail]
        self.register_buffer("knots", torch.linspace(-self.knot_range, self.knot_range, self.n_knots + 1))
        self.raw_sigma = nn.Parameter(torch.zeros(self.rank, self.n_knots + 2))
        self._set_raw_sigma_from_target(self._sigma_init_target())

    def _spline_uncentered(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t = self.knots
        width = t[1:] - t[:-1]                                                   # (K,)
        inner = torch.minimum(torch.relu(z.unsqueeze(-1) - t[:-1]), width)     # (..., r, K)
        h = (inner * c[:, 1:-1]).sum(-1)
        h = h + c[:, 0] * torch.clamp(z - t[0], max=0.0)
        return h + c[:, -1] * torch.clamp(z - t[-1], min=0.0)

    def spline(self, z: torch.Tensor) -> torch.Tensor:
        """h(z) with h_i(0) = 0, applied coordinate-wise to z of shape (..., rank)."""
        c = self.sigmas()
        return self._spline_uncentered(z, c) - self._spline_uncentered(torch.zeros_like(c[:, 0]), c)

    def local_gains(self, x: torch.Tensor) -> torch.Tensor:
        """h'(V^T x): the exact singular values of the Jacobian at x, shape (..., rank)."""
        _, v = self._factors()
        z = x @ v
        seg = torch.bucketize(z.detach(), self.knots)     # 0 = left tail, K+1 = right tail
        return self.sigmas()[torch.arange(self.rank, device=z.device), seg]

    def weight_matrix(self) -> torch.Tensor:
        raise NotImplementedError("BiLipSpline is nonlinear; use local_gains(x) for its Jacobian spectrum.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = U h(V^T x) + b = h(x @ V) @ U^T + b
        u, v = self._factors()
        y = self.spline(x @ v) @ u.transpose(0, 1)
        if self.bias is not None:
            y = y + self.bias
        return y

    @torch.no_grad()
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """x = V h^{-1}(U^T (y - b)); exact for y in the layer's range (square/expansion)."""
        if self.out_features < self.in_features:
            raise ValueError("BiLipSpline.inverse: a compression layer is not injective.")
        u, v = self._factors()
        if self.bias is not None:
            y = y - self.bias
        w = y @ u                                                  # target values of h, (..., r)
        c = self.sigmas()
        H = self.spline(self.knots.unsqueeze(-1).expand(-1, self.rank)).transpose(0, 1).contiguous()  # h_i(t_k), (r, K+1)
        flat = w.reshape(-1, self.rank).transpose(0, 1).contiguous()                                # (r, N)
        seg = torch.searchsorted(H, flat)                          # 0 = left tail, K+1 = right tail
        anchor = (seg - 1).clamp(min=0)                            # knot the segment starts from
        rows = torch.arange(self.rank, device=y.device).unsqueeze(-1)
        z = self.knots[anchor] + (flat - H[rows, anchor]) / c[rows, seg]
        return z.transpose(0, 1).reshape(w.shape) @ v.transpose(0, 1)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", n_knots={self.n_knots}, knot_range={_format_float(self.knot_range)}"


# ---------------------------------------------------------------------------
# BiLipNetLayer (square layers via ACFR's BiLipNet)
# ---------------------------------------------------------------------------

class BiLipNetLayer(nn.Module):
    """
    Square (n -> n) bi-Lipschitz layer backed by ACFR's BiLipNet, vendored 1:1 in
    ``src/acfr`` (RobustNeuralNetworks, MIT; Wang, Dvijotham & Manchester 2024):

        f(x) = P_d M_d( ... P_1 M_1( P_0 x ) ... )

    P_k: orthogonal (Cayley) linear layers with bias, ||P_k x - P_k y|| = ||x - y||.
    M_k: MonLipNet blocks  M(x) = 1/2 ((mu_k + nu_k) x + sqrt(nu_k - mu_k) yh(x) Q) + b_y,
         each (mu_k, nu_k)-bi-Lipschitz with mu_k = s_min^(1/d), nu_k = s_max^(1/d).
    Hence  s_min ||x - y|| <= ||f(x) - f(y)|| <= s_max ||x - y||  for every x, y,
    with mu, nu FIXED (is_*_fixed=True).  Exposes the BiLipLinear reporting
    interface so BiLipMLP can mix it with BiLipLinear / BiLipSpline layers.

    ``s_min`` / ``s_max`` are the certified values from ``BiLipNet.get_bounds()``
    (float32-rounded, so e.g. 0.49999998 for a requested 0.5).  Requires the
    ``acfr`` package (which imports jax + flax), loaded only when this layer is built.
    """

    def __init__(
        self,
        features: int,
        s_min: float,
        s_max: float,
        unit_features: Optional[Sequence[int]] = None,
        depth: int = 1,
    ):
        super().__init__()
        if not (0.0 < s_min < s_max):
            raise ValueError(f"BiLipNetLayer needs 0 < s_min < s_max, got s_min={s_min}, s_max={s_max}.")
        # lazy: bilip_layers works without acfr (and jax/flax) unless a BiLipNet layer is built
        try:
            from acfr import BiLipNet            # src/ on sys.path (experiment drivers)
        except ModuleNotFoundError:
            from .acfr import BiLipNet           # imported as the package module src.bilip_layers

        self.in_features = self.out_features = self.rank = int(features)
        self.depth = int(depth)
        self.unit_features = list(unit_features) if unit_features is not None else [self.in_features]
        self.net = BiLipNet(
            features=self.in_features, unit_features=self.unit_features, mu=float(s_min), nu=float(s_max),
            depth=self.depth, is_mu_fixed=True, is_nu_fixed=True, is_tau_fixed=True,
        )
        mu, nu, _ = self.net.get_bounds()
        self.s_min, self.s_max = float(mu), float(nu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def sigmas(self) -> torch.Tensor:
        """Certified gain band [s_min, s_max] (BiLipNet does not expose per-direction gains)."""
        return torch.tensor([self.s_min, self.s_max], device=self.net.orth_layers[0].weight.device)

    def lipschitz_lower(self) -> float:
        return self.s_min

    def lipschitz_upper(self) -> float:
        return self.s_max

    def rms_lipschitz_lower(self) -> float:
        return self.s_min

    def extra_repr(self) -> str:
        return (f"features={self.in_features}, unit_features={self.unit_features}, depth={self.depth}, "
                f"lower_lipschitz_cert={_format_float(self.s_min)}, upper_lipschitz_cert={_format_float(self.s_max)}")


# ---------------------------------------------------------------------------
# BiLipMLP
# ---------------------------------------------------------------------------

class BiLipMLP(nn.Module):
    """MLP composed of BiLipLinear layers and a Bi-Lipschitz activation.

    Two modes
    ---------
    Per-layer mode (``budget_from_global=False``, default for backward compat):
        ``s_min`` / ``s_max`` are applied directly to every BiLipLinear layer.
        The resulting global MLP BiLip bounds are
        ``[s_min^L * m_phi^(L-1), s_max^L * L_phi^(L-1)]``.

    Global-budget mode (``budget_from_global=True``):
        ``s_min`` / ``s_max`` are the *desired global* MLP BiLip bounds.
        Per-layer weight-matrix bounds are derived from the product formula
        (see ``_budget_per_layer_bounds``).  The default activation switches
        to ``BiLipSoftplus`` (globally (softplus_a, 1)-BiLip), which ensures
        a strictly positive lower-Lipschitz constant at every layer.

    Args:
        input_dim, output_dim, hidden_dims: network shape.
        activation_cls: callable ``() -> nn.Module``.  Defaults to
            ``BiLipSoftplus(softplus_a)`` when ``budget_from_global=True``,
            or ``nn.ELU`` otherwise.
        s_min, s_max: per-layer (mode=False) or global (mode=True) bounds.
        budget_from_global: if True, treat s_min/s_max as global targets and
            derive per-layer bounds from them and the activation constants.
        softplus_a: the ``a`` parameter for the default BiLipSoftplus
            activation (only used when ``budget_from_global=True`` and
            ``activation_cls`` is None).
        layer_type: ``"linear"`` (BiLipLinear, default) or ``"spline"``
            (BiLipSpline, y = U h(V^T x) + b).  The spline is itself the
            nonlinearity, so with ``layer_type="spline"`` and no
            ``activation_cls`` the activation is ``nn.Identity`` (bounds (1, 1)):
            the global bounds are then [s_min_layer^L, s_max_layer^L] with no
            activation factor m_phi^(L-1).
        spline_knots, spline_range: ``n_knots`` / ``knot_range`` of BiLipSpline.
        square_layer: ``"same"`` (default: square layers use ``layer_type``) or
            ``"bilipnet"``: every layer with in_dim == out_dim becomes a
            BiLipNetLayer (ACFR BiLipNet, bounds [s_min_layer, s_max_layer]),
            while dimension-changing layers stay BiLipLinear / BiLipSpline.  The
            budget formula is unchanged (each layer is still
            (s_min_layer, s_max_layer)-bounded); activations between layers
            follow the usual defaults.  ``bias``, ``init``, ``factorization``,
            ``orthogonal_map`` and ``freeze_*`` do not apply to BiLipNet layers.
        bilipnet_units, bilipnet_depth: ``unit_features`` (default [dim]) and
            ``depth`` of each BiLipNet layer.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        activation_cls: Optional[Callable[[], nn.Module]] = None,
        s_min: float = 0.01,
        s_max: float = 0.99,
        bias: bool = True,
        init: str = "random",
        factorization: str = "thin",
        budget_from_global: bool = False,
        softplus_a: float = 0.8,
        orthogonal_map: str = "cayley",
        factored: bool = True,
        freeze_U: bool = False,
        freeze_V: bool = False,
        layer_type: str = "linear",
        spline_knots: int = 8,
        spline_range: float = 3.0,
        square_layer: str = "same",
        bilipnet_units: Optional[Sequence[int]] = None,
        bilipnet_depth: int = 1,
    ):
        super().__init__()
        if layer_type not in {"linear", "spline"}:
            raise ValueError(f"Unknown layer_type '{layer_type}', use 'linear' or 'spline'.")
        if square_layer not in {"same", "bilipnet"}:
            raise ValueError(f"Unknown square_layer '{square_layer}', use 'same' or 'bilipnet'.")
        self.layer_type = layer_type
        self.square_layer = square_layer
        self.orthogonal_map = orthogonal_map
        self.factored = bool(factored)
        self.freeze_U = bool(freeze_U)
        self.freeze_V = bool(freeze_V)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dims = [int(h) for h in hidden_dims]
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.factorization = factorization
        self.init = init
        self.budget_from_global = bool(budget_from_global)
        self.softplus_a = float(softplus_a)

        # Resolve activation factory
        if activation_cls is None:
            if layer_type == "spline":
                self.activation_factory = nn.Identity
            elif budget_from_global:
                _a = self.softplus_a
                self.activation_factory: Callable[[], nn.Module] = lambda: BiLipSoftplus(_a)
            else:
                self.activation_factory = nn.ELU
        else:
            self.activation_factory = activation_cls

        # Probe activation bounds using a temporary instance
        _act_sample = self.activation_factory()
        act_lower, act_upper = _activation_bilip_bounds(_act_sample)
        self._act_lower = act_lower
        self._act_upper = act_upper

        # Resolve per-layer weight bounds
        dims: List[int] = [self.input_dim, *self.hidden_dims, self.output_dim]
        n_layers = len(dims) - 1

        if budget_from_global:
            dims_pairs = list(zip(dims[:-1], dims[1:]))
            layer_s_min, layer_s_max = _budget_per_layer_bounds(
                n_layers=n_layers,
                global_lower=s_min,
                global_upper=s_max,
                act_lower=act_lower,
                act_upper=act_upper,
                layer_dims=dims_pairs,
            )
            if act_lower == 0.0:
                warnings.warn(
                    f"BiLipMLP(budget_from_global=True): activation '{type(_act_sample).__name__}' "
                    f"has m_phi=0 — lower-gain budget cannot be distributed through activations. "
                    f"Use BiLipSoftplus or LeakyReLU for a positive lower-gain budget.",
                    UserWarning,
                    stacklevel=2,
                )
            if layer_s_min > layer_s_max:
                # Heavy compression makes the RMS lower-gain target infeasible.
                # Compute what IS achievable and warn clearly.
                r_dims = _dim_ratio_factor(dims_pairs)
                n_acts = max(n_layers - 1, 0)
                achievable = (layer_s_max ** n_layers) * r_dims * (act_lower ** n_acts if act_lower > 0 else 1.0)
                compression_layers = [(i, o) for i, o in dims_pairs if o < i]
                warnings.warn(
                    f"BiLipMLP: RMS lower-gain target {s_min} is infeasible for this architecture "
                    f"— compression layers {compression_layers} reduce too much of the budget "
                    f"(required layer_s_min={layer_s_min:.4f} > layer_s_max={layer_s_max:.4f}). "
                    f"Maximum achievable RMS lower gain with this architecture: {achievable:.4f}. "
                    f"Fix: increase bilip_softplus_a, widen hidden dims, or lower the target. "
                    f"Clamping layer_s_min to layer_s_max for now.",
                    UserWarning,
                    stacklevel=2,
                )
                layer_s_min = layer_s_max
        else:
            layer_s_min, layer_s_max = s_min, s_max

        self._layer_s_min = layer_s_min
        self._layer_s_max = layer_s_max

        common = dict(
            s_min=layer_s_min, s_max=layer_s_max, bias=bias, init=self.init,
            factorization=self.factorization, orthogonal_map=self.orthogonal_map,
            freeze_U=self.freeze_U, freeze_V=self.freeze_V,
        )
        def make_layer(i: int) -> nn.Module:
            if square_layer == "bilipnet" and dims[i] == dims[i + 1]:
                return BiLipNetLayer(dims[i], layer_s_min, layer_s_max,
                                     unit_features=bilipnet_units, depth=bilipnet_depth)
            if layer_type == "spline":
                return BiLipSpline(dims[i], dims[i + 1], n_knots=spline_knots,
                                   knot_range=spline_range, **common)
            return BiLipLinear(dims[i], dims[i + 1], factored=self.factored, **common)

        self.linears = nn.ModuleList([make_layer(i) for i in range(n_layers)])
        self.activations = nn.ModuleList([self.activation_factory() for _ in range(n_layers - 1)])

    def _activation_1_lipschitz(self) -> bool:
        return all(_activation_bilip_bounds(act)[1] <= 1.0 + 1e-9 for act in self.activations)

    def _activation_lower_bound(self) -> Optional[float]:
        if not self.activations:
            return 1.0
        m = _safe_product(_activation_bilip_bounds(act)[0] for act in self.activations)
        return m if m > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if i < len(self.activations):
                x = self.activations[i](x)
        return x

    def lipschitz_upper(self) -> float:
        """Certified upper Lipschitz bound: product of per-layer s_max times activation upper bounds."""
        upper_linear = _safe_product(layer.lipschitz_upper() for layer in self.linears)
        return upper_linear * (self._act_upper ** len(self.activations))

    def lipschitz_lower(self) -> Optional[float]:
        """Certified lower Lipschitz bound.

        For square/expansion-only networks: strict worst-case lower bound
        (product of per-layer s_min × activation lower bounds).

        For networks with compression layers: RMS lower-gain bound
        (product of per-layer rms_lipschitz_lower × activation lower bounds),
        which accounts for the dimension reduction via sqrt(out/in) per layer.
        This is not a strict worst-case lower-Lipschitz certificate.
        Returns None if the activation has zero lower bound.
        """
        act_lower = self._activation_lower_bound()
        if act_lower is None or act_lower <= 0.0:
            return None
        all_no_compression = all(layer.lipschitz_lower() > 0.0 for layer in self.linears)
        if all_no_compression:
            return _safe_product(layer.lipschitz_lower() for layer in self.linears) * act_lower
        # Fall back to an RMS lower-gain bound (not a strict certificate).
        return _safe_product(layer.rms_lipschitz_lower() for layer in self.linears) * act_lower

    def extra_repr(self) -> str:
        layer_sigma_mins = [float(layer.sigmas().detach().min().item()) for layer in self.linears]
        layer_sigma_maxs = [float(layer.sigmas().detach().max().item()) for layer in self.linears]
        upper_linear = _safe_product(layer.lipschitz_upper() for layer in self.linears)
        act_1_lip = self._activation_1_lipschitz()
        act_lower = self._activation_lower_bound()

        global_upper = upper_linear * (self._act_upper ** len(self.activations)) if not math.isinf(self._act_upper) else float("inf")

        # Strict worst-case lower bound (unavailable if any compression layer)
        all_no_compression = all(layer.lipschitz_lower() > 0.0 for layer in self.linears)
        if all_no_compression and act_lower is not None and act_lower > 0.0:
            strict_lower = _safe_product(layer.lipschitz_lower() for layer in self.linears) * act_lower
        else:
            strict_lower = None

        # RMS lower-gain bound: dimension-compensated (always computable)
        # act_lower is already the product over all activations (from _activation_lower_bound)
        rms_lower_linear = _safe_product(layer.rms_lipschitz_lower() for layer in self.linears)
        rms_lower = rms_lower_linear * act_lower if (act_lower is not None and act_lower > 0.0) else None

        has_compression = any(layer.in_features > layer.out_features for layer in self.linears)
        all_square = all(layer.in_features == layer.out_features for layer in self.linears)
        invertible_if_input_eq_output = (
            self.input_dim == self.output_dim
            and all_square
            and strict_lower is not None
            and strict_lower > 0.0
        )
        if invertible_if_input_eq_output:
            guarantee_type = "bilip_embedding_certified"
        elif has_compression and rms_lower is not None and rms_lower > 0.0 and self.budget_from_global:
            guarantee_type = "rms_lower_gain_bound"
        else:
            guarantee_type = "upper_bound_only"

        regime = _map_regime(self.s_min, self.s_max)
        degenerate = self.s_min == 0.0

        strict_lower_text = _format_float(strict_lower) if strict_lower is not None else "unavailable"
        rms_lower_text = _format_float(rms_lower) if rms_lower is not None else "unavailable"
        sigma_mins_text = "[" + ",".join(_format_float(v) for v in layer_sigma_mins) + "]"
        sigma_maxs_text = "[" + ",".join(_format_float(v) for v in layer_sigma_maxs) + "]"

        budget_text = (
            f"budget_from_global=True, global_s_min={_format_float(self.s_min)}, "
            f"global_s_max={_format_float(self.s_max)}, "
            f"layer_s_min={_format_float(self._layer_s_min)}, layer_s_max={_format_float(self._layer_s_max)}, "
        ) if self.budget_from_global else (
            f"budget_from_global=False, "
        )

        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, hidden_dims={self.hidden_dims}, "
            f"n_linear_layers={len(self.linears)}, layer_type={self.layer_type}, "
            f"square_layer={self.square_layer}, "
            f"activation={type(self.activations[0]).__name__ if self.activations else 'none'}, "
            f"factorization={self.factorization}, init={self.init}, "
            + budget_text +
            f"layer_sigma_mins={sigma_mins_text}, layer_sigma_maxs={sigma_maxs_text}, "
            f"regime={regime}, degenerate={str(degenerate).lower()}, "
            f"activation_1_lipschitz={str(act_1_lip).lower()}, "
            f"aggregate_upper_lipschitz_cert={_format_float(global_upper)}, "
            f"aggregate_lower_lipschitz_cert={strict_lower_text}, "
            f"aggregate_rms_lower_gain_bound={rms_lower_text}, "
            f"guarantee_type={guarantee_type}, "
            f"invertible_if_input_eq_output={str(invertible_if_input_eq_output).lower()}"
        )

# ---------------------------------------------------------------------------
# Self-test:  python bilipmlp.py
# ---------------------------------------------------------------------------

def _check(f: nn.Module, d_in: int, name: str) -> None:
    m, L = f.lipschitz_lower(), f.lipschitz_upper()
    x, y = 3 * torch.randn(20000, d_in), 3 * torch.randn(20000, d_in)
    with torch.no_grad():
        r = (f(x) - f(y)).norm(dim=1) / (x - y).norm(dim=1)
    ok = bool(r.min() >= m - 1e-6 and r.max() <= L + 1e-6)
    print(f"[{'ok' if ok else 'FAIL'}] {name:34s} certified [{m:.4f}, {L:.4f}]  "
          f"observed gain [{r.min():.4f}, {r.max():.4f}]")
    assert ok


def _self_test() -> None:
    torch.manual_seed(0)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)          # exact orthogonality for the checks
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for layer_type in ("linear", "spline"):
                f = BiLipMLP(8, 8, hidden_dims=[8, 8], s_min=0.1, s_max=2.0,
                             budget_from_global=True, layer_type=layer_type)
                with torch.no_grad():                   # spread parameters over the whole band
                    for p in f.parameters():
                        p.add_(0.5 * torch.randn_like(p))
                _check(f, 8, f"{layer_type} 8->8->8->8")

            layer = BiLipSpline(8, 12, s_min=0.3, s_max=1.5)
            with torch.no_grad():
                layer.raw_sigma.normal_(0.0, 3.0)
                layer.bias.normal_()
            x = 4 * torch.randn(1000, 8)
            err = float((layer.inverse(layer(x)) - x).abs().max())
            print(f"[{'ok' if err < 1e-9 else 'FAIL'}] BiLipSpline inverse round trip, max error {err:.1e}")
            assert err < 1e-9

            try:
                f = BiLipMLP(6, 24, hidden_dims=[12, 12], s_min=0.1, s_max=2.0,
                             budget_from_global=True, square_layer="bilipnet")
            except ImportError as exc:
                print(f"[skip] BiLipNet hybrid: {exc} (needs acfr/ next to this file + jax, flax)")
            else:
                kinds = ", ".join(type(l).__name__ for l in f.linears)
                _check(f, 6, "hybrid 6->12->12->24 (init)")
                opt = torch.optim.Adam(f.parameters(), lr=1e-2)
                xs, ys = torch.randn(256, 6), torch.randn(256, 24)
                for _ in range(200):
                    opt.zero_grad()
                    F.mse_loss(f(xs), ys).backward()
                    opt.step()
                _check(f, 6, "hybrid 6->12->12->24 (trained)")
                print(f"     layers: {kinds}")
    finally:
        torch.set_default_dtype(prev)
    print("all checks passed")


if __name__ == "__main__":
    _self_test()
