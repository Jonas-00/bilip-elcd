"""MonLipNetD: PER-DIMENSION-band monotone Lipschitz network (OUR file, not upstream's).

Upstream ``MonLipNet`` (``monlipnet_torch.py``, Wang/Dvijotham/Manchester ICML 2024,
arXiv:2402.01344) certifies one SCALAR band: F(x) = mu*x + H(x) is mu-strongly-monotone
and nu-Lipschitz, gam = nu - mu.  In the "normalized" hidden coordinates zhat (the
diagonal Psi cancels for relu) upstream's explicit form is::

    zhat_k = relu( V_k zhat_{k-1} + sqrt(2*gam) * S_k x + bhat_k ),   zhat_0 = 0
    y      = mu*x + sqrt(gam/2) * S^T zhat + b_y

with S (stacked, (m_total, n)) and V_k = 2 B_k A_{k-1}^T built from Cayley
(column-orthonormal) factors -- mirrored here 1:1 from upstream's
``MonLipNet.direct_to_explicit`` (monlipnet_torch.py).

THE GENERALIZATION (derived + verified numerically):
replace scalar mu -> M = diag(mu_1..mu_n) > 0 and scalar gam -> Gamma =
diag(gamma_1..gamma_n) > 0, per-dim nu_i := mu_i + gamma_i.  The ONLY changes::

    input scaling:  sqrt(2*gam) * S_k x   ->   sqrt(2) * S_k (Gamma^{1/2} x)
                    (row-vector batch: pre += sqrt(2) * (x * gamma.sqrt()) @ S_k.T)
    output:         y = M x + (1/sqrt(2)) * Gamma^{1/2} S^T zhat + b_y
                    (row-vector batch: y = x * mu + (zhat @ S) * gamma.sqrt()/sqrt(2) + b_y)

WHY THE SAME CAYLEY CONSTRUCTION STILL CERTIFIES IT.  The incremental certificate
LMI is

    2*Lambda - Lambda@W - W.T@Lambda >= 2 * Lambda @ U @ Gamma^{-1} @ U.T @ Lambda

with U_k = sqrt(2) * Psi_k^{-1} S_k Gamma^{1/2} (here Psi = I, the normalized form),
W the strictly-block-lower matrix of the V_k, Yhat = U.T @ Lambda.  Substituting
U = sqrt(2) * S @ Gamma^{1/2} makes U @ Gamma^{-1} @ U.T = 2 * S @ S.T -- the
Gamma^{1/2} * Gamma^{-1} * Gamma^{1/2} CANCELS -- so with Lambda = I/2 the LMI is

    I - (W + W.T)/2 - S @ S.T >= 0,

exactly the scalar-case certificate that the Cayley construction satisfies for ANY
parameter values.  Certified properties:

  (P1) the LMI above, eigvalsh >= -1e-10;
  (P2) monotonicity ellipsoid  <F(a)-F(c), a-c> >= (a-c)^T M (a-c);
  (P3) weighted cocoercivity of H = F - Mx - b_y:  dH^T Gamma^{-1} dH <= dx^T Gamma dx
       (H in the variable u = Gamma^{1/2} x is upstream's normalized 1-Lipschitz,
       monotone H-tilde; that is where P2/P3 come from);
  (P4) per-AXIS floors: along each axis e_i the directional response dF_i/dx_i >= mu_i;
  (P5) global band: incremental gain in [min_i mu_i, max_i nu_i].

NOT certified: per-dimension SINGULAR VALUES of the Jacobian (that would need a
frame-aligned diagonal Jacobian, which this is not); the upper bound is the weighted
ellipsoid of (P3), whose cross-gain between axes i and j scales like
sqrt(gamma_i * gamma_j), not min/max of the two.

LEARNABLE BOUNDS IN LADDER BRACKETS.  The floors mu_i live in per-rung brackets
[lo_i, hi_i] from the log-spaced ladder (``ladder_bounds``, copied from elcd.py):
mu_i = lo_i + (hi_i - lo_i) * sigmoid(s_i), s_i learnable, initialized so mu_i
starts at the rung center.  gamma_i = gamma_min + (nu_target - hi_i - gamma_min)
* sigmoid(g_i), so nu_i = mu_i + gamma_i < nu_target ALWAYS -- training cannot
break the certificate, only move inside it.

CONJUGATED FRAME.  ``ConjugatedMonLipNetD``: G(x) = P^T F(P x) with P = Cayley of
a learnable square generator (exactly orthogonal).  Then
<dG, dx> = <dF, P dx> >= (P dx)^T M (P dx): per-MODE floors mu_i along the learned
directions v_i = i-th ROW of P (row-vector convention: P v_i^T = e_i since
P P^T = I).  Inverse: G(x) = y  ==>  P^T F(P x) = y  ==>  F(P x) = P y
==>  P x = F^{-1}(P y)  ==>  x = P^T F^{-1}(P y).

"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .orthogonal_torch import cayley, norm


# ---------------------------------------------------------------------------
# copied from bilip_jepa/models/elcd.py ladder_bounds to keep _acfr self-contained
# ---------------------------------------------------------------------------
def ladder_bounds(
    n: int,
    sigma_slow: float,
    sigma_fast: float,
    delta: float,
    sigma_floor: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-rung bands of the log-spaced ladder.

    Rung centers are geometric in (1 - sigma):

        1 - c_i = (1 - sigma_slow) * r**i,
        r = ((1 - sigma_fast) / (1 - sigma_slow)) ** (1 / (n - 1))

    with multiplicative bands lo_i = 1 - (1-c_i)e^{+delta},
    hi_i = 1 - (1-c_i)e^{-delta}. The fast-end lower bounds are clipped at
    sigma_floor, which doubles as the certified global lower bound. Bands are
    pairwise disjoint (spectral collapse impossible by construction) iff
    exp(2*delta) < r; larger delta gives an overlapping "soft" ladder.

    Returns (centers, lo, hi), each of shape (n,), sorted slow -> fast.
    """
    if not (0.0 < sigma_fast < sigma_slow < 1.0):
        raise ValueError(
            f"need 0 < sigma_fast ({sigma_fast}) < sigma_slow ({sigma_slow}) < 1"
        )
    i = torch.arange(n, dtype=torch.float64)
    r = ((1.0 - sigma_fast) / (1.0 - sigma_slow)) ** (1.0 / max(n - 1, 1))
    one_minus_c = (1.0 - sigma_slow) * r ** i
    centers = 1.0 - one_minus_c
    lo = torch.clamp(1.0 - one_minus_c * math.exp(+delta), min=sigma_floor)
    hi = 1.0 - one_minus_c * math.exp(-delta)
    if (lo >= hi).any():
        raise ValueError(
            "ladder bands collapsed (lo >= hi); raise sigma_fast or lower sigma_floor/delta"
        )
    return centers.float(), lo.float(), hi.float()


class MonLipNetD(nn.Module):
    """Per-dimension-band MonLipNet: F(x) = M x + H(x), M = diag(mu_1..mu_n) > 0.

    Certified (see module docstring, P1-P5):
      * <F(a)-F(c), a-c> >= (a-c)^T M (a-c)   -- per-dim monotonicity floors mu_i;
      * dH^T Gamma^{-1} dH <= dx^T Gamma dx for H = F - Mx - b_y, Gamma = diag(gamma_i);
      * global incremental band [min_i mu_i, max_i nu_i], nu_i = mu_i + gamma_i.

    Exactly ONE of ``mu_ladder`` / ``mu_bounds`` / ``mu_fixed`` selects the floors:
      * mu_ladder = dict(slow=..., fast=..., delta=..., floor=...): learnable
        mu_i = lo_i + (hi_i - lo_i)*sigmoid(s_i) inside the log-spaced ladder rungs,
        initialized at the rung centers;
      * mu_bounds = (lo, hi): explicit per-dim brackets, learnable, init at midpoints;
      * mu_fixed = vector (or float): fixed floors, not learnable.

    gamma_i: with ``gamma_fixed=None`` (default), learnable
    gamma_i = gamma_min + (nu_target - hi_i - gamma_min)*sigmoid(g_i), which pins
    nu_i = mu_i + gamma_i < nu_target for ALL parameter values (requires
    nu_target > max_i hi_i + gamma_min).  ``gamma_fixed`` freezes gamma instead.
    """

    def __init__(
        self,
        features: int,
        unit_features: Sequence[int],
        mu_ladder: Optional[Dict[str, float]] = None,
        mu_bounds: Optional[Tuple] = None,
        mu_fixed: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        nu_target: Optional[float] = None,
        gamma_min: float = 1e-3,
        gamma_fixed: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
    ):
        super().__init__()
        self.features = int(features)
        self.units = list(int(u) for u in unit_features)
        self.gamma_min = float(gamma_min)
        n = self.features

        # ----- floors mu_i --------------------------------------------------
        n_modes = sum(x is not None for x in (mu_ladder, mu_bounds, mu_fixed))
        if n_modes != 1:
            raise ValueError("give exactly one of mu_ladder / mu_bounds / mu_fixed")
        if mu_ladder is not None:
            centers, lo, hi = ladder_bounds(
                n=n,
                sigma_slow=float(mu_ladder["slow"]),
                sigma_fast=float(mu_ladder["fast"]),
                delta=float(mu_ladder["delta"]),
                sigma_floor=float(mu_ladder.get("floor", 0.05)),
            )
        elif mu_bounds is not None:
            lo = torch.as_tensor(mu_bounds[0], dtype=torch.float32).reshape(-1).expand(n).clone()
            hi = torch.as_tensor(mu_bounds[1], dtype=torch.float32).reshape(-1).expand(n).clone()
            centers = 0.5 * (lo + hi)
        else:
            mu = torch.as_tensor(mu_fixed, dtype=torch.float32).reshape(-1).expand(n).clone()
            lo, hi, centers = mu.clone(), mu.clone(), mu.clone()
        if not bool((lo > 0).all()):
            raise ValueError(f"need lo_i > 0, got min {float(lo.min())}")
        self.register_buffer("mu_lo", lo)
        self.register_buffer("mu_hi", hi)
        if mu_fixed is not None:
            self.mu_s = None
        else:
            if not bool((hi > lo).all()):
                raise ValueError("need hi_i > lo_i for learnable floors")
            # sigmoid-bracket init at the RUNG CENTER (pattern of elcd.py's
            # _init_sigma_head_bias: target -> unit interval -> logit):
            #   mu_i(s_i=logit(frac_i)) = lo_i + (hi_i-lo_i)*frac_i = c_i.
            frac = ((centers - lo) / (hi - lo)).clamp(1e-6, 1.0 - 1e-6)
            self.mu_s = nn.Parameter(torch.logit(frac))

        # ----- gains gamma_i (nu_i = mu_i + gamma_i) ------------------------
        if gamma_fixed is not None:
            g = torch.as_tensor(gamma_fixed, dtype=torch.float32).reshape(-1).expand(n).clone()
            if not bool((g > 0).all()):
                raise ValueError("need gamma_i > 0")
            self.register_buffer("gamma_fixed_vec", g)
            self.gamma_g = None
            self.nu_target = float((hi + g).max()) if nu_target is None else float(nu_target)
        else:
            if nu_target is None:
                raise ValueError("nu_target is required for learnable gamma")
            self.nu_target = float(nu_target)
            if not self.nu_target > float(hi.max()) + self.gamma_min:
                raise ValueError(
                    f"need nu_target ({self.nu_target}) > max_i hi_i ({float(hi.max())})"
                    f" + gamma_min ({self.gamma_min})"
                )
            self.gamma_g = nn.Parameter(torch.zeros(n))

        # ----- map parameters (mirror upstream MonLipNet.__init__) ----------
        self.Fq = nn.Parameter(torch.empty(sum(self.units), n))
        nn.init.xavier_normal_(self.Fq)
        self.fq = nn.Parameter(torch.empty((1,)))
        nn.init.constant_(self.fq, norm(self.Fq))
        self.by = nn.Parameter(torch.zeros(n))
        Fr, fr, b = [], [], []
        nz_1 = 0
        for nz in self.units:
            R = nn.Parameter(torch.empty((nz, nz + nz_1)))
            nn.init.xavier_normal_(R)
            r = nn.Parameter(torch.empty((1,)))
            nn.init.constant_(r, norm(R))
            Fr.append(R)
            fr.append(r)
            b.append(nn.Parameter(torch.zeros(nz)))
            nz_1 = nz
        self.Fr = nn.ParameterList(Fr)
        self.fr = nn.ParameterList(fr)
        self.bs = nn.ParameterList(b)
        self._cache: Optional[Tuple[List[torch.Tensor], torch.Tensor]] = None

    # ----- bounds -----------------------------------------------------------
    @property
    def mu_vec(self) -> torch.Tensor:
        """Per-dim monotonicity floors mu_i, always inside [mu_lo, mu_hi]."""
        if self.mu_s is None:
            return self.mu_lo
        return self.mu_lo + (self.mu_hi - self.mu_lo) * torch.sigmoid(self.mu_s)

    @property
    def gamma_vec(self) -> torch.Tensor:
        """Per-dim gains gamma_i > 0 with mu_i + gamma_i < nu_target always."""
        if self.gamma_g is None:
            return self.gamma_fixed_vec
        ceil = self.nu_target - self.mu_hi - self.gamma_min  # > 0 by constructor check
        return self.gamma_min + ceil * torch.sigmoid(self.gamma_g)

    @property
    def nu_vec(self) -> torch.Tensor:
        """Per-dim upper bounds nu_i = mu_i + gamma_i (< nu_target when learnable)."""
        return self.mu_vec + self.gamma_vec

    def get_bounds(self) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """(mu_vec, nu_vec, tau_max = max_i nu_i/mu_i), detached."""
        mu = self.mu_vec.detach()
        nu = self.nu_vec.detach()
        return mu, nu, float((nu / mu).max())

    # ----- Cayley frames (mirror upstream MonLipNet.direct_to_explicit) -------
    def _compute_frames(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """(V, S): V[k-1] = 2 B_k A_{k-1}^T (m_k, m_{k-1}); S stacked (m_total, n).

        Identical construction to upstream ``MonLipNet.direct_to_explicit``, in torch.
        """
        QT = cayley((self.fq / norm(self.Fq.T, eps=0)) * self.Fq.T)
        Q = QT.T
        V: List[torch.Tensor] = []
        S: List[torch.Tensor] = []
        idx = 0
        ATk_1 = None
        Qk_1 = None
        for k, nz in enumerate(self.units):
            Qk = Q[idx:idx + nz, :]
            Fab = self.Fr[k].T
            ABT = cayley((self.fr[k] / norm(Fab, eps=0)) * Fab)
            ATk, BTk = ABT[:nz, :], ABT[nz:, :]
            if k > 0:
                Ak, Bk = ATk.T, BTk.T
                V.append(2 * Bk @ ATk_1)              # (m_k, m_{k-1})
                S.append(Ak @ Qk - Bk @ Qk_1)         # (m_k, n)
            else:
                Ak = ATk.T
                S.append(ABT.T @ Qk)
            ATk_1, Qk_1 = Ak.T, Qk
            idx += nz
        return V, torch.cat(S, dim=0)

    def _frames(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Training: recompute (differentiable).  Eval: cache with no_grad
        (upstream MonLipNet.forward's caching pattern)."""
        if self.training:
            self._cache = None
            return self._compute_frames()
        if self._cache is None:
            with torch.no_grad():
                self._cache = self._compute_frames()
        return self._cache

    def train(self, mode: bool = True):
        self._cache = None
        return super().train(mode)

    def bhat(self) -> torch.Tensor:
        return torch.cat([b for b in self.bs], dim=0)

    # ----- forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """zhat_k = relu(V_k zhat_{k-1} + sqrt(2) S_k (Gamma^{1/2} x) + b_k);
        y = M x + (1/sqrt(2)) Gamma^{1/2} S^T zhat + b_y  (row-vector batch)."""
        V, S = self._frames()
        mu = self.mu_vec
        sg = torch.sqrt(self.gamma_vec)
        xg = math.sqrt(2.0) * (x * sg)                    # sqrt(2) * Gamma^{1/2} x
        zs: List[torch.Tensor] = []
        zk: Optional[torch.Tensor] = None
        idx = 0
        for k, nz in enumerate(self.units):
            Sk = S[idx:idx + nz, :]
            pre = xg @ Sk.transpose(0, 1) + self.bs[k]
            if k > 0:
                pre = pre + zk @ V[k - 1].transpose(0, 1)
            zk = torch.relu(pre)
            zs.append(zk)
            idx += nz
        z = torch.cat(zs, dim=-1)
        return x * mu + (z @ S) * sg / math.sqrt(2.0) + self.by


class ConjugatedMonLipNetD(nn.Module):
    """G(x) = P^T F(P x): MonLipNetD floors along LEARNED mode directions.

    RELATION TO BiLipNet: this class IS a depth-1 BiLipNet (upstream
    ``bilipnet_torch.BiLipNet``: orthogonal O_0 -> monotone M_0 -> orthogonal O_1)
    restricted to TIED orthogonal layers P_2 = P_1^T (and zero orthogonal biases
    q = 0), with the monotone block replaced by the per-dim ``MonLipNetD``.
    The tie is LOAD-BEARING for the per-mode floors.  Write G(x) = P_2 F(P_1 x)
    with independent orthogonal P_1, P_2.  The pairing that monotonicity needs is

        <dG, dx> = <P_2 dF, dx> = <dF, P_2^T dx>,

    but the monotone block's ellipsoid is stated against ITS OWN input
    difference du = P_1 dx:

        <dF, P_1 dx> >= (P_1 dx)^T M (P_1 dx).

    The two align if and only if P_2^T = P_1, i.e. P_2 = P_1^T -- exactly this
    class.  Untied P_1, P_2 is upstream BiLipNet's DELIBERATE choice: it breaks
    monotonicity of the composite on purpose, for expressivity (a bi-Lipschitz
    map need not be monotone), and with it the per-mode floors go too (measured).  The GLOBAL band
    [min_i mu_i, max_i nu_i] survives either way, tied or untied, because
    orthogonal layers preserve norms.

    P = cayley(P_gen) is exactly orthogonal for every value of the square
    generator ``P_gen`` (Cayley of the skew part; reuses ``orthogonal_torch.cayley``,
    which for a square input reduces to (I+A)^{-1}(I-A), A = W - W.T).

    Guarantee: <G(a)-G(c), a-c> = <F(Pa)-F(Pc), P(a-c)> >= (P(a-c))^T M (P(a-c)),
    i.e. per-MODE floors mu_i along v_i = i-th row of P (row-vector convention;
    P maps v_i to e_i because P P^T = I).

    Row-vector batches: forward y_r = F(x_r @ P.T) @ P.
    """

    def __init__(self, features: int, unit_features: Sequence[int],
                 frame_init_scale: float = 0.1, **monlip_kwargs):
        super().__init__()
        self.F = MonLipNetD(features, unit_features, **monlip_kwargs)
        self.P_gen = nn.Parameter(frame_init_scale * torch.randn(features, features))

    def frame(self) -> torch.Tensor:
        """The orthogonal frame P (rows = certified mode directions)."""
        return cayley(self.P_gen)

    @property
    def mu_vec(self) -> torch.Tensor:
        return self.F.mu_vec

    @property
    def gamma_vec(self) -> torch.Tensor:
        return self.F.gamma_vec

    @property
    def nu_vec(self) -> torch.Tensor:
        return self.F.nu_vec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        P = self.frame()
        return self.F(x @ P.transpose(0, 1)) @ P
