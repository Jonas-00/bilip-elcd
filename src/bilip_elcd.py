"""
Bi-Lipschitz Contracting Dynamics (CT-ELCD)

Continuous-time dynamical system with certified contraction and bounded skew:

    ẋ = f(x, a) = A(x, a, x*)(x - x*)

    A(x) = S(x) + K(x)
    S(x) = -Q(x)^T Lambda(x) Q(x) - alpha I      (symmetric, contracting)
    K(x) = Qk^T blkdiag(w1(x)*J, ...) Qk         (skew-symmetric, bounded)

Properties by construction:
  - Contraction:   sym(A) ≼ -(lambda_min_eff + alpha) I
  - Bounded skew:  ||K(x)||_2 <= omega_max
  - Bi-Lipschitz:  (lambda_min_eff - omega_max)||delta|| <= ||f|| <= (lambda_max + alpha + omega_max)||delta||
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

try:
    from .bilip_layers import BiLipMLP
except ImportError:
    try:
        from bilip_layers import BiLipMLP
    except ImportError:
        BiLipMLP = None


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    n_hidden: int = 2,
    use_bilip_mlp: bool = False,
    bilip_kwargs: Optional[Dict[str, object]] = None,
) -> nn.Module:
    if use_bilip_mlp:
        effective_hidden = [hidden_dim] * max(int(n_hidden), 1)
        return BiLipMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=effective_hidden,
            activation_cls=nn.Tanh,
            **(bilip_kwargs or {}),
        )
    layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
    layers.append(nn.Linear(hidden_dim, output_dim))
    net = nn.Sequential(*layers)
    with torch.no_grad():
        net[-1].weight.mul_(0.01)
        net[-1].bias.zero_()
    return net


class ELCDOrthogonalCTGatedK(nn.Module):
    """
    Continuous-time ELCD with state-dependent bounded skew term K(x).

        A(x) = S(x) + K(x)
        S(x) = -Q(x)^T Lambda(x) Q(x) - alpha I
        K(x) = Qk^T blkdiag(w1(x)*J, ..., wm(x)*J) Qk
        wi(x) = omega_max * tanh(gi(x))

    ||K(x)||_2 <= omega_max for all x. lambda_min is reparametrized as
        lambda_min_eff = omega_max + 1e-3 + softplus(lambda_min_param)
    so that lambda_min_eff > omega_max is guaranteed by construction,
    ensuring the full Jacobian of f remains positive-definite.

    Args:
        dim:              State dimension
        action_dim:       Action dimension
        alpha:            Additional contraction rate shift (default 0)
        lambda_min:       Initial lower eigenvalue bound (learnable)
        lambda_max:       Upper eigenvalue bound (fixed)
        omega_max:        Maximum skew magnitude ||K||_2 <= omega_max
        hidden_dim:       Hidden units per MLP layer
        n_hidden:         Number of hidden MLP layers
        use_bilip_mlp:    Use Bi-Lipschitz MLP generators (certified Lipschitz)
        bilip_s_min:      BiLipMLP per-layer Lipschitz lower bound
        bilip_s_max:      BiLipMLP per-layer Lipschitz upper bound
    """

    def __init__(
        self,
        dim: int,
        action_dim: int,
        alpha: float = 0.0,
        lambda_min: float = 0.01,
        lambda_max: float = 1.0,
        omega_max: float = 1.0,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        use_bilip_mlp: bool = False,
        bilip_s_min: float = 0.01,
        bilip_s_max: float = 0.99,
        bilip_init: str = "random",
        bilip_factorization: str = "thin",
    ):
        super().__init__()
        self.dim = dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.lambda_max = lambda_max
        self.omega_max = omega_max
        self.n_hidden = n_hidden
        self._m_blocks = dim // 2

        init_val = math.log(math.expm1(max(lambda_min, 1e-6)))
        self.lambda_min_param = nn.Parameter(torch.tensor(init_val))

        inp = dim + action_dim + dim
        bilip_kwargs: Dict[str, object] = {
            "s_min": bilip_s_min,
            "s_max": bilip_s_max,
            "init": bilip_init,
            "factorization": bilip_factorization,
            "square_layer": "bilipnet",
        }
        self.S_Q_net   = _build_mlp(inp, dim * dim,       hidden_dim, n_hidden, use_bilip_mlp, bilip_kwargs)
        self.lambda_net = _build_mlp(inp, dim,             hidden_dim, n_hidden, use_bilip_mlp, bilip_kwargs)
        self.omega_net  = _build_mlp(inp, self._m_blocks,  hidden_dim, n_hidden, use_bilip_mlp, bilip_kwargs)

        A_rand = torch.randn(dim, dim)
        Qk, R = torch.linalg.qr(A_rand)
        Qk = Qk * torch.sign(torch.diag(R)).unsqueeze(0)
        self.register_buffer("Qk", Qk)

    def _lambda_min_eff(self) -> torch.Tensor:
        raw = self.omega_max + 1e-3 + torch.nn.functional.softplus(self.lambda_min_param)
        return torch.clamp(raw, max=self.lambda_max - 1e-3)

    @staticmethod
    def _cayley_batch(S: torch.Tensor) -> torch.Tensor:
        S_skew = S - S.transpose(-2, -1)
        d = S.shape[-1]
        I = torch.eye(d, device=S.device, dtype=S.dtype).unsqueeze(0)
        return torch.linalg.solve(I + S_skew, I - S_skew)

    def _build_K_batch(self, omegas: torch.Tensor) -> torch.Tensor:
        B = omegas.shape[0]
        d = self.dim
        B_block = torch.zeros(B, d, d, device=omegas.device, dtype=omegas.dtype)
        for i in range(self._m_blocks):
            idx = 2 * i
            B_block[:, idx,     idx + 1] = -omegas[:, i]
            B_block[:, idx + 1, idx    ] =  omegas[:, i]
        Qk = self.Qk.unsqueeze(0).expand(B, -1, -1)
        return torch.bmm(torch.bmm(Qk.transpose(-2, -1), B_block), Qk)

    def get_A(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        x_star: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return A(x, a, x*) as (B, dim, dim) tensor."""
        if x_star is None:
            x_star = torch.zeros_like(x)
        squeeze = x.dim() == 1
        if squeeze:
            x, a, x_star = x.unsqueeze(0), a.unsqueeze(0), x_star.unsqueeze(0)

        B = x.shape[0]
        inp = torch.cat([x, a, x_star], dim=-1)

        S_Q = self.S_Q_net(inp).view(B, self.dim, self.dim)
        Q = self._cayley_batch(S_Q)

        lambda_min_eff = self._lambda_min_eff()
        lambdas = lambda_min_eff + (self.lambda_max - lambda_min_eff) * torch.sigmoid(self.lambda_net(inp))
        Lambda = torch.diag_embed(lambdas)

        I = torch.eye(self.dim, device=x.device, dtype=x.dtype).unsqueeze(0)
        S = -torch.bmm(torch.bmm(Q.transpose(-2, -1), Lambda), Q) - self.alpha * I

        omegas = self.omega_max * torch.tanh(self.omega_net(inp))
        K = self._build_K_batch(omegas)

        A = S + K
        if squeeze:
            A = A.squeeze(0)
        return A

    def forward(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        x_star: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute ẋ = A(x, a, x*)(x - x*)."""
        if x_star is None:
            x_star = torch.zeros_like(x)
        A = self.get_A(x, a, x_star)
        delta = x - x_star
        if x.dim() == 1:
            return A @ delta
        return torch.bmm(A, delta.unsqueeze(-1)).squeeze(-1)

    def get_contraction_rate(self) -> float:
        return self._lambda_min_eff().item() + self.alpha

    def get_theoretical_bounds(self) -> Tuple[float, float]:
        """Return (m, M) bi-Lipschitz bounds on ||f(x)||/||x - x*||."""
        m = self._lambda_min_eff().item() + self.alpha - self.omega_max
        M = self.lambda_max + self.alpha + self.omega_max
        return m, M

    def set_x_eq(self, x_eq):
        pass


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def integrate_ode(
    model: nn.Module,
    x0: torch.Tensor,
    a: torch.Tensor,
    x_star: Optional[torch.Tensor] = None,
    t_span: Tuple[float, float] = (0.0, 1.0),
    dt: float = 0.01,
    method: str = "euler",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Integrate ẋ = f(x, a) with a fixed action.

    Args:
        model:  ELCDOrthogonalCTGatedK instance
        x0:     Initial state (dim,) or (batch, dim)
        a:      Action (action_dim,) or (batch, action_dim)
        x_star: Equilibrium (default: zeros)
        t_span: (t_start, t_end)
        dt:     Time step
        method: "euler" or "rk4"

    Returns:
        t (num_steps,), trajectory (num_steps, ..., dim)
    """
    if x_star is None:
        x_star = torch.zeros_like(x0)
    t_start, t_end = t_span
    num_steps = int((t_end - t_start) / dt) + 1
    t = torch.linspace(t_start, t_end, num_steps, device=x0.device, dtype=x0.dtype)
    trajectory = [x0.clone()]
    x = x0.clone()
    for _ in range(num_steps - 1):
        if method == "euler":
            x = x + dt * model(x, a, x_star)
        elif method == "rk4":
            k1 = model(x, a, x_star)
            k2 = model(x + 0.5 * dt * k1, a, x_star)
            k3 = model(x + 0.5 * dt * k2, a, x_star)
            k4 = model(x + dt * k3, a, x_star)
            x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"Unknown method: {method!r}")
        trajectory.append(x.clone())
    return t, torch.stack(trajectory)


def check_contraction(
    model: nn.Module,
    x: torch.Tensor,
    a: torch.Tensor,
    x_star: Optional[torch.Tensor] = None,
) -> Tuple[bool, float]:
    """Check sym(A) ≼ −alpha I at a single point. Returns (passes, max_eigenvalue)."""
    if x_star is None:
        x_star = torch.zeros_like(x)
    A = model.get_A(x, a, x_star)
    if A.dim() == 3:
        A = A.squeeze(0)
    sym_A = 0.5 * (A + A.T)
    max_eig = torch.linalg.eigvalsh(sym_A).max().item()
    return max_eig <= -model.get_contraction_rate() + 1e-6, max_eig


def check_bilipschitz(
    model: nn.Module,
    x1: torch.Tensor,
    x2: torch.Tensor,
    a: torch.Tensor,
    x_star: Optional[torch.Tensor] = None,
    m: float = 0.0,
    M: float = float('inf'),
) -> Tuple[bool, float]:
    """Check m‖δ‖ ≤ ‖f(x1)−f(x2)‖ ≤ M‖δ‖. Returns (passes, ratio)."""
    if x_star is None:
        x_star = torch.zeros_like(x1)
    f1, f2 = model(x1, a, x_star), model(x2, a, x_star)
    x_diff = torch.norm(x1 - x2).item()
    if x_diff < 1e-10:
        return True, 0.0
    ratio = torch.norm(f1 - f2).item() / x_diff
    return m - 1e-6 <= ratio <= M + 1e-6, ratio
