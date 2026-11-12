"""Vendored copy of the ACFR ``RobustNeuralNetworks`` PyTorch layers.

Provides upstream's BiLipNet / MonLipNet (static bi-Lipschitz maps) and RENs
(``RENBase``, ``BiLipschitzREN``).

Upstream: https://github.com/acfr/RobustNeuralNetworks (MIT).  Every ``.py`` file
here EXCEPT ``__init__.py`` and ``monlipnet_d.py`` (both ours) is a
BYTE-FOR-BYTE copy of upstream at commit
``417b70716b795c1db0d8b19037d03afd15ac2a26`` -- no edits to upstream source at
all; see ``VENDORED.md`` and ``HANDOFF.md``.

``solvers.py`` (Davis-Yin splitting) and ``utils.py`` are upstream's JAX modules,
vendored unmodified.  ``jax`` and ``flax`` are therefore hard dependencies of this
package (declared in ``requirements.txt`` and ``pyproject.toml``) rather than
being patched away.

Upstream's own ``import`` lines read ``from robustnn.X import ...``.  Rather than
rewrite those lines (which would break the byte-for-byte guarantee) we register
THIS package under the name ``robustnn`` in ``sys.modules`` before importing the
submodules, so ``robustnn.orthogonal_torch`` resolves to
``bilip_jepa.models._acfr.orthogonal_torch``.  The submodules are then aliased
under both names so each is a single module object.

Usage::

    from bilip_jepa.models._acfr import BiLipNet, MonLipNet

    f = BiLipNet(features=8, unit_features=[64, 64], mu=0.6, nu=0.95, depth=2,
                 is_mu_fixed=True, is_nu_fixed=True, is_tau_fixed=True)

Constructor note: at least two of ``mu`` (lower Lipschitz bound), ``nu`` (upper
Lipschitz bound), ``tau = nu / mu`` (distortion) must be given; pass
``is_mu_fixed=True, is_nu_fixed=True, is_tau_fixed=True`` to keep the bounds
FIXED instead of learnable.
"""

import sys as _sys

_existing = _sys.modules.get("robustnn")
if _existing is None:
    _sys.modules["robustnn"] = _sys.modules[__name__]
elif _existing is not _sys.modules[__name__]:
    raise ImportError(
        "The real `robustnn` package is already imported; the vendored copy in "
        "bilip_jepa.models._acfr cannot claim that name.  Import the upstream "
        "MonLipNet/BiLipNet directly instead of using this vendored copy."
    )

from robustnn.orthogonal_torch import Params, Unitary, cayley, norm  # noqa: E402
from robustnn.monlipnet_torch import MonLipNet  # noqa: E402
from robustnn.bilipnet_torch import BiLipNet  # noqa: E402
from robustnn.ren_torch import (  # noqa: E402
    BiLipschitzREN,
    ExplicitRENParams,
    RENBase,
    solve_full_layer,
    tril_equilibrium_layer,
)

# Make `bilip_jepa.models._acfr.<mod>` and `robustnn.<mod>` the SAME module object,
# so importing by either path cannot produce two copies of MonLipNet.
for _m in ("orthogonal_torch", "monlipnet_torch", "bilipnet_torch", "ren_torch"):
    _sys.modules[__name__ + "." + _m] = _sys.modules["robustnn." + _m]
del _m, _existing

# MODIFIED (ours): MonLipNetD / ConjugatedMonLipNetD live in OUR file
# `monlipnet_d.py` -- a PER-DIMENSION-band generalization of upstream's
# MonLipNet (scalar mu, nu -> diagonal M = diag(mu_i), Gamma = diag(gamma_i);
# same Cayley construction, certificate LMI unchanged because
# Gamma^{1/2} Gamma^{-1} Gamma^{1/2} cancels), with learnable per-dim floors in
# log-spaced ladder brackets, a conjugated-frame wrapper (= depth-1 BiLipNet
# with TIED orthogonal layers P_2 = P_1^T). No upstream file is touched.
from .monlipnet_d import ConjugatedMonLipNetD, MonLipNetD  # noqa: E402

__all__ = [
    "MonLipNet", "BiLipNet", "Unitary", "Params", "cayley", "norm",
    "RENBase", "BiLipschitzREN", "ExplicitRENParams",
    "tril_equilibrium_layer", "solve_full_layer",
    # ours:
    "MonLipNetD", "ConjugatedMonLipNetD",
]
