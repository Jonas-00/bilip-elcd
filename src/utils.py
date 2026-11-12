import numpy
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import torch
import torch.nn as nn
import numpy as np

from manifold_flow.vector_transforms import create_vector_transform
from manifold_flow.transforms.projections import CompositeProjection


class ELCD_Transform(nn.Module):
    def __init__(self, x_eq, transform, model):
        super().__init__()
        self.transform = transform
        self.model = model
        self.x_eq = x_eq
        self.x_eq.requires_grad = False
        self.model.set_x_eq(self.transform(x_eq)[0])

    def to(self, device):
        super().to(device)
        self.x_eq = self.x_eq.to(device)
        return self

    def forward(self, x):
        z_eq = self.transform(self.x_eq)[0]
        z, _ = self.transform(x, full_jacobian=False)
        a = torch.zeros(z.shape[0], 1, device=z.device)
        x_star = z_eq.expand(z.shape[0], -1)
        z_dot = self.model(z, a, x_star)
        _, inv_jacobian = self.transform.inverse(z, full_jacobian=True)
        z_dot = z_dot.view(z.shape[0], z.shape[1], 1)
        return torch.bmm(inv_jacobian, z_dot).squeeze(-1)


def plot_manifold(transform, scale, grid_step_size, axis_step_size):
    xs = np.arange(-scale, scale, grid_step_size)
    ys = np.arange(-scale, scale, grid_step_size)
    lines = []
    for x in xs:
        y_vals = np.arange(-scale, scale, axis_step_size)
        x_vals = [x] * len(y_vals)
        lines.append(np.array([x_vals, y_vals]))
    for y in ys:
        x_vals = np.arange(-scale, scale, axis_step_size)
        y_vals = [y] * len(x_vals)
        lines.append(np.array([x_vals, y_vals]))
    for line in lines:
        tl = transform(torch.Tensor(line).T)[0].T.detach().numpy()
        plt.plot(tl[0], tl[1], c='b')
    plt.show()


def plot_2d_vector_field(model, x_min, x_mx, y_min, y_max, normalize=True, scale=None, points=20):
    f, ax = plt.subplots(figsize=(8, 8))
    x = numpy.linspace(x_min, x_mx, points)
    y = numpy.linspace(y_min, y_max, points)
    xy = numpy.meshgrid(x, y)
    pts = torch.Tensor(np.array(xy).reshape(2, -1).T)
    vf = model(pts)
    U = vf[:, 0].view(points, points).detach().numpy()
    V = vf[:, 1].view(points, points).detach().numpy()
    if normalize:
        norm = numpy.sqrt(U ** 2 + V ** 2)
        U, V = U / norm, V / norm
    if scale is not None:
        ax.quiver(xy[0], xy[1], U, V, units='width', scale=scale)
    else:
        ax.quiver(xy[0], xy[1], U, V)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlim([x_min, x_mx]);  ax.set_ylim([y_min, y_max])
    return ax


def solve_ivp(f, x_0, t_start, t_end, t_step=1):
    t = numpy.arange(t_start, t_end, t_step)
    X = odeint(f, x_0, t)
    x_dot = numpy.zeros(X.shape)
    for i in range(X.shape[0]):
        x_dot[i] = f(X[i], 0)
    return X, x_dot


def solve_ivp_no_dot(f, x_0, t_start, t_end, t_step=1):
    t = numpy.arange(t_start, t_end, t_step)
    return odeint(f, x_0, t)


def model_to_f(model):
    return lambda x, t: model(torch.Tensor(x))


def single_ode45_step(f, x, dt):
    k1 = dt * f(x);  k2 = dt * f(x + k1 / 2)
    k3 = dt * f(x + k2 / 2);  k4 = dt * f(x + k3)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def forward_step(f, x0, t_steps, dt):
    X = torch.zeros((t_steps, x0.shape[0], x0.shape[1]))
    X[0] = x0
    for i in range(1, t_steps):
        X[i] = f(X[i - 1], dt)
    return X


def forward_step_until_converge(f, x0, dt, x_eq=None, max_steps=1000, tol=1e-5):
    X = torch.zeros((max_steps, x0.shape[0], x0.shape[1]))
    X[0] = x0
    for i in range(1, max_steps):
        X[i] = f(X[i - 1], dt)
        if torch.norm(X[i] - X[i - 1]) < tol:
            break
    return X[:i + 1]


def DTWD(trajectory, reference):
    n = trajectory.shape[0];  N = len(reference)
    dists = torch.zeros((n, N))
    for i in range(n):
        for j in range(N):
            dists[i, j] = torch.norm(trajectory[i] - reference[j])
    return (torch.min(dists, dim=0).values.sum() / N
            + torch.min(dists, dim=1).values.sum() / n).item()


def dtwd_np(traj, reference):
    assert traj.shape[1] == reference[0].shape[0]
    n = traj.shape[0];  N = len(reference)
    dists = numpy.zeros((n, N))
    for i in range(n):
        for j in range(N):
            dists[i, j] = numpy.linalg.norm(traj[i] - reference[j])
    return numpy.mean(numpy.min(dists, axis=0)) + numpy.mean(numpy.min(dists, axis=1))


def generate_mf_transform(flow_steps, input_dim, latent_dim,
                           linear_transform_type='lu',
                           base_transform_type='rq-coupling',
                           num_bins=10):
    rq_transform = create_vector_transform(
        input_dim, flow_steps,
        linear_transform_type=linear_transform_type,
        base_transform_type=base_transform_type,
        hidden_features=30,
        num_transform_blocks=2,
        dropout_probability=0.0,
        use_batch_norm=False,
        num_bins=num_bins,
        tail_bound=10,
        apply_unconditional_transform=False,
        context_features=None,
    )
    return CompositeProjection(rq_transform, input_dim, latent_dim)
