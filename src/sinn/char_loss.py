# ==== char_loss.py ====
import torch
import numpy as np


def flatten(xss):
    return [x for xs in xss for x in xs]


def compute_theta_cum(theta):
    B, m = theta.shape
    theta_cum = torch.zeros(B, m, m, device=theta.device, dtype=theta.dtype)
    for i in range(m):
        for j in range(m):
            if j > i:
                theta_cum[:, i, j] = theta[:, i:j].sum(dim=1)
    return theta_cum.view(B, -1)


def compute_slices(t, G):
    m = len(t)
    A = torch.zeros(m, m)
    for i in range(m):
        for j in range(m):
            if i > j:
                A[i, j] = 0.0
            elif i > 0 and j < m - 1:
                A[i, j] = -(
                    G(t[j + 1] - t[i])
                    - G(t[j + 1] - t[i - 1])
                    + G(t[j] - t[i - 1])
                    - G(t[j] - t[i])
                )
            elif i > 0 and j == m - 1:
                A[i, j] = G(t[j] - t[i]) - G(t[j] - t[i - 1])
            elif i == 0 and j < m - 1:
                A[i, j] = G(t[j] - t[i]) - G(t[j + 1] - t[i])
            elif i == 0 and j == m - 1:
                A[i, j] = G(t[j] - t[i])
    return A.flatten()


def char_func_analytic(theta, times, seed_cumulant, g, slice_leb=None):
    theta_cum = compute_theta_cum(theta)
    if slice_leb is None:
        slice_leb = compute_slices(times, g).to(theta.device)
    cumulant = seed_cumulant(theta_cum) * slice_leb
    total = cumulant.sum(dim=1)
    return torch.exp(total)


def char_func_mc_model(theta, data):
    device = data.device
    theta = theta.to(device)
    data_flat = data.squeeze(-1)
    exp_values = torch.exp(1j * torch.matmul(data_flat.T, theta.T))
    return exp_values.mean(dim=0)


def _cf_prediction(theta, data):
    return char_func_mc_model(theta.to(data.device), data)


def _cf_target(theta, times, seed_cumulant, g, slice_leb):
    return char_func_analytic(
        theta.to(slice_leb.device), times, seed_cumulant, g, slice_leb=slice_leb
    )


def char_loss_analytic(
    times, seed_cumulant, g, data, mc_bound, mc_points, device, slice_leb=None
):
    def joint_cf_error(theta):
        return torch.abs(
            _cf_prediction(theta, data)
            - _cf_target(theta, times, seed_cumulant, g, slice_leb)
        )

    theta = np.random.uniform(-mc_bound, mc_bound, (mc_points, times.shape[0])).astype(
        np.float32
    )
    theta = torch.from_numpy(theta).to(device)
    return joint_cf_error(theta).mean()
