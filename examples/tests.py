# ==== tests.py ====
import numpy as np
import matplotlib.pyplot as plt
import torch

from config import sec_length, batch, delta_t
from toolbox import make_loss, StatLoss
from ini_sinn_data import Traj, T, get_training_components, train_sinn

# Get model and train
net, optimizer, times, slice_leb, g = get_training_components()
val_data = np.random.normal(0, 1, (2 * sec_length, batch)).astype(np.float32)
val_set = torch.from_numpy(val_data.reshape(2 * sec_length, batch, 1)).to("cuda")
net, T_error, V_error, Step = train_sinn(net, optimizer, times, g, slice_leb, val_set)

# Use generated data
Traj = Traj.astype(np.float32)
target = torch.from_numpy(Traj.reshape(sec_length, batch, 1)).to("cuda")

# Generate input and predictions
ran_ini = np.random.normal(0, 1, (2 * sec_length, batch)).astype(np.float32)
ran_input = torch.from_numpy(ran_ini.reshape(2 * sec_length, batch, 1)).to("cuda")
Pred, _ = net(ran_input)
prediction = Pred[-sec_length:, :, :].detach().cpu().numpy().reshape(sec_length, batch)

# Loss functions
LAGS = 400
N = 400
loss_acf_fft = make_loss("acf[fft]", target, lags=LAGS, device="cuda")
loss_acf_q2 = make_loss("acf[fft]", target**2, lags=LAGS, device="cuda")
loss_acf_bruteforce = make_loss("acf[bruteforce]", target, lags=LAGS, device="cuda")
loss_acf_randbrute = make_loss(
    "acf[randombrute]", target, lags=LAGS, sample_lags=20, device="cuda"
)
loss_pdf_empirical = make_loss("pdf", target, lower=0, upper=10, n=N, device="cuda")

# Plotting
skip = 0
plt.figure(0)
plt.title("MD trajectories", fontsize=15)
for i in [0, 1, 2, -1, -2, -3]:
    plt.plot(T[skip:] - T[skip], Traj[skip:, i])
plt.ylim([-3, 3])
plt.xlabel(r"$t$", fontsize=15)
plt.ylabel(r"$x(t)$", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure(1)
plt.title("SINN trajectories", fontsize=15)
for i in [0, 1, 2, -1, -2, -3]:
    plt.plot(T[skip:] - T[skip], prediction[skip:, i])
plt.ylim([-3, 3])
plt.xlabel(r"$t$", fontsize=15)
plt.ylabel(r"$x(t)$", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# ACF comparison
acf_p = StatLoss.acf(Pred[-sec_length:, :, :])
acf_t = StatLoss.acf(target)
corr_p = acf_p.mean(axis=1).detach().cpu().numpy()
corr_t = acf_t.mean(axis=1).detach().cpu().numpy()

plt.figure(2)
plt.title("Normalized ACF", fontsize=15)
plt.plot(T, corr_t, "r", label="Exact correlation of $x(t)$")
plt.plot(T, corr_p, "b--", label="Correlation of the output")
plt.xlabel(r"$t$", fontsize=15)
plt.ylabel(r"$C(t)/C(0)$", fontsize=15)
plt.legend(frameon=False, fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# PDF comparison
rho_data = StatLoss.gauss_kde(target, -1, 10, 100).cpu()
rho_input = StatLoss.gauss_kde(ran_input, -1, 10, 100).cpu()
rho_prediction = StatLoss.gauss_kde(
    Pred[-sec_length:, :, :].detach(), -1, 10, 100
).cpu()

x_plot = np.linspace(-1, 10, 100)
plt.figure(3)
plt.title("Equilibrium PDF", fontsize=15)
plt.plot(x_plot, rho_data.numpy(), "r", label="Exact PDF")
plt.plot(x_plot, rho_input.numpy(), "b:", label="Input PDF")
plt.plot(x_plot, rho_prediction.numpy(), "g--", label="Output PDF")
plt.legend(frameon=False, fontsize=15)
plt.xlabel(r"$x$", fontsize=15)
plt.ylabel(r"$\rho(x)$", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Higher-order ACF
acf2 = StatLoss.acf(target**2)
pred_norm = torch.from_numpy(prediction.reshape(sec_length, batch, 1))
acf2_app = StatLoss.acf(pred_norm**2)

plt.figure(4)
plt.title("Normalized ACF of $x^2(t)$", fontsize=15)
plt.plot(T, acf2.mean(axis=1).detach().cpu().numpy(), "r", label="Exact")
plt.plot(T, acf2_app.mean(axis=1).detach().cpu().numpy(), "g--", label="Output")
plt.xlabel(r"$t$", fontsize=15)
plt.ylabel(r"$C(t)/C(0)$", fontsize=15)
plt.legend(frameon=False, fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Extrapolation
[a, b, c] = ran_input.size()
scale = 10
burnin = 3
skip = 20
batch_new = 50
ran_ini_ext = np.random.normal(0, 1, ((scale - burnin) * a, batch_new, c)).astype(
    np.float32
)
ran_input_ext = torch.from_numpy(
    ran_ini_ext.reshape((scale - burnin) * a, batch_new, c)
).to("cuda")
Pred_L, _ = net(ran_input_ext)
prediction_ext = Pred_L.detach().cpu().numpy().reshape((scale - burnin) * a, batch_new)

LT = np.linspace(0, (scale - burnin) * a * delta_t, num=(scale - burnin) * a + 1)[:-1]

plt.figure(5)
plt.title("Short-time trajectories", fontsize=15)
plt.plot(T - T[0], Traj[:, 0], label="MD simulation")
plt.plot(LT[skip:] - LT[skip], prediction_ext[skip:, 1], "--", label="SINN output")
plt.xlabel(r"$t$", fontsize=15)
plt.ylabel(r"$x(t)$", fontsize=15)
plt.legend(frameon=False, fontsize=15)
plt.grid(True)
plt.ylim([-3, 3])
plt.xticks(fontsize=15)
plt.yticks([0, 2, 4, 6, 8, 10], fontsize=15)

plt.figure(6, figsize=(16, 4))
plt.title("Long-time trajectories", fontsize=15)
plt.plot(T - T[0], Traj[:, 0], label="MD simulation")
plt.plot(LT[skip:] - LT[skip], prediction_ext[skip:, 1], "--", label="SINN output")
plt.xlabel(r"$t$", fontsize=15)
plt.ylabel(r"$x(t)$", fontsize=15)
plt.legend(frameon=False, fontsize=15)
plt.grid(True)
plt.ylim([-3, 3])
plt.xticks(fontsize=15)
plt.yticks([-3, -2, -1, 0, 1, 2, 3], fontsize=15)
