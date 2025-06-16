# ==== init_sinn_data.py ====
import numpy as np
import torch
from config import sec_length, batch, delta_t
from char_loss import compute_slices, char_loss_analytic

# Parameters for OU process
theta = 1
sigma = 1.0
dt = 0.01
mu_t = 0

# Time settings
Time = 5 * delta_t * sec_length
length = int(Time / dt) + 1
t = np.linspace(0, Time, length)
gap = int(delta_t / dt)
t_store = t[0:-1:gap]
q_store = np.zeros([t_store.size + 1, batch])

# OU process solver
q0 = np.zeros((1, batch))
q1 = np.zeros((1, batch))
j = 0
for i in range(1, length):
    Wt = np.random.normal(0, np.sqrt(dt), size=(1, batch))
    q1 = q0 + theta * (mu_t - q0) * dt + sigma * Wt
    if i % gap == 0:
        q_store[j, :] = q1
        j = j + 1
    q0 = q1

# Cut transient
Toss = int(t_store.size / 4)
q_store = q_store[Toss:-1, :]

# Construct sample data
ini = np.random.normal(0, 1, (sec_length, batch))
Traj = q_store[0:sec_length, :]
T = np.arange(len(Traj[:, 1])) * delta_t
x = np.linspace(0, 10, sec_length)

# Export usable items
__all__ = ["ini", "Traj", "T", "x", "get_training_components", "train_sinn"]


# Training-related parameters
def get_training_components():
    import torch.optim as optim
    from toolbox import SINN

    input_size = 1
    hidden_layers = 1
    hidden_size = 1
    output = 1
    lr = 1e-3

    net = SINN(input_size, hidden_layers, hidden_size, output).to("cuda")
    optimizer = optim.Adam(net.parameters(), lr)
    times = np.linspace(0, 1, 25)

    def g(h):
        return np.exp(-abs(h)) / 2

    slice_leb = compute_slices(times, g).to("cuda")

    return net, optimizer, times, slice_leb, g


# Training loop function
def train_sinn(net, optimizer, times, g, slice_leb, val_set, device="cuda"):
    T_error = []
    V_error = []
    Step = []
    for step in range(2000):
        optimizer.zero_grad()
        ini = np.random.normal(0, 1, (2 * sec_length, batch)).astype(np.float32)
        input = torch.from_numpy(ini.reshape(2 * sec_length, batch, 1)).to(device)

        prediction, _ = net(input)
        prediction = prediction[-25:, :, :].to(device)
        loss = char_loss_analytic(
            times, lambda t: -0.5 * t**2, g, prediction, slice_leb
        )
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            with torch.no_grad():
                pred_val, _ = net(val_set)
                pred_val = pred_val[-25:, :, :]
                loss_val = char_loss_analytic(
                    times, lambda t: -0.5 * t**2, g, pred_val, slice_leb
                )
                print(f"[{step}]-th step loss: {loss:.3f}, {loss_val:.3f}")
                T_error.append(loss.detach().cpu().numpy())
                V_error.append(loss_val.detach().cpu().numpy())
                Step.append(step)

        if loss <= 0.09 and loss_val <= 0.09:
            break

    print("Training finished")
    return net, T_error, V_error, Step
