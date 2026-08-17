'''
Sweep through different r0, rho, and thick values to observe the effect on saliency detection.
'''

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import copy
from visual_attention.helpers_visual_att import initialise_attention, run_attention
import torch

device = torch.device("cpu")

def load_events(path):
    data = np.load(path, allow_pickle=True)
    if data.dtype.names:
        names = {n.lower(): n for n in data.dtype.names}
        x = data[names['x']].astype(int)
        y = data[names['y']].astype(int)
        p = data[names[next(k for k in ('p', 'pol', 'polarity') if k in names)]]
        t = data[names[next(k for k in ('t', 'ts', 'timestamp', 'time') if k in names)]].astype(float)
    else:
        diffs = np.diff(data, axis=0)
        t_col = np.argmax((diffs >= 0).mean(axis=0))
        rest = [c for c in range(data.shape[1]) if c != t_col]
        t = data[:, t_col].astype(float)
        x, y, p = data[:, rest[0]].astype(int), data[:, rest[1]].astype(int), data[:, rest[2]]
    t = t - t.min()
    t = t / 1e3
    return x, y, p, t

x, y, p, t = load_events("/home/rocharay/kws_attention/data/6_weird_jitter_objects_346x260.npy")

max_x = x.max() + 1
max_y = y.max() + 1
resolution = (max_y, max_x)

class Config:
    ATTENTION_PARAMS = {
        'size_krn': 32, 'r0': 7, 'rho': 0.015, 'theta': np.pi*3/2,
        'thetas': np.arange(0, 2*np.pi, np.pi/4), 'thick': 12,
        'fltr_resize_perc': [2, 2], 'offsetpxs': 0, 'offset': (0, 0),
        'num_pyr': 6, 'tau_mem': 0.3, 'stride': 1, 'out_ch': 1,
    }
config = Config()

# isolated tau test - check if the tau is the same across windows
params_a = copy.deepcopy(config.ATTENTION_PARAMS); params_a['tau_mem'] = 0.05
params_b = copy.deepcopy(config.ATTENTION_PARAMS); params_b['tau_mem'] = 1.2
net_a = initialise_attention(device, params_a)
net_b = initialise_attention(device, params_b)

test_window = torch.zeros((1, max_y, max_x), dtype=torch.float32)
mask = (t >= 0) & (t < 100)
for xi, yi in zip(x[mask], y[mask]):
    test_window[0][yi][xi] = 255

map_a, coords_a = run_attention(test_window, net_a, device, resolution, params_a['num_pyr'])
map_b, coords_b = run_attention(test_window, net_b, device, resolution, params_b['num_pyr'])

print("identical arrays?", np.array_equal(map_a, map_b))
print("max abs diff:", np.abs(map_a - map_b).max())
# ----------------------------------------------------------------------
def tau_decay_test(tau_values, stim_ms=100, n_empty_windows=20, window_period=100):
    results = {}
    empty_window = torch.zeros((1, max_y, max_x), dtype=torch.float32)

    stim_window = torch.zeros((1, max_y, max_x), dtype=torch.float32)
    mask = (t >= 0) & (t < stim_ms)
    for xi, yi in zip(x[mask], y[mask]):
        stim_window[0][yi][xi] = 255

    for tau in tau_values:
        params = copy.deepcopy(config.ATTENTION_PARAMS)
        params['tau_mem'] = tau
        net = initialise_attention(device, params)

        saliency_map, _ = run_attention(stim_window, net, device, resolution, params['num_pyr'])
        print(f"tau={tau} stim: max={saliency_map.max()}, has_nan={np.isnan(saliency_map).any()}")
        decay_curve = [saliency_map.max()]

        for i in range(n_empty_windows):
            saliency_map, _ = run_attention(empty_window, net, device, resolution, params['num_pyr'])
            has_nan = np.isnan(saliency_map).any()
            print(f"  step {i}: max={saliency_map.max()}, has_nan={has_nan}")
            decay_curve.append(saliency_map.max())

        results[tau] = np.array(decay_curve)
    return results
tau_values = [0.05, 0.1, 0.3, 0.6, 1.2]
decay_results = tau_decay_test(tau_values)

plt.figure(figsize=(8, 4))
for tau, curve in decay_results.items():
    plt.plot(np.arange(len(curve)) * 100, curve, marker='o', label=f"tau_mem={tau}")
plt.xlabel("Time after stimulus (ms)")
plt.ylabel("Max saliency (raw)")
plt.legend()
plt.title("Decay after a single stimulus, no further input")
plt.show()