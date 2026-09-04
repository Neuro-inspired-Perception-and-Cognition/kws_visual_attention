'''
Get only saliency maps from event data
'''

import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
import numpy as np
from visual_attention.helpers_visual_att import initialise_attention, run_attention
import torch
import cv2
import torchvision
import os

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
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
    t = t / 1e3  # IEBCS timestamps are always us, convert directly to ms

    return x, y, p, t

# Load event data from a .npy file
x, y, p, t = load_events("/home/rocharay/kws_attention/data/shapes_jitter_5_events.npy")

# diagnostic print
print(f"duration: {t.max()-t.min():.1f} ms")
print(f"n windows at 100ms: {(t.max()-t.min())/100:.1f}")
# ----------------------------------------------------------

# Determine the resolution based on the maximum coordinates
max_x = x.max() + 1  # Maximum x coordinate + 1 for resolution
max_y = y.max() + 1  # Maximum y coordinate + 1 for resolution
resolution = (max_y, max_x)  # Resolution tuple for attention processing

# QUickly visualize the first 100ms of the events
m = (t >= t[0]) & (t < t[0] + 100)
preview = np.zeros((max_y, max_x))
preview[y[m], x[m]] = np.where(p[m] > 0, 1, -1) # Vectorized insertion of polarity (+1 for ON, -1 for OFF)
plt.imshow(preview, cmap='bwr', vmin=-1, vmax=1); plt.title("Event Data Preview (First 100 ms)"); plt.colorbar(); plt.show()


##### Attention Mechanism #####
# Configuration class to store attention parameters
class Config:
    # Attention Parameters
    ATTENTION_PARAMS = {
        'size_krn': 32, 'r0': 7, 'rho': 0.015, 'theta': np.pi*3/2,
        'thetas': np.arange(0, 2*np.pi, np.pi/4), 'thick': 12,
        'fltr_resize_perc': [2, 2], 'offsetpxs': 0, 'offset': (0, 0),
        'num_pyr': 6, 'tau_mem': 0.3, 'stride': 1, 'out_ch': 1,
    }

# Initialize the configuration
config = Config()

# Initialize saliency map and coordinates for maximum saliency
saliency_map = np.zeros((max_y, max_x), dtype=np.float32)  # Saliency map initialized to zero
salmax_coords = np.zeros((2,), dtype=np.int32)  # Array to hold coordinates of maximum saliency

##### Attention Mechanism #####
# Initialize the attention modules with the specified device and parameters
net_attention = initialise_attention(device, config.ATTENTION_PARAMS)

# Set the time window period for processing events (in milliseconds)
window_period = 100  # Time window in milliseconds -> This value can be adjusted based on the desired temporal resolution of the saliency detection
time = window_period  # Initialize the time variable
window = torch.zeros((1, max_y, max_x), dtype=torch.float32)  # Create a tensor to hold the current window of events

# Create output folder for saved frames
output_dir = "saliency_frames"
os.makedirs(output_dir, exist_ok=True)
frame_idx = 0  # counts how many windows have been processed

# Iterate through the event data
for xi, yi, pi, ti in zip(x, y, p, t):
    if ti <= time:
        # If the event time is within the current time window, update the window
        window[0][yi][xi] = 255  # Mark the pixel corresponding to the event
    else:
        # If the event time exceeds the current time window, process the attention
        saliency_map[:], salmax_coords[:] = run_attention(window, net_attention, device, resolution,
                                                          config.ATTENTION_PARAMS['num_pyr'])

        # PLOTS
        window_map_jet = cv2.applyColorMap(window.detach().cpu().numpy().squeeze(0).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.putText(window_map_jet, 'Events map', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(window_map_jet, (int(salmax_coords[1]), int(salmax_coords[0])), 6, (255, 255, 255), 4)

        sal_normalized = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        saliency_map_color = cv2.applyColorMap(sal_normalized, cv2.COLORMAP_JET)
        cv2.putText(saliency_map_color, 'Saliency map', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(saliency_map_color, (int(salmax_coords[1]), int(salmax_coords[0])), 6, (255, 255, 255), 4)

        side_by_side = cv2.hconcat([window_map_jet, saliency_map_color])

        # Save the frame to disk
        out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(out_path, side_by_side)
        frame_idx += 1

        # Display the side-by-side visualization
        cv2.imshow('Events and Saliency Map', side_by_side)
        cv2.waitKey(1)

        # Increment the time by the window period for the next iteration
        time += window_period

        # Reset the window for the next time period
        window = torch.zeros((1, max_y, max_x), dtype=torch.float32)

# Clean up by closing the OpenCV window after processing all events
cv2.destroyAllWindows()
print(f"Saved {frame_idx} frames to {output_dir}/")