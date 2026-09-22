# KWS accuracy evaluation - prompted trials from the microphone, results written to CSV.
# For each trial: the target word is shown, press Enter and say it.
# Controls: Enter = record   s = skip   q = quit and save

# KWS accuracy evaluation - prompted trials from the microphone, results written to CSV.
# Countdown, then a recording window: speak when it says "speak now".
# Controls: Enter = record   s = skip   q = quit and save

import os
import sys
import csv
import time
import struct
import random
from datetime import datetime
import ok

bit_file = r"bitstreams/ok_top_wrapper.bit"
out_csv = os.path.join("Kws_eval_results", f"kws_eval_{datetime.now():%Y%m%d_%H%M%S}.csv")

words_comm = [
    "yes", "no", "up", "down", "left",
    "right", "on", "off", "stop", "go", "unknown"
]

bytes_per_frame = 16
confidence_threshold = 190
min_frames_per_read = 25    # read in batches, like continuous mode does
record_seconds = 3.0
trials_per_word = 10           # 10 trials per word, 10 words = 100 trials total
eval_words = words_comm[:10]   # add "unknown" to also test silence/non-keywords -: [:11]
fallback_frames = 5            # if nothing passes the threshold, use the N most confident frames

# FPGA init
devices = ok.FrontPanelDevices()
dev = devices.Open("")
if dev is None:
    print("Device could not be opened")
    sys.exit(1)

cfg_result = dev.ConfigureFPGA(bit_file)
if cfg_result != 0:
    print(f"FPGA configuration failed, error code {cfg_result}")
    sys.exit(1)
print("FPGA configured successfully.")

dp = dev.GetFPGADataPortClassic()
if dp is None:
    print("Failed to get FPGA Data Port.")
    sys.exit(1)


def set_wire(addr, value):
    dp.SetWireInValue(addr, value, 1)
    dp.UpdateWireIns()
    time.sleep(0.01)


def restart_pipeline():
    set_wire(0x01, 0)   # disarm
    set_wire(0x00, 1)   # reset pulse
    set_wire(0x00, 0)
    set_wire(0x01, 1)   # arm


def frames_available():
    dp.UpdateWireOuts()
    return int(dp.GetWireOutValue(0x20) / 4)


def read_frame_batch(no_frames):
    buf = bytearray(no_frames * bytes_per_frame)
    dp.ReadFromPipeOut(0xA0, buf)

    frames = []
    for i in range(0, len(buf), bytes_per_frame):
        word1, word2, word3, word4 = struct.unpack('<IIII', buf[i:i + bytes_per_frame])
        if word1 == 0 and word2 == 0 and word3 == 0 and word4 == 0:
            continue
        scores = [(word4 >> s) & 0xFF for s in (8, 16, 24)] \
               + [(word3 >> s) & 0xFF for s in (0, 8, 16, 24)] \
               + [(word2 >> s) & 0xFF for s in (0, 8, 16, 24)]
        scores[10] -= 70   # 'unknown' class penalty
        frames.append({"confidence": word4 & 0xFF, "scores": scores})
    return frames


def drain_fifo():
    while (n := frames_available()) >= min_frames_per_read:
        read_frame_batch(n)


def record_trial():
    drain_fifo()
    for c in ("3...", "2...", "1...", "speak now"):
        print(f"   {c}", end="\r", flush=True)
        time.sleep(0.5)

    frames = []
    t_end = time.time() + record_seconds
    while time.time() < t_end:
        n = frames_available()
        if n >= min_frames_per_read:
            frames.extend(read_frame_batch(n))
        time.sleep(0.02)
    return frames


def classify(frames):
    good = [f for f in frames if f["confidence"] > confidence_threshold]
    fallback = 0
    if not good and frames:
        good = sorted(frames, key=lambda f: f["confidence"], reverse=True)[:fallback_frames]
        fallback = 1
    if not good:
        return None, 0, 0
    avg = [sum(f["scores"][c] for f in good) / len(good) for c in range(11)]
    ranked = sorted(range(11), key=lambda i: avg[i], reverse=True)
    return [words_comm[i] for i in ranked[:3]], len(good) if not fallback else 0, fallback


# Evaluation loop 
os.makedirs("kws_eval_results", exist_ok=True)
schedule = [w for w in eval_words for _ in range(trials_per_word)]
random.shuffle(schedule)

restart_pipeline()   # armed once, like continuous mode - no reset between trials
n_done = n_correct = 0

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["trial", "target", "top1", "top2", "top3", "correct",
                     "valid_frames", "total_frames", "max_conf", "fallback"])
    try:
        for idx, target in enumerate(schedule, start=1):
            cmd = input(f"[{idx}/{len(schedule)}] Say '{target}' -> Enter: ").strip().lower()
            if cmd == "q":
                break
            if cmd == "s":
                continue

            frames = record_trial()
            top, valid, fallback = classify(frames)
            top = top or ["", "", ""]
            correct = int(top[0] == target)
            max_conf = max((f["confidence"] for f in frames), default=0)

            writer.writerow([idx, target, *top, correct, valid,
                             len(frames), max_conf, fallback])
            f.flush()
            n_done += 1
            n_correct += correct
            tag = "OK " if correct else "ERR"
            note = " [fallback]" if fallback else ""
            print(f"   {tag} top1={top[0] or '-':<8} valid {valid}/{len(frames)} "
                  f"max_conf {max_conf}{note}    ")
    finally:
        set_wire(0x01, 0)

if n_done:
    print(f"\nAccuracy: {n_correct}/{n_done} = {n_correct / n_done:.1%}")
print(f"Saved: {out_csv}")