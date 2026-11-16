"""KWS-on-FPGA server for the GSCD HW-in-the-loop experiments.

Runs in the Opal Kelly python env (needs `ok` + sounddevice), started as a subprocess by
gscd_hw_loop_core.py. Protocol: one JSON object per line on stdin, one JSON reply per line on stdout.

    request : {"wav": "/path/to/sample.wav"}
    reply   : {"label": "left"|"right"|"up"|"down"|"unknown"|"none", "conf": 0-1,
               "peak_conf": 0-255, "scores": [5 raw uint8], "n_frames": int, "latency_s": float}

Per word: reset the KWS pipeline (wire 0x00) -> arm capture (wire 0x01) -> play the wav through
the jack (same as play_audio_file.py: raw samples, native rate) -> read every frame in the FIFO.
Decision ("peak" rule): class argmax in the frame with the highest confidence, with UNKNOWN_SUPPRESS
subtracted from the 'unknown' score; if that peak confidence <= CONF_THR the word is reported "none".
Frame layout of the 5-class VoiceNAS bitstream: {64'0, valid_cnt, cls[4..0], conf}; FIFO count in 32-bit words.
"""
import json, os, struct, sys, time
import numpy as np
import ok
import sounddevice as sd
from scipy.io import wavfile

CLASSES = ["up", "down", "left", "right", "unknown"]
BIT_FILE = os.environ.get("KWS_BITFILE",
    "/home/pwz/Repo/VoiceNAS/NAS-GNN-KWS/HW/vivado/kws/kws_system.runs/impl_1/ok_top_wrapper.bit")
CAPTURE_S = float(os.environ.get("KWS_CAPTURE_S", 1.5))
CONF_THR = int(os.environ.get("KWS_CONF_THR", 200))
UNKNOWN_SUPPRESS = int(os.environ.get("KWS_UNKNOWN_SUPPRESS", 25))

sd.default.samplerate = 48000
sd.default.channels = 1
sd.default.dtype = 'float32'
sd.default.latency = ('high', 'high')
sd.default.blocksize = 2048


def log(*a):
    print("[kws]", *a, file=sys.stderr, flush=True)


dev = ok.FrontPanelDevices().Open("")
if dev is None:
    log("no Opal Kelly device"); sys.exit(1)
if os.environ.get("KWS_NO_CONFIG", "0") != "1":
    rc = dev.ConfigureFPGA(BIT_FILE)
    if rc != 0:
        log("ConfigureFPGA failed", rc); sys.exit(1)
    time.sleep(0.3)
    log("configured", BIT_FILE)
dp = dev.GetFPGADataPortClassic()


def wire(addr, v):
    dp.SetWireInValue(addr, v, 1); dp.UpdateWireIns()


def classify(wav):
    t0 = time.time()
    wire(0x00, 1); time.sleep(0.01); wire(0x00, 0)
    wire(0x01, 1)
    sr, audio = wavfile.read(wav)
    sd.play(audio, sr)
    time.sleep(CAPTURE_S)
    wire(0x01, 0)
    sd.wait()
    dp.UpdateWireOuts()
    n = (dp.GetWireOutValue(0x20) & 0xFFF) // 4
    fr = np.zeros((0, 6), int)
    if n:
        buf = bytearray(n * 16)
        dp.ReadFromPipeOut(0xA0, buf)
        rows = []
        for i in range(0, len(buf), 16):
            w1, w2, w3, w4 = struct.unpack('<IIII', buf[i:i + 16])
            rows.append([w4 & 0xFF, (w4 >> 8) & 0xFF, (w4 >> 16) & 0xFF, (w4 >> 24) & 0xFF,
                         w3 & 0xFF, (w3 >> 8) & 0xFF])
        fr = np.array(rows)
    out = {"label": "none", "conf": 0.0, "peak_conf": 0, "scores": [], "n_frames": int(len(fr))}
    if len(fr):
        i = int(np.argmax(fr[:, 0]))
        s = fr[i, 1:6].astype(float)
        out["peak_conf"] = int(fr[i, 0]); out["scores"] = fr[i, 1:6].tolist()
        out["conf"] = round(fr[i, 0] / 255.0, 3)
        if fr[i, 0] > CONF_THR:
            s[4] -= UNKNOWN_SUPPRESS
            out["label"] = CLASSES[int(np.argmax(s))]
    out["latency_s"] = round(time.time() - t0, 3)
    return out


print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    if req.get("quit"):
        break
    try:
        res = classify(req["wav"])
    except Exception as e:  # never kill the experiment on one bad read
        res = {"label": "none", "conf": 0.0, "error": repr(e)}
    print(json.dumps(res), flush=True)
wire(0x01, 0)
