"""
auto_experiment.py -- automated (random25) runs of kws_mod_live_words_new_exp.py.

Everything here is OFF unless the environment variable VA_MODE is set (the runners in written_auto/ set it).
Without it, kws_mod_live_words_new_exp.py behaves exactly as before: typed commands, live window, original
scoring. With it, the same script runs one trial headless and reproducibly:

  * command source  -- commands are generated instead of typed (VA_MODE = list | possible | random25);
                       random25: before every command the possible / impossible directions are computed from the
                       current attention maximum with the scoring oracle, and with probability VA_P_IMPOSSIBLE
                       (default 0.25) an impossible one is drawn (seeded per batch + trial)
  * timing          -- first command VA_INIT windows after the first fixation; next one after the controller LOCK
                       + a 0.3-0.8 s reaction, or after VA_TIMEOUT windows (3 s) without a lock
  * spoken commands -- LINGUISTIC = 1: the drawn command is the INTENDED one; a random GSCD v2 test-split utterance
                       of it is played through the audio jack into the NAS-KWS on the FPGA (kws/gscd_hw_server.py,
                       Opal Kelly env); the RECOGNISED word is applied like a typed one (even a wrong direction);
                       unknown / no detection = the fovea is not moved. The KWS pipeline is reset before every
                       utterance.
  * scoring         -- rule 1: an impossible command (no object that way) is compliant if attention did not jump to
                       another object; rule 2 ("either"): after a command that ended WITHOUT a lock, landing on the
                       start object (if it lies in the commanded direction) or on the nearest other object that way
                       are both compliant. Extra CSV columns: correct, start_inside, prev_locked, impossible,
                       planned_impossible, mode (+ recognized, kws_conf, wav for spoken runs).
  * camera input    -- VISUAL = 1: pre-recorded DAVIS346 events, processed like kws_mod_cam_words_exp.py
                       (background-activity noise filter, clear-peak fixation gate, follow the peak before the
                       first command; r0 = 7 is set in the main script).

See written_auto/README.md for the protocol, the metrics and how to run the four experiments.
"""
import csv
import json
import os
import random
import subprocess
import threading

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
ENABLED = "VA_MODE" in os.environ          # automated run requested by a runner
GUI = os.environ.get("VA_GUI", "0") == "1"  # optional: show the live window during an automated run
RECORD = os.environ.get("VA_RECORD", "0") == "1"


# ============================== experiment block ==============================
def experiment_settings():
    """(LINGUISTIC, VISUAL, TRIAL, BATCH, NPY_PATH, MASK_PATH, NAMES, RESULTS_DIR) from the environment."""
    return (int(os.environ.get("VA_LING", 0)), int(os.environ.get("VA_VIS", 0)),
            int(os.environ["VA_TRIAL"]), int(os.environ["VA_BATCH"]),
            os.environ["VA_NPY"], os.environ["VA_MASK"], json.loads(os.environ["VA_NAMES"]),
            os.environ["VA_RESULTS"])


# ============================== camera recordings ==============================
MIN_EVENTS_FIX = 300          # kws_mod_cam_words_exp.py: events a window needs before the first fixation
FIX_PEAK_RATIO = 2.0          # ... and the saliency peak must be this many times the map mean
FOLLOW_BEFORE_COMMAND = True  # ... until the first command the fovea follows the saliency peak


def camera_noise_filter(npy_path, data, ev_x, ev_y, ev_t_us, noise_us=None):
    """dv_processing BackgroundActivityNoiseFilter over the whole stream (same filter as the live camera script,
    state carried across windows). Cached as .npz in written_auto/davis_cache/. Returns filtered x, y, t [us]."""
    noise_us = int(os.environ.get("VA_NOISE_US", 500)) if noise_us is None else noise_us
    if not noise_us:
        return ev_x, ev_y, ev_t_us
    cache_dir = os.environ.get("VA_EVENT_CACHE", os.path.join(REPO, "written_auto", "davis_cache"))
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, os.path.basename(npy_path).replace(".npy", f"_ba{noise_us}us.npz"))
    if os.path.exists(cache):
        z = np.load(cache)
        fx, fy, ft = z["x"].astype(int), z["y"].astype(int), z["t"].astype(float)
    else:
        import dv_processing as dv
        from datetime import timedelta
        o = np.argsort(ev_t_us, kind="stable")
        names = data.dtype.names
        if names is not None and ("p" in names or "polarity" in names):
            pol = data["p" if "p" in names else "polarity"]
        else:
            pol = np.ones(len(ev_t_us))
        pol = pol.astype(bool)[o]
        store = dv.EventStore()
        for t, x, y, p in zip(ev_t_us[o].astype(np.int64) + 1, ev_x[o], ev_y[o], pol):   # +1: dv rejects t = 0
            store.push_back(int(t), int(x), int(y), bool(p))
        filt = dv.noise.BackgroundActivityNoiseFilter((346, 260), backgroundActivityDuration=timedelta(microseconds=noise_us))
        filt.accept(store)
        a = filt.generateEvents().numpy()
        fx, fy, ft = a["x"].astype(int), a["y"].astype(int), (a["timestamp"] - 1).astype(float)
        np.savez(cache, x=fx.astype(np.int16), y=fy.astype(np.int16), t=ft.astype(np.int64))
    print(f"noise filter {noise_us} us: kept {len(ft)} of {len(ev_t_us)} events")
    return fx, fy, ft


def raw_window_counts(raw_t_ms, t0, n_frames, window_ms):
    """Number of RAW (unfiltered) events per window -- what the live camera script gates the fixation on."""
    raw_t_ms = np.sort(raw_t_ms)
    return np.bincount(np.clip(((raw_t_ms - t0) // window_ms).astype(int), 0, n_frames - 1), minlength=n_frames)


def clear_peak(saliency, salmax, n_raw):
    """True if the window has enough events and a saliency peak well above the mean (camera script rule)."""
    py, px = int(salmax[0]), int(salmax[1])
    peak, mean = float(saliency[py, px]), float(saliency.mean())
    return (n_raw >= MIN_EVENTS_FIX and mean > 0 and peak >= FIX_PEAK_RATIO * mean), py, px


# ============================== scoring rules ==============================
def inside_id(mask, p):
    """Object whose footprint contains the point (no snap tolerance), else None."""
    x, y = int(round(p[0])), int(round(p[1]))
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
        return int(mask[y, x])
    return None


def target_for(mask, cents, snap, ref, direction, prev_locked, object_near, oracle):
    """Used to decide whether a direction is POSSIBLE (non-empty targets_for <=> this is not None): the start object
    is excluded only if the previous command ended with a LOCK."""
    start = object_near(mask, cents, ref, snap)
    return oracle(cents, ref, direction, exclude=start if prev_locked else None)


def targets_for(mask, cents, snap, ref, direction, prev_locked, object_near, oracle):
    """Acceptable landing objects (rule 2, "either"). After a LOCKED command (or the first one): the nearest other
    object in the commanded direction, as in the original oracle. After an UNLOCKED command the true fovea can be up
    to READOUT_R away from the logged attended point, so whether the controller (lock only after MIN_TRAVEL) can
    re-acquire the start object cannot be judged: the start object (if it lies that way) and the nearest other
    object that way are both compliant."""
    start = object_near(mask, cents, ref, snap)
    if prev_locked or start is None:
        return {oracle(cents, ref, direction, exclude=start)} - {None}
    return {oracle(cents, ref, direction, exclude=None), oracle(cents, ref, direction, exclude=start)} - {None}


def score_step(mask, cents, snap, ref, direction, prev_locked, start_id, landed, object_near, oracle, label):
    """Rules 1 + 2. Returns (target, verdict, ok)."""
    targets = targets_for(mask, cents, snap, ref, direction, prev_locked, object_near, oracle)
    target = min(targets) if targets else None
    if target is not None and landed in targets:
        return target, "OK", True
    if target is None:      # rule 1: impossible command - ok unless another object captured attention
        ok = landed is None or landed == start_id
        return target, ("no object that way" if ok else "no object that way - jumped to another object"), ok
    if landed == start_id:
        return target, "FAILED - did not move", False
    return target, f"WRONG - expected {label(target)}", False


# ============================== command source ==============================
class Commander:
    """Generates the commands of one automated trial and, for spoken runs, drives the KWS on the FPGA."""

    def __init__(self, dirs, cmd_queue, threshold, command_limit, batch, trial, spoken):
        self.dirs, self.q, self.threshold, self.spoken = dirs, cmd_queue, threshold, spoken
        self.mode = os.environ.get("VA_MODE", "list")
        n = int(os.environ.get("VA_NCMDS", command_limit))
        self.p_imp = float(os.environ.get("VA_P_IMPOSSIBLE", 0.25))
        self.cmds = ([c.strip() for c in os.environ.get("VA_CMDS", "").split(",") if c.strip()]
                     if self.mode == "list" else [None] * n)
        # separate seeded streams, drawn in a fixed order -> reproducible trials
        self.rng = random.Random(f"react-{batch}-{trial}")
        self.rng_cmd = random.Random(f"cmd-{batch}-{trial}")
        self.init = int(os.environ.get("VA_INIT", 10))
        self.timeout = int(os.environ.get("VA_TIMEOUT", 30))
        self.rmin, self.rmax = 3, 8                         # reaction after a lock, in 100 ms windows
        self.st = {"i": 0, "issued_at": None, "lock_at": None, "react": None, "first_fix": None, "done": False}
        self.planned = []                                   # per command: cmd, planned_impossible, n_possible, prev_locked
        self.meta_active = None
        self.kws_log = []
        self.kws = {"busy": False, "result": None}
        if spoken:
            self._start_kws(batch, trial)

    # ---- spoken commands: GSCD sample -> KWS server (Opal Kelly env) ----
    def _start_kws(self, batch, trial):
        audio = os.environ.get("VA_GSCD_AUDIO", "/home/pwz/Music/NAS/gsc_v2")
        test = os.environ.get("VA_GSCD_TEST", "/media/pwz/KINGSTON/NAS-KWS/datasets-spike/gsc_v2_voice_spike/testing_list.txt")
        ok_py = os.environ.get("VA_OK_PYTHON", "/home/pwz/Repo/OK-FINAL/NAS-GNN-KWS/SW/ok/ok-venv/bin/python")
        self.audio = audio
        self.rng_wav = random.Random(f"gscd-{batch}-{trial}")
        self.test_by = {}
        for line in open(test):
            line = line.strip()
            if line:
                self.test_by.setdefault(line.split("/")[0], []).append(line)
        self.srv = subprocess.Popen([ok_py, os.path.join(REPO, "kws", "gscd_hw_server.py")],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        if '"ready"' not in self.srv.stdout.readline():
            raise RuntimeError("KWS server did not start")

    def _kws_worker(self, intended, wav):
        self.srv.stdin.write(json.dumps({"wav": wav}) + "\n"); self.srv.stdin.flush()
        self.kws["result"] = (intended, wav, json.loads(self.srv.stdout.readline()))

    # ---- called once per window from the main loop, after the attended point is known ----
    def feed(self, win, is_locked, fixated, ref, target_fn):
        a = self.st
        if a["done"] or not fixated:
            return
        if a["first_fix"] is None:
            a["first_fix"] = win
        if self.spoken and self.kws["result"] is not None:      # the spoken word was classified -> apply it now
            intended, wav, res = self.kws["result"]; self.kws["result"] = None; self.kws["busy"] = False
            meta = {"intended": intended, "recognized": res.get("label", "none"),
                    "kws_conf": res.get("conf", 0.0), "wav": wav}
            self.kws_log.append({"step": a["i"], "window": win, **meta,
                                 **{k: res.get(k) for k in ("peak_conf", "scores", "latency_s")}})
            if meta["recognized"] in self.dirs:
                self.q.put({"type": "word", "word": meta["recognized"],
                            "conf": max(meta["kws_conf"], self.threshold), "meta": meta})
            else:
                self.q.put({"type": "miss", "meta": meta})
            a["issued_at"] = win; a["lock_at"] = None
            return
        if self.spoken and self.kws["busy"]:
            return
        if a["issued_at"] is None:
            due = win - a["first_fix"] >= self.init
        else:
            if is_locked and a["lock_at"] is None:
                a["lock_at"] = win; a["react"] = self.rng.randint(self.rmin, self.rmax)
            due = (a["lock_at"] is not None and win - a["lock_at"] >= a["react"]) or (win - a["issued_at"] >= self.timeout)
        if not due:
            return
        if a["i"] >= len(self.cmds):
            self.q.put("quit"); a["done"] = True
            return
        prev_locked = True if a["issued_at"] is None else bool(is_locked)
        poss = [d for d in self.dirs if target_fn(ref, d, prev_locked) is not None]
        imp = [d for d in self.dirs if d not in poss]
        if self.mode == "list":
            c = self.cmds[a["i"]]
        elif self.mode == "possible":
            c = self.rng_cmd.choice(poss or imp)
        else:  # random25
            want_imp = self.rng_cmd.random() < self.p_imp
            pool = (imp or poss) if want_imp else (poss or imp)
            c = self.rng_cmd.choice(pool)
        self.cmds[a["i"]] = c
        self.planned.append({"cmd": c, "planned_impossible": int(c in imp), "n_possible": len(poss), "prev_locked": prev_locked})
        if self.spoken:
            wav = os.path.join(self.audio, self.rng_wav.choice(self.test_by[c]))
            self.kws["busy"] = True
            threading.Thread(target=self._kws_worker, args=(c, wav), daemon=True).start()
            a["i"] += 1
        else:
            self.q.put(c)
            a["i"] += 1; a["issued_at"] = win; a["lock_at"] = None

    # ---- logging helpers used by close_command() / drain_commands() / write_results() ----
    def set_meta(self, meta):
        self.meta_active = meta

    def annotate(self, entry, word):
        """Spoken runs: the logged direction is the INTENDED command; keep the recognised one alongside."""
        if self.spoken:
            m = self.meta_active or {}
            entry["direction"] = m.get("intended", word)
            entry.update(recognized=word, kws_conf=m.get("kws_conf", ""), wav=m.get("wav", ""))
        return entry

    @staticmethod
    def miss_entry(meta, att_x, att_y):
        """Row for a spoken command that was not recognised as a direction: the fovea did not move."""
        return {"direction": meta["intended"], "k": 1, "recognized": meta["recognized"],
                "kws_conf": meta["kws_conf"], "wav": meta["wav"],
                "ref_x": round(att_x, 1), "ref_y": round(att_y, 1),
                "fovea_x": round(att_x, 1), "fovea_y": round(att_y, 1)}

    def prev_locked(self, i):
        return self.planned[i - 1]["prev_locked"] if i - 1 < len(self.planned) else True

    def planned_impossible(self, i):
        return self.planned[i - 1]["planned_impossible"] if i - 1 < len(self.planned) else ""

    def close(self, results_dir, trial_id):
        """Stop the KWS server and write {trial_id}_kws.csv (spoken runs)."""
        if not self.spoken:
            return
        try:
            self.srv.stdin.write(json.dumps({"quit": True}) + "\n"); self.srv.stdin.flush(); self.srv.wait(5)
        except Exception:
            self.srv.kill()
        if self.kws_log:
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, f"{trial_id}_kws.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.kws_log[0].keys())); w.writeheader(); w.writerows(self.kws_log)
            ok = sum(r["intended"] == r["recognized"] for r in self.kws_log)
            print(f"KWS: {ok}/{len(self.kws_log)} spoken commands recognised correctly")
