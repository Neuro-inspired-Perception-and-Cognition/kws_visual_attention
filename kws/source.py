"""
source.py
Turns the FPGA's 100 Hz prediction stream into discrete attention commands.

The classification matches the reference inference script for this head:

    1. collect a batch of frames (25 = 250 ms, what the reference reads)
    2. keep only frames with conf > accept_conf
    3. average each class score over those
    4. take the top class -- but if even the best averaged score is still
       negative, nothing cleared the -125 offset and the batch holds no word
       (the reference's "Bleh" case), so nothing is emitted
    5. if the winner is a direction, emit a command
    6. reload the bitstream, as the reference does after every prediction

Commands are pushed as dicts identical to the ones the keyboard reader
produces, so the attention loop never learns where a command came from:

    {"type": "word", "word": "left", "conf": 0.86, "src": "kws"}

This head has no "stop" word, so the spotter never emits a clear.

BACKENDS
    frontpanel : live board over USB3
    replay     : recorded CSV (t_ms,label,conf) - lets ground_truth.py score
                 without the board attached
    off        : no-op, typed commands only
"""

import csv
import threading
import time

from .protocol import (
    CLEARING, CONF_MAX, DIRECTIONS, FRAME_BYTES, N_CLASSES, PIPE_OUT_PRED,
    SEQ_MOD, UNKNOWN_INDEX, WIRE_IN_ARM, WIRE_IN_RESET, WIRE_OUT_FIFO_COUNT,
    WORDS, WORDS_PER_FRAME, decode_buffer,
)


class FrontPanelBackend:
    """Live board: configure, arm, drain the fifo once a backlog has built up."""

    def __init__(self, bitfile=None, serial="", min_backlog_frames=25,
                 max_frames_per_poll=256):
        import ok  # lazy: the replay path doesn't need the sdk
        self.dev = ok.FrontPanelDevices().Open(serial)
        if self.dev is None:
            raise RuntimeError("no Opal Kelly device opened "
                               "(is the FrontPanel app still holding it?)")
        self.bitfile = bitfile
        self.configure()
        self.min_backlog = min_backlog_frames
        self.max_frames = max_frames_per_poll
        self.reset()
        self.arm(True)

    def configure(self):
        """Load the bitstream and re-acquire the data port handle.

        The handle has to be fetched again after every configuration, and the
        device needs a moment to settle before its endpoints are touched --
        a WireIn too soon after reprogramming can break FrontPanel.
        """
        if self.bitfile:
            rc = self.dev.ConfigureFPGA(self.bitfile)
            if rc != 0:
                raise RuntimeError(f"ConfigureFPGA failed: {rc}")
            time.sleep(0.3)
        self.dp = self.dev.GetFPGADataPortClassic()
        if self.dp is None:
            raise RuntimeError("GetFPGADataPortClassic returned None")

    def _wire(self, addr, val):
        self.dp.SetWireInValue(addr, val, 1)
        self.dp.UpdateWireIns()

    def reset(self):
        self._wire(WIRE_IN_RESET, 1)
        time.sleep(0.01)
        self._wire(WIRE_IN_RESET, 0)
        time.sleep(0.01)

    def arm(self, on):
        self._wire(WIRE_IN_ARM, 1 if on else 0)
        time.sleep(0.01)

    def resync(self):
        """Disarm, reset, rearm. Drops in-flight audio -- only call between words."""
        self.arm(False)
        self.reset()
        self.arm(True)

    def reload(self):
        """Full restart: reload the bitstream, then reset and re-arm.

        What the reference does after every accepted prediction. Takes about a
        second, during which no audio is being classified.
        """
        self.arm(False)
        self.configure()
        self.reset()
        self.arm(True)

    def frames_available(self):
        self.dp.UpdateWireOuts()
        return self.dp.GetWireOutValue(WIRE_OUT_FIFO_COUNT) // WORDS_PER_FRAME

    def poll(self):
        """Return [(conf, scores, seq)], or [] until min_backlog frames are queued."""
        n = self.frames_available()
        if n < self.min_backlog:
            return []
        n = min(n, self.max_frames)
        buf = bytearray(n * FRAME_BYTES)
        rc = self.dp.ReadFromPipeOut(PIPE_OUT_PRED, buf)
        if rc < 0:
            raise RuntimeError(f"ReadFromPipeOut failed: {rc}")
        return decode_buffer(buf, rc)

    def close(self):
        try:
            self.arm(False)
        except Exception:
            pass


class ReplayBackend:
    """Recorded commands, wall-clock paced. csv columns: t_ms,label,conf."""

    def __init__(self, path, speed=1.0):
        self.rows = []
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                self.rows.append((float(r["t_ms"]), r["label"], int(r["conf"])))
        self.rows.sort(key=lambda r: r[0])
        self.speed = speed
        self.i = 0
        self.t0_wall = None
        self.t0_ms = 0.0          # logged times are already relative to session start

    def poll_commands(self):
        if self.i >= len(self.rows):
            return []
        if self.t0_wall is None:
            self.t0_wall = time.monotonic()
        elapsed = (time.monotonic() - self.t0_wall) * 1e3 * self.speed
        out = []
        while self.i < len(self.rows) and (self.rows[self.i][0] - self.t0_ms) <= elapsed:
            _, label, conf = self.rows[self.i]
            out.append((label, conf))
            self.i += 1
        return out

    def close(self):
        pass


def make_backend(kind, **kw):
    if kind == "frontpanel":
        return FrontPanelBackend(bitfile=kw.get("bitfile"),
                                 serial=kw.get("serial", ""),
                                 min_backlog_frames=kw.get("min_backlog_frames", 25))
    if kind == "replay":
        return ReplayBackend(kw["path"], speed=kw.get("speed", 1.0))
    if kind == "off":
        return None
    raise ValueError(f"unknown kws backend: {kind!r}")


class KWSSource(threading.Thread):
    """Polls a backend, emits commands into out_queue. Never raises into the caller.

    If the board fails mid-run the thread stops, sets `.error`, and prints one
    line -- the attention loop keeps running and typed commands still work.
    """

    def __init__(self, backend, out_queue, accept_conf=179, unknown_penalty=0,
                 batch_frames=25, min_valid_frames=1, refractory_ms=800,
                 resync_every_batches=0, reload_after_command=True,
                 poll_ms=5, log_path=None, verbose=True):
        super().__init__(daemon=True)
        self.backend = backend
        self.q = out_queue
        self.accept_conf = accept_conf
        self.unknown_penalty = unknown_penalty
        self.batch_frames = batch_frames
        self.min_valid_frames = min_valid_frames
        self.refractory_s = refractory_ms / 1000.0
        self.resync_every_batches = resync_every_batches
        self.reload_after_command = reload_after_command
        self.poll_s = poll_ms / 1000.0
        self.log_path = log_path
        self.verbose = verbose

        self._stop = threading.Event()
        self._pending = []        # frames not yet formed into a batch
        self._last_label = None
        self._last_t = -1e9
        self._n_batches = 0
        self._last_seq = None
        self.t0 = None
        self.accepted = []        # (t_ms, label, conf) -- replayable log
        self.n_frames = 0
        self.n_dropped = 0
        self.error = None

    def stop(self):
        self._stop.set()

    def _classify(self, batch):
        """Returns (label, mean_conf, n_valid), or None if the batch holds no word."""
        hi = [f for f in batch if f[0] > self.accept_conf]
        if len(hi) < self.min_valid_frames:
            return None
        avg = [0.0] * N_CLASSES
        for _, scores, _ in hi:
            for i in range(N_CLASSES):
                avg[i] += scores[i]
        avg = [v / len(hi) for v in avg]
        avg[UNKNOWN_INDEX] -= self.unknown_penalty
        best = max(range(N_CLASSES), key=lambda i: avg[i])
        # the reference's "Bleh" test: every averaged score negative means no
        # class cleared the offset, so this batch is not a word at all
        if avg[best] < 0:
            return None
        mean_conf = sum(f[0] for f in hi) / len(hi)
        return WORDS[best], int(mean_conf), len(hi)

    def _handle_batch(self, batch):
        self._n_batches += 1
        r = self._classify(batch)
        if r is None:
            return
        label, conf, n_valid = r
        if label not in DIRECTIONS and label not in CLEARING:
            if self.verbose:
                print(f"\n  [kws] heard '{label}' ({n_valid} frames) -- not a command")
            return
        now = time.monotonic()
        # one word often spans two batches: don't fire it twice
        if label == self._last_label and (now - self._last_t) < self.refractory_s:
            return
        self._last_label, self._last_t = label, now
        self._emit(label, conf, n_valid)
        # the reference reloads the bitstream after every accepted prediction
        if self.reload_after_command and hasattr(self.backend, "reload"):
            self.backend.reload()
            self._pending.clear()
            self._last_seq = None

    def _emit(self, label, conf, n_valid=0):
        if self.t0 is None:
            self.t0 = time.monotonic()
        t_ms = (time.monotonic() - self.t0) * 1e3
        if label in CLEARING:
            self.q.put({"type": "stop", "src": "kws"})
        else:
            self.q.put({"type": "word", "word": label,
                        "conf": conf / CONF_MAX, "src": "kws"})
        self.accepted.append((round(t_ms, 1), label, conf))
        if self.verbose:
            # leading \n so we don't land on the main loop's \r status line
            detail = f" ({n_valid} frames)" if n_valid else ""
            print(f"\n  [kws] {label:6} conf={conf}{detail}")

    def _maybe_resync(self):
        """Periodic disarm/reset/rearm. Off by default: the reload after each
        command does the same job. Never fires right after a word."""
        if not self.resync_every_batches or not self._n_batches:
            return
        if self._n_batches % self.resync_every_batches:
            return
        if time.monotonic() - self._last_t < 1.0:
            return
        self.backend.resync()
        self._pending.clear()
        self._last_seq = None     # the frame counter restarts after a reset
        self._n_batches += 1      # don't resync twice on the same count

    def run(self):
        if self.backend is None:
            return
        self.t0 = time.monotonic()
        replay = hasattr(self.backend, "poll_commands")
        try:
            while not self._stop.is_set():
                if replay:
                    for label, conf in self.backend.poll_commands():
                        self._emit(label, conf)
                else:
                    for frame in self.backend.poll():
                        self.n_frames += 1
                        seq = frame[2]
                        if self._last_seq is not None:
                            self.n_dropped += (seq - self._last_seq - 1) % SEQ_MOD
                        self._last_seq = seq
                        self._pending.append(frame)
                    while len(self._pending) >= self.batch_frames:
                        batch = self._pending[:self.batch_frames]
                        del self._pending[:self.batch_frames]
                        self._handle_batch(batch)
                    self._maybe_resync()
                time.sleep(self.poll_s)
        except Exception as e:
            self.error = e
            print(f"\n  [kws] stopped: {e}  -- typed commands still work")

    def write_log(self):
        if not (self.log_path and self.accepted):
            return
        with open(self.log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_ms", "label", "conf"])
            w.writerows(self.accepted)
        print(f"wrote {self.log_path}  ({len(self.accepted)} commands, "
              f"{self.n_frames} frames, {self.n_dropped} dropped)")

    def close(self):
        self.stop()
        if self.backend is not None:
            try:
                self.backend.close()
            except Exception:
                pass
        self.write_log()