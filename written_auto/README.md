# Automated attention-modulation experiments (random25 protocol)

All four experiments run the **original** experiment script `kws_mod_live_words_new_exp.py`. An automated run is
switched on by the environment variable `VA_MODE`, which the runners in this folder set. The added logic lives in
`auto_experiment.py` (repo root), and every hook in the original script is marked `# CHANGED (random25)` or
`# CHANGED (camera)`. Without `VA_MODE`, the script behaves exactly as before: typed commands, live window, original
scoring, results in `results/`.

| # | Visual input | Commands | `VA_LING` / `VA_VIS` | Trial-ID prefix | Runner | Results |
|---|---|---|---|---|---|---|
| 1 | Synthetic (IEBCS events, `data/*.npy`) | written | 0 / 0 | `00_` | `run_written_random25.py` | `results_random25_rule2fix/` |
| 2 | Synthetic (IEBCS events) | spoken: GSCD v2 test samples -> NAS-KWS on FPGA | 1 / 0 | `10_` | `run_gscd_hw_random25.py` | `results_random25_gscd_hw_rule2fix/` |
| 3 | DAVIS346 recording of the stimuli on a monitor | written | 0 / 1 | `01_` | `run_davis_written_random25.py` | `results_random25_davis/` |
| 4 | DAVIS346 recording of the stimuli on a monitor | spoken: GSCD v2 -> NAS-KWS on FPGA | 1 / 1 | `11_` | `run_davis_gscd_hw_random25.py` | `results_random25_davis_gscd_hw/` (not run yet) |

Files are named `{prefix}_tTT_bBB.csv` (trial TT, batch BB). Spoken runs also write `..._kws.csv`, one row per spoken
command: intended word, recognised word, KWS confidence, wav file and latency.

---

## 1. Protocol (all four experiments)

* **Stimuli: the 10 Set-1 batches** (`batches.py`):

  | batch | scene | batch | scene |
  |---|---|---|---|
  | 1 | 4 circles / nobg / mono | 2 | 4 circles / bg / mono |
  | 3 | 6 circles / nobg / mono | 4 | 6 circles / bg / mono |
  | 5 | 6 objects / nobg / mono | 6 | 6 objects / bg / mono |
  | 7 | 6 objects / nobg / color | 8 | 6 objects / bg / color |
  | 9 | 9 objects / nobg / color | 10 | 9 objects / bg / color |

  "bg" is always the **jittering** hatch background (`*_bg_jitter`). Odd batches have no clutter, even batches have clutter.
* **10 trials per batch, 10 commands per trial** (100 trials per experiment).
* **Command generation (`VA_MODE=random25`)**: before every command, the possible and impossible directions are
  computed from the *current* attention maximum with the same oracle that is used for scoring. With probability 0.25
  an impossible direction is drawn, otherwise a possible one. The draws are seeded per batch and trial, so runs are
  reproducible. A command is *impossible* if no object lies in that direction (the commanded axis must dominate:
  `|dy| > |dx|` for up/down, `|dx| > |dy|` for left/right).
* **Timing**: the first command comes 1–3 s (10–30 windows, depending on the trial) after the first fixation.
  The next command comes after the controller LOCK plus a 0.3–0.8 s "reaction", or after 3 s (30 windows) without a lock.
  Windows are 100 ms of events.
* **Spoken commands (experiments 2 and 4)**: the drawn command is the **intended** one. A random GSCD v2 *test-split*
  utterance of that word is played through the audio jack into the NAS-KWS bitstream. The **recognised** word is applied
  to the attention exactly like a typed one, even when it is the wrong direction. `unknown` or no detection means the
  fovea is not moved. All scoring is against the intended command. The KWS pipeline is **reset before every utterance**
  (wire 0x00), so the GRU starts from zero shortly before the word, as in training. Without the reset, recognition
  degrades in continuous operation; see `HANDOFF.md`. Headless spoken runs are paced in real time (100 ms per window),
  so the ~1.5 s KWS latency costs the same number of windows as in the live system.

## 2. Metrics and scoring

* **Command compliance**: % of correct steps in a trial, averaged over trials (mean ± population std).
* **Lock rate**: % of steps that end with attention on *any* object (`landed_id != 0`).
* **Compliance for possible / impossible commands**, reported separately, because impossible commands are almost
  always compliant and would otherwise inflate the overall score, especially with clutter.
* **KWS accuracy** (spoken runs): % of commands whose recognised word equals the intended one.

Scoring rules in automated runs (`auto_experiment.score_step`, called from `write_results()`):

1. **Impossible command** (no object in that direction): compliant if attention did not jump to another object,
   i.e. it stays on the start object or ends on the background (the lock rate tells which).
   *Original:* staying on the start object was always "FAILED - did not move".
2. **Command after a command that ended without a LOCK** (rule "either"): the logged reference point is the attention
   maximum, not the fovea, so it cannot be decided whether the controller (which may lock only after `MIN_TRAVEL` = 50 px)
   can re-acquire the start object. Both the start object (if it lies in the commanded direction) and the nearest
   *other* object in that direction are accepted. After a locked command, and for the first command, the start object
   is excluded, as in the original oracle.

Unchanged: `object_near()` (footprint distance + snap = 2 × median radius, clamped to 0.45 × the edge-to-edge gap),
`oracle()`, the attention network and the membrane/fovea controller with all its parameters. Extra CSV columns in
automated runs: `correct`, `start_inside`, `prev_locked`, `impossible`, `planned_impossible`, `mode`, plus
`recognized`, `kws_conf`, `wav` for spoken runs.

## 3. Prerequisites

| What | Where |
|---|---|
| Attention env (torch, sinabs, cv2, dv_processing) | `venv/` of this repo |
| Opal Kelly env (`ok`, sounddevice), used only by the KWS server | `/home/pwz/Repo/OK-FINAL/NAS-GNN-KWS/SW/ok/ok-venv/bin/python` (override: `VA_OK_PYTHON`) |
| KWS bitstream (5 classes: up, down, left, right, unknown) | `/home/pwz/Repo/VoiceNAS/NAS-GNN-KWS/HW/vivado/kws/kws_system.runs/impl_1/ok_top_wrapper.bit` (override: `KWS_BITFILE`) |
| GSCD v2 audio / test list | `/home/pwz/Music/NAS/gsc_v2`, `/media/pwz/KINGSTON/NAS-KWS/datasets-spike/gsc_v2_voice_spike/testing_list.txt` (`VA_GSCD_AUDIO`, `VA_GSCD_TEST`) |
| DAVIS recordings | `/media/pwz/KINGSTON/NAS-KWS/Experiments/Event and Grayscale data - DAVIS-.../raw/{events,gs_videos}` |
| Hardware (spoken runs) | XEM7310 on USB + PC audio jack -> NAS I2S input. Nothing else may use the board or the jack during a run, and the jack must stay plugged in. |

`kws/protocol.py` in this repo decodes an **older** class layout (unknown at index 6, score offsets) and does **not** match
the VoiceNAS bitstream. The spoken experiments use `kws/gscd_hw_server.py` instead.

**Load**: one trial process ≈ 1 GB RAM. 8 parallel jobs × 2 threads froze the PC once, so the runners default to light
settings: `nice 19`, 1 thread per process; synthetic written 4 jobs, DAVIS written 2 jobs, spoken 1 job (one board).

## 4. How to run

All commands from the repo root (`~/Repo/kws_visual_attention`). Every runner accepts `--trials N`,
`--batches 1,2` / `--batches 1-10`, `--out DIR` and `--metrics-only` (re-prints the tables from existing CSVs).

```bash
# 1. synthetic + written        (~5-10 min)
venv/bin/python written_auto/run_written_random25.py --trials 10
# 2. synthetic + GSCD-HW        (~70 min, sequential, board + jack)
venv/bin/python written_auto/run_gscd_hw_random25.py --trials 10
# 3. DAVIS + written            (~10 min; once before: ground-truth masks, check the previews)
venv/bin/python written_auto/davis_ground_truth.py
venv/bin/python written_auto/run_davis_written_random25.py --trials 10
# 4. DAVIS + GSCD-HW            (~70 min, sequential, board + jack)
venv/bin/python written_auto/run_davis_gscd_hw_random25.py --trials 10
```

Quick check on a small subset, e.g. one trial of batches 1 and 2:
```bash
venv/bin/python written_auto/run_davis_written_random25.py --trials 1 --batches 1,2 --out written_auto/_check
```

A single trial without a runner, e.g. to watch it live (`VA_GUI=1`; `VA_RECORD=1` also saves an mp4):
```bash
VA_MODE=random25 VA_LING=0 VA_VIS=0 VA_TRIAL=1 VA_BATCH=1 VA_GUI=1 \
VA_NPY=data/4_circles_mono_nobg_346x260.npy \
VA_MASK=stimuli/ground_truth_masks/4_circles_mono_nobg_346x260.mask.npy \
VA_NAMES='["top-left","top-right","bottom-left","bottom-right"]' VA_RESULTS=written_auto/_check \
venv/bin/python kws_mod_live_words_new_exp.py
```
Other optional variables: `VA_INIT` (windows before the first command), `VA_TIMEOUT` (30), `VA_P_IMPOSSIBLE` (0.25),
`VA_NCMDS` (10), `VA_CMDS` + `VA_MODE=list` (replay a fixed list), `VA_NOISE_US` (camera filter, 500).

The first DAVIS run builds the noise-filtered event caches in `written_auto/davis_cache/`. Parallel jobs must not build
the same cache, so run trial 1 of each batch alone first (e.g. `--trials 1 --jobs 1`) if the cache is missing.

### Re-scoring old results
```bash
python3 written_auto/rescore.py IN_DIR OUT_DIR --rule either      # = auto_experiment.score_step; --rule old = fork's rule
```
`object_near` / `oracle` are taken verbatim (via `ast`) from `kws_mod_live_words_new_exp.py`. With `--rule old` the
script reproduces the stored `correct` column exactly. Original values are kept as `*_old` columns.

## 5. DAVIS specifics (experiments 3, 4)

* **Recordings**: 30 s DAVIS346 recordings of the rendered `*_346x260.mp4` stimuli shown on a monitor (events `.npy`
  with fields `t` [us], `x`, `y`, `p`, plus the APS video). All 10 files were checked visually against the stimuli and
  match their names.
* **Ground truth (`davis_ground_truth.py`)**: the existing `camera_*.mask.npy` come from earlier live sessions with a
  different camera placement and do **not** line up with these recordings (offsets of half an object or more;
  `camera_6_objects_color_nobg` is a 9-object mask; there is no 4-circles-bg mask). New masks
  `stimuli/ground_truth_masks/davis_<stem>_346x260.{mask.npy,truth.csv,preview.png}` are built per recording:
  average of the APS video -> threshold (Otsu; top-hat for the dim 9-object colour scenes) -> N largest components ->
  ids in reading order (as the synthetic masks and `NAMES`) -> downsampled ×2. Objects whose APS footprint is too small
  (the dark green tree) take the synthetic shape mapped with a homography fitted on the other objects' centroids.
  IoU(APS segmentation, mapped synthetic) is 0.6–0.95. SAM 2, the project's usual tool, is not installed in `venv`.
* **Camera processing** (`VISUAL == 1`), taken from `kws_mod_cam_words_exp.py`: `r0 = 7`; dv_processing
  `BackgroundActivityNoiseFilter` 500 us over the whole stream (cached); the first fixation only on a window with
  ≥ 300 raw events and peak ≥ 2 × saliency mean; before the first command the fovea follows the saliency peak.
  The processing grid is fixed to the sensor (173 × 130).
* The warning "only N% of event pixels fall on the mask's objects" is expected with the camera, because sparse stripe
  noise covers the whole sensor. In the nobg recordings 76–81 % of the *events* fall on the objects; with the
  jittering background it is 25 %.

## 6. Results so far (rule 2 "either", mean ± std over 50 trials per group)

| Experiment | group | compliance | lock rate | compl. possible | compl. impossible | KWS acc |
|---|---|---|---|---|---|---|
| 1 synthetic + written | no clutter | 97.6 ± 5.1 | 97.6 ± 6.8 | 96.5 ± 7.6 | 100.0 ± 0.0 | – |
| | clutter | 41.4 ± 20.2 | 32.0 ± 18.4 | 22.7 ± 21.4 | 99.7 ± 2.4 | – |
| 2 synthetic + GSCD-HW | no clutter | 78.4 ± 12.4 | 95.4 ± 9.2 | 71.5 ± 16.0 | 97.0 ± 14.7 | 72.2 ± 14.3 |
| | clutter | 37.4 ± 18.2 | 35.4 ± 19.5 | 19.0 ± 18.9 | 97.7 ± 9.2 | 72.0 ± 14.0 |
| 3 DAVIS + written | no clutter | 98.2 ± 4.3 | 97.8 ± 7.6 | 97.5 ± 6.2 | 100.0 ± 0.0 | – |
| | clutter | 72.6 ± 15.3 | 72.4 ± 20.1 | 64.2 ± 20.0 | 96.7 ± 10.3 | – |
| 4 DAVIS + GSCD-HW | – | not run yet | | | | |

Per-batch tables: run the runner with `--metrics-only`.

**Equivalence check after moving the logic into the original script.** Trials 1–2 of every batch (20 trials) of
experiments 1 and 3 were re-run with the original script and compared with the earlier results (which came from the
generated copies): 20/20 trials identical in every column for synthetic + written (vs `results_random25_rule2fix/`) and
20/20 for DAVIS + written (vs `results_random25_davis/`). The spoken variants (not deterministic: real-time hardware)
ran end to end for batches 1–2. Manual mode (no `VA_MODE`) was checked: typed command, original scoring and columns.

## 7. Files

| File | Role |
|---|---|
| `../kws_mod_live_words_new_exp.py` | the original experiment script, with commented `CHANGED` hooks for automated runs |
| `../auto_experiment.py` | automated runs: command generator (`list` / `possible` / `random25`), spoken commands through the KWS server, scoring rules 1–2, DAVIS processing |
| `../kws/gscd_hw_server.py` | KWS-on-FPGA server (Opal Kelly env): reset -> arm -> play wav -> read frames -> peak-confidence decision (`KWS_CONF_THR` = 200, `KWS_UNKNOWN_SUPPRESS` = 25, `KWS_CAPTURE_S` = 1.5) |
| `batches.py` | batch -> synthetic `.npy`, synthetic mask, object names |
| `run_written_random25.py`, `run_gscd_hw_random25.py`, `run_davis_written_random25.py`, `run_davis_gscd_hw_random25.py` | runners + metric tables |
| `davis_ground_truth.py` | ground-truth masks + previews for the DAVIS recordings |
| `rescore.py` | offline re-scoring of saved CSVs |
| `replay_camera_baseline.py` | diagnostic: replays the typed sequences of the live-camera baseline (`results/01*.csv`) on the DAVIS recordings |
| `README.md`, `HANDOFF.md` | this file; context for whoever continues the work |

Result folders: `results_random25/` and `results_random25_gscd_hw/` are the raw runs scored with the **old** rule 2
(kept for reference); `*_rule2fix/` are the same runs re-scored with "either". `results_random25_davis/` was produced
with "either" directly. `davis_cache/` holds regenerable filtered event streams, and `_verify/` the outputs of the
equivalence check (neither needs to be committed).

## 8. What changed in pre-existing code

* `kws_mod_live_words_new_exp.py`: hooks only, each marked `CHANGED`. In order:
  1. import of `auto_experiment` and `time`;
  2. the EXPERIMENT block is overridden from the environment in automated runs (`{L}{V}_tTT_bBB` trial ids);
  3. `r0 = 7` when `VISUAL == 1` (camera value);
  4. headless flags (`SHOW`; `RECORD` / `DEBUG` off unless `VA_GUI` / `VA_RECORD`);
  5. `AX.Commander` replaces the stdin thread in automated runs;
  6. camera noise filter, sensor-sized grid and raw event counts when `VISUAL == 1`;
  7. `close_command()` logs the intended and recognised words for spoken runs, and `drain_commands()` handles a `miss`;
  8. `write_results()` applies rules 1–2 and adds the per-step flag columns in automated runs;
  9. camera fixation gate / follow-the-peak when `VISUAL == 1`;
  10. `commander.feed(...)` once per window;
  11. drawing and window only when `SHOW`, real-time sleep for headless spoken runs, KWS server shutdown at the end.

  Untouched: everything a manual session does.
* New: `auto_experiment.py`, `kws/gscd_hw_server.py`, this folder, and `stimuli/ground_truth_masks/davis_*`. The
  `camera_*` masks were not touched.
* Not part of the random25 protocol: the older fixed-sequence scripts in the repo root (`GSCD_HW_loop_<stimulus>.py` +
  `gscd_hw_loop_core.py`, `replay_batch.py` + `auto_exp_gui.py`). They still use generated copies of the script and the
  original scoring.
