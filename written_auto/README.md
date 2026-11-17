# Automated attention-modulation experiments

All experiments run the **original** experiment script `kws_mod_live_words_new_exp.py`. An automated run is switched
on by the environment variable `VA_MODE`, which the runners in this folder set. The added logic lives in
`auto_experiment.py` (repo root), and every hook in the original script is marked `# CHANGED (random25)` or
`# CHANGED (camera)`. Without `VA_MODE`, the script behaves exactly as before: typed commands, live window, original
scoring, results in `results/` (a manual run there **overwrites** the baseline files of the same trial id).

## 1. Experiments

Three visual inputs × two kinds of commands, **only possible commands** (every command has a target object),
**5 trials × 10 commands per scene**, i.e. the size of the manual baseline sessions in `results/`:

| Visual input | Scenes | Commands | Runner (`--spoken` / spoken runner) | Trial-ID prefix | Results |
|---|---|---|---|---|---|
| Synthetic events (IEBCS, `data/*.npy`) | 10 (5 no clutter, 5 clutter) | written / GSCD-HW | `run_written_random25.py` / `run_gscd_hw_random25.py` | `00_` / `10_` | `results_possible_t5c10/sim_{written,gscd_hw}/` |
| DAVIS346 pointed at a monitor showing the same stimuli | 10 (5 + 5) | written / GSCD-HW | `run_davis_written_random25.py` / `run_davis_gscd_hw_random25.py` | `01_` / `11_` | `results_possible_t5c10/davis_{written,gscd_hw}/` |
| DAVIS346 pointed at real objects | 12 (6 + 6) | written / GSCD-HW | `run_rwo_random25.py` (`--spoken`) | `01_` / `11_` | `results_possible_t5c10/rwo_{written,gscd_hw}/` |

Per visual input: 50 trials (60 for real objects), half without clutter, 10 commands each. GSCD-HW = utterances of the
keyword from the GSCD v2 test split, played into the NAS and classified by the GNN keyword spotter on the FPGA.
Live speech into a microphone (planned for the real objects) is **not implemented yet**.

Files are named `{prefix}_tTT_bBB.csv` (trial TT, scene/batch BB), one row per command. Spoken runs also write
`..._kws.csv`: intended word, recognised word, KWS confidence, wav file and latency.

### Scenes

Synthetic and monitor (`batches.py`; "bg" = the jittering hatch background `*_bg_jitter`):

| batch | scene | batch | scene |
|---|---|---|---|
| 1 | 4 circles / nobg / mono | 2 | 4 circles / bg / mono |
| 3 | 6 circles / nobg / mono | 4 | 6 circles / bg / mono |
| 5 | 6 objects / nobg / mono | 6 | 6 objects / bg / mono |
| 7 | 6 objects / nobg / color | 8 | 6 objects / bg / color |
| 9 | 9 objects / nobg / color | 10 | 9 objects / bg / color |

Real objects (`run_rwo_random25.SCENES`; ball, cube, cylinder, cup on a table; 2 and 3 objects in one row, 4 objects in a
2 × 2 grid; camera moved by hand: small jitter or a circular motion; clutter = dark cloth, a cable and folds, mostly at the
image border, camera farther away so the objects are smaller):

| scene | objects | camera | clutter | scene | objects | camera | clutter |
|---|---|---|---|---|---|---|---|
| 1 | 2 | jitter | no | 2 | 2 | jitter | yes |
| 3 | 2 | circular | no | 4 | 2 | circular | yes |
| 5 | 3 | jitter | no | 6 | 3 | jitter | yes |
| 7 | 3 | circular | no | 8 | 3 | circular | yes |
| 9 | 4 | jitter | no | 10 | 4 | jitter | yes |
| 11 | 4 | circular | no | 12 | 4 | circular | yes |

With only possible commands, the 2- and 3-object scenes allow horizontal commands only (one row), so they are easy by
construction; the 4-object scenes are the informative ones.

## 2. Protocol

* **Command generation**: before every command the possible directions are computed from the *current* attention
  maximum with the same oracle that is used for scoring, and one of them is drawn at random. A direction is possible if
  an object lies that way with the commanded axis dominating (`|dy| > |dx|` for up/down, `|dx| > |dy|` for left/right).
  Draws are seeded per scene and trial (`cmd-{batch}-{trial}`), so runs are reproducible. The seeds are the same for every
  visual input and both kinds of commands.
* **Timing**: the first command comes 1–3 s (10–30 windows, depending on the trial) after the first fixation. The next
  command comes 0.3–0.8 s after the controller LOCK, or after 3 s without a lock. Windows are 100 ms of events. A slower
  reaction (`VA_REACT=10,20`, 1–2 s) gave the same results on the real objects (`results_rwo/written_react1-2s/`).
* **Spoken commands**: the drawn command is the **intended** one. A random GSCD v2 *test-split* utterance of that word
  (seed `gscd-{batch}-{trial}`) is played through the audio jack into the NAS-KWS bitstream. The **recognised** word is
  applied to the attention like a typed one, even when it is the wrong direction; `unknown` / no detection does not move
  the fovea. The KWS pipeline is **reset before every utterance** (wire 0x00), i.e. the word onset is known (must be stated
  in the paper). `KWS_UNKNOWN_SUPPRESS = 35` (tuned on the test split: disclose, or re-tune on `validation_list`).
  Headless spoken runs are paced in real time, so the ~1.5 s KWS latency costs the same number of windows as live.

## 3. Metrics and scoring

* **Command compliance**: % of correct commands in a trial, mean ± population std over the trials of a group.
* **Lock rate**: % of commands that end with attention on *any* object (`landed_id != 0`, within the snap tolerance).
  It is a landing rate, not the controller's LOCK flag.
* **KWS accuracy** (spoken): % of commands whose recognised word equals the intended one.

Scoring (`auto_experiment.score_step`, called from `write_results()`): a command is correct if attention ends on the
nearest object in the commanded direction (`object_near()`: footprint distance + snap = 2 × median object radius,
clamped to 0.45 × the edge-to-edge gap; `oracle()`: nearest centroid with the commanded axis dominating), and:

* **Rule 2 (v2)**: when the fovea is "lost" (the last command that moved it ended without a lock), the start object
  (if it lies that way) is also accepted, because the logged reference is the attention maximum, not the fovea.
* **Rule 3 (spoken)**: a command the KWS did not recognise correctly (`unknown`, none or a wrong direction) is
  **not compliant**.

`written_auto/verify_scoring.py DIR ...` re-implements these rules independently and checks every stored row.

### Allowing impossible commands

Impossible commands (no object in the commanded direction) can be mixed in with any runner:

```bash
venv/bin/python written_auto/run_rwo_random25.py --trials 5 --p-impossible 0.25 --out written_auto/results_xyz
# or, for a single trial / hand-written runs: VA_P_IMPOSSIBLE=0.25
```

With probability `p` an impossible direction is drawn (when one exists), otherwise a possible one. An impossible command
is scored by **rule 1**: compliant unless attention jumps to *another object* (staying on the start object or ending
on the background is fine). The tables then also report compliance for possible and impossible commands separately.
Two caveats, measured with `p = 0.25` (folders `results_v2/`, `results_rwo/written/`):

* impossible commands are almost always compliant (97–100 % written), so they add easy points where the system is weak
  (synthetic clutter: 41 % overall vs 21–26 % with possible commands only);
* they leave the fovea at the image border or beside an object, which makes the *next* command harder (no clutter:
  most failures came right after an impossible command), so possible-only runs are better there.

Rule 1 does not count a lock on a background distractor as an error; with clutter on the real objects 36–63 % of the
impossible commands end that way. A strict variant (a lock on anything but the start object = error) would measure the
false-lock rate; it is not implemented.

`VA_RULE=either` / `VA_KWS_WRONG=off` restore the earlier rules (see `HANDOFF.md`).

## 4. Prerequisites

| What | Where |
|---|---|
| Attention env (torch, sinabs, cv2, dv_processing) | `venv/` of this repo |
| Opal Kelly env (`ok`, sounddevice), used only by the KWS server | `/home/pwz/Repo/OK-FINAL/NAS-GNN-KWS/SW/ok/ok-venv/bin/python` (override: `VA_OK_PYTHON`) |
| KWS bitstream (5 classes: up, down, left, right, unknown) | `/home/pwz/Repo/VoiceNAS/NAS-GNN-KWS/HW/vivado/kws/kws_system.runs/impl_1/ok_top_wrapper.bit` (override: `KWS_BITFILE`) |
| GSCD v2 audio / test list | `/home/pwz/Music/NAS/gsc_v2`, `/media/pwz/KINGSTON/NAS-KWS/datasets-spike/gsc_v2_voice_spike/testing_list.txt` (`VA_GSCD_AUDIO`, `VA_GSCD_TEST`) |
| DAVIS recordings, monitor | `/media/pwz/KINGSTON/NAS-KWS/Experiments/Event and Grayscale data - DAVIS-.../raw/events` (`DAVIS` in the DAVIS runners) |
| DAVIS recordings, real objects | `/media/pwz/KINGSTON/NAS-KWS/Experiments/Real World Objects-.../raw/events` (`RAW` in `run_rwo_random25.py`) |
| Hardware (spoken runs) | XEM7310 on USB + PC audio jack -> NAS I2S input. Nothing else may use the board or the jack during a run. |

The recording paths are hard-coded in the runners: edit `DAVIS` / `RAW` on another machine. `kws/protocol.py` decodes
an **older** class layout and does not match the VoiceNAS bitstream; the spoken runs use `kws/gscd_hw_server.py`.

**Load**: one trial process ≈ 1 GB RAM; 8 parallel jobs froze the PC once. Runners use `nice 19`, 1 thread per process,
4 jobs (synthetic) or 2 (DAVIS) for written commands, and 1 job for spoken commands (one board).

## 5. How to run

From the repo root (`~/Repo/kws_visual_attention`):

```bash
written_auto/run_possible_written.sh 5 10 written_auto/results_possible_t5c10   # written, all three inputs (~10 min)
written_auto/run_possible_spoken.sh  5 10 written_auto/results_possible_t5c10   # GSCD-HW, all three (~2.5 h, board + jack)
python3 written_auto/tables_v2.py  written_auto/results_possible_t5c10          # synthetic + monitor tables
python3 written_auto/tables_rwo.py written_auto/results_possible_t5c10/rwo_written   # real objects, per scene
```

The scripts take `TRIALS NCMDS OUT_DIR` and export `VA_RULE=v2 VA_KWS_WRONG=all` (and `KWS_UNKNOWN_SUPPRESS=35`).
Before a spoken run, check the audio: feed ~10 wavs to the server and expect ~70–80 % correct; all `unknown` means the
jack is not working (e.g. after a reboot from Windows):

```bash
KWS_UNKNOWN_SUPPRESS=35 /home/pwz/Repo/OK-FINAL/NAS-GNN-KWS/SW/ok/ok-venv/bin/python kws/gscd_hw_server.py < requests.jsonl
# requests.jsonl: one {"wav": "/home/pwz/Music/NAS/gsc_v2/up/<file>.wav"} per line
```

Single runners: `--trials N`, `--p-impossible P`, `--batches 1,2` / `1-10`, `--out DIR`, `--metrics-only`; the number of
commands per trial is `VA_NCMDS` (default 10).

**Watch one trial live** (same settings as the experiment, reproduces the stored CSV; output in `written_auto/_check/`):

```bash
venv/bin/python written_auto/show_trial.py real 12 1 --compare written_auto/results_possible_t5c10/rwo_written
# SET = sim | monitor | real, then scene and trial; --p-impossible / --ncmds as in the runners
```

Other variables of `auto_experiment.py`: `VA_INIT` (windows before the first command), `VA_TIMEOUT` (30),
`VA_REACT` (`3,8` windows after a lock), `VA_NOISE_US` (camera noise filter, 500; 0 = off), `VA_GUI`, `VA_RECORD`.

The first monitor run builds the noise-filtered event caches in `written_auto/davis_cache/` (175 MB, not in git).
Parallel jobs must not build the same cache: run `--trials 1 --jobs 1` first if it is missing.

## 6. Camera specifics

* **Monitor recordings**: 30 s DAVIS346 recordings of the rendered stimuli shown on a monitor (events `.npy` with fields
  `t` [us], `x`, `y`, `p`, plus the APS video). Ground truth `stimuli/ground_truth_masks/davis_*` from the APS video
  (`davis_ground_truth.py`: Otsu / top-hat threshold, ids in reading order, a homography for the dim tree). The
  `camera_*` masks of the older live sessions do not line up with these recordings.
* **Real-object recordings**: 30 s, recorded with `record_camera.py` (on the drive, `code/`), which already applies the
  dv background-activity filter (500 us). The runners therefore set `VA_NOISE_US=0` (no second filter pass).
  Ground truth: the masks shipped with the recordings, copied to `stimuli/ground_truth_masks/rwo_*`. The camera motion
  shifts the objects by only 1–3 px on the 173 × 130 grid, so a static mask is enough.
* **Processing** (`VISUAL == 1`, from `kws_mod_cam_words_exp.py`): `r0 = 7`; the first fixation only on a window with
  ≥ 300 raw events and peak ≥ 2 × saliency mean; before the first command the fovea follows the saliency peak; grid
  fixed to the sensor (173 × 130). Attention and controller parameters are the same for all inputs.

## 7. Results (only possible commands, 5 trials × 10 commands per scene, mean ± std over trials)

Written commands:

| Input | group | trials | compliance | lock rate |
|---|---|---|---|---|
| Synthetic | no clutter | 25 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| | clutter | 25 | 21.2 ± 16.8 | 40.4 ± 17.5 |
| Monitor | no clutter | 25 | 98.4 ± 3.7 | 98.8 ± 3.2 |
| | clutter | 25 | 68.8 ± 20.1 | 81.6 ± 14.9 |
| Real objects | no clutter | 30 | 99.0 ± 3.0 | 100.0 ± 0.0 |
| | clutter | 30 | 90.7 ± 18.8 | 93.3 ± 13.5 |
| | 4 objects, no clutter | 10 | 97.0 ± 4.6 | 100.0 ± 0.0 |
| | 4 objects, clutter | 10 | 72.0 ± 23.2 | 80.0 ± 16.7 |

Real objects per scene: 100 % in all 2- and 3-object scenes; scene 9: 94.0 ± 4.9, 10: 68.0 ± 27.1, 11: 100.0 ± 0.0,
12: 76.0 ± 17.4. Full tables: `results_possible_t5c10/tables_*.md`.

GSCD-HW: see `results_possible_t5c10/` (tables as above; spoken tables also give KWS accuracy and compliance split by
the KWS outcome: heard correctly / not recognised / wrong direction).

Why clutter hurts much more on the monitor than on the real objects: the hatch texture lies right next to the objects
(object event density only 2.4–4 × the background within ~12 px), while the real clutter is at the image border
(objects ~300 × denser than their surroundings). On the real objects clutter matters when the fovea pans towards the
border past a small object (the ball in scenes 10 and 12).

## 8. Files

| File | Role |
|---|---|
| `../kws_mod_live_words_new_exp.py` | the original experiment script, with commented `CHANGED` hooks for automated runs |
| `../auto_experiment.py` | automated runs: command generator, spoken commands through the KWS server, scoring rules, camera processing |
| `../kws/gscd_hw_server.py` | KWS-on-FPGA server (Opal Kelly env): reset -> arm -> play wav -> read frames -> peak-confidence decision (`KWS_CONF_THR` 200, `KWS_UNKNOWN_SUPPRESS` 25 by default, 35 in the runs, `KWS_CAPTURE_S` 1.5) |
| `batches.py` | synthetic / monitor batch -> `.npy`, mask, object names |
| `run_written_random25.py`, `run_gscd_hw_random25.py`, `run_davis_written_random25.py`, `run_davis_gscd_hw_random25.py`, `run_rwo_random25.py` | runners |
| `run_possible_written.sh`, `run_possible_spoken.sh` | all three inputs, only possible commands |
| `tables_v2.py`, `tables_rwo.py` | result tables (synthetic + monitor; real objects per scene / factor) |
| `show_trial.py` | one trial with the live window |
| `verify_scoring.py` | independent re-implementation of the scoring, checks every stored row |
| `davis_ground_truth.py` | ground-truth masks for the monitor recordings |
| `HANDOFF.md` | history, earlier protocols (random25 with impossible commands) and their result folders, traps |

## 9. What changed in pre-existing code

* `kws_mod_live_words_new_exp.py`: hooks only, each marked `CHANGED`: import of `auto_experiment`; the EXPERIMENT block
  overridden from the environment in automated runs; `r0 = 7` for the camera; headless flags; `AX.Commander` instead of
  the stdin thread; camera noise filter, sensor grid and raw event counts; logging of intended / recognised words;
  scoring in `write_results()` (`AX.score_step`) with per-step flag columns; camera fixation gate; `commander.feed(...)`
  once per window; drawing only when `SHOW`, real-time pacing of spoken runs, KWS server shutdown. Everything a manual
  session does is untouched.
* New: `auto_experiment.py`, `kws/gscd_hw_server.py`, this folder, `stimuli/ground_truth_masks/davis_*` and `rwo_*`.
