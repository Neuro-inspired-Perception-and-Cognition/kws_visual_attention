# Hand-off notes: NAS-KWS on FPGA + command-modulated visual attention

Written for the next agent picking this up. Read `written_auto/README.md` first: it covers how to run the four
random25 experiments. This file holds the context, the findings, the traps and the open work.

## The user and the way of working
* Researcher (AGH, EVS group) writing a paper on spoken-command modulation of a spiking visual attention model,
  with a neuromorphic auditory sensor (NAS) and a GNN keyword spotter (KWS) on an Opal Kelly XEM7310 FPGA.
* Writes in **Polish**, informal. Answer in Polish. **Code comments must be in English.**
* Wants experiments run end to end and reported as tables of mean ± std. Wants honest diagnosis: say so when something
  does not reproduce.
* Nothing has been committed. Do not commit or push unless asked.

## Repositories and environments
| Path | What |
|---|---|
| `~/Repo/kws_visual_attention` | attention model + experiments (this repo). Env: `venv/` (torch, sinabs, cv2, dv_processing; no `ok`, no `ultralytics`) |
| `~/Repo/VoiceNAS/NAS-GNN-KWS` | HW (Vivado: NAS + GCNN + GRU head) and `SW/ok/` board scripts. KWS bitstream: `HW/vivado/kws/kws_system.runs/impl_1/ok_top_wrapper.bit` (5 classes). No `rec` bitstream is built in this repo. |
| `~/Repo/Praga/SW/NAS-GNN-KWS/SW` | SW model. Checkpoint used on HW: `less_spikes_voice_freq/CHECKPOINTS/{config.yaml,best_model_calibration.pth}` (trained on Cyfronet; see `test_quantization.py` for paths) |
| `~/Repo/OK-FINAL/NAS-GNN-KWS/SW/ok/ok-venv/bin/python` | Opal Kelly env (`ok`, sounddevice). Its torchvision is broken, so attention runs in the repo `venv` and the board is driven by a subprocess (`kws/gscd_hw_server.py`) |
| `/home/pwz/Music/NAS/gsc_v2` | GSCD v2 wavs. Test list: `/media/pwz/KINGSTON/NAS-KWS/datasets-spike/gsc_v2_voice_spike/testing_list.txt` (spike dataset = `.wav.aedat`) |
| `/media/pwz/KINGSTON/NAS-KWS/Experiments/Event and Grayscale data - DAVIS-.../raw` | DAVIS346 recordings of the stimuli on a monitor (10 scenes, 30 s, `events/*.npy` + `gs_videos/*.mp4`) |

Hardware: the board on USB, the PC audio jack into the NAS I2S input. A run plays audio, so nothing else may use the board
or the jack meanwhile.

## What was found about the KWS (VoiceNAS)
* **The HW matches the SW model 1:1**: weights, quantisation constants, LUTs and class order were checked, and per-frame
  timelines agree. Quantised SW model on the spike test set: 60 % (float 59.5 %). Strong `unknown` bias: 85.7 % of the
  5-class training set is `unknown`, with no class weighting.
* **FIFO**: `wire 0x20` counts 32-bit words; 1 frame = 4 words (128-bit `{64'0, valid_cnt, cls[4..0], conf}`).
  Reading `cnt//4` frames is correct; over-reading returns garbage.
* **The GRU state drifts without a reset.** The same `right` utterance scores right = 209 at 0 s after reset and
  116 / 101 / 108 at 3 / 6 / 9 s (then `down` becomes the top keyword, which is the user's "right -> down"). In silence the
  confidence climbs to 190–207, so thresholds around 190 give false triggers. Hence: **reset the pipeline (wire 0x00)
  right before each utterance**. Re-loading the bitstream is not needed. The paper must mention this reset (the user agreed).
* Decision rule used everywhere now: class argmax in the frame with the highest confidence, accepted if conf > 200/255,
  with 25 subtracted from the raw `unknown` score (tuned on 200 HW samples: keyword recall 0.57 -> 0.73, non-keywords 95 %
  still `unknown`). KWS accuracy in the loops ≈ 72–74 %. Errors are almost all `unknown`/missed, rarely a wrong direction.
* VoiceNAS `SW/ok` changes (uncommitted): new `play_audio_file.py` (plays a random test sample of a class) and
  `kws_hw_eval.py` (per-file HW eval, dumps all frames); `UNKNOWN_SUPPRESS = 25` added to `kws_continuous_reset.py`,
  `kws_continuous.py`, `kws_sys_infer.py` and `kws_plot.py`.
* Training advice already given (not implemented): train on long concatenated streams with real NAS silence between
  words, random initial GRU states or TBPTT, cls loss on all steps, class weights, fixed `pos_weight`, and calibrate the
  hidden-state observers on long sequences.

## Attention experiments: history and numbers
1. **Fixed command lists** (older, repo root): `GSCD_HW_loop_<stimulus>.py` (+ `gscd_hw_loop_core.py`) and
   `replay_batch.py` (+ `auto_exp_gui.py`). They replay the user's typed sequences (`results/00{t}{b}.csv`) or the PDF
   lists with the original scoring.
   * Synthetic + written baseline (user's files): no clutter 99.6 ± 2.0, clutter 19.6 ± 17.5 compliance.
   * Automated replay of the user's sequences: 100 ± 0 / 17.6 ± 15.6. PDF lists: 73.2 / 21.2 (they contain impossible moves).
   * Synthetic + GSCD-HW on the user's sequences: no clutter 59.2 ± 21.3, clutter 16.0 ± 12.6, KWS 73.8 ± 12.9.
   * One KWS miss derails a fixed sequence (the later commands no longer fit the trajectory), so fixed lists over-penalise
     the KWS. This motivated random25.
   * **The CSVs of these older runs were in /tmp and were lost in a reboot**; only the numbers above survive.
2. **random25 protocol** (`written_auto/`, see README): experiments 1–3 done, 4 implemented and smoke-tested
   (batches 1–2), but **not run** in full.
   * Rule 2 history: the fork's rule (start object never excluded after an unlocked command) gave impossible
     expectations. A "reachability" fix (>= MIN_TRAVEL − READOUT_R) was tried and **rejected**: the CSV ref is the
     attention maximum, not the fovea. Final rule: "either" (README §2).
   * With clutter, overall compliance is inflated by impossible commands. Report possible/impossible separately.
3. **Camera baseline discrepancy**: the user's live-camera written sessions (`results/01{t}{b}.csv`, own batch numbering,
   `camera_*` masks) reach clutter 90.4 ± 10.8 / lock 91.6, versus DAVIS + written random25 72.6 / 72.4. The main reason is
   that 20 of the 25 baseline clutter sessions used the **static** hatch background (`*_bg`), while all new recordings are
   `*_bg_jitter` (75 % of their events come from the hatches). Ruled out: command protocol (a replay of the user's
   sequences gives 61.6 %), scoring (the camera script's centroid-based `object_near` is stricter: 57.8 %), sensor noise
   and monitor flicker (the only rate peak is the 20 fps stimulus). Unexplained: the one jitter-bg baseline scene
   (6 circles) at 96 %, which came from an older session and possibly another stimulus version. Suggested: record
   static-bg variants with the DAVIS.

## Traps
* **Heavy parallel runs froze the PC** (8 jobs × 2 threads on DAVIS). One trial ≈ 1 GB. Use `nice 19`, 1 thread, ≤ 2–4 jobs.
  Spoken runs are strictly sequential.
* **Keep results inside the repo**: the session scratchpad in /tmp is wiped by a reboot.
* `pkill -f <pattern>` also matches your own shell command line. List the processes with `ps ... | grep "[p]attern"` first.
* The `camera_*` masks do not match the DAVIS recordings. Use `davis_*` masks (check the `*.preview.png`).
* `kws/protocol.py` (repo) decodes an old class layout. Do not use it with the VoiceNAS bitstream.
* Automated runs use the ORIGINAL `kws_mod_live_words_new_exp.py` (hooks marked `CHANGED`) + `auto_experiment.py`,
  switched on by `VA_MODE`. Earlier, generated copies (`make_core*.py` -> `*_auto_core.py`) were used. They were removed
  after an equivalence check (40/40 trials identical in every column). The user requires the original scripts,
  the same procedure (random25) and the same improved metrics. Keep it that way.
* A manual run of the original script writes to `results/{trial_id}.csv` and OVERWRITES the user's baseline files
  (e.g. `results/00510.csv`). For tests, use an automated run (`VA_RESULTS`) or a copy with another `RESULTS_DIR`.
* The DAVIS noise-filter cache (`written_auto/davis_cache/`) must exist before parallel DAVIS runs.
* Two parallel sessions (forks) of the user also touched `written_auto/` (they created `batches.py`, the first `make_core.py` and
  `run_written_random25.py`).

## Open work
1. Run experiment 4 (DAVIS + GSCD-HW, `run_davis_gscd_hw_random25.py`, ~70 min, board + jack) and report it next to
   experiment 3, with KWS accuracy and a split by KWS-correct / missed / wrong.
2. Optionally: a retry-on-miss variant (repeat an unrecognised command with another utterance, up to 2×), DAVIS
   recordings with a static background, SAM 2 masks (needs `ultralytics` in `venv`, so ask first), and re-creating the
   lost fixed-sequence CSVs.

## Bring-up prompt for the next agent
> You are continuing work for a researcher (answer in Polish, code comments in English) on spoken-command modulation of a
> spiking visual-attention model. Speech goes through a neuromorphic auditory sensor and a GNN keyword spotter on an Opal
> Kelly XEM7310 FPGA. First read `~/Repo/kws_visual_attention/written_auto/README.md` and
> `~/Repo/kws_visual_attention/written_auto/HANDOFF.md`, then skim
> `auto_experiment.py`, the `CHANGED` hooks in `kws_mod_live_words_new_exp.py` and `kws/gscd_hw_server.py`.
> Four automated experiments share the random25 protocol (10 Set-1 batches × 10 trials × 10 commands, 25 % impossible
> commands, LOCK-or-3 s timing) and the scoring with rule 1 (impossible = no jump to another object) and rule 2 "either".
> They are: synthetic + written, synthetic + GSCD-HW, DAVIS + written (done) and DAVIS + GSCD-HW (implemented, not run).
> Report compliance, lock rate, compliance for possible / impossible commands and KWS accuracy as mean ± std over trials,
> split into no clutter and clutter.
> Constraints: keep runs light (`nice 19`, 1 thread, ≤ 2–4 jobs; spoken runs sequential, exclusive use of the board and
> the audio jack); keep all outputs inside the repo (not /tmp); run experiments through the original script + `auto_experiment.py` (no generated copies), never overwrite `results/` baseline files; do not commit
> unless asked; the KWS pipeline must be reset before each utterance.
> Next task: run DAVIS + GSCD-HW (`venv/bin/python written_auto/run_davis_gscd_hw_random25.py --trials 10`)
> and compare it with DAVIS + written.
