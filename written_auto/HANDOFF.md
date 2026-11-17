# Hand-off notes: NAS-KWS on FPGA + command-modulated visual attention

Written for the next agent picking this up. Read `written_auto/README.md` first: it covers the current experiments
(only possible commands, synthetic / monitor / real objects, written / GSCD-HW). This file holds the context, the
history (including the earlier random25 protocol with impossible commands), the traps and the open work.

## The user and the way of working
* Researcher (AGH, EVS group) writing a paper on spoken-command modulation of a spiking visual attention model,
  with a neuromorphic auditory sensor (NAS) and a GNN keyword spotter (KWS) on an Opal Kelly XEM7310 FPGA.
* Writes in **Polish**, informal. Answer in Polish. **Code comments must be in English.**
* Wants experiments run end to end and reported as tables of mean ± std. Wants honest diagnosis: say so when something
  does not reproduce.
* Do not commit or push unless asked.

## Repositories and environments
| Path | What |
|---|---|
| `~/Repo/kws_visual_attention` | attention model + experiments (this repo). Env: `venv/` (torch, sinabs, cv2, dv_processing; no `ok`, no `ultralytics`) |
| `~/Repo/VoiceNAS/NAS-GNN-KWS` | HW (Vivado: NAS + GCNN + GRU head) and `SW/ok/` board scripts. KWS bitstream: `HW/vivado/kws/kws_system.runs/impl_1/ok_top_wrapper.bit` (5 classes). No `rec` bitstream is built in this repo. |
| `~/Repo/Praga/SW/NAS-GNN-KWS/SW` | SW model. Checkpoint used on HW: `less_spikes_voice_freq/CHECKPOINTS/{config.yaml,best_model_calibration.pth}` (trained on Cyfronet; see `test_quantization.py` for paths) |
| `~/Repo/OK-FINAL/NAS-GNN-KWS/SW/ok/ok-venv/bin/python` | Opal Kelly env (`ok`, sounddevice). Its torchvision is broken, so attention runs in the repo `venv` and the board is driven by a subprocess (`kws/gscd_hw_server.py`) |
| `/home/pwz/Music/NAS/gsc_v2` | GSCD v2 wavs. Test list: `/media/pwz/KINGSTON/NAS-KWS/datasets-spike/gsc_v2_voice_spike/testing_list.txt` (spike dataset = `.wav.aedat`) |
| `/media/pwz/KINGSTON/NAS-KWS/Experiments/Event and Grayscale data - DAVIS-.../raw` | DAVIS346 recordings of the stimuli on a monitor (10 scenes, 30 s, `events/*.npy` + `gs_videos/*.mp4`) |
| `/media/pwz/KINGSTON/NAS-KWS/Experiments/Real World Objects-.../Real World Objects` | DAVIS346 recordings of real objects (12 scenes, 30 s, events already noise-filtered at recording), masks in `raw/ground_truth_masks`, recorder in `code/record_camera.py` (no attention code there) |

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
  still `unknown`). KWS accuracy in the loops ≈ 72–74 %. For `results_v2` the suppression was raised to 35
  (env `KWS_UNKNOWN_SUPPRESS`; sweep: recall 0.78, 91 % non-keywords `unknown`; 50 would give 0.82 / 72 %) -> KWS 73–76 %.
  The sweep used `testing_list`, the split the experiments draw from (leakage: disclose or re-tune on `validation_list`). Errors are almost all `unknown`/missed, rarely a wrong direction.
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
2. **random25 protocol** (`written_auto/`, see README): all four experiments done (2026-11-17, `results_v2/`, tables in `results_v2/tables_v2.md`).
   Replication before the change: experiments 1 and 3 re-run with the original script matched the reference CSVs
   100/100 trials (only the stale `accuracy` column of `*_rule2fix` differs).
   * Rule 2 history: the fork's rule (start object never excluded after an unlocked command) gave impossible
     expectations. A "reachability" fix (>= MIN_TRAVEL − READOUT_R) was tried and **rejected**: the CSV ref is the
     attention maximum, not the fovea. Then rule "either" (start object accepted after any unlocked step). It also
     credited KWS misses (the fovea did not move, yet the start object counted), so the user asked for: **rule v2**
     (start object accepted only when the last command that moved the fovea ended without a lock) and **any KWS error
     = not compliant**, for possible and impossible commands (README §3). `VA_RULE=either` reproduces the old CSVs 20/20.
   * Result: with the KWS right, spoken ≈ written (DAVIS 98.1 / 73.2 % vs 98.2 / 72.6 %, no clutter / clutter); the whole
     gap is the ~25 % of KWS errors.
   * With clutter, overall compliance is inflated by impossible commands. Report possible/impossible separately.
   * Old result folders (kept, all random25): `results_random25/`, `results_random25_gscd_hw/` (fork's original rule 2),
     `*_rule2fix/` (same runs re-scored with "either"; stale `accuracy` column, use `correct`), `results_random25_davis/`
     ("either"), `results_v2/` (rule v2, the four experiments; `run_all_v2.sh`, `tables_v2.md`),
     `results_random25_davis_gscd_hw_INVALID_noaudio/` (no audio reached the NAS: do not use, do not commit).
     `rescore.py IN OUT --rule either|old` re-scores saved CSVs.
3. **Camera baseline discrepancy**: the user's live-camera written sessions (`results/01{t}{b}.csv`, own batch numbering,
   `camera_*` masks) reach clutter 90.4 ± 10.8 / lock 91.6, versus DAVIS + written random25 72.6 / 72.4. The main reason is
   that 20 of the 25 baseline clutter sessions used the **static** hatch background (`*_bg`), while all new recordings are
   `*_bg_jitter` (75 % of their events come from the hatches). Ruled out: command protocol (a replay of the user's
   sequences gives 61.6 %), scoring (the camera script's centroid-based `object_near` is stricter: 57.8 %), sensor noise
   and monitor flicker (the only rate peak is the 20 fps stimulus). Unexplained: the one jitter-bg baseline scene
   (6 circles) at 96 %, which came from an older session and possibly another stimulus version. Suggested: record
   static-bg variants with the DAVIS.
4. **Real World Objects** (2026-11-17, `run_rwo_random25.py`, `results_rwo/`): 12 scenes, random25 written:
   all 90.9 %, clutter 85.8 %; the 4-object clutter scenes fail on the small ball (top left), which loses to the cable at
   the image border. The user found the numbers suspicious; checks: a live replay reproduced the CSV row by row, the
   generator draws impossible commands at 25 % on average (one unlucky seed drew 5 in a row), and a slower reaction after
   a lock (`VA_REACT=10,20`, 1–2 s, `results_rwo/written_react1-2s/`) changed nothing (all 91.4 vs 90.9 %).
5. **Only possible commands** (current protocol, README): the user decided impossible commands distort the score.
   They do so in both directions: free points where the system is weak (synthetic clutter 41 % random25 vs 21–26 %),
   and a worse start for the next command where it is good (no clutter: 9 of 12 synthetic failures came right after an
   impossible command). `results_possible5/` = 10 trials × 5 commands (first try), `results_possible_t5c10/` = 5 trials
   × 10 commands (= baseline size, the one to report). `verify_scoring.py` re-implements the scoring independently: all
   stored rows of every folder match, and it catches deliberately broken rules (mutation test).
   Rule 1 does not count a lock on a background distractor as an error (36–63 % of impossible commands on the real
   objects with clutter); a strict false-lock metric would be the meaningful use of impossible commands.

## Traps
* **Dual boot**: after Windows the Ubuntu audio jack may not work, or the wrong audio output is selected. Then the KWS
  hears silence (everything `unknown`, accuracy near 0 %). Before a spoken run, do a ~10-sample sanity check
  and check KWS accuracy in the first `_kws.csv` files.
* The DAVIS recordings are on the KINGSTON drive (`/media/pwz/KINGSTON/...`); it must be mounted (FileNotFoundError otherwise).
* **Heavy parallel runs froze the PC** (8 jobs × 2 threads on DAVIS). One trial ≈ 1 GB. Use `nice 19`, 1 thread, ≤ 2–4 jobs.
  Spoken runs are strictly sequential.
* **Keep results inside the repo**: the session scratchpad in /tmp is wiped by a reboot.
* `pkill -f <pattern>` (and a `pgrep -f` wait loop) also matches your own shell command line. List the processes with `ps ... | grep "[p]attern"` first.
* The `camera_*` masks do not match the DAVIS recordings. Use `davis_*` masks (check the `*.preview.png`).
* `kws/protocol.py` (repo) decodes an old class layout. Do not use it with the VoiceNAS bitstream.
* Automated runs use the ORIGINAL `kws_mod_live_words_new_exp.py` (hooks marked `CHANGED`) + `auto_experiment.py`,
  switched on by `VA_MODE`. Earlier, generated copies (`make_core*.py` -> `*_auto_core.py`) were used. They were removed
  after an equivalence check (40/40 trials identical in every column). The user requires the original scripts,
  the same procedure (random25) and the same improved metrics. Keep it that way.
* A manual run of the original script writes to `results/{trial_id}.csv` and OVERWRITES the user's baseline files
  (e.g. `results/00510.csv`). For tests, use an automated run (`VA_RESULTS`) or a copy with another `RESULTS_DIR`.
* The DAVIS noise-filter cache (`written_auto/davis_cache/`) must exist before parallel DAVIS runs.
* The real-object recordings are already filtered; a second filter pass removes another 50–70 % of the events.
  `run_rwo_random25.py` sets `VA_NOISE_US=0`; keep it.
* With only possible commands, the 2- and 3-object real scenes (one row) allow left/right only and give 100 %: report
  the 4-object scenes separately.
* Two parallel sessions (forks) of the user also touched `written_auto/` (they created `batches.py`, the first `make_core.py` and
  `run_written_random25.py`).

## Open work
1. Finish / report GSCD-HW with only possible commands for the three inputs (`run_possible_spoken.sh 5 10
   written_auto/results_possible_t5c10`), then `tables_v2.py` and `tables_rwo.py` on that folder.
2. Live speech into a microphone for the real objects (not implemented): the speaker needs the command shown on screen,
   and the KWS reset needs a trigger (push-to-talk, or reset when the command is shown).
3. Paper caveats: the GRU reset before each word assumes a known word onset; `KWS_UNKNOWN_SUPPRESS` tuned on the test
   split; std is the population std over trials of mixed scenes; the 9-object monitor masks are weaker (apple, mug
   handle, homography of the tree); "lock rate" is a landing rate; the 2-/3-object real scenes are trivial.
4. Optionally: a strict false-lock metric for impossible commands in clutter scenes; a retry-on-miss variant; monitor
   recordings with a static background; re-tuning the suppression on `validation_list`.

## Bring-up prompt for the next agent
> You are continuing work for a researcher (answer in Polish, code comments in English) on spoken-command modulation of a
> spiking visual-attention model. Speech goes through a neuromorphic auditory sensor and a GNN keyword spotter on an Opal
> Kelly XEM7310 FPGA. First read `~/Repo/kws_visual_attention/written_auto/README.md` and
> `~/Repo/kws_visual_attention/written_auto/HANDOFF.md`, then skim
> `auto_experiment.py`, the `CHANGED` hooks in `kws_mod_live_words_new_exp.py` and `kws/gscd_hw_server.py`.
> The current experiments use only possible commands, 5 trials × 10 commands per scene, LOCK-or-3 s timing, for three
> visual inputs (synthetic events, DAVIS on a monitor: 10 scenes each; DAVIS on real objects: 12 scenes) and two kinds
> of commands (written, GSCD-HW). Scoring: rule 2 "v2" (start object also accepted while the fovea is lost) and rule 3
> (any KWS error = not compliant); impossible commands can be enabled with `--p-impossible` (rule 1).
> Report compliance and lock rate (+ KWS accuracy) as mean ± std over trials, split into no clutter and clutter; real
> objects also per scene.
> Constraints: keep runs light (`nice 19`, 1 thread, ≤ 2–4 jobs; spoken runs sequential, exclusive use of the board and
> the audio jack); keep all outputs inside the repo (not /tmp); run experiments through the original script + `auto_experiment.py` (no generated copies), never overwrite `results/` baseline files; do not commit
> unless asked; the KWS pipeline must be reset before each utterance.
> Runs: `written_auto/run_possible_written.sh` / `run_possible_spoken.sh 5 10 OUT`; tables: `tables_v2.py`, `tables_rwo.py`.

## Asking the original agent
The session that did this work is still available on this machine as a local Claude Code session titled
**"KWS-attention experiments (original agent)"** (id `local_3b7de26d-5d43-4d80-a942-269f0bc28836`). Find it with
`ListAgents` and ask it with `SendMessage` (to = that id or name) when something in this file, the README or the
code is unclear. Its context is very long, so ask specific questions.
