# Toeplitz/FFT Privacy Amplification — Template Attack (CW312)

A profiled (template) side-channel attack that recovers the secret **key**
input to a seeded Toeplitz/FFT privacy-amplification hash running on a
ChipWhisperer **CW312 / SAM4S** target.

The attack targets **stage 1 of the key FFT**. That stage is a sequence of
radix-2 butterflies, each mixing one pair of input bits. Every butterfly's
power signature depends only on its two input bits (4 possibilities:
`00 01 10 11`), so one template per class is enough to read the key back
two bits at a time.

---

## Repository layout

```
scripts/
  toeplitz_fft_controller.py   host <-> firmware comms + a numpy reference; standalone CLI to verify the firmware
  tools.py                     capture / segmentation / template / attack-scoring library (class Tools)
  template_attack.py           the attack driver — CLI with `profile` and `attack` subcommands
src/privacy_amplification/FFT/
  Toeplitz_FFT.c               the firmware (one source, two build modes)
  makefile                     builds Toeplitz_FFT-ROWLEN<n>-{FULL,KEYONLY}-CW312_SAM4S.hex
  gen_twiddle_table.py         regenerates twiddle_table.h for a given ROWLEN (run by the makefile)
results/                       captured traces, templates, plots (git-ignored)
```

## Requirements

- A ChipWhisperer capture board + CW312/SAM4S target, with the
  `chipwhisperer` Python package installed and importable.
- Python 3.10+ (the code uses `match`), `numpy`, `matplotlib`.
- An ARM toolchain (`arm-none-eabi-gcc`) to (re)build the firmware.

---

## 1. Build & flash the firmware

The C source builds into **two** executables, selected by `KEY_FFT_ONLY`:

| Build | `make` flags | Protocol | Used for |
|-------|--------------|----------|----------|
| **FULL**    | `KEY_FFT_ONLY=0` (default) | key → echo → seed → echo → result | correctness verification |
| **KEYONLY** | `KEY_FFT_ONLY=1` | key → echo (triggered key FFT only) | trace gathering / the attack |

`ROWLEN` is the key size **in bytes** and must match on both sides. The
makefile defaults to `ROWLEN=128`; the Python side defaults to
`ROWLEN=16` — pass `--rowlen` to the scripts, or `ROWLEN=` to `make`, so
they agree.

```bash
cd src/privacy_amplification/FFT

# Attack / trace-gathering firmware (KEYONLY), 16-byte key:
make clean && make ROWLEN=16 KEY_FFT_ONLY=1

# Correctness firmware (FULL), same key size:
make clean && make ROWLEN=16 KEY_FFT_ONLY=0
```

> `make clean` between builds is required — the object dir is keyed only on
> `PLATFORM`, so a stale rebuild would silently keep objects from the
> previous `ROWLEN` / `KEY_FFT_ONLY`.

The Python code flashes the matching `.hex` for you (see
`toeplitz_fft_controller.connect()`), so you normally only need to *build*
it, not flash it by hand.

## 2. (Optional) Verify the firmware is correct

Flash the **FULL** build, then check the device against the numpy
reference implementation:

```bash
python scripts/toeplitz_fft_controller.py --rowlen 16 --trials 100
```

Exit status is 0 iff every trial matched with no unresolved UART desyncs.
`--raw SECONDS` (optionally with `--key-only` for the KEYONLY build) dumps
the raw bytes the device sends instead of verifying — useful when
debugging comms.

---

## 3. Profile — build the templates

Flash the **KEYONLY** build first. Profiling has two modes:

```bash
# (a) Capture fresh raw traces from the ChipWhisperer, then build templates:
python scripts/template_attack.py profile --online

# (b) Rebuild templates from a previously captured raw-trace .npz — no hardware:
python scripts/template_attack.py profile
```

Options: `--repeats N` (traces averaged per class, default 3). `--rowlen`
and `--results-dir` are global — put them *before* the subcommand
(`template_attack.py --rowlen 16 profile`).

Writes to `results/` (for `--rowlen 16`):

| File | Written by | Contents |
|------|-----------|----------|
| `raw_traces_rowlen16.npz` / `.png` | `--online` only | every pre-segmentation capture (warmup + 4 classes × repeats) |
| `templates_rowlen16.npz` / `.png` | always | the 4 per-butterfly mean templates `(4, n_seg, seg_len)` + metadata |
| `aligment_trace.npy` | always | the alignment reference window every later capture is segmented against |

Splitting capture (`--online`) from template-building lets you iterate on
segmentation / averaging offline against a fixed raw-trace set.

## 4. Attack — recover keys

Flash the **KEYONLY** build. Loads the saved templates + alignment
reference (no re-profiling), then for each random key: captures a target
trace, segments it, and scores every butterfly against the 4 templates by
**SAD** (lowest wins → that butterfly's two key bits).

```bash
python scripts/template_attack.py attack --tests 20
```

Prints `Correct key bits: c/N` per test; **exit status is 0 iff every
test recovered the full key**.

Both functions are importable for notebook / REPL use:

```python
import template_attack as ta
ta.profile(rowlen=16, repeats=5)          # -> templates .npz path
ok = ta.attack(rowlen=16, tests=10)       # -> bool
```

---

## Tuning notes

- **Capture window** — `Tools.configure_scope(samples=, offset=, gain=,
  adc_mul=)` must be wide enough to hold *all* `ROWLEN*4` stage-1
  butterflies under one trigger. If a capture is clipped,
  `sliding_window_segmenter()` finds fewer than `N//2 - 1` segments and
  asserts. Compare the `Trig count:` print against `samples`.
- **Segmentation landmarks** — `allign_p0` / `allign_p1` (module constants
  in `tools.py`) and `sad_thresh` / `window_below` / `window_above` inside
  `sliding_window_segmenter()` are measured for this hardware + `ROWLEN`.
  Re-measure against a real trace if you change the setup.
- **Clock hiccups** — changing `adc_mul` can unlock the ADC PLL
  (`CLKGEN Failed to load divider value` spam). `configure_scope()` calls
  `scope.clock.reset_adc()`; `Tools.reset_target()` recovers the target
  side.
- **RAM ceiling** — a larger `ROWLEN` needs more target RAM. See the
  header docstring of `toeplitz_fft_controller.py` for the tested limits
  (KEYONLY tolerates roughly double the FULL build).
```
