"""Driver for the profiled SAD template attack against the KEY_FFT_ONLY
Toeplitz/FFT firmware on a CW312 target (algorithms live in tools.py).

Two phases, exposed as subcommands:

  profile  -- build the 4 per-butterfly templates and save them (plus the
              alignment reference) under --results-dir.
                --online   : capture fresh raw traces from the ChipWhisperer
                             first (save_raw_traces).
                (default)  : rebuild the templates from an existing
                             raw-trace .npz -- no hardware needed.
              Writes templates_rowlen<ROWLEN>.npz and aligment_trace.npy.

  attack   -- load the saved templates + alignment reference, capture a
              target trace for each random key, and score every stage-1
              butterfly against the 4 templates by SAD to recover the key
              two bits at a time (run_profiled_attack). Exit status is 0
              iff every test recovered the full key.

Examples:
  python template_attack.py profile                  # offline, from saved raw traces
  python template_attack.py profile --online         # capture raw traces first
  python template_attack.py attack --tests 20
"""
import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import toeplitz_fft_controller as tfc
import tools

# Key/seed size in BYTES -- must match the KEY_FFT_ONLY=1 firmware's
# build-time ROWLEN (see toeplitz_fft_controller.py / Toeplitz_FFT.c).
DEFAULT_ROWLEN = 16
DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _paths(results_dir, rowlen):
    """(raw-trace, templates, alignment-reference) file paths for a build."""
    return (
        os.path.join(results_dir, "raw_traces_rowlen{0}.npz".format(rowlen)),
        os.path.join(results_dir, "templates_rowlen{0}.npz".format(rowlen)),
        os.path.join(results_dir, "aligment_trace.npy"),
    )


def profile(rowlen=DEFAULT_ROWLEN, repeats=3, online=False,
            results_dir=DEFAULT_RESULTS_DIR):
    """Build the 4 per-butterfly mean templates (one per input bit-pair
    00/01/10/11) and save them, with the alignment reference, under
    `results_dir`.

    online=True captures fresh raw traces from the ChipWhisperer first;
    online=False rebuilds the templates from an existing raw-trace .npz
    with no hardware attached. Returns the templates .npz path.
    """
    N = rowlen * 8
    raw_path, templates_path, align_path = _paths(results_dir, rowlen)

    if online:
        scope, target = tfc.connect(rowlen=rowlen, key_fft_only=True)
        t = tools.Tools(scope, target, rowlen=rowlen)
        t.configure_scope()
        t.save_raw_traces(N, repeats, path=raw_path)
    else:
        t = tools.Tools(None, None, rowlen=rowlen)

    # Build the templates from the raw .npz -- runs offline; only
    # save_raw_traces() above needs the ChipWhisperer.
    d = t.load_raw_traces(raw_path)
    _, aligment_trace = t.get_trace_from_raw(d["warmup"], int(d["N"]))
    np.save(align_path, aligment_trace)

    templates = []
    for i in range(4):
        traces = [t.get_trace_from_raw(d["template_raw"][i][r], int(d["N"]), aligment_trace)[0]
                  for r in range(repeats)]
        templates.append(np.mean(traces, axis=0))

    t.save_templates(N, templates, path=templates_path,
                     inputs=d["template_inputs"], alignment_trace=aligment_trace,
                     repeats=repeats)
    return templates_path


def attack(rowlen=DEFAULT_ROWLEN, tests=10, results_dir=DEFAULT_RESULTS_DIR):
    """Load the saved templates + alignment reference and recover `tests`
    random keys from fresh target captures, printing the correct-bit count
    per test. Returns True iff every test recovered the full key.
    """
    N = rowlen * 8
    _, templates_path, align_path = _paths(results_dir, rowlen)

    scope, target = tfc.connect(rowlen=rowlen, key_fft_only=True)
    t = tools.Tools(scope, target, rowlen=rowlen)
    t.configure_scope()

    tmpl = t.load_templates(templates_path)
    templates = tmpl["templates"]          # (4, n_seg, seg_len)
    aligment_trace = np.load(align_path)
    print("Loaded templates {0} from {1}".format(templates.shape, templates_path))

    all_ok = True
    for test in range(tests):
        key = np.random.randint(0, 2, N)
        target_trace = t._capture_raw(key)
        segmented_trace = t.sliding_window_segmenter(target_trace, N, aligment_trace)[0]
        returned_key = t.run_profiled_attack(segmented_trace, templates, N)

        corr_count = int(np.sum(returned_key == key))
        all_ok &= corr_count == N
        print(f"Test: {test} - Correct key bits: {corr_count}/{N}")
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rowlen", type=int, default=DEFAULT_ROWLEN,
                        help="key size in BYTES; must match the firmware's build-time ROWLEN")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                        help="directory for the raw-trace / template / alignment files")
    sub = parser.add_subparsers(dest="phase", required=True)

    p_prof = sub.add_parser("profile", help="build and save the templates")
    p_prof.add_argument("--repeats", type=int, default=3,
                        help="traces averaged per template pattern")
    p_prof.add_argument("--online", action="store_true",
                        help="capture fresh raw traces from the ChipWhisperer first "
                             "(default: rebuild from an existing raw-trace .npz, no hardware)")

    p_atk = sub.add_parser("attack", help="run the profiled attack against the target")
    p_atk.add_argument("--tests", type=int, default=10,
                       help="number of random keys to recover")

    args = parser.parse_args()

    if args.phase == "profile":
        profile(rowlen=args.rowlen, repeats=args.repeats, online=args.online,
                results_dir=args.results_dir)
        ok = True
    else:
        ok = attack(rowlen=args.rowlen, tests=args.tests, results_dir=args.results_dir)

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
