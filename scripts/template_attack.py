import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import toeplitz_fft_controller as tfc
import tools
importlib.reload(tools)

# Key/seed size in BYTES -- must match the KEY_FFT_ONLY=1 firmware's
# build-time ROWLEN (see toeplitz_fft_controller.py / Toeplitz_FFT.c).
ROWLEN = 16
N = ROWLEN * 8  # key size in BITS -- the attack guesses individual input bits

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")



online = False
profiling = False
repeats = 3
x = np.arange(0,400) ##debug


RAW_PATH = os.path.join(RESULTS_DIR, "raw_traces_rowlen{0}.npz".format(ROWLEN))
TEMPLATES_PATH = os.path.join(RESULTS_DIR, "templates_rowlen{0}.npz".format(ROWLEN))
ALIGN_PATH = os.path.join(RESULTS_DIR, "aligment_trace.npy")

if profiling:
    if online:
        scope, target = tfc.connect(rowlen=ROWLEN, key_fft_only=True)
        t = tools.Tools(scope, target, rowlen=ROWLEN)
        t.configure_scope()
        t.save_raw_traces(N, repeats, path=RAW_PATH)
    else:
        t = tools.Tools(None, None, rowlen=ROWLEN)

    # Build the 4 per-butterfly mean templates from the raw .npz -- runs
    # offline; only save_raw_traces() above needs the ChipWhisperer.
    d = t.load_raw_traces(RAW_PATH)
    align, aligment_trace = t.get_trace_from_raw(d["warmup"], int(d["N"]))
    np.save(ALIGN_PATH, aligment_trace)
    templates = []
    for i in range(4):
        traces = []
        for r in range(repeats):
            traces.append(t.get_trace_from_raw(d["template_raw"][i][r], int(d["N"]), aligment_trace)[0])
        template = np.mean(traces, axis=0)
        templates.append(template)

    t.save_templates(N, templates, path=TEMPLATES_PATH,
                     inputs=d["template_inputs"], alignment_trace=aligment_trace,
                     repeats=repeats)


if profiling == False:

    scope, target = tfc.connect(rowlen=ROWLEN, key_fft_only=True)
    t = tools.Tools(scope, target, rowlen=ROWLEN)
    t.configure_scope()
    # Attack stage: read the templates back (no hardware, no re-profiling).
    tmpl = t.load_templates(TEMPLATES_PATH)
    templates = tmpl["templates"]          # (4, n_seg, seg_len)
    template_inputs = tmpl["template_inputs"]
    aligment_trace = np.load(ALIGN_PATH)
    print("Loaded templates {0} from {1}".format(templates.shape, TEMPLATES_PATH))

    for test in range(10):
        key = np.random.randint(0,2,N)
        target_trace = t._capture_raw(key)
        segmented_trace = t.sliding_window_segmenter(target_trace, N, aligment_trace)[0]
        returned_key = t.run_profiled_attack(segmented_trace, templates, N)

        corr_count = 0
        for n in range(N):
            if returned_key[n] == key[n]:
                corr_count = corr_count + 1

        print(f"Test: {test} - Correct key bits: {corr_count}/{N}")

# try:
#     cor_count, time_taken = tools_.run_attack(repeats, N)
#     results.append(cor_count)
#     if cor_count == N:
#         timing.append(time_taken)

#     rec_rate = np.asarray(results).mean(axis=0) / N
#     print("Recovery rate: {0}".format(rec_rate))

#     timing = np.asarray(timing)
#     np.savetxt(os.path.join(RESULTS_DIR, "timing_128byte.csv"), timing, delimiter=",")
#     np.savetxt(os.path.join(RESULTS_DIR, "recovery_counts_128byte.csv"), np.asarray(results), delimiter=",")
# finally:
#     tfc.disconnect(scope, target)
