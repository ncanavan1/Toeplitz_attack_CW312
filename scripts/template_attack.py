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

scope, target = tfc.connect(rowlen=ROWLEN, key_fft_only=True)
tools_ = tools.Tools(scope, target, rowlen=ROWLEN)
tools_.configure_scope()

results = []
timing = []

repeats = 3

try:
    cor_count, time_taken = tools_.run_attack(repeats, N)
    results.append(cor_count)
    if cor_count == N:
        timing.append(time_taken)

    rec_rate = np.asarray(results).mean(axis=0) / N
    print("Recovery rate: {0}".format(rec_rate))

    timing = np.asarray(timing)
    np.savetxt(os.path.join(RESULTS_DIR, "timing_128byte.csv"), timing, delimiter=",")
    np.savetxt(os.path.join(RESULTS_DIR, "recovery_counts_128byte.csv"), np.asarray(results), delimiter=",")
finally:
    tfc.disconnect(scope, target)
