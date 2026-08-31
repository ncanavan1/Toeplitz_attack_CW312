"""Tools and algorithms for the profiled SAD template attack against the
Toeplitz/FFT firmware (see template_attack.py for the driver).

Pipeline:
  save_raw_traces()      capture every pre-segmentation trace once, to a .npz
  load_raw_traces()      read that .npz back for offline development
  get_trace_from_raw()   -> sliding_window_segmenter(): cut a flat capture
                            into one aligned window per stage-1 butterfly
  plot_warmup_telescope() 3-level zoom of the warmup capture (whole window
                          -> 16 butterflies -> one butterfly + align window),
                          written once per save_raw_traces()
  save_templates() /
  load_templates()       persist / reload the 4 averaged per-butterfly
                         templates (one per input bit-pair 00/01/10/11)
  run_profiled_attack()  score a target capture against the templates,
                         recovering the key two bits per butterfly by SAD
"""
import os
import shutil
import time

import numpy as np
import matplotlib.pyplot as plt

import toeplitz_fft_controller as tfc


# Sample window, relative to a segmentation candidate's start, that
# sliding_window_segmenter() matches against to locate each butterfly.
allign_p0 = 515
allign_p1 = 585


class Tools:
    def __init__(self, scope, target, rowlen=tfc.ROWLEN):
        """`scope`/`target` are a connected ChipWhisperer pair (see
        toeplitz_fft_controller.connect()); pass None for both to use only
        the offline methods (load_*/segmentation/templates/attack scoring).
        `rowlen` is the key size in BYTES and must match the flashed
        firmware's build-time ROWLEN.
        """
        self.scope = scope
        self.target = target
        self.rowlen = rowlen

    ##########################
    ### SCOPE / TARGET #######
    ##########################

    def configure_scope(self, samples=2000, offset=0, gain=25, adc_mul=1, ROWLEN=64):
        """Size the capture window and ADC gain for this attack. Call once
        after toeplitz_fft_controller.connect() (which only runs
        scope.default_setup()) and before the first _capture_raw().

        samples: must be large enough to hold the *entire* triggered
        stage-1 window -- all N/2 = rowlen*4 butterflies captured
        back-to-back under one trigger (Toeplitz_FFT.c's DIT_FFT raises the
        trigger once before stage 1 and drops it at the start of stage 2).
        Too few samples truncates the capture mid-butterfly and
        sliding_window_segmenter() then finds fewer than the expected 63
        segments. scope.adc.trig_count (printed by _capture_raw()) is the
        actual number of samples the trigger was held high for -- compare
        it against `samples` to check the window wasn't clipped.

        offset:  ADC sample offset from the trigger edge.
        gain:    ADC gain in dB -- raise if traces look flat/low-amplitude,
                 lower if they clip.
        adc_mul: scope.clock.adc_mul, if the default ADC rate undersamples
                 the target clock for this ROWLEN.

        These defaults are starting points, not measured values -- tune
        them against a real captured trace/trig_count before relying on them.
        """
        self.scope.adc.stream_mode = False
        self.scope.adc.samples = samples * adc_mul * ROWLEN
        self.scope.adc.offset = offset
        self.scope.gain.db = gain
        self.scope.clock.adc_mul = adc_mul
        self.scope.adc.timeout = 10
        # Changing adc_mul can drop the target's clock momentarily (Husky
        # logs "Target clock may drop; you may need to reset your target")
        # and leave the ADC's PLL unlocked, which then shows up as repeated
        # "CLKGEN Failed to load divider value" errors on the next capture.
        # reset_target() recovers the target side; this recovers the scope side.
        self.scope.clock.reset_adc()
        self.reset_target()
    

    def reset_target(self):
        """Pulse nRST low then high to force the target firmware back to a
        fresh state (recovers a UART desync or an unlocked clock)."""
        self.scope.io.nrst = False
        time.sleep(0.001)
        self.scope.io.nrst = True
        time.sleep(0.001)

    def _capture_raw(self, key_bits, seed_bytes=None):
        """Load `key_bits` (a length rowlen*8 bit array, MSB-first per
        byte) plus a throwaway seed into the firmware, arm the scope, fire
        the hash, and return the raw power capture: all N/2 stage-1
        butterflies of the key FFT back-to-back under one trigger, before
        sliding_window_segmenter() cuts it up. Only the key FFT is
        triggered, so the seed value doesn't affect the trace -- it
        defaults to all-zero. save_raw_traces() dumps exactly this so the
        rest of the pipeline can run offline (see get_trace_from_raw()).
        """
        key_bytes = np.packbits(np.asarray(key_bits, dtype=np.uint8)).tolist()
        if seed_bytes is None:
            #seed_bytes = [1] * self.rowlen
            seed_bytes = np.random.randint(0,256,self.rowlen).tolist()
        # Load key + seed *before* arming: these are slow (a command byte,
        # rowlen payload bytes and a rowlen echo each, at ~38400 baud with
        # settling sleeps). Only the one-byte CMD_GO in trigger_hash() runs
        # after arm(), so the trigger fires within microseconds of it --
        # required for scope.adc.stream_mode, which streams from arm and
        # would run out its sample budget during a full key/seed exchange
        # ("no trigger seen").
        reference_result = tfc.reference_hash(key_bytes, seed_bytes)

        reader = tfc.ByteReader(self.target)
        tfc.load_key(self.target, key_bytes, reader, rowlen=self.rowlen)
        tfc.load_seed(self.target, list(seed_bytes), reader, rowlen=self.rowlen)
        self.scope.arm()
        result = tfc.trigger_hash(self.target, reader, rowlen=self.rowlen)
        if self.scope.capture():
            if reference_result != result:
                raise RuntimeError("Capture failed, Incorrect Result")
            else:
                raise RuntimeError("Capture failed, Correct Result...CW Issue most likely")
        trace = self.scope.get_last_trace()
        print("Trig count: {0}".format(self.scope.adc.trig_count))
        return trace

    ##########################################
    ### RAW TRACE CAPTURE / OFFLINE REPLAY ###
    ##########################################

    def _default_raw_path(self):
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        return os.path.join(results_dir, "raw_traces_rowlen{0}.npz".format(self.rowlen))

    def template_keys(self, N):
        """The 4 full keys the template stage profiles on -- one per input
        bit-pair pattern in [[0,0],[0,1],[1,0],[1,1]], with *every*
        gen_pairs() pair set to that pattern so every stage-1 butterfly
        sees the same input. Returns (inputs (4, 2), keys (4, N)).
        """
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
        pairs = self.gen_pairs(N)
        keys = np.zeros((4, N), dtype=np.uint8)
        for k, inp in enumerate(inputs):
            for pair in pairs:
                keys[k][pair[0]] = inp[0]
                keys[k][pair[1]] = inp[1]
        return inputs, keys

    def save_raw_traces(self, N, repeats, path=None, warmup_key=None,
                        correct_key=None, plot=True):
        """Capture every raw (pre-segmentation) trace the warmup/alignment
        and template stages consume and dump them to one .npz, so
        sliding_window_segmenter() / save_templates() / run_profiled_attack()
        can be developed and re-run offline with no ChipWhisperer attached
        (see load_raw_traces() / get_trace_from_raw()).

        .npz contents:
          rowlen, N, repeats                 capture parameters
          warmup_key      (N,)               key used for the warmup trace
          warmup          (raw_len,)         one raw capture for warmup_key
          template_inputs (4, 2)             the 4 bit-pair patterns
          template_keys   (4, N)             full key per pattern
          template_raw    (4, repeats, raw_len)
          correct_key     (N,)               only if correct_key was given
          target_raw      (raw_len,)         only if correct_key was given
        """
        if path is None:
            path = self._default_raw_path()
        if warmup_key is None:
            warmup_key = np.random.randint(0,2,N)
        warmup_key = np.asarray(warmup_key, dtype=np.uint8)

        self.reset_target()
        warmup = self._capture_raw(warmup_key)

        inputs, tkeys = self.template_keys(N)
        template_raw = np.array([
            [self._capture_raw(tkeys[k]) for _ in range(repeats)]
            for k in range(4)
        ])

        out = {
            "rowlen": np.int64(self.rowlen),
            "N": np.int64(N),
            "repeats": np.int64(repeats),
            "warmup_key": warmup_key,
            "warmup": warmup,
            "template_inputs": inputs,
            "template_keys": tkeys,
            "template_raw": template_raw,
        }
        if correct_key is not None:
            correct_key = np.asarray(correct_key, dtype=np.uint8)
            out["correct_key"] = correct_key
            out["target_raw"] = self._capture_raw(correct_key)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **out)
        print("Saved raw traces -> {0}".format(path))

        if plot:
            self.plot_raw_traces(data=out, save_png=os.path.splitext(path)[0] + ".png")
            tele_png = os.path.join(
                os.path.dirname(path),
                "warmup_telescope_rowlen{0}.png".format(self.rowlen))
            self.plot_warmup_telescope(warmup=warmup, N=N, save_png=tele_png)
            # results/ is git-ignored; keep a committed copy for the README.
            docs_png = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "docs", "warmup_telescope.png")
            os.makedirs(os.path.dirname(docs_png), exist_ok=True)
            shutil.copyfile(tele_png, docs_png)
            print("Copied telescope figure -> {0}".format(docs_png))
        return path

    @staticmethod
    def load_raw_traces(path):
        """Load a save_raw_traces() .npz into a plain dict (arrays
        materialised, not a lazy NpzFile). Keys are as documented on
        save_raw_traces().
        """
        with np.load(path, allow_pickle=False) as npz:
            return {k: npz[k] for k in npz.files}

    def get_trace_from_raw(self, raw_trace, N, allignment_trace=None):
        """sliding_window_segmenter() on a raw capture loaded from disk --
        the offline equivalent of capturing + segmenting, minus the
        ChipWhisperer. Returns (segments, allignment_trace).
        """
        return self.sliding_window_segmenter(np.asarray(raw_trace), N, allignment_trace)

    ##########################
    ### SEGMENTATION #########
    ##########################

    def SAD_inner(self, a, b):
        """Sum of absolute differences between two equal-length sample
        windows -- the alignment and template-scoring metric."""
        assert len(a) == len(b)
        total = 0
        for idx in range(len(a)):
            total = total + np.abs(a[idx] - b[idx])
        return total

    def sliding_window_segmenter(self, trace, N, allignment_trace=None):
        """Cut a flat capture (all N/2 stage-1 butterflies back-to-back
        under one trigger) into one fixed-length window per butterfly.

        Slides `allignment_trace` -- an (allign_p1 - allign_p0)-sample
        reference lifted from the first butterfly, or supplied by the
        caller so every trace segments against the *same* reference --
        across `trace` one sample at a time; wherever the SAD drops below
        `sad_thresh` that position is a butterfly landmark and the
        surrounding [-window_below, +window_above] samples are emitted as
        a segment.

        Returns (segments, allignment_trace) where segments is a list of
        N//2 - 1 arrays (63 for N=128) -- one short of N/2 because the last
        butterfly's window runs off the end of the capture. Asserts on that
        count so a clipped or misaligned capture fails loudly here, not
        downstream.
        """
        sad_thresh = 2
        offset = 0
        window_below = 300
        window_above = 100

        if allignment_trace is None:
            allignment_trace = trace[offset + allign_p0 : offset + allign_p1]

        window_stride = allign_p1 - allign_p0
        max_stride = len(trace) - window_stride

        segments = []
        for window in range(offset, max_stride, 1):
            sad = self.SAD_inner(allignment_trace, trace[window : window + window_stride])
            if sad < sad_thresh:
                segments.append(trace[window - window_below : window + window_above])

        assert len(segments) == N // 2 - 1
        return segments, allignment_trace

    ##########################
    ### TEMPLATE SAVE / LOAD #
    ##########################

    def _default_templates_path(self):
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        return os.path.join(results_dir, "templates_rowlen{0}.npz".format(self.rowlen))

    def save_templates(self, N, templates, path=None, inputs=None,
                       alignment_trace=None, repeats=None, plot=True):
        """Persist the 4 per-butterfly mean templates built by the
        profiling stage (one per bit-pair pattern in
        [[0,0],[0,1],[1,0],[1,1]]) so the attack stage can score a target
        capture offline without re-profiling (see load_templates()).

        templates: sequence of 4 arrays, each (n_segments, seg_len) -- the
        mean over `repeats` of get_trace_from_raw(...)[0] for that
        pattern's full key (as assembled in template_attack.py).

        .npz contents:
          rowlen, N                          capture parameters
          repeats            scalar          traces averaged per template (if given)
          template_inputs    (4, 2)          the 4 bit-pair patterns
          templates          (4, n_seg, seg_len)
          alignment_trace    (align_len,)    only if alignment_trace was given
        """
        if path is None:
            path = self._default_templates_path()
        if inputs is None:
            inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
        inputs = np.asarray(inputs, dtype=np.uint8)

        templates = np.asarray(templates, dtype=np.float64)
        if templates.ndim != 3 or templates.shape[0] != inputs.shape[0]:
            raise ValueError(
                "expected templates of shape ({0}, n_seg, seg_len), got {1}"
                .format(inputs.shape[0], templates.shape))

        out = {
            "rowlen": np.int64(self.rowlen),
            "N": np.int64(N),
            "template_inputs": inputs,
            "templates": templates,
        }
        if repeats is not None:
            out["repeats"] = np.int64(repeats)
        if alignment_trace is not None:
            out["alignment_trace"] = np.asarray(alignment_trace, dtype=np.float64)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **out)
        print("Saved templates -> {0}".format(path))

        if plot:
            self.plot_templates(data=out, save_png=os.path.splitext(path)[0] + ".png")
        return path

    @staticmethod
    def load_templates(path):
        """Load a save_templates() .npz into a plain dict (arrays
        materialised, not a lazy NpzFile). data["templates"] is the
        (4, n_seg, seg_len) array run_profiled_attack() scores against,
        indexed as templates[pattern][segment].
        """
        with np.load(path, allow_pickle=False) as npz:
            return {k: npz[k] for k in npz.files}

    ##########################
    ### PLOTTING #############
    ##########################

    def plot_raw_traces(self, path=None, data=None, save_png=None):
        """Overlay the saved raw traces (warmup, first capture per template
        pattern, and the target if present) for a quick visual check. Pass
        either a .npz `path` or a `data` dict from load_raw_traces().
        """
        if data is None:
            data = self.load_raw_traces(path or self._default_raw_path())

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(data["warmup"], lw=0.8, label="warmup")
        for k in range(data["template_raw"].shape[0]):
            inp = data["template_inputs"][k]
            ax.plot(data["template_raw"][k][0], lw=0.8,
                    label="template {0}{1}".format(int(inp[0]), int(inp[1])))
        if "target_raw" in data:
            ax.plot(data["target_raw"], lw=0.8, color="k", label="target")
        ax.set_xlabel("sample")
        ax.set_ylabel("power")
        ax.set_title("Raw traces (rowlen={0}, N={1}, repeats={2})".format(
            int(data["rowlen"]), int(data["N"]), int(data["repeats"])))
        ax.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        if save_png:
            fig.savefig(save_png, dpi=120)
            print("Saved plot -> {0}".format(save_png))
        plt.show()
        return fig

    def _butterfly_landmarks(self, trace, sad_thresh=2):
        """Sample indices where the (allign_p0:allign_p1) reference window
        lifted from the first butterfly re-matches -- one per stage-1
        butterfly, the exact positions sliding_window_segmenter() keys its
        segments off. Same SAD < sad_thresh test, but vectorised: SAD_inner
        is a Python loop, fine for 63 segments, far too slow to sweep a
        whole raw trace just for a plot.
        """
        trace = np.asarray(trace, dtype=np.float64)
        w = allign_p1 - allign_p0
        ref = trace[allign_p0:allign_p1]
        n = len(trace) - w + 1
        if n <= 0:
            return np.empty(0, dtype=int)
        sad = np.zeros(n)
        for j in range(w):
            sad += np.abs(trace[j:j + n] - ref[j])
        return np.flatnonzero(sad < sad_thresh)

    def plot_warmup_telescope(self, warmup=None, N=None, path=None, data=None,
                              save_png=None, start_bf=0, n_bf_zoom=16,
                              bf_index=None, show=True):
        """Three-level telescopic view of one warmup capture, each panel a
        sub-interval of the one above (the expanded slice shaded on the
        parent):

          1. the whole trigger window -- all N/2 stage-1 key-FFT butterflies
             back-to-back under one trigger
          2. `n_bf_zoom` consecutive butterfly segments, with the landmarks
             sliding_window_segmenter() finds marked
          3. one butterfly unit -- the same [-window_below, +window_above]
             window the segmenter emits and a template scores -- with the
             alignment window (allign_p0:allign_p1, the 70-sample reference
             the segmenter slides across the trace) shaded

        Called once by save_raw_traces(); also usable standalone by passing
        `warmup`+`N`, or a raw-trace .npz `path` / `data` dict.
        """
        if warmup is None:
            if data is None:
                data = self.load_raw_traces(path or self._default_raw_path())
            warmup = data["warmup"]
            if N is None:
                N = int(data["N"])
        warmup = np.asarray(warmup, dtype=np.float64)

        # match sliding_window_segmenter()'s constants
        window_below, window_above = 300, 100
        align_w = allign_p1 - allign_p0

        lm = self._butterfly_landmarks(warmup)
        expected = (N // 2 - 1) if N else None
        if expected is not None and len(lm) != expected:
            print("plot_warmup_telescope: {0} butterfly landmarks found, "
                  "expected {1} -- warmup may be clipped or misaligned"
                  .format(len(lm), expected))
        if len(lm) < 2:
            print("plot_warmup_telescope: too few landmarks to telescope; "
                  "plotting the full window only")

        # panel 2 span: n_bf_zoom butterflies starting at start_bf
        start_bf = int(np.clip(start_bf, 0, max(0, len(lm) - 2)))
        end_bf = min(len(lm) - 1, start_bf + n_bf_zoom) if len(lm) >= 2 else 0
        p2_lo = max(0, int(lm[start_bf]) - window_below) if len(lm) else 0
        p2_hi = (min(len(warmup), int(lm[end_bf]) + window_above)
                 if len(lm) >= 2 else len(warmup))

        # panel 3: one butterfly inside that span
        if bf_index is None:
            bf_index = min(len(lm) - 1, start_bf + n_bf_zoom // 2) if len(lm) else 0
        bf = int(lm[bf_index]) if len(lm) else len(warmup) // 2
        p3_lo = max(0, bf - window_below)
        p3_hi = min(len(warmup), bf + window_above)

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        ax = axes[0]
        ax.plot(warmup, lw=0.6)
        ax.axvspan(p2_lo, p2_hi, color="tab:orange", alpha=0.25,
                   label="panel 2 ({0} butterflies)".format(max(0, end_bf - start_bf)))
        ax.set_xlim(0, len(warmup))
        ax.set_title("Warmup capture -- full stage-1 window "
                     "({0} butterfly landmarks)".format(len(lm)))
        ax.legend(fontsize=8, loc="upper right")

        ax = axes[1]
        x2 = np.arange(p2_lo, p2_hi)
        ax.plot(x2, warmup[p2_lo:p2_hi], lw=0.8)
        for i in range(start_bf, end_bf + 1):
            ax.axvline(int(lm[i]), color="0.6", lw=0.7, ls=":")
        ax.axvspan(p3_lo, p3_hi, color="tab:green", alpha=0.25,
                   label="panel 3 (butterfly #{0})".format(bf_index))
        ax.set_xlim(p2_lo, p2_hi)
        ax.set_title("Telescope 1 -- {0} consecutive butterfly segments "
                     "(dotted = landmarks)".format(max(0, end_bf - start_bf)))
        ax.legend(fontsize=8, loc="upper right")

        ax = axes[2]
        x3 = np.arange(p3_lo, p3_hi)
        ax.plot(x3, warmup[p3_lo:p3_hi], lw=1.0)
        ax.axvspan(bf, bf + align_w, color="tab:red", alpha=0.30,
                   label="alignment window [allign_p0:allign_p1]")
        ax.axvline(bf, color="tab:red", lw=1.0)
        ax.set_xlim(p3_lo, p3_hi)
        ax.set_title("Telescope 2 -- one butterfly unit "
                     "(segment = landmark [-{0}, +{1}])"
                     .format(window_below, window_above))
        ax.legend(fontsize=8, loc="upper right")

        for ax in axes:
            ax.set_xlabel("sample")
            ax.set_ylabel("power")
        fig.tight_layout()

        if save_png is None:
            results_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
            save_png = os.path.join(
                results_dir, "warmup_telescope_rowlen{0}.png".format(self.rowlen))
        os.makedirs(os.path.dirname(save_png), exist_ok=True)
        fig.savefig(save_png, dpi=120)
        print("Saved plot -> {0}".format(save_png))
        if show:
            plt.show()
        return fig

    def plot_templates(self, path=None, data=None, save_png=None, segment=0):
        """Overlay the 4 templates for one butterfly `segment` for a quick
        visual check. Pass either a .npz `path` or a `data` dict from
        load_templates()/save_templates().
        """
        if data is None:
            data = self.load_templates(path or self._default_templates_path())

        templates = np.asarray(data["templates"])
        inputs = np.asarray(data["template_inputs"])

        fig, ax = plt.subplots(figsize=(12, 5))
        for k in range(templates.shape[0]):
            inp = inputs[k]
            ax.plot(templates[k][segment], lw=0.9,
                    label="template {0}{1}".format(int(inp[0]), int(inp[1])))
        ax.set_xlabel("sample")
        ax.set_ylabel("power")
        ax.set_title("Templates (rowlen={0}, N={1}, segment={2})".format(
            int(data["rowlen"]), int(data["N"]), segment))
        ax.legend(fontsize=8)
        fig.tight_layout()
        if save_png:
            fig.savefig(save_png, dpi=120)
            print("Saved plot -> {0}".format(save_png))
        plt.show()
        return fig

    ##########################
    ### BIT-PAIR ORDERING ####
    ##########################

    def reverse_bits(self, n, bitSize):
        """Reverse the low `bitSize` bits of integer `n`."""
        result = 0
        for i in range(bitSize):
            if n & (1 << i):
                result |= 1 << (bitSize - 1 - i)
        return result

    def bit_reverse(self, x):
        """In-place bit-reversal permutation of array `x` (length must be a
        power of 2) -- the reordering DIT_FFT applies to its input before
        stage 1."""
        N = x.shape[0]
        s = int(np.log2(N))
        for i in range(N):
            rb = self.reverse_bits(i, s)
            if i < rb:
                tmp = x[i]
                x[i] = x[rb]
                x[rb] = tmp
        return x

    def gen_pairs(self, N):
        """Bit-index pairs in the order the firmware's stage-1 DIT_FFT
        butterfly processes them: X is bit-reversed before the trigger goes
        high, then stage 1 (span=2) butterflies consecutive pairs
        (X[2i], X[2i+1]) in order -- so pair i here is exactly the i-th
        butterfly captured in a single stage-1 trace.
        """
        pairs = []
        order_list = np.arange(N)
        reverse_order = self.bit_reverse(order_list)
        for i in range(int(N / 2)):
            pairs.append([reverse_order[int(2 * i)], reverse_order[int(2 * i + 1)]])
        return pairs

    ##########################
    ### PROFILED ATTACK ######
    ##########################

    def score_function(self, a, b):
        """Distance between a template window and the target window -- lower
        is a better match. SAD; swap the metric here to try another."""
        return self.SAD_inner(a, b)

    def run_profiled_attack(self, target_trace, templates, N):
        """Recover a key from one segmented target capture.

        For each stage-1 butterfly n, score its window against all 4
        templates[.][n] and take the lowest-SAD pattern as that butterfly's
        two input bits, writing them to the key positions gen_pairs()[n]
        couples. Iterates n over range(N//2 - 1) to match the 63 segments
        sliding_window_segmenter() produces. Returns the length-N key guess.
        """
        guessed_key = np.zeros(N)
        pairs = self.gen_pairs(N)
        bit_pair = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for n in range((N // 2) - 1):
            scores = [self.score_function(templates[k][n], target_trace[n]) for k in range(4)]
            best = int(np.argmin(scores))
            guessed_key[pairs[n][0]] = bit_pair[best][0]
            guessed_key[pairs[n][1]] = bit_pair[best][1]

        return guessed_key
