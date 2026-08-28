##########Contains the tools and algorithms for the template attacks####
import chipwhisperer as cw
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy import stats
import sys
import os
import time
import scipy.signal

import toeplitz_fft_controller as tfc


allign_p0 = 515
allign_p1 = 585

class Tools:
    def __init__(self, scope, target, rowlen=tfc.ROWLEN):
        self.scope = scope
        self.target = target
        self.rowlen = rowlen


    ##########################
    ### SCOPE SETTINGS #######
    ##########################
    # toeplitz_fft_controller.connect() only runs scope.default_setup()
    # before flashing -- call configure_scope() once after connecting to
    # size the capture window and ADC gain for this attack, before the
    # first get_trace().

    def configure_scope(self, samples=2000, offset=0, gain=25, adc_mul=1):
        """samples: must be large enough to hold the *entire* triggered
        stage-1 window, i.e. all N/2 = rowlen*4 butterflies captured
        back-to-back in one trace (see Toeplitz_FFT.c's DIT_FFT: trigger
        goes high once, before stage 1, and only drops at the start of
        stage 2). Too few samples truncates get_trace()'s trace mid-
        butterfly -- if guess_sequentially_by_2's segment slicing looks
        wrong, raise this first. scope.adc.trig_count (printed by
        get_trace()) is the actual number of ADC samples the trigger was
        held high for on the last capture -- compare it against `samples`
        to check the window wasn't clipped.

        offset: ADC sample offset from the trigger edge.
        gain: ADC gain in dB -- raise if traces look flat/low-amplitude,
        lower if they're clipping.
        adc_mul: scope.clock.adc_mul, in case the default ADC sample rate
        undersamples the target's clock for this ROWLEN.

        These defaults are starting points, not measured values -- tune
        them against a captured trace/trig_count for your actual hardware
        setup before relying on them.
        """
        self.scope.adc.samples = samples*adc_mul*16
        self.scope.adc.offset = offset
        self.scope.gain.db = gain
        self.scope.clock.adc_mul = adc_mul
        # Changing adc_mul can drop the target's clock momentarily (Husky
        # logs "Target clock may drop; you may need to reset your target"
        # when this happens) and leave the ADC's own PLL unlocked, which
        # then shows up as repeated "CLKGEN Failed to load divider value"
        # errors on the next capture. reset_target() (called in run_attack())
        # recovers the target side; this recovers the scope side.
        self.scope.clock.reset_adc()


    def plot_overlay(self, traces):

        fig = cw.plot()
        for trace in traces:
            fig *= cw.plot(trace)
        return fig

    def plot_difference(self, trace1, trace2):
        return cw.plot(trace1 - trace2)

    def plot_segment_divisions(self, trace, offset, n_seg, seg_len):
        """Ravel a (n_seg, seg_len) segmented capture back into one trace and
        draw a vertical line at every segment boundary (x=0, seg_len, 2*seg_len,
        ..., n_seg*seg_len) for visual inspection of the reconjoin.
        """
        segs = np.arange(offset, offset + seg_len*n_seg, seg_len)
        plt.plot(trace)
        plt.vlines(segs,ymin=-0.5,ymax=0.5,colors="r",linestyles="dashed")
        plt.show()
        k=7


    def SAD_inner(self, a, b):

        sum = 0
        assert len(a) == len(b)
        for idx in range(len(a)):
            sum = sum + np.abs(a[idx] - b[idx])

        return sum


    def SADs(self, ref_pattern, target_pattern):

        sads = []

        ref_len = len(ref_pattern)
        target_len = len(target_pattern)

        max_stride = target_len - ref_len

        for window_index in range(max_stride):
            target_start = window_index
            target_end = window_index + ref_len

            target_window = target_pattern[target_start:target_end]

            SAD = self.SAD_inner(ref_pattern, target_window)
            sads.append(SAD)

        return sads


    def align_traces(self, pattern, traces, allign_p0, allign_p1):
        aligned_traces = []

        for trace in traces:
            sads = self.SADs(pattern, trace)
            min_sad = np.argmin(sads)
            # constrain shift if desired
            if min_sad >= allign_p0:
                shift = allign_p0 - min_sad
            else:
                shift = min_sad - allign_p0

            aligned_trace = np.roll(trace, -shift)  # Shift the trace
            # Zero-fill the wrapped-around region instead of circular rolling
            if shift > 0:
                aligned_trace[:shift] = 0  # shifting forward: zero the start
            elif shift < 0:
                aligned_trace[shift:] = 0  # shifting back: zero the end
            aligned_traces.append(aligned_trace)

        return np.array(aligned_traces)



    def _capture_raw(self, key_bits):
        """Arm the scope, send `key_bits` to the KEY_FFT_ONLY firmware and
        return the single flat power capture -- all N/2 stage-1 butterflies
        back-to-back under one trigger -- *before* full_trace_segmenter()
        slices and aligns it. save_raw_traces() dumps exactly this so the
        rest of the pipeline can run offline (see get_trace_from_raw()).
        """
        key_bytes = np.packbits(np.asarray(key_bits, dtype=np.uint8)).tolist()
        self.scope.arm()
        tfc.send_key_only(self.target, key_bytes, rowlen=self.rowlen)
        if self.scope.capture():
            raise RuntimeError("Capture failed")
        trace = self.scope.get_last_trace()
        print("Trig count: {0}".format(self.scope.adc.trig_count))
        return trace

    def get_trace(self, key_bits, allignment_trace=None):
        """Capture one power trace of the triggered stage-1 key FFT for a
        key given as a bit array (length rowlen*8, MSB-first per byte --
        matches the firmware's bytes_to_bits()/the reference in
        toeplitz_fft_controller.reference_hash). Requires the target to be
        flashed with the KEY_FFT_ONLY=1 firmware (see toeplitz_fft_controller
        connect()/reprogram()).
        """
        trace = self._capture_raw(key_bits)
        segmented_trace = self.full_trace_segmenter(trace, len(key_bits), allignment_trace)
        return segmented_trace

    ##########################################
    ### RAW TRACE CAPTURE / OFFLINE REPLAY ###
    ##########################################

    def _default_raw_path(self):
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        return os.path.join(results_dir, "raw_traces_rowlen{0}.npz".format(self.rowlen))

    def template_keys(self, N):
        """The 4 full keys templates_gen() captures on -- one per bit-pair
        pattern in [[0,0],[0,1],[1,0],[1,1]], with every gen_pairs() pair
        set to that pattern. Returns (inputs (4, 2), keys (4, N)).
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
        full_trace_segmenter() / templates_gen() / the MSE guessing can be
        developed and re-run offline with no ChipWhisperer attached
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
            warmup_key = np.ones(N, dtype=np.uint8)
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
        """full_trace_segmenter() on a raw capture loaded from disk -- the
        offline equivalent of get_trace(), minus the ChipWhisperer.
        """
        #return self.full_trace_segmenter(np.asarray(raw_trace), N, allignment_trace)
        return self.sliding_window_segmenter(np.asarray(raw_trace), N, allignment_trace)

    ##################################
    ### TEMPLATE SAVE / LOAD       ###
    ##################################

    def _default_templates_path(self):
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        return os.path.join(results_dir, "templates_rowlen{0}.npz".format(self.rowlen))

    def save_templates(self, N, templates, path=None, inputs=None,
                       alignment_trace=None, repeats=None, plot=True):
        """Persist the 4 per-butterfly mean templates built by the profiling
        stage (one per bit-pair pattern in [[0,0],[0,1],[1,0],[1,1]]) so the
        attack stage can score a target capture offline without re-profiling
        (see load_templates()).

        templates: sequence of 4 arrays, each (n_segments, seg_len) -- the
        mean over `repeats` of get_trace_from_raw(...)[0] for that pattern's
        full key (as assembled in template_attack.py).

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
        materialised, not a lazy NpzFile). Keys are as documented on
        save_templates(); data["templates"] is the (4, n_seg, seg_len) array
        that infer_from_traces() indexes as template_traces[i][segment].
        """
        with np.load(path, allow_pickle=False) as npz:
            return {k: npz[k] for k in npz.files}

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


    def sliding_window_segmenter(self, trace, N, allignment_trace=None):

        sad_thresh = 2
        offset = 0
        segments = []
        seg_len = 591 - offset ##samples of a single bfu
        n_segments = int(N // 2)

        if allignment_trace is None:
            allignment_trace = trace[offset + allign_p0 : offset + allign_p1]
            # plt.plot(trace)
            # plt.vlines([offset+allign_p0,offset+allign_p1],ymin=min(trace),ymax=max(trace),linestyles="dashed",color="k")
            # plt.show()

        window_stride = allign_p1 - allign_p0
        max_stride = len(trace) - window_stride

        window_below = 300
        window_above = 100
        
        for window in range(offset, max_stride, 1):
            sad = self.SAD_inner(allignment_trace, trace[window: window+window_stride])
            if sad < sad_thresh:
                segments.append(trace[window-window_below:window+window_above])

        assert len(segments) == 63 

        return segments, allignment_trace

    def full_trace_segmenter(self, trace, N, allignment_trace=None):
        """Slice a single flat capture (all N/2 stage-1 butterflies
        back-to-back under one trigger -- see get_trace()) into one
        fixed-length window per butterfly, aligning each against the first
        via cross-correlation to correct for small timing drift between
        butterflies.

        offset/seg_len are measured constants for this hardware/ROWLEN
        (samples before the first butterfly, and samples spanning one), not
        derived -- re-measure against get_trace()'s trig_count print if they
        drift.

        Each window is sliced with `align_margin` samples of headroom on
        both sides, then align_segments() searches that margin and trims
        back to exactly seg_len -- see its docstring for why the margin is
        required (aligning equal-length windows has nowhere to search).
        Returns a (N//2, seg_len) 2-D array; raises ValueError up front if
        `trace` is too short to hold every window with margin, rather than
        silently producing a ragged/malformed result.
        """
        offset = 163 ##samples before initial bfu
        seg_len = 591 - offset ##samples of a single bfu
        n_segments = int(N // 2)
        margin = 150 ##required to allow for overlapping as the regular segments are not perfect

        #self.plot_segment_divisions(trace, offset, n_segments, seg_len)
        padded_segments = []
        for n in range(n_segments):
            start = (offset + seg_len * n) - margin
            end = offset + seg_len * (n + 1) + margin
            if start < 0:
                start = 0
            if end > len(trace):
                end = len(trace)
            padded_segments.append(trace[start:end])

        if allignment_trace is None:
            allignment_trace = padded_segments[0]


        segments_alligned = self.align_traces(allignment_trace[allign_p0:allign_p1], padded_segments, allign_p0, allign_p1)

        assert segments_alligned.ndim == 2, \
            "expected (segments, samples), got shape {0}".format(segments_alligned.shape)
        return segments_alligned


    ##################################
    ### USED IN FORMAL MGD METHOD ####
    ##################################





    def calculate_mean_sample(self, samples):
        return np.mean(samples,axis=0)


    def compute_noise_vector(self, sample, mean, points):
        n_vec = []
        for p in points:
            n_vec.append(np.abs(sample[p] - mean[p]))
        return np.asarray(n_vec)

    def compute_noise_covarience_matrix(self, samples, mean, points):
        mat_size = points.shape[0]
        cov_mat = np.zeros([mat_size, mat_size])
        p = 0
        for u in points:
            q = 0
            for v in points:
                n1 = self.compute_noise_vector(u, mean, points)
                n2 = self.compute_noise_vector(v, mean, points)
                cov =  np.cov(n1,n2)[0][1]
                cov_mat[p,q] = cov
                q=q+1
            p=p+1
        return cov_mat


    def compress_samples(self, sample_matrix_in, points):
        sample_size = sample_matrix_in.shape[0]
        sample_matrix_out = np.zeros([sample_size, points.shape[0]])

        for s in range(sample_size):
            q = 0
            for p in points:
                sample_matrix_out[s,q] = sample_matrix_in[s,p]
                q = q+1
        return sample_matrix_out

    #assumes that input matrix is already reduced to dimension of use
    def compute_noise_cov_mat(self, sample_matrix, mean):
        sample_size = sample_matrix.shape[0]
        sample_points = sample_matrix.shape[1]
        cov_mat = np.zeros([sample_points, sample_points])

        for i in range(sample_size):
            n_vec = sample_matrix[i] - mean
            mat = np.matmul(n_vec.T,n_vec)
            cov_mat = cov_mat + mat

        return (1/(sample_size-1))*cov_mat

    def calc_p_vec(self, n, det_sig, inv_sig):
        N = n.shape[0]
        e1 = np.matmul(n.T,inv_sig)
        e = np.exp(-0.5*np.matmul(e1,n))
        s = 1/(np.sqrt(2*np.pi)**N * det_sig)
        return s*e

    def compute_p_vec(self, sample, means, points, dets, invs):
        test_vec0 = self.compute_noise_vector(sample, means[0], points)
        test_vec1 = self.compute_noise_vector(sample, means[1], points)
        test_vec2 = self.compute_noise_vector(sample, means[2], points)
        test_vec3 = self.compute_noise_vector(sample, means[3], points)

        p_0 = self.calc_p_vec(test_vec0, dets[0], invs[0])
        p_1 = self.calc_p_vec(test_vec1, dets[1], invs[1])
        p_2 = self.calc_p_vec(test_vec2, dets[2], invs[2])
        p_3 = self.calc_p_vec(test_vec3, dets[3], invs[3])

        p_sum = p_0 + p_1 + p_2 + p_3
        p_0 = p_0/p_sum
        p_1 = p_1/p_sum
        p_2 = p_2/p_sum
        p_3 = p_3/p_sum


        return np.asarray([p_0, p_1, p_2, p_3])


    def compute_pearson_vec(self, sample, means):
        p0, d0 = stats.pearsonr(sample, means[0])
        p1, d1 = stats.pearsonr(sample, means[1])
        p2, d2 = stats.pearsonr(sample, means[2])
        p3, d3 = stats.pearsonr(sample, means[3])

        d0 = 1 - d0
        d1 = 1 - d1
        d2 = 1 - d2
        d3 = 1 - d3

        sum = d0 + d1 + d2 + d3
        d0 = d0/sum
        d1 = d1/sum
        d2 = d2/sum
        d3 = d3/sum

        return np.asarray([d0,d1,d2,d3])

    def class_accuracy_exp(self, samples, points, means, dets, invs, L):
        classes = 4
        Confusion_matrix = np.zeros([classes,classes])
        confidence_matrix = np.zeros([classes,classes])
        ##4 classes
        for k in range(classes):
            for l in range(L):
                p_vec = self.compute_p_vec(samples[k*L + l], means, points, dets, invs)
                #p_vec = compute_pearson_vec(samples[k*L + l], means)
                winner = np.argmax(p_vec)
                Confusion_matrix[k,winner] = Confusion_matrix[k,winner]+1
                confidence_matrix[k] = confidence_matrix[k] + p_vec
        confidence_matrix = confidence_matrix/L
        return Confusion_matrix, confidence_matrix


    ################################
    ### USED IN INFORMAL METHOD ####
    ################################

    def reverse_bits(self, n, bitSize):
        result = 0
        for i in range(bitSize):
            if n & (1 << i):
                result |= 1 << (bitSize - 1 - i)
        return result

    def bit_reverse(self, x):
        N = x.shape[0]
        s = int(np.log2(N))
        for i in range(N):
            rb = self.reverse_bits(i,s)
            if(i < rb):
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
        for i in range(int(N/2)):
            pairs.append([reverse_order[int(2*i)], reverse_order[int(2*i + 1)]])
        return pairs

    def infer_from_traces(self, target_trace, template_traces, segment):
        """Compare butterfly `segment` of the target's segmented capture
        against the same segment in each of the 4 hypothesis traces.
        target_trace/template_traces are (N//2, seg_len) arrays from
        get_trace() (via full_trace_segmenter()/align_segments()) -- segment
        i is a real, separately-aligned window for butterfly i, not a slice
        of one long trace.
        """
        target_seg = target_trace[segment]

        MSE = []
        PC = []
        print("\n\nSegment {0}".format(segment))
        for i in range(4):
            template_seg = template_traces[i][segment]
            MSE.append(mean_squared_error(target_seg, template_seg))
            PC.append(stats.pearsonr(target_seg, template_seg))
            print("Guess {0}. MSE: {1}, PC: {2}".format(i,MSE[i], PC[i]))

        best = np.argsort(MSE)[0]
        result = []
        match best:
            case 0:
                result = [0,0]
            case 1:
                result = [0,1]
            case 2:
                result = [1,0]
            case 3:
                result = [1,1]

        return result


    def guess_sequentially_by_2(self, target_trace, repeats, N):
        """target_trace: (N//2, seg_len) array from get_trace() -- segment i
        lines up with gen_pairs(N)[i], the pair this loop is guessing on
        iteration i.
        """
        pairs = self.gen_pairs(N)
        key_hyp = np.zeros(N)
        segment = 0

        for pair in pairs:

            self.reset_target()
            warmup = self.get_trace(np.ones(N))

            traces_avg = []
            guesses = [[0,0], [0,1], [1,0], [1,1]]
            for guess in guesses:
                key_hyp[pair[0]] = guess[0]
                key_hyp[pair[1]] = guess[1]
                traces = [self.get_trace(key_hyp) for _ in range(repeats)]
                traces_avg.append(np.mean(traces,axis=0))

            result = self.infer_from_traces(target_trace, traces_avg, segment)
            key_hyp[pair[0]] = result[0]
            key_hyp[pair[1]] = result[1]
            segment += 1
        return key_hyp


    def templates_gen(self, N, repeats, alignment_trace):

        templates = []
        inputs = [[0,0],[0,1],[1,0],[1,1]]
        pairs = self.gen_pairs(N)
        for input in inputs:
            key = np.zeros(N)
            for pair in pairs:
                key[pair[0]] = input[0]
                key[pair[1]] = input[1]
            traces = [self.align_traces(alignment_trace[allign_p0: allign_p1], self.get_trace(key), allign_p0, allign_p1) for _ in range(repeats)]
            trace_avg = np.mean(traces,axis=0)
            templates.append(trace_avg)
        return templates

    def score_function(self, a, b):
        return self.SAD_inner(a,b)

    def run_profiled_attack(self, target_trace, templates, N):

        guessed_key = np.zeros(N)
        pairs = self.gen_pairs(N)

        for n in range((N//2) - 1):
            scores = []
            templates_n = [templates[0][n], templates[1][n], templates[2][n], templates[3][n]]
            for template in templates_n:
                score = self.score_function(template, target_trace[n])
                scores.append(score)


            if np.argmin(scores) == 0:
                guessed_key[pairs[n][0]] = 0
                guessed_key[pairs[n][1]] = 0

            if np.argmin(scores) == 1:
                guessed_key[pairs[n][0]] = 0
                guessed_key[pairs[n][1]] = 1

            if np.argmin(scores) == 2:
                guessed_key[pairs[n][0]] = 1
                guessed_key[pairs[n][1]] = 0

            if np.argmin(scores) == 3:
                guessed_key[pairs[n][0]] = 1
                guessed_key[pairs[n][1]] = 1

        return guessed_key

        
    def run_attack(self, repeats, N):
        self.reset_target()
        correct_key = np.random.randint(2,size=N)
        print(correct_key)
        rand_key = np.random.randint(0,1,N)
        for _ in range(1):
            warmup = self.get_trace(rand_key)

        allignment_trace = warmup[0]
        templates = self.templates_gen(N,repeats, allignment_trace)



        target_trace = self.get_trace(correct_key, allignment_trace)

        start = time.time()
        guess = self.guess_sequentially_by_2(target_trace, repeats, N)
        end = time.time()

        time_taken = end - start

        guess = np.asarray(guess,dtype=int)
        correct_key = np.asarray(correct_key,dtype=int)
        print("\n{0} : Eve's Guess".format(guess))
        print("{0} : Correct Key".format(correct_key))
        cor_count = 0
        for i in range(N):
            if guess[i] == correct_key[i]:
                cor_count = cor_count+1

        if(guess == correct_key).all():
            print("Successfull Key Recovery")
        else:
            print("Incorrect Key Recovery, {0}% Correct".format(cor_count/N *100))
        print("Time Taken: {0}s".format(time_taken))

        return cor_count, time_taken

    def reset_target(self):
        self.scope.io.nrst = False
        time.sleep(0.001)
        self.scope.io.nrst = True
        time.sleep(0.001)
