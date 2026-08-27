##########Contains the tools and algorithms for the template attacks####
import chipwhisperer as cw
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy import stats
import sys
import time
import scipy.signal

import toeplitz_fft_controller as tfc

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


    def allign_traces_vec(self, pattern, traces, max_shift=None):
        aligned_traces = np.zeros(traces.shape)

        for i, trace_set in enumerate(traces):
            aligned_traces[i] = self.align_traces(pattern, trace_set, max_shift)
        return aligned_traces

    def align_traces(self, pattern, traces, max_shift=None):
        aligned_traces = []

        for trace in traces:
            correlation = np.correlate(trace, pattern, mode="valid")  # Compute cross-correlation
            shift = np.argmax(correlation)
            if max_shift is not None:
            # constrain shift if desired
                shift = np.clip(shift, 0, max_shift)
            aligned_trace = np.roll(trace, -shift)  # Shift the trace
            aligned_traces.append(aligned_trace)

        return np.array(aligned_traces)



    def get_trace(self, key_bits, allignment_trace=None):
        """Capture one power trace of the triggered stage-1 key FFT for a
        key given as a bit array (length rowlen*8, MSB-first per byte --
        matches the firmware's bytes_to_bits()/the reference in
        toeplitz_fft_controller.reference_hash). Requires the target to be
        flashed with the KEY_FFT_ONLY=1 firmware (see toeplitz_fft_controller
        connect()/reprogram()).
        """
        key_bytes = np.packbits(np.asarray(key_bits, dtype=np.uint8)).tolist()
        self.scope.arm()
        tfc.send_key_only(self.target, key_bytes, rowlen=self.rowlen)
        if self.scope.capture():
            raise RuntimeError("Capture failed")
        trace = self.scope.get_last_trace()
        active_trig_count = self.scope.adc.trig_count
        print("Trig count: {0}".format(active_trig_count))

        segmented_trace = self.full_trace_segmenter(trace,len(key_bits), allignment_trace)
        return segmented_trace


    def full_trace_segmenter(self, trace, N, alligment_trace=None):
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
        seg_len = 590 - offset ##samples of a single bfu
        n_segments = int(N // 2)
        align_margin = 60

        padded_segments = []
        for n in range(n_segments):
            start = offset + seg_len * n - align_margin
            end = offset + seg_len * (n + 1) + align_margin
            if start < 0 or end > trace.shape[0]:
                raise ValueError(
                    "full_trace_segmenter: butterfly {0}'s window [{1}:{2}] falls "
                    "outside the {3}-sample trace -- configure_scope()'s `samples` "
                    "is too small for N={4} (need at least {5})".format(
                        n, start, end, trace.shape[0], N,
                        offset + seg_len * n_segments + align_margin))
            padded_segments.append(trace[start:end])

        if alligment_trace == None:
            alligment_trace = padded_segments[0]


        segments_alligned = self.align_traces(alligment_trace[170:230], padded_segments)

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


    def run_attack(self, repeats, N):
        self.reset_target()
        correct_key = np.random.randint(2,size=N)
        print(correct_key)

        for _ in range(1):
            warmup = self.get_trace(np.ones(N))

        allignment_trace = warmup[0]

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
