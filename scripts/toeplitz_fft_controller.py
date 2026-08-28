#!/usr/bin/env python3
"""
Controller + correctness verification for the seeded Toeplitz/FFT privacy
amplification firmware (src/privacy_amplification/FFT/Toeplitz_FFT.c)
running on a CW312 target.

That C source builds into two different executables, selected at build
time by KEY_FFT_ONLY (`make KEY_FFT_ONLY=0|1`, 0 is the default) and
matched here by the `key_fft_only` argument to connect()/reprogram()/
hex_path() or the `--key-only` CLI flag:

  KEY_FFT_ONLY=0 (full hash) -- speaks a raw binary protocol over the CW
  UART, no framing, no text, no newlines (read_bytes()/write_bytes() via
  getch()/putch()):

    host   -> target : ROWLEN raw bytes   (key)
    target -> host   : ROWLEN raw bytes   (echo of the key, for comms
                                            integrity checking)
    host   -> target : ROWLEN raw bytes   (seed)
    target -> host   : ROWLEN raw bytes   (echo of the seed)
    target -> host   : ROWLEN raw bytes   (result)
    (loops forever)

  Bit-level (GF(2)): each byte is expanded into 8 bits, MSB-first, before
  hashing, and the result's bits are packed back into bytes the same way.
  The device performs, at the bit level:
    output = mod2(round(real(IFFT(FFT(key_bits) * FFT(seed_bits)))))
  i.e. the circular convolution of key and seed, reduced mod 2. run_hash()
  and verify() drive this build.

  KEY_FFT_ONLY=1 (key-FFT-only) -- same key phase, nothing else:

    host   -> target : ROWLEN raw bytes   (key)
    target -> host   : ROWLEN raw bytes   (echo of the key)
    (loops forever)

  For SCA trace gathering: the triggered key FFT (trigger_high/trigger_low
  around DIT_FFT's stage 1) is the only thing a capture is ever armed
  around, so dropping the seed/IFFT/mod2 stages shortens the per-trace
  round trip without changing what a trace actually captures. Also uses
  much less RAM (no seed_fft, no seed/output byte or bit arrays), so it
  tolerates a larger ROWLEN than the full build -- see the RAM note below.
  send_key_only() drives this build; verify() does not apply to it (there's
  no result to check against a reference).

ROWLEN here is a byte count and MUST match the value the firmware was
built with -- there is no handshake to auto-detect it, and a mismatch
will desync the reader (which only knows how many bytes to expect per
phase, not what a valid line looks like).

Note: ROWLEN*8 (the bit-vector length) must fit in this target's 64KB of
RAM. The full build needs two double-buffered float-complex FFT arrays
(8 bytes/sample) plus three int-per-bit arrays (4 bytes/bit): tested
ceiling 128 bytes (1024 bits, ~51% RAM) comfortably, 256 bytes builds but
at ~95% (works, not recommended), 512+ fails to link. The key-only build
only needs one FFT array and one bit array, roughly half the RAM: 256
bytes is the comfortable ceiling there (~44%), 512 builds tight (~82%),
1024 fails to link. A true 1024-BYTE (8192-bit) full hash needs roughly
227KB of RAM -- 3.5x more than this SAM4S target has -- and isn't
achievable with this single-FFT-in-RAM approach on this hardware, in
either build.
"""
import argparse
import time

import numpy as np
import chipwhisperer as cw
from chipwhisperer.capture.api.programmers import SAM4SProgrammer

ROWLEN = 16  # bytes; must match the firmware's build-time ROWLEN
KEY_FFT_ONLY = False  # must match the firmware's build-time KEY_FFT_ONLY

hex_dir = "/home/40265864@ecit.qub.ac.uk/CSIT/Toeplitz_Attack_CW312/src/privacy_amplification/FFT/"


class CommsError(Exception):
    """Raised when the device's echo doesn't match what we sent -- a UART
    desync, not an algorithm problem. Callers should drain and retry."""


def hex_path(rowlen=ROWLEN, key_fft_only=KEY_FFT_ONLY):
    """Path to the .hex for a given (rowlen, key_fft_only) build, matching
    the makefile's TARGET naming (Toeplitz_FFT-ROWLEN<n>-FULL or
    -KEYONLY-CW312_SAM4S.hex)."""
    mode = "KEYONLY" if key_fft_only else "FULL"
    return hex_dir + f"Toeplitz_FFT-ROWLEN{rowlen}-{mode}-CW312_SAM4S.hex"


def reprogram(scope, target, rowlen=ROWLEN, key_fft_only=KEY_FFT_ONLY):
    """Flash the firmware and wait for it to come up. Used by connect() for
    the initial flash; call it directly if you need to force the device
    back to a known, fresh state for any other reason (a comms desync
    a plain drain+retry couldn't recover from, etc.) -- verify()'s own
    retry path currently just drains and resends, it does not reprogram."""
    cw.program_target(scope, SAM4SProgrammer, hex_path(rowlen, key_fft_only))
    # The freshly-reset target's UART/firmware needs a moment to come up;
    # writing to it immediately risks the first byte(s) landing before it's
    # ready. Give it a moment, then discard whatever startup noise is
    # sitting in the RX buffer.
    time.sleep(0.3)
    _drain(target)


def connect(rowlen=ROWLEN, key_fft_only=KEY_FFT_ONLY):
    scope = cw.scope()
    target = cw.target(scope, cw.targets.SimpleSerial)
    scope.default_setup()
    reprogram(scope, target, rowlen, key_fft_only)
    return scope, target


def _drain(target, duration_s=0.1, poll_s=0.02):
    """Discard whatever is currently arriving on the target's UART."""
    deadline = time.time() + duration_s
    while time.time() < deadline:
        target.read(timeout=int(poll_s * 1000))


def disconnect(scope, target):
    target.dis()
    scope.dis()


def gen_bytes(n, rng=None):
    rng = rng or np.random.default_rng()
    return rng.integers(0, 256, size=n, dtype=np.uint8).tolist()


def _write_bytes(target, byte_values):
    # target.write() converts a plain list of ints to a bytearray
    # internally -- this is the supported way to send raw binary data
    # rather than a text string.
    target.write(list(byte_values))


class ByteReader:
    """Reads exact byte counts off `target`, buffering anything read past
    what was asked for instead of discarding it. A single low-level
    target.read() call returns whatever's currently available, which can
    legitimately span into the next logical message if e.g. the key's
    echo and the start of the seed's echo arrive close together -- without
    carrying that spillover forward, those bytes would either vanish or
    get misattributed to the wrong phase.

    One instance is reused across all reads within a single run_hash()
    call (key echo, seed echo, result) so spillover between those phases
    is handled correctly; a fresh instance per run_hash() call is fine
    since the device never sends anything for the next exchange before
    its next key is written.
    """
    def __init__(self, target):
        self.target = target
        self.buf = bytearray()

    def read_exact(self, n, timeout_s=2.0, poll_s=0.02):
        """Return exactly `n` raw bytes as a `bytes` object."""
        deadline = time.time() + timeout_s
        while len(self.buf) < n and time.time() < deadline:
            # target.read() returns a latin-1 decoded str (each char maps
            # 1:1 to a byte value 0-255); re-encode back to raw bytes --
            # latin-1 round-trips every byte value losslessly, unlike utf-8.
            chunk = self.target.read(timeout=int(poll_s * 1000))
            if chunk:
                self.buf += chunk.encode('latin-1')
            else:
                time.sleep(poll_s)
        if len(self.buf) < n:
            raise TimeoutError(f"only received {len(self.buf)}/{n} bytes within "
                                f"{timeout_s}s; got: {list(self.buf)}")
        result = bytes(self.buf[:n])
        del self.buf[:n]
        return result


def run_hash(target, key_bytes, seed_bytes, rowlen=ROWLEN, timeout_s=2.0):
    """Send key + seed to the device and return its computed output_key
    (as a list of ints, one per byte).

    The firmware echoes back both the key and seed it received before
    computing the result. We check those echoes against what we actually
    sent: if they don't match, the UART desynced and the result that
    follows can't be trusted -- raise CommsError so the caller can drain
    and retry rather than reporting a false algorithm mismatch.

    The seed is deliberately NOT sent until the key's echo has actually
    been read back. Firmware's write_bytes()/putch() is a blocking,
    bit-banged ~38400 baud transmit with nothing polling getch() while it
    runs; if the seed's bytes are written too early, they pile up faster
    than they're serviced and the target's UART (a single holding
    register, no real FIFO) drops all but whatever's left when the CPU
    finally checks. Waiting for the key's echo is a real signal that the
    device is done with the key and blocked in getch() again. The small
    sleeps around each write are extra settling margin on top of that
    signal -- empirically still needed even with the echo-wait, so left in
    rather than assumed away.
    """
    assert len(key_bytes) == rowlen and len(seed_bytes) == rowlen
    reader = ByteReader(target)

    time.sleep(0.1)
    _write_bytes(target, key_bytes)
    key_echo = list(reader.read_exact(rowlen, timeout_s=timeout_s))
    if key_echo != list(key_bytes):
        raise CommsError(f"echo mismatch (UART desync): sent key={key_bytes}, "
                          f"device echoed key={key_echo}")

    time.sleep(0.1)
    _write_bytes(target, seed_bytes)
    time.sleep(0.1)
    seed_echo = list(reader.read_exact(rowlen, timeout_s=timeout_s))
    if seed_echo != list(seed_bytes):
        raise CommsError(f"echo mismatch (UART desync): sent seed={seed_bytes}, "
                          f"device echoed seed={seed_echo}")

    result = list(reader.read_exact(rowlen, timeout_s=timeout_s))
    return result


def send_key_only(target, key_bytes, rowlen=ROWLEN, timeout_s=2.0):
    """Send a key to a device flashed with the KEY_FFT_ONLY=1 firmware and
    confirm its echo -- there's no seed and no result in this build (see
    the KEY_FFT_ONLY comment in Toeplitz_FFT.c), just the triggered key
    FFT running between this write and the next one. This is the per-trace
    primitive for the trace-gathering phase: arm the scope's capture, call
    this, then read back the trace. Raises CommsError on an echo mismatch
    (a UART desync), same as run_hash() -- the caller should drain and
    retry rather than trust whatever trace was captured during a desynced
    exchange.
    """
    assert len(key_bytes) == rowlen
    reader = ByteReader(target)
    time.sleep(0.1)
    _write_bytes(target, key_bytes)
    key_echo = list(reader.read_exact(rowlen, timeout_s=timeout_s))
    if key_echo != list(key_bytes):
        raise CommsError(f"echo mismatch (UART desync): sent key={key_bytes}, "
                          f"device echoed key={key_echo}")


def reference_hash(key_bytes, seed_bytes):
    """Independent numpy computation of the same algorithm the firmware
    runs: each byte is expanded to 8 bits MSB-first (matching the
    firmware's bytes_to_bits()), circular convolution of the two bit
    vectors via FFT, mod 2, then packed back into bytes MSB-first
    (matching bits_to_bytes()). np.unpackbits/packbits default to
    bitorder='big', i.e. MSB-first, so no manual bit-twiddling needed."""
    key_bits = np.unpackbits(np.asarray(key_bytes, dtype=np.uint8)).astype(float)
    seed_bits = np.unpackbits(np.asarray(seed_bytes, dtype=np.uint8)).astype(float)
    conv = np.fft.ifft(np.fft.fft(key_bits) * np.fft.fft(seed_bits))
    result_bits = (np.round(conv.real).astype(int) % 2).astype(np.uint8)
    return np.packbits(result_bits).tolist()


def shannon_entropy_bits_per_symbol(byte_values, alphabet_size=256):
    """Order-0 Shannon entropy of `byte_values` (an iterable of ints in
    [0, alphabet_size)), in bits per symbol. Max is log2(alphabet_size)
    (8.0 for bytes), achieved only when every symbol value that appears is
    equally frequent -- this says nothing about higher-order structure
    (e.g. correlations between positions), just the flatness of the
    single-byte value distribution."""
    byte_values = np.asarray(byte_values, dtype=np.int64)
    if byte_values.size == 0:
        return 0.0
    counts = np.bincount(byte_values, minlength=alphabet_size)
    probs = counts[counts > 0] / byte_values.size
    return max(0.0, float(-np.sum(probs * np.log2(probs))))  # avoid -0.0 on a constant input


def verify(target, rowlen=ROWLEN, trials=100, seed=None, timeout_s=2.0, verbose=True, max_retries=5):
    """Run `trials` random key/seed byte-array pairs through the device
    and check its output against the independent numpy reference. Returns
    True iff every trial matched with no unresolved comms errors.

    A trial whose echoes don't match what was sent (CommsError) or that
    times out is a UART desync, not evidence the algorithm is wrong -- it's
    drained and retried up to `max_retries` times before being counted as
    a comms failure, kept separate from genuine hash mismatches.

    Output bytes from trials that verified correct (device matched the
    reference) are pooled and checked for entropy: reported against the
    theoretical max for that much data (8 bits/byte), as a sanity check
    that the hash's output looks like it's actually mixing the input
    rather than e.g. collapsing to mostly-constant bytes. This is only an
    order-0, single-byte-distribution check (see
    shannon_entropy_bits_per_symbol) -- it can't catch subtler structure,
    and isn't a substitute for a real statistical test suite (NIST STS,
    dieharder, etc.) if that level of rigor is ever needed.
    """
    rng = np.random.default_rng(seed)
    failures = []
    comms_failures = []
    verified_output_bytes = []
    for i in range(trials):
        key_bytes = gen_bytes(rowlen, rng)
        seed_bytes = gen_bytes(rowlen, rng)
        expected_out = reference_hash(key_bytes, seed_bytes)

        device_out = None
        last_err = None
        for attempt in range(max_retries):
            try:
                device_out = run_hash(target, key_bytes, seed_bytes, rowlen, timeout_s)
                break
            except (CommsError, TimeoutError) as e:
                last_err = e
                if verbose:
                    print(f"[{i+1}/{trials}] comms error (attempt {attempt+1}/{max_retries}): {e}")
                _drain(target, 0.2)

        if device_out is None:
            comms_failures.append((i, key_bytes, seed_bytes, last_err))
            if verbose:
                print(f"[{i+1}/{trials}] COMMS FAILURE after {max_retries} attempts, skipping trial")
            continue

        ok = device_out == expected_out
        if not ok:
            failures.append((i, key_bytes, seed_bytes, device_out, expected_out))
        else:
            verified_output_bytes.extend(device_out)
        if verbose:
            status = "OK" if ok else "MISMATCH"
            print(f"[{i+1}/{trials}] {status}  key={key_bytes} seed={seed_bytes} "
                  f"device={device_out} expected={expected_out}")

    if verbose:
        passed = trials - len(failures) - len(comms_failures)
        print(f"\n{passed}/{trials} trials passed "
              f"({len(failures)} hash mismatches, {len(comms_failures)} unresolved comms failures)")
        for i, key_bytes, seed_bytes, device_out, expected_out in failures:
            print(f"  hash mismatch, trial {i}: key={key_bytes} seed={seed_bytes} "
                  f"device={device_out} expected={expected_out}")
        for i, key_bytes, seed_bytes, err in comms_failures:
            print(f"  comms failure, trial {i}: key={key_bytes} seed={seed_bytes} ({err})")

        n_bytes = len(verified_output_bytes)
        max_entropy_bits = 8.0 * n_bytes
        if n_bytes > 0:
            bits_per_byte = shannon_entropy_bits_per_symbol(verified_output_bytes)
            calculated_entropy_bits = bits_per_byte * n_bytes
            print(f"\nEntropy check over {n_bytes} output bytes from verified-correct trials:")
            print(f"  calculated: {calculated_entropy_bits:.2f} bits  "
                  f"({bits_per_byte:.4f} bits/byte)")
            print(f"  max:        {max_entropy_bits:.2f} bits  (8.0 bits/byte)")
            # Rule of thumb: an order-0 byte-value entropy estimate needs
            # several samples per possible value (256 of them) to converge
            # anywhere near the true value -- with too few bytes it reads
            # artificially low even for genuinely well-mixed output, which
            # would misread as a broken hash rather than an undersampled
            # estimate.
            min_reliable_bytes = 10 * 256
            if n_bytes < min_reliable_bytes:
                print(f"  note: only {n_bytes} bytes sampled over a 256-value alphabet -- "
                      f"below ~{min_reliable_bytes} needed for this estimate to be reliable; "
                      f"a low number here is likely undersampling, not evidence of a weak hash. "
                      f"Increase --trials and/or --rowlen for a trustworthy reading.")
        else:
            print("\nEntropy check: no verified-correct trials, nothing to measure "
                  f"(max for 0 bytes: {max_entropy_bits:.2f} bits)")

    return len(failures) == 0 and len(comms_failures) == 0


def monitor_raw(target, duration_s=10.0, poll_s=0.05):
    """Print everything the target sends as raw byte values (the protocol
    is binary now, not text, so printing characters directly would just
    show garbled control codes for most of the 0-255 range) for
    `duration_s` seconds."""
    print(f"--- raw monitor for {duration_s}s (Ctrl+C to stop early) ---")
    deadline = time.time() + duration_s
    total = bytearray()
    try:
        while time.time() < deadline:
            chunk = target.read(timeout=int(poll_s * 1000))
            if chunk:
                data = chunk.encode('latin-1')
                total += data
                print(list(data), end=" ", flush=True)
            else:
                time.sleep(poll_s)
    except KeyboardInterrupt:
        pass
    print(f"\n--- end raw monitor ({len(total)} bytes total) ---")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rowlen", type=int, default=ROWLEN,
                         help="key/seed size in BYTES; must match the firmware's build-time ROWLEN")
    parser.add_argument("--trials", type=int, default=10,
                         help="number of random key/seed pairs to verify")
    parser.add_argument("--seed", type=int, default=None,
                         help="seed for the RNG that generates test key/seed pairs (for reproducibility)")
    parser.add_argument("--timeout", type=float, default=10.0,
                         help="per-trial UART read timeout in seconds")
    parser.add_argument("--retries", type=int, default=5,
                         help="max retries for a trial after a UART comms desync before giving up on it")
    parser.add_argument("--raw", type=float, default=None, metavar="SECONDS",
                         help="skip verification; connect, program, send one key(+seed), "
                              "then print everything the device sends for SECONDS as raw byte values")
    parser.add_argument("--key-only", action="store_true",
                         help="target is flashed with the KEY_FFT_ONLY=1 firmware (see "
                              "Toeplitz_FFT.c) -- only a triggered key FFT, no seed or result. "
                              "Requires --raw, since there's no hash result to verify against a "
                              "reference with this firmware; use send_key_only() directly for "
                              "trace gathering.")
    args = parser.parse_args()

    if args.key_only and args.raw is None:
        parser.error("--key-only requires --raw (there's no result to verify with this firmware)")

    scope, target = connect(rowlen=args.rowlen, key_fft_only=args.key_only)
    try:
        if args.key_only:
            key_bytes = gen_bytes(args.rowlen)
            print(f"sending key={key_bytes}")
            reader = ByteReader(target)
            _write_bytes(target, key_bytes)
            key_echo = list(reader.read_exact(args.rowlen, timeout_s=args.raw))
            print(f"key echo: {key_echo}")
            if reader.buf:
                print(f"(already buffered ahead of the next key: {list(reader.buf)})")
            monitor_raw(target, duration_s=args.raw)
            ok = True
        elif args.raw is not None:
            key_bytes = gen_bytes(args.rowlen)
            seed_bytes = gen_bytes(args.rowlen)
            print(f"sending key={key_bytes}")
            _write_bytes(target, key_bytes)
            # Wait for the key's echo before sending the seed -- see the
            # comment on run_hash() for why sending both immediately loses
            # the seed to a UART overrun.
            reader = ByteReader(target)
            key_echo = list(reader.read_exact(args.rowlen, timeout_s=args.raw))
            print(f"key echo: {key_echo}")
            if reader.buf:
                print(f"(already buffered ahead of the seed write: {list(reader.buf)})")

            print(f"sending seed={seed_bytes}")
            _write_bytes(target, seed_bytes)
            seed_echo = list(reader.read_exact(args.rowlen, timeout_s=args.raw))
            print(f"seed echo: {seed_echo}")
            if reader.buf:
                print(f"(already buffered ahead of the key write: {list(reader.buf)})")

            monitor_raw(target, duration_s=args.raw)
            ok = True
        else:
            ok = verify(target, rowlen=args.rowlen, trials=args.trials,
                        seed=args.seed, timeout_s=args.timeout, max_retries=args.retries)
    finally:
        disconnect(scope, target)

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
