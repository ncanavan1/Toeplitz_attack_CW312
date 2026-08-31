#!/usr/bin/env python3
"""
Controller + correctness verification for the seeded Toeplitz/FFT privacy
amplification firmware (src/privacy_amplification/FFT/Toeplitz_FFT.c)
running on a CW312 target.

A single firmware build serves both correctness verification and SCA
trace gathering: it computes the full seeded hash, and only stage 1 of
the key FFT is triggered (trigger_high/trigger_low around DIT_FFT's
stage 1), so that's the window a capture is armed around while the
key/seed/result exchange still produces a checkable result.

The device speaks a raw binary protocol over the CW UART -- no framing,
no text, no newlines (read_bytes()/write_bytes() via getch()/putch()).
Every exchange is a single command byte followed by its payload:

    host -> target : 'k' + ROWLEN raw bytes   (load key)
    target -> host : ROWLEN raw bytes         (echo of the key, for comms
                                                integrity checking)
    host -> target : 's' + ROWLEN raw bytes   (load seed)
    target -> host : ROWLEN raw bytes         (echo of the seed)
    host -> target : 'g'                      (run the hash on the loaded
                                                key + seed)
    target -> host : ROWLEN raw bytes         (result)
    (loops forever; key and seed persist until reloaded)

Key and seed are loaded by separate commands so a capture can be armed
after both slow transfers are done and before the one-byte 'g' -- the
trigger then fires microseconds after scope.arm() instead of after a
full key/seed/echo exchange, which matters for scope.adc.stream_mode
(see tools._capture_raw()).

Bit-level (GF(2)): each byte is expanded into 8 bits, MSB-first, before
hashing, and the result's bits are packed back into bytes the same way.
The device performs, at the bit level:
    output = mod2(round(real(IFFT(FFT(key_bits) * FFT(seed_bits)))))
i.e. the circular convolution of key and seed, reduced mod 2. run_hash()
and verify() drive it; tools._capture_raw() uses load_key()/load_seed()
before scope.arm() and trigger_hash() after it to capture the triggered
key FFT (with a throwaway seed).

ROWLEN here is a byte count and MUST match the value the firmware was
built with -- there is no handshake to auto-detect it, and a mismatch
will desync the reader (which only knows how many bytes to expect per
phase, not what a valid line looks like).

Note: ROWLEN*8 (the bit-vector length) must fit in this target's 64KB of
RAM. The build needs two double-buffered float-complex FFT arrays
(8 bytes/sample) plus three int-per-bit arrays (4 bytes/bit): tested
ceiling 128 bytes (1024 bits, ~51% RAM) comfortably, 256 bytes builds but
at ~95% (works, not recommended), 512+ fails to link. A true 1024-BYTE
(8192-bit) hash needs roughly 227KB of RAM -- 3.5x more than this SAM4S
target has -- and isn't achievable with this single-FFT-in-RAM approach
on this hardware.
"""
import argparse
import time

import numpy as np
import chipwhisperer as cw
from chipwhisperer.capture.api.programmers import SAM4SProgrammer

ROWLEN = 64  # bytes; must match the firmware's build-time ROWLEN

# Single-byte protocol commands -- must match the #defines in Toeplitz_FFT.c.
CMD_LOAD_KEY = ord('k')
CMD_LOAD_SEED = ord('s')
CMD_GO = ord('g')

hex_dir = "/home/40265864@ecit.qub.ac.uk/CSIT/Toeplitz_Attack_CW312/src/privacy_amplification/FFT/"


class CommsError(Exception):
    """Raised when the device's echo doesn't match what we sent -- a UART
    desync, not an algorithm problem. Callers should drain and retry."""


def hex_path(rowlen=ROWLEN):
    """Path to the .hex for a given rowlen build, matching the makefile's
    TARGET naming (Toeplitz_FFT-ROWLEN<n>-CW312_SAM4S.hex)."""
    return hex_dir + f"Toeplitz_FFT-ROWLEN{rowlen}-CW312_SAM4S.hex"


def reprogram(scope, target, rowlen=ROWLEN):
    """Flash the firmware and wait for it to come up. Used by connect() for
    the initial flash; call it directly if you need to force the device
    back to a known, fresh state for any other reason (a comms desync
    a plain drain+retry couldn't recover from, etc.) -- verify()'s own
    retry path currently just drains and resends, it does not reprogram."""
    cw.program_target(scope, SAM4SProgrammer, hex_path(rowlen))
    # The freshly-reset target's UART/firmware needs a moment to come up;
    # writing to it immediately risks the first byte(s) landing before it's
    # ready. Give it a moment, then discard whatever startup noise is
    # sitting in the RX buffer.
    time.sleep(0.3)
    _drain(target)


def connect(rowlen=ROWLEN):
    scope = cw.scope()
    target = cw.target(scope, cw.targets.SimpleSerial)
    scope.default_setup()
    reprogram(scope, target, rowlen)
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

    One instance is reused across all reads of a single exchange (key
    echo, seed echo, result -- spanning load_key()/load_seed()/
    trigger_hash()) so spillover between those phases is handled
    correctly; a fresh instance per exchange is fine since the device
    never sends anything for the next exchange before its next command
    byte is written.
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


def _load_vector(target, cmd, name, values, reader, rowlen=ROWLEN, timeout_s=2.0):
    """Send one command byte + `values` (key or seed) and check the echo.

    The firmware echoes back what it received before it's ready for the
    next command. We check that echo against what we sent: a mismatch is a
    UART desync, not an algorithm problem -- raise CommsError so the
    caller can drain and retry rather than reporting a false hash mismatch.

    Nothing is sent for the next phase until this echo has been read back.
    Firmware's write_bytes()/putch() is a blocking, bit-banged ~38400 baud
    transmit with nothing polling getch() while it runs; bytes written too
    early pile up faster than they're serviced and the target's UART (a
    single holding register, no real FIFO) drops all but the last. Reading
    the echo is a real signal that the device is done and blocked in
    getch() again; the small sleeps are extra settling margin on top of
    that -- empirically still needed, so left in.
    """
    assert len(values) == rowlen
    time.sleep(0.1)
    _write_bytes(target, [cmd])
    _write_bytes(target, list(values))
    echo = list(reader.read_exact(rowlen, timeout_s=timeout_s))
    if echo != list(values):
        raise CommsError(f"echo mismatch (UART desync): sent {name}={list(values)}, "
                          f"device echoed {name}={echo}")


def load_key(target, key_bytes, reader, rowlen=ROWLEN, timeout_s=2.0):
    """Load (and echo-check) the key. Persists on the device until reloaded."""
    _load_vector(target, CMD_LOAD_KEY, "key", key_bytes, reader, rowlen, timeout_s)


def load_seed(target, seed_bytes, reader, rowlen=ROWLEN, timeout_s=2.0):
    """Load (and echo-check) the seed. Persists on the device until reloaded."""
    _load_vector(target, CMD_LOAD_SEED, "seed", seed_bytes, reader, rowlen, timeout_s)


def trigger_hash(target, reader, rowlen=ROWLEN, timeout_s=2.0, pre_go_sleep=0.0):
    """Send the one-byte CMD_GO and return the device's result (list of
    ints). The hash runs on whatever key/seed are currently loaded. This
    is the only host->target traffic that happens after scope.arm(), so
    the trigger fires ~immediately.

    No settling sleep by default: unlike the load steps, nothing is being
    transmitted either way when this is called (the seed echo was fully
    read by load_seed(), the device is blocked in getch()), so the
    pile-up hazard doesn't apply -- and any delay here lands between arm()
    and the trigger, which is exactly what stream_mode can't absorb. Pass
    pre_go_sleep only if a specific target proves to need it.
    """
    if pre_go_sleep:
        time.sleep(pre_go_sleep)
    _write_bytes(target, [CMD_GO])
    return list(reader.read_exact(rowlen, timeout_s=timeout_s))


def run_hash(target, key_bytes, seed_bytes, rowlen=ROWLEN, timeout_s=2.0):
    """Load key + seed, run the hash, and return the device's output_key
    (list of ints, one per byte). For a scope capture, call load_key() /
    load_seed() before scope.arm() and trigger_hash() after it instead of
    this all-in-one helper (see tools._capture_raw()).
    """
    assert len(key_bytes) == rowlen and len(seed_bytes) == rowlen
    reader = ByteReader(target)
    load_key(target, key_bytes, reader, rowlen, timeout_s)
    load_seed(target, seed_bytes, reader, rowlen, timeout_s)
    return trigger_hash(target, reader, rowlen, timeout_s)


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
                         help="skip verification; connect, program, send one key+seed, "
                              "then print everything the device sends for SECONDS as raw byte values")
    args = parser.parse_args()

    scope, target = connect(rowlen=args.rowlen)
    try:
        if args.raw is not None:
            key_bytes = gen_bytes(args.rowlen)
            seed_bytes = gen_bytes(args.rowlen)
            reader = ByteReader(target)

            print(f"sending key={key_bytes}")
            _write_bytes(target, [CMD_LOAD_KEY])
            _write_bytes(target, key_bytes)
            # Wait for the key's echo before sending the seed -- see the
            # comment on _load_vector() for why sending both immediately
            # loses the seed to a UART overrun.
            key_echo = list(reader.read_exact(args.rowlen, timeout_s=args.raw))
            print(f"key echo: {key_echo}")
            if reader.buf:
                print(f"(already buffered ahead of the seed write: {list(reader.buf)})")

            print(f"sending seed={seed_bytes}")
            _write_bytes(target, [CMD_LOAD_SEED])
            _write_bytes(target, seed_bytes)
            seed_echo = list(reader.read_exact(args.rowlen, timeout_s=args.raw))
            print(f"seed echo: {seed_echo}")
            if reader.buf:
                print(f"(already buffered ahead of the result: {list(reader.buf)})")

            print("sending go")
            _write_bytes(target, [CMD_GO])
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
