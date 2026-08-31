#!/usr/bin/env python3
"""
Read / write / compute breakdown for the seeded Toeplitz/FFT firmware
(src/privacy_amplification/FFT/Toeplitz_FFT.c) on the CW312-SAM4S.

Non-invasive: no firmware change, no reflash. Drives the normal binary
protocol (toeplitz_fft_controller) and splits one full hash operation into
three buckets, as the firmware sees them:

  read     getch() of  'k' + key(ROWLEN) + 's' + seed(ROWLEN) + 'g'   = 2*ROWLEN+3 B
  write    putch() of  key echo + seed echo + result                  = 3*ROWLEN   B
  compute  Toeplitz_hash_fft_seeded(): 2x DIT_FFT + pointwise mul + DIT_IFFT
           (+ bytes_to_bits / bits_to_bytes)

UART is deterministic: 38400 baud, 8N1 -> 10 bit/byte -> 260.42 us/byte,
so read and write are reported at their exact theoretical values. Only
`compute` is measured: t(CMD_GO on the wire) -> t(first result byte back),
minus one byte each way, minus a per-session USB round-trip offset measured
from the key-echo path.

Usage:  python fft_timing.py [--rowlen 64] [--iters 20]
"""
import argparse
import statistics
import time

import toeplitz_fft_controller as tfc

BYTE_S = 10 / 38400.0            # 8N1: start + 8 data + stop
CPU_HZ = 7.37e6


def tight_read(target, n, timeout_s=8.0):
    """Poll with no sleep; return (data, t_first_byte, t_last_byte)."""
    buf = bytearray()
    t_first = None
    deadline = time.perf_counter() + timeout_s
    while len(buf) < n and time.perf_counter() < deadline:
        chunk = target.read(timeout=0)
        if chunk:
            t_first = t_first or time.perf_counter()
            buf += chunk.encode("latin-1")
    if len(buf) < n:
        raise TimeoutError(f"got {len(buf)}/{n} bytes")
    return bytes(buf[:n]), t_first, time.perf_counter()


def usb_offset(target, rowlen, reps=15):
    """Non-firmware round-trip overhead (USB-serial latency + scheduling),
    from the key-load echo: t(payload sent) -> t(first echo byte), with the
    known on-wire cost of (rowlen payload + 1 cmd + 1 echo byte) removed."""
    key, offs = tfc.gen_bytes(rowlen), []
    for _ in range(reps):
        time.sleep(0.05)
        tfc._write_bytes(target, [tfc.CMD_LOAD_KEY])
        t0 = time.perf_counter()
        tfc._write_bytes(target, list(key))
        _, t_first, _ = tight_read(target, rowlen)
        offs.append(max(0.0, (t_first - t0) - (rowlen + 2) * BYTE_S))
    return statistics.median(offs)


def measure(rowlen, iters):
    scope, target = tfc.connect(rowlen=rowlen)
    try:
        off = usb_offset(target, rowlen)
        comp = []
        for _ in range(iters):
            r = tfc.ByteReader(target)
            tfc.load_key(target, tfc.gen_bytes(rowlen), r, rowlen=rowlen)
            tfc.load_seed(target, tfc.gen_bytes(rowlen), r, rowlen=rowlen)
            tfc._write_bytes(target, [tfc.CMD_GO])
            t_go = time.perf_counter()
            _, t_first, _ = tight_read(target, rowlen)
            comp.append((t_first - t_go) - 2 * BYTE_S - off)

        c_m = statistics.mean(comp)
        c_sd = statistics.stdev(comp) if len(comp) > 1 else 0.0
        read_b, write_b = 2 * rowlen + 3, 3 * rowlen
        read_s, write_s = read_b * BYTE_S, write_b * BYTE_S
        total = read_s + write_s + c_m

        print(f"\nROWLEN {rowlen} B ({rowlen*8} bits)   {iters} iters   "
              f"CPU {CPU_HZ/1e6:.2f} MHz   UART 38400 8N1 ({BYTE_S*1e6:.1f} us/byte)")
        print(f"USB offset removed from compute: {off*1e3:.2f} ms\n")
        w = f"{'phase':8}{'bytes':>8}{'time':>13}{'share':>9}"
        print(w)
        print("-" * len(w))
        print(f"{'read':8}{read_b:>8}{read_s*1e3:>10.1f} ms{read_s/total:>9.1%}")
        print(f"{'write':8}{write_b:>8}{write_s*1e3:>10.1f} ms{write_s/total:>9.1%}")
        print(f"{'compute':8}{'-':>8}{c_m*1e3:>10.1f} ms{c_m/total:>9.1%}"
              f"   (+/- {c_sd*1e3:.1f} ms, ~{c_m*CPU_HZ/1e6:.1f} Mcyc)")
        print("-" * len(w))
        print(f"{'total':8}{'':>8}{total*1e3:>10.1f} ms{1.0:>9.1%}")
        print(f"\nfull-op throughput: {1/total:.2f} hash/s")
    finally:
        tfc.disconnect(scope, target)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rowlen", type=int, default=tfc.ROWLEN)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    measure(a.rowlen, a.iters)
