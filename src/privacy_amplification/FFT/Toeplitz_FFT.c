#include "hal.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <time.h>

// Single-byte protocol commands (see toeplitz_fft_controller.py). Key and
// seed are loaded by their own commands and kept in memory; CMD_GO then
// runs the hash on whatever is currently loaded. This lets the host
// finish every slow UART transfer *before* it arms the capture and send
// only CMD_GO afterwards, so the trigger fires within microseconds of the
// arm instead of after a full key+seed+echo exchange (a delay long enough
// to break scope.adc.stream_mode captures -- "no trigger seen").
#define CMD_LOAD_KEY  'k'
#define CMD_LOAD_SEED 's'
#define CMD_GO        'g'

#define PI 3.1415926535897932384626433832795  // Value of Pi
#define BUFLEN 64

// Key/seed size in BYTES. Set at build time with `make ROWLEN=<n>`; changing
// it requires a rebuild since the FFT length is baked into the binary.
// The Toeplitz hash itself is bit-level (GF(2)), so each byte is expanded
// into 8 individual bits before hashing -- ROWBITS is that bit-vector
// length, and it's what actually needs to be a power of two for the
// radix-2 DIT FFT.
// The default is 64 bytes (512-bit hash). 128 bytes (1024 bits) is a
// tested, comfortable ceiling for this SAM4S target's 64KB of RAM (~51%
// used); 256 bytes builds but leaves under 5% headroom, and 512+ fails to
// link outright -- see the RAM comment on Toeplitz_hash_fft_seeded.
#ifndef ROWLEN
#define ROWLEN 64
#endif

#define ROWBITS (ROWLEN * 8)

#if (ROWBITS & (ROWBITS - 1)) != 0
#error "ROWLEN*8 must be a power of two (required by the radix-2 DIT FFT)"
#endif

// One build serves both purposes: the full Toeplitz hash (key FFT, seed
// FFT, pointwise multiply, IFFT, mod2) produces a real, correct
// privacy-amplification key for verification, and only stage 1 of the
// key FFT is triggered, so the same firmware is what SCA traces are
// captured from (see the `trigger` comment on DIT_FFT).

uint8_t memory[BUFLEN];
uint8_t tmp[BUFLEN];
char asciibuf[BUFLEN];
uint8_t pt[16];

static void delay_2_ms(void);


static void delay_2_ms()
{
  for (volatile unsigned int i=0; i < 0xfff; i++ ){
    ;
  }
}

// Raw binary byte I/O for the key/seed/result protocol -- no framing, no
// text. getch() blocks until a real byte is received (confirmed from the
// SAM4S HAL: uart_read() itself spins until data is available, so unlike
// the old line-based my_read() there's no "0 means not ready yet" sentinel
// to guard against; every byte returned is genuine received data,
// including a legitimate 0x00).
void read_bytes(uint8_t *buf, int len){
  for(int i = 0; i < len; i++){
    buf[i] = getch();
  }
}

void write_bytes(const uint8_t *buf, int len){
  for(int i = 0; i < len; i++){
    putch(buf[i]);
  }
}

// MSB-first: bit 7 of bytes_in[i] becomes bits_out[i*8], bit 0 becomes
// bits_out[i*8+7]. Must match bits_to_bytes()'s packing order exactly, and
// the Python controller's unpacking convention.
void bytes_to_bits(const uint8_t *bytes_in, int *bits_out, int nbytes){
  for(int i = 0; i < nbytes; i++){
    for(int b = 0; b < 8; b++){
      bits_out[i*8 + b] = (bytes_in[i] >> (7 - b)) & 1;
    }
  }
}

void bits_to_bytes(const int *bits_in, uint8_t *bytes_out, int nbytes){
  for(int i = 0; i < nbytes; i++){
    uint8_t byte = 0;
    for(int b = 0; b < 8; b++){
      byte = (byte << 1) | (bits_in[i*8 + b] & 1);
    }
    bytes_out[i] = byte;
  }
}

void long_delay(){
  for(int i = 0; i < 1000; i++){
    asm volatile(
    "nop"       "\n\t"
    "nop"       "\n\t"
    "nop"       "\n\t"
    "nop"       "\n\t"
    "nop"       "\n\t"
    "nop"       "\n\t"
    "nop"       "\n\t"
    "nop"       "\n\t"
    "nop"       "\n\t"
    "nop"       "\n\t"
    ::
    );
  }
}


////////////////////////
/// FFT STUFF //////////


void pad_key(complex float *padded_key, int *key, int padLen, int keyLen){
  for(int i = 0; i < keyLen; i++){
    padded_key[i] = key[i] + 0*I;
  }
  for(int i = keyLen; i < padLen; i++){
    padded_key[i] = 0 + 0*I;
  }
}


void conj_array(float complex *arr, int N){
  for(int i = 0; i < N; i++){
    arr[i] = conj(arr[i]);
  }
}


int reverse_bit(int num, int s) {
    int res = 0;
    for (int i = 0; i < s; i++) {
        if (num & (1 << i))
            res |= 1 << (s - 1 - i);
    }
    return res;
}

void reverse_array(float complex *X, int N, int s){
    for(int i = 0; i < N; i ++){
        int rb = reverse_bit(i,s);
        if(i < rb){
          float complex tmp = X[i];
            X[i] = X[rb];
            X[rb] = tmp;
        }
    }
}


// Precomputed twiddle table, indexed as table[n * (N/span)] for a stage
// with span = 2^stage: exp(-2*pi*i*n/span) == exp(-2*pi*i*(n*(N/span))/N),
// so a single length-N/2 table (the N-point DFT's twiddle factors) covers
// every stage via a stride lookup, rather than calling cexp() (needing
// soft-emulated sin/cos on this FPU-less target) on every butterfly.
// `static const` here means the values live in .rodata (flash), not
// .bss/.data (RAM) -- verified against the actual toolchain. The table is
// computed offline by gen_twiddle_table.py (C has no constexpr, so cexp()
// can't run in a static initializer) and regenerated for the current
// ROWLEN on every build; see twiddle_table.h.
#include "twiddle_table.h"

//https://github.com/Swati-Verma671/Computation-of-DFT-using-Radix-2-DIT-FFT-algorithm/blob/main/code.m
// `trigger`: when nonzero, raises the CW trigger for the duration of stage 1
// of the butterfly (dropping it again once stage 2 begins). Callers pass 0
// for every FFT call that should NOT be visible in a power trace (the seed's
// FFT, and the FFT performed internally by DIT_IFFT). Only stage 1 of the
// key FFT is ever captured; the remaining log2(N)-1 stages still run -- they
// are needed for a correct hash result -- but never appear in a trace.
void DIT_FFT(float complex *X, int N, int trigger){

    //int collected = 0;

    int s = round(log2(N));
    reverse_array(X,N,s);
    if(trigger){
      trigger_high();
    }
    for(int stage=1; stage <= s; stage++){
        // Powers of two, computed with bit-shifts rather than pow(2,...):
        // this target has no hardware FPU (-mfloat-abi=soft), so pow() is
        // a software-emulated transcendental call, and the old code was
        // calling it (plus a redundant one inside the while-condition,
        // evaluated on every iteration) tens of thousands of times per
        // hash at larger N -- easily seconds of wasted time for what's
        // always just an integer power of two.
        int half_span = 1 << (stage - 1);
        int span = 1 << stage;
        int table_stride = N >> stage;  // N/span
        int p = 0;
        int q = 0 + half_span;
        int n = 0;
        while (n <= half_span - 1 && q <= N)
        {

            if (stage == 2 && trigger){
              trigger_low();
            }

            float complex w = twiddle_table[n * table_stride];

            float complex y = X[p];
            float complex z = X[q];


            // for(int i = 0; i < 20; i++){
            //     __asm__ volatile ("nop");
            // }

            z *= w;
            X[p] = y+z;
            X[q] = y-z;

            p++;
            q++;
            n++;
            if(q % span == 0){
                p = p + half_span;
                q = q + half_span;
                n=0;
            }
        }
    }
}

void DIT_IFFT(float complex *X, int N){
    conj_array(X, N);
    DIT_FFT(X,N,0);
    conj_array(X,N);

    for(int i = 0; i < N; i++){
      X[i] = X[i]/N;
    }
}



// Seeded variant: instead of deriving the circulant from a Toeplitz row/col
// spec, a random seed of the same length as the key is FFT'd and pointwise
// multiplied with the key's FFT (circular convolution of key and seed),
// then IFFT'd and reduced mod 2. Only the key's FFT is triggered so power
// traces capture leakage from the key-dependent NTT butterfly and nothing
// else -- this one build serves both SCA trace gathering and correctness
// verification.
void Toeplitz_hash_fft_seeded(int *input_key, int *seed, int *output_key, int N){
    // Fixed-size, not malloc'd: ROWBITS is a compile-time constant and N is
    // always ROWBITS in this firmware, so there's no need for heap
    // allocation here. static so these live in .bss, not on the stack --
    // at large ROWLEN (e.g. 1024 bytes = 8192 bits) two ROWBITS-sized
    // float complex arrays (8 bytes each) don't come close to fitting in
    // this target's 4KB stack, but comfortably fit in its 64KB of RAM.
    // Both are fully overwritten before being read on every call, so not
    // re-zeroing between calls is harmless.
    static float complex key_fft[ROWBITS];
    static float complex seed_fft[ROWBITS];

    for(int i = 0; i < N; i++){
      key_fft[i]  = input_key[i] + 0*I;
      seed_fft[i] = seed[i] + 0*I;
    }

    DIT_FFT(key_fft, N, 1);   // triggered: key NTT (only stage 1 is captured)
    DIT_FFT(seed_fft, N, 0);  // untriggered: seed NTT

    for(int i = 0; i < N; i++){
      key_fft[i] = key_fft[i] * seed_fft[i];
    }

    DIT_IFFT(key_fft, N);     // untriggered

    for(int i = 0; i < N; i++){
      int bit_rounded = round(creal(key_fft[i]));
      output_key[i] = ((bit_rounded % 2) + 2) % 2;
    }
}

//////////////////////////
/// END FFT STUFF ////////

int main(void)
{
    platform_init();
    init_uart();
    trigger_setup();

    // static: see the comment on key_fft/seed_fft in
    // Toeplitz_hash_fft_seeded -- these don't fit on the stack at large
    // ROWLEN, and are fully overwritten each iteration regardless.
    // input_key_bits/seed_bits are zero-initialised (static), so a CMD_GO
    // issued before anything is loaded hashes all-zero inputs rather than
    // reading uninitialised memory.
    static uint8_t input_key_bytes[ROWLEN];
    static int input_key_bits[ROWBITS];
    static uint8_t seed_bytes[ROWLEN];
    static uint8_t output_bytes[ROWLEN];
    static int seed_bits[ROWBITS];
    static int output_bits[ROWBITS];

    while(1){
      int cmd = getch();
      switch(cmd){

        case CMD_LOAD_KEY:
          read_bytes(input_key_bytes, ROWLEN);
          write_bytes(input_key_bytes, ROWLEN);  // echo, so the host can detect a UART desync
          bytes_to_bits(input_key_bytes, input_key_bits, ROWLEN);
          break;

        case CMD_LOAD_SEED:
          read_bytes(seed_bytes, ROWLEN);
          write_bytes(seed_bytes, ROWLEN);  // echo
          bytes_to_bits(seed_bytes, seed_bits, ROWLEN);
          break;

        case CMD_GO:
          // Nothing slow between here and trigger_high() (raised inside
          // DIT_FFT): just the getch() that already returned plus the
          // float-complex load and bit-reversal. Arm the capture, send
          // this one byte, and the trigger follows immediately.
          Toeplitz_hash_fft_seeded(input_key_bits, seed_bits, output_bits, ROWBITS);
          bits_to_bytes(output_bits, output_bytes, ROWLEN);
          write_bytes(output_bytes, ROWLEN);  // result
          break;

        default:
          // Unknown byte (startup noise, a desync): ignore it rather than
          // consuming ROWLEN bytes behind it and shifting the protocol.
          break;
      }
    }
    return 1;
  }


