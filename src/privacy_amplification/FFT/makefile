# Hey Emacs, this is a -*- makefile -*-
#----------------------------------------------------------------------------
#
# Makefile for ChipWhisperer SimpleSerial-AES Program
#
#----------------------------------------------------------------------------
# On command line:
#
# make all = Make software.
#
# make clean = Clean out built project files.
#
# make coff = Convert ELF to AVR COFF.
#
# make extcoff = Convert ELF to AVR Extended COFF.
#
# make program = Download the hex file to the device, using avrdude.
#                Please customize the avrdude settings below first!
#
# make debug = Start either simulavr or avarice as specified for debugging,
#              with avr-gdb or avr-insight as the front end for debugging.
#
# make filename.s = Just compile filename.c into the assembler code only.
#
# make filename.i = Create a preprocessed source file for use in submitting
#                   bug reports to the GCC project.
#
# To rebuild project do "make clean" then "make all".
#----------------------------------------------------------------------------

# List C source files here.
# Header files (.h) are automatically pulled in.
SRC += Toeplitz_FFT.c

# -----------------------------------------------------------------------------
ifeq ($(CRYPTO_OPTIONS),)
CRYPTO_OPTIONS = AES128C
endif

SS_VER = SS_VER_1_0

# Defaults for this project's target hardware. Override on the command line
# (e.g. `make PLATFORM=CW308_STM32F3`) to build for something else; this
# firmware doesn't use any crypto code, hence CRYPTO_TARGET=NONE.
PLATFORM ?= CW312_SAM4S
CRYPTO_TARGET ?= NONE

# Key/seed size in bytes (ROWLEN*8 must be a power of two, checked at
# compile time in Toeplitz_FFT.c). Default 64 bytes (512-bit hash);
# override on the command line, e.g. `make ROWLEN=128`.
#
# Changing ROWLEN requires a rebuild - do `make clean` first when
# switching, since object files built for a different ROWLEN won't be
# recompiled by make alone (OBJDIR is keyed only on PLATFORM, not on
# ROWLEN), and a stale rebuild would silently keep the *old* binary's
# FFT length despite the new filename.
ROWLEN ?= 64
CDEFS += -DROWLEN=$(ROWLEN)

# twiddle_table.h holds DIT_FFT's precomputed twiddle factors as compile-
# time float literals (see gen_twiddle_table.py -- C has no constexpr, so
# cexp() can't run in a static initializer, hence generating it offline).
# Generated here via $(shell ...), which runs immediately as this line is
# parsed -- i.e. before any target/prerequisite is evaluated -- so the
# header unconditionally exists and matches the current $(ROWLEN) by the
# time anything tries to compile Toeplitz_FFT.c, with no dependency-
# ordering to get wrong.
$(shell python3 gen_twiddle_table.py $(ROWLEN) > twiddle_table.h)

# Target file name (without extension). Includes ROWLEN so builds for
# different key sizes don't overwrite each other's output files.
TARGET = Toeplitz_FFT-ROWLEN$(ROWLEN)

#Add simpleserial project to build
include ../simpleserial/Makefile.simpleserial

FIRMWAREPATH = ../.
include $(FIRMWAREPATH)/Makefile.inc

# The shared `clean` target above only removes files matching the *current*
# TARGET (i.e. the current ROWLEN) by name. Since TARGET encodes ROWLEN,
# sweep every ROWLEN's leftover output files and object dirs too, so a
# plain `make clean` clears everything regardless of what was last built.
.PHONY: clean-all-rowlens
clean: clean-all-rowlens
clean-all-rowlens:
	rm -f Toeplitz_FFT-ROWLEN*.hex Toeplitz_FFT-ROWLEN*.eep Toeplitz_FFT-ROWLEN*.bin \
	      Toeplitz_FFT-ROWLEN*.cof Toeplitz_FFT-ROWLEN*.elf Toeplitz_FFT-ROWLEN*.map \
	      Toeplitz_FFT-ROWLEN*.sym Toeplitz_FFT-ROWLEN*.lss
	rm -f twiddle_table.h
	rm -rf objdir-*

