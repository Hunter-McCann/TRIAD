# TRIAD

TRIAD is an experimental amateur-radio digital mode for automatic weather observation.
It represents data as simultaneous three-note chords chosen from an Ionian
major scale. A total of three tones start together, move together, and appear as three tracks on a waterfall.

This repository contains a first Python prototype:

- `triad/modem.py` generates and decodes TRIAD audio.
- `triad/cli.py` provides a small command-line interface.
- `examples/demo.py` creates a sample packet and decodes it again.

## Design Defaults

- Ionian tonic: `720 Hz`
- Scale: Ionian / major scale, using equal-temperament intervals
- Chord roots: 8 scale positions from tonic through octave
- Chord shapes: 8 simultaneous block voicings, including major and minor colors
- Symbol alphabet: `8 * 8 = 64` chords, or `6 bits/symbol`
- Symbol rate: `2.5 baud`, so each chord lasts `0.4 seconds`
- Preamble: four fixed TRIAD symbols
- End marker: a `1.5 second` tonic major chord
- Modulation: block chord FSK with a short fade on each chord

## Quick Start

Install NumPy if needed:

```bash
python3 -m pip install numpy
```

For system-wide use from any folder, install this project in editable mode from
the repository root:

```bash
python3 -m pip install -e .
```

Decode a file:

```bash
python3 -m triad.cli decode out/triad.wav
```

Render a lab waterfall image from the WAV:

```bash
python3 -m triad.cli waterfall out/triad.wav out/triad_waterfall.png
```

Generate a repeated low-level OTA test WAV with callsign-bearing packets:

```bash
python3 -m triad.cli ota-test out/triad_ota_test.wav --callsign KO6NGX
```

Add a synthesized voice lead-in before the TRIAD packet:

```bash
python3 -m triad.cli ota-test out/triad_ota_test.wav --callsign KO6NGX --voice-intro
```

## Notes

TRIAD is currently being shaped for NFM operation, so the present prototype
should be treated as generated audio to feed into an FM transmitter path rather
than an SSB-specific waveform. The important next checks are occupied audio
bandwidth, FM deviation, receiver limiter behavior, and whether the decoder can
survive realistic NFM noise and squelch edges.

This is a lab prototype, not recommended for extended over-the-air (OTA) use. The decoder assumes known
symbol timing and clean audio. Good next steps are pilot-aided drift correction, interleaving, and a stronger FEC layer.
