from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .voice import station_intro_text, synthesize_voice_prompt


PREAMBLE_SYMBOLS = (0, 21, 42, 63)
IONIAN_SCALE = (0, 2, 4, 5, 7, 9, 11)
DEFAULT_CHORD_SHAPES = (
    (0, 2, 4),
    (0, 2, 5),
    (0, 3, 5),
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 3),
    (0, 1, 4),
    (0, 3, 4),
)


@dataclass(frozen=True)
class TriadConfig:
    sample_rate: int = 48_000
    symbol_rate: float = 2.5
    root_hz: float = 720.0
    amplitude: float = 0.25
    gap_symbols: int = 2
    block_fade_ms: float = 12.0
    eot_duration_seconds: float = 1.5
    eot_chord_degrees: tuple[int, int, int] = (0, 2, 4)
    chord_shapes: tuple[tuple[int, int, int], ...] = DEFAULT_CHORD_SHAPES

    @property
    def chord_count(self) -> int:
        return 8 * len(self.chord_shapes)

    @property
    def samples_per_symbol(self) -> int:
        return round(self.sample_rate / self.symbol_rate)


def encode_to_wav(payload: bytes, path: str | Path, config: TriadConfig | None = None) -> None:
    config = config or TriadConfig()
    samples = encode_payload(payload, config)
    write_wav(path, samples, config.sample_rate)


def encode_ota_test_to_wav(
    path: str | Path,
    *,
    callsign: str,
    message: str | None = None,
    repeats: int = 3,
    leading_silence_seconds: float = 2.0,
    gap_seconds: float = 4.0,
    voice_intro: bool = False,
    voice_prefix: str | None = None,
    voice_name: str | None = None,
    config: TriadConfig | None = None,
) -> str:
    config = config or TriadConfig(amplitude=0.18)
    payload = _ota_test_payload(callsign, message)
    packet = encode_payload(payload.encode("ascii"), config)
    leading_silence = np.zeros(round(config.sample_rate * leading_silence_seconds), dtype=np.float64)
    gap = np.zeros(round(config.sample_rate * gap_seconds), dtype=np.float64)

    chunks: list[np.ndarray] = [leading_silence]
    if voice_intro:
        intro = synthesize_voice_prompt(
            station_intro_text(callsign, prefix=voice_prefix),
            sample_rate=config.sample_rate,
            voice=voice_name,
        )
        chunks.append(intro)
        chunks.append(np.zeros(round(config.sample_rate * 0.75), dtype=np.float64))

    for index in range(repeats):
        chunks.append(packet)
        if index != repeats - 1:
            chunks.append(gap)

    chunks.append(leading_silence)
    write_wav(path, np.concatenate(chunks), config.sample_rate)
    return payload


def decode_wav(path: str | Path, config: TriadConfig | None = None) -> bytes:
    config = config or TriadConfig()
    sample_rate, samples = read_wav(path)
    if sample_rate != config.sample_rate:
        raise ValueError(f"Expected {config.sample_rate} Hz WAV, got {sample_rate} Hz")
    return decode_payload(samples, config)


def _ota_test_payload(callsign: str, message: str | None) -> str:
    normalized_callsign = callsign.strip().upper()
    if not normalized_callsign:
        raise ValueError("A callsign is required for an OTA test payload")
    if not normalized_callsign.isascii():
        raise ValueError("Callsign must be ASCII")

    if message is None:
        message = "TRIAD OTA TEST WX:T=18.7C RH=61 P=1014.2"

    payload = f"{normalized_callsign} {message.strip()}"
    payload.encode("ascii")
    return payload


def encode_payload(payload: bytes, config: TriadConfig) -> np.ndarray:
    framed = _frame_payload(payload)
    symbols = list(PREAMBLE_SYMBOLS) + list(_bytes_to_symbols(framed))
    return synthesize_symbols(symbols, config)


def decode_payload(samples: np.ndarray, config: TriadConfig) -> bytes:
    symbols = detect_symbols(samples, config)
    start = _find_preamble(symbols)
    if start is None:
        raise ValueError("TRIAD preamble was not found")

    data_symbols = symbols[start + len(PREAMBLE_SYMBOLS) :]
    framed = _symbols_to_bytes(data_symbols)
    if len(framed) < 2:
        raise ValueError("Decoded packet is too short")

    expected_len = int.from_bytes(framed[:2], "big")
    payload = framed[2 : 2 + expected_len]
    if len(payload) != expected_len:
        raise ValueError("Decoded packet ended before the declared payload length")
    return payload


def synthesize_symbols(symbols: Iterable[int], config: TriadConfig) -> np.ndarray:
    samples_per_symbol = config.samples_per_symbol
    envelope = _block_envelope(samples_per_symbol, config)
    chunks: list[np.ndarray] = []

    for symbol in symbols:
        freqs = _symbol_frequencies(symbol, config)
        tone = np.zeros(samples_per_symbol, dtype=np.float64)

        for freq in freqs:
            phase_step = 2.0 * math.pi * freq / config.sample_rate
            phase_offsets = phase_step * np.arange(samples_per_symbol)
            tone += np.sin(phase_offsets)

        chunks.append((config.amplitude / 3.0) * tone * envelope)

    chunks.append(_synthesize_eot_chord(config))

    if config.gap_symbols:
        chunks.append(np.zeros(samples_per_symbol * config.gap_symbols, dtype=np.float64))

    return np.concatenate(chunks)


def detect_symbols(samples: np.ndarray, config: TriadConfig) -> list[int]:
    samples_per_symbol = config.samples_per_symbol
    symbol_count = len(samples) // samples_per_symbol
    symbols: list[int] = []

    for offset in range(symbol_count):
        start = offset * samples_per_symbol
        frame = samples[start : start + samples_per_symbol]
        if np.sqrt(np.mean(frame * frame)) < 0.005:
            continue
        symbols.append(_detect_symbol(frame, config))

    return symbols


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * np.iinfo(np.int16).max).astype("<i2")

    with wave.open(str(destination), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError("Only mono WAV files are supported")
        if wav.getsampwidth() != 2:
            raise ValueError("Only 16-bit PCM WAV files are supported")

        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    samples /= np.iinfo(np.int16).max
    return sample_rate, samples


def _frame_payload(payload: bytes) -> bytes:
    if len(payload) > 65_535:
        raise ValueError("Payload is too large for a TRIAD prototype frame")
    return len(payload).to_bytes(2, "big") + payload


def _bytes_to_symbols(data: bytes) -> Iterable[int]:
    accumulator = 0
    bit_count = 0

    for byte in data:
        accumulator = (accumulator << 8) | byte
        bit_count += 8
        while bit_count >= 6:
            bit_count -= 6
            yield (accumulator >> bit_count) & 0b111111

    if bit_count:
        yield (accumulator << (6 - bit_count)) & 0b111111


def _symbols_to_bytes(symbols: Iterable[int]) -> bytes:
    accumulator = 0
    bit_count = 0
    output = bytearray()

    for symbol in symbols:
        accumulator = (accumulator << 6) | (symbol & 0b111111)
        bit_count += 6
        while bit_count >= 8:
            bit_count -= 8
            output.append((accumulator >> bit_count) & 0xFF)

    return bytes(output)


def _symbol_frequencies(symbol: int, config: TriadConfig) -> tuple[float, float, float]:
    root_degree, shape_index = _symbol_to_chord_indexes(symbol, config)
    shape = config.chord_shapes[shape_index]
    return tuple(
        config.root_hz * (2.0 ** (_ionian_degree_to_semitones(root_degree + degree) / 12.0))
        for degree in shape
    )


def _block_envelope(sample_count: int, config: TriadConfig) -> np.ndarray:
    envelope = np.ones(sample_count, dtype=np.float64)
    fade_samples = round(config.sample_rate * config.block_fade_ms / 1000.0)
    fade_samples = max(0, min(fade_samples, sample_count // 2))
    if fade_samples == 0:
        return envelope

    fade = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, fade_samples))
    envelope[:fade_samples] = fade
    envelope[-fade_samples:] = fade[::-1]
    return envelope


def _synthesize_eot_chord(config: TriadConfig) -> np.ndarray:
    sample_count = round(config.sample_rate * config.eot_duration_seconds)
    envelope = _block_envelope(sample_count, config)
    tone = np.zeros(sample_count, dtype=np.float64)

    for degree in config.eot_chord_degrees:
        freq = config.root_hz * (2.0 ** (_ionian_degree_to_semitones(degree) / 12.0))
        phase_step = 2.0 * math.pi * freq / config.sample_rate
        tone += np.sin(phase_step * np.arange(sample_count))

    return (config.amplitude / 3.0) * tone * envelope


def _symbol_to_chord_indexes(symbol: int, config: TriadConfig) -> tuple[int, int]:
    if not 0 <= symbol < config.chord_count:
        raise ValueError(f"TRIAD symbol out of range: {symbol}")
    return (symbol >> 3, symbol & 0b111)


def _chord_indexes_to_symbol(root_degree: int, shape_index: int) -> int:
    return (root_degree << 3) | shape_index


def _detect_symbol(frame: np.ndarray, config: TriadConfig) -> int:
    windowed = frame * np.hanning(len(frame))
    spectrum = np.abs(np.fft.rfft(windowed))
    bin_hz = config.sample_rate / len(windowed)
    best_symbol = 0
    best_power = -math.inf

    for symbol in range(config.chord_count):
        power = sum(
            _fft_bin_power(spectrum, freq, bin_hz)
            for freq in _symbol_frequencies(symbol, config)
        )
        if power > best_power:
            best_symbol = symbol
            best_power = power

    return best_symbol


def _ionian_degree_to_semitones(degree: int) -> int:
    octave, scale_index = divmod(degree, len(IONIAN_SCALE))
    return octave * 12 + IONIAN_SCALE[scale_index]


def _fft_bin_power(spectrum: np.ndarray, target_hz: float, bin_hz: float) -> float:
    center = round(target_hz / bin_hz)
    start = max(0, center - 1)
    stop = min(len(spectrum), center + 2)
    return float(np.sum(spectrum[start:stop] ** 2))


def _find_preamble(symbols: list[int]) -> int | None:
    preamble = list(PREAMBLE_SYMBOLS)
    limit = len(symbols) - len(preamble) + 1
    for start in range(max(0, limit)):
        if symbols[start : start + len(preamble)] == preamble:
            return start
    return None
