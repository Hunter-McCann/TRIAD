from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .modem import read_wav


def render_waterfall(
    wav_path: str | Path,
    output_path: str | Path,
    *,
    min_hz: float = 300.0,
    max_hz: float = 2700.0,
    fft_size: int = 4096,
    hop_size: int = 512,
) -> None:
    sample_rate, samples = read_wav(wav_path)
    spectrogram, freqs, times = _spectrogram(samples, sample_rate, fft_size, hop_size)
    band = (freqs >= min_hz) & (freqs <= max_hz)
    band_power = spectrogram[band, :]

    db = 20.0 * np.log10(np.maximum(band_power, 1e-12))
    top = float(np.max(db))
    floor = top - 70.0
    normalized = np.clip((db - floor) / (top - floor), 0.0, 1.0)
    waterfall = _colorize(normalized)
    waterfall = np.flipud(waterfall)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    image = Image.fromarray(waterfall, mode="RGB")
    image = image.resize((1400, 760), Image.Resampling.BICUBIC)
    image = _add_labels(image, times[-1], min_hz, max_hz)
    image.save(destination)


def _spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    fft_size: int,
    hop_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(samples) < fft_size:
        samples = np.pad(samples, (0, fft_size - len(samples)))

    window = np.hanning(fft_size)
    frame_count = 1 + (len(samples) - fft_size) // hop_size
    frames = np.lib.stride_tricks.sliding_window_view(samples, fft_size)[::hop_size]
    frames = frames[:frame_count] * window

    spectrum = np.abs(np.fft.rfft(frames, axis=1)).T
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    times = (np.arange(frame_count) * hop_size + fft_size / 2) / sample_rate
    return spectrum, freqs, times


def _colorize(values: np.ndarray) -> np.ndarray:
    stops = np.array(
        [
            [0, 0, 20],
            [0, 30, 90],
            [0, 110, 160],
            [40, 190, 120],
            [235, 220, 80],
            [255, 245, 225],
        ],
        dtype=np.float64,
    )
    scaled = values * (len(stops) - 1)
    lower = np.floor(scaled).astype(int)
    upper = np.clip(lower + 1, 0, len(stops) - 1)
    blend = scaled - lower
    rgb = stops[lower] * (1.0 - blend[..., None]) + stops[upper] * blend[..., None]
    return rgb.astype(np.uint8)


def _add_labels(image: Image.Image, max_time: float, min_hz: float, max_hz: float) -> Image.Image:
    margin_left = 95
    margin_right = 24
    margin_top = 56
    margin_bottom = 68
    canvas = Image.new(
        "RGB",
        (image.width + margin_left + margin_right, image.height + margin_top + margin_bottom),
        (12, 16, 24),
    )
    canvas.paste(image, (margin_left, margin_top))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    text = (230, 236, 245)
    grid = (75, 88, 105)

    draw.text((margin_left, 20), "TRIAD waterfall", fill=text, font=font)
    draw.text((margin_left, canvas.height - 34), "Time (seconds)", fill=text, font=font)
    draw.text((12, 20), "Hz", fill=text, font=font)

    for index in range(6):
        x = margin_left + round(index * image.width / 5)
        t = index * max_time / 5
        draw.line((x, margin_top, x, margin_top + image.height), fill=grid)
        draw.text((x - 12, margin_top + image.height + 12), f"{t:.1f}", fill=text, font=font)

    for index in range(7):
        y = margin_top + round(index * image.height / 6)
        hz = max_hz - index * (max_hz - min_hz) / 6
        draw.line((margin_left, y, margin_left + image.width, y), fill=grid)
        draw.text((22, y - 6), f"{hz:.0f}", fill=text, font=font)

    return canvas
