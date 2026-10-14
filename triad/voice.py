from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np


ITU_PHONETICS = {
    "A": "Alpha",
    "B": "Bravo",
    "C": "Charlie",
    "D": "Delta",
    "E": "Echo",
    "F": "Foxtrot",
    "G": "Golf",
    "H": "Hotel",
    "I": "India",
    "J": "Juliett",
    "K": "Kilo",
    "L": "Lima",
    "M": "Mike",
    "N": "November",
    "O": "Oscar",
    "P": "Papa",
    "Q": "Quebec",
    "R": "Romeo",
    "S": "Sierra",
    "T": "Tango",
    "U": "Uniform",
    "V": "Victor",
    "W": "Whiskey",
    "X": "X-ray",
    "Y": "Yankee",
    "Z": "Zulu",
    "0": "Zero",
    "1": "One",
    "2": "Two",
    "3": "Three",
    "4": "Four",
    "5": "Five",
    "6": "Six",
    "7": "Seven",
    "8": "Eight",
    "9": "Nine",
}


def station_intro_text(callsign: str, prefix: str | None = None) -> str:
    phonetics = ". ".join(callsign_to_phonetics(callsign))
    intro = (
        "TRIAD automatic environmental observation system. "
        f"{prefix.strip()}. " if prefix and prefix.strip() else
        "TRIAD automatic environmental observation system. "
    )
    return f"{intro}station identification. {phonetics}."


def callsign_to_phonetics(callsign: str) -> list[str]:
    words = []
    for character in callsign.strip().upper():
        if character in ITU_PHONETICS:
            words.append(ITU_PHONETICS[character])
    if not words:
        raise ValueError("Callsign must include at least one letter or digit")
    return words


def synthesize_voice_prompt(
    text: str,
    *,
    sample_rate: int,
    amplitude: float = 0.7,
    voice: str | None = None,
) -> np.ndarray:
    if shutil.which("say") is None or shutil.which("afconvert") is None:
        raise RuntimeError("Voice intro requires macOS 'say' and 'afconvert' commands")

    with tempfile.TemporaryDirectory(prefix="triad-voice-") as directory:
        workdir = Path(directory)
        aiff_path = workdir / "voice.aiff"
        wav_path = workdir / "voice.wav"
        say_command = ["say"]
        if voice:
            say_command.extend(["-v", voice])
        say_command.extend(["-o", str(aiff_path), text])
        subprocess.run(say_command, check=True)
        subprocess.run(
            [
                "afconvert",
                "-f",
                "WAVE",
                "-d",
                f"LEI16@{sample_rate}",
                str(aiff_path),
                str(wav_path),
            ],
            check=True,
        )
        samples = _read_any_channel_wav(wav_path)

    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    if peak > 0:
        samples = samples / peak * amplitude
    return samples


def _read_any_channel_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        if sample_width != 2:
            raise ValueError("Synthesized voice conversion did not produce 16-bit PCM")
        raw = wav.readframes(wav.getnframes())

    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm / np.iinfo(np.int16).max
