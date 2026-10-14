import math
from pathlib import Path
import unittest

from triad.modem import (
    IONIAN_SCALE,
    PREAMBLE_SYMBOLS,
    TriadConfig,
    decode_payload,
    detect_symbols,
    encode_payload,
    encode_ota_test_to_wav,
    read_wav,
    _symbol_frequencies,
)
from triad.voice import station_intro_text


class TriadModemTests(unittest.TestCase):
    def test_payload_round_trip(self) -> None:
        config = TriadConfig(symbol_rate=20.0)
        payload = b"WX:T=18.7C RH=61 P=1014.2"

        samples = encode_payload(payload, config)

        self.assertEqual(decode_payload(samples, config), payload)

    def test_preamble_is_detectable(self) -> None:
        config = TriadConfig(symbol_rate=20.0)
        samples = encode_payload(b"ABC", config)

        symbols = detect_symbols(samples, config)

        self.assertEqual(tuple(symbols[:4]), PREAMBLE_SYMBOLS)

    def test_custom_root_round_trip(self) -> None:
        config = TriadConfig(root_hz=660.0, symbol_rate=20.0)
        payload = b"WX:WIND=220/11"

        samples = encode_payload(payload, config)

        self.assertEqual(decode_payload(samples, config), payload)

    def test_default_chord_duration_is_slow_block(self) -> None:
        config = TriadConfig()

        self.assertAlmostEqual(config.samples_per_symbol / config.sample_rate, 0.4)

    def test_transmission_includes_end_major_chord(self) -> None:
        payload = b"WX"
        with_eot = TriadConfig(eot_duration_seconds=1.5, gap_symbols=0)
        without_eot = TriadConfig(eot_duration_seconds=0.0, gap_symbols=0)

        extra_samples = len(encode_payload(payload, with_eot)) - len(
            encode_payload(payload, without_eot)
        )

        self.assertEqual(extra_samples, round(with_eot.sample_rate * 1.5))

    def test_symbols_are_unique_ionian_chords(self) -> None:
        config = TriadConfig()
        chords = {
            tuple(round(freq, 3) for freq in _symbol_frequencies(symbol, config))
            for symbol in range(config.chord_count)
        }

        self.assertEqual(len(chords), 64)
        for chord in chords:
            self.assertEqual(len(chord), 3)
            for freq in chord:
                semitones = round(12 * math.log2(freq / config.root_hz))
                self.assertIn(semitones % 12, IONIAN_SCALE)
                self.assertGreaterEqual(freq, 300)
                self.assertLessEqual(freq, 2700)

    def test_ota_test_wav_contains_callsign_payload(self) -> None:
        path = Path("/tmp/triad_ota_test.wav")
        config = TriadConfig(symbol_rate=20.0, amplitude=0.18)

        payload = encode_ota_test_to_wav(
            path,
            callsign="KO6NGX",
            message="TRIAD OTA TEST",
            repeats=1,
            leading_silence_seconds=0.1,
            gap_seconds=0.1,
            config=config,
        )
        sample_rate, samples = read_wav(path)

        self.assertEqual(sample_rate, config.sample_rate)
        self.assertEqual(payload, "KO6NGX TRIAD OTA TEST")
        self.assertIn(b"KO6NGX", decode_payload(samples, config))

    def test_station_intro_uses_phonetic_callsign(self) -> None:
        intro = station_intro_text("KO6NGX")

        self.assertEqual(
            intro,
            "TRIAD automatic environmental observation system. "
            "station identification. Kilo. Oscar. Six. November. Golf. X-ray.",
        )


if __name__ == "__main__":
    unittest.main()
