from __future__ import annotations

import argparse
from pathlib import Path

from .modem import TriadConfig, decode_wav, encode_ota_test_to_wav, encode_to_wav
from .waterfall import render_waterfall


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TRIAD audio modem prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode = subparsers.add_parser("encode", help="encode text into a TRIAD WAV")
    encode.add_argument("message", help="ASCII telemetry payload")
    encode.add_argument("output", type=Path, help="output WAV path")
    encode.add_argument("--baud", type=float, default=2.5, help="chords per second")
    encode.add_argument("--root", type=float, default=720.0, help="Ionian tonic in Hz")

    decode = subparsers.add_parser("decode", help="decode a TRIAD WAV")
    decode.add_argument("input", type=Path, help="input WAV path")
    decode.add_argument("--baud", type=float, default=2.5, help="chords per second")
    decode.add_argument("--root", type=float, default=720.0, help="Ionian tonic in Hz")

    waterfall = subparsers.add_parser("waterfall", help="render a waterfall PNG from a WAV")
    waterfall.add_argument("input", type=Path, help="input WAV path")
    waterfall.add_argument("output", type=Path, help="output PNG path")
    waterfall.add_argument("--min-hz", type=float, default=300.0, help="lowest displayed audio frequency")
    waterfall.add_argument("--max-hz", type=float, default=2700.0, help="highest displayed audio frequency")

    ota_test = subparsers.add_parser("ota-test", help="generate a repeated OTA test WAV")
    ota_test.add_argument("output", type=Path, help="output WAV path")
    ota_test.add_argument("--callsign", default="KO6NGX", help="station callsign to include")
    ota_test.add_argument(
        "--message",
        default=None,
        help="ASCII payload after the callsign",
    )
    ota_test.add_argument("--repeats", type=int, default=3, help="number of packets to send")
    ota_test.add_argument("--gap", type=float, default=4.0, help="seconds between repeated packets")
    ota_test.add_argument("--lead", type=float, default=2.0, help="seconds of silence before and after")
    ota_test.add_argument("--baud", type=float, default=2.5, help="chords per second")
    ota_test.add_argument("--root", type=float, default=720.0, help="Ionian tonic in Hz")
    ota_test.add_argument("--amplitude", type=float, default=0.18, help="audio amplitude from 0.0 to 1.0")
    ota_test.add_argument("--voice-intro", action="store_true", help="prepend a synthesized voice ID")
    ota_test.add_argument(
        "--voice-prefix",
        default=None,
        help="optional spoken text between the TRIAD system phrase and station ID",
    )
    ota_test.add_argument("--voice", default=None, help="optional macOS voice name")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "encode":
        config = TriadConfig(symbol_rate=args.baud, root_hz=args.root)
        encode_to_wav(args.message.encode("ascii"), args.output, config=config)
        print(f"Wrote {args.output}")
    elif args.command == "decode":
        config = TriadConfig(symbol_rate=args.baud, root_hz=args.root)
        decoded = decode_wav(args.input, config=config)
        print(decoded.decode("ascii", errors="replace"))
    elif args.command == "waterfall":
        render_waterfall(args.input, args.output, min_hz=args.min_hz, max_hz=args.max_hz)
        print(f"Wrote {args.output}")
    elif args.command == "ota-test":
        config = TriadConfig(
            symbol_rate=args.baud,
            root_hz=args.root,
            amplitude=args.amplitude,
        )
        payload = encode_ota_test_to_wav(
            args.output,
            callsign=args.callsign,
            message=args.message,
            repeats=args.repeats,
            leading_silence_seconds=args.lead,
            gap_seconds=args.gap,
            voice_intro=args.voice_intro,
            voice_prefix=args.voice_prefix,
            voice_name=args.voice,
            config=config,
        )
        print(f"Wrote {args.output}")
        print(f"Payload: {payload}")


if __name__ == "__main__":
    main()
