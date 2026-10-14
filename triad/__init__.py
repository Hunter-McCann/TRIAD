"""TRIAD amateur-radio digital mode prototype."""

from .modem import TriadConfig, decode_wav, encode_to_wav

__all__ = ["TriadConfig", "decode_wav", "encode_to_wav"]
