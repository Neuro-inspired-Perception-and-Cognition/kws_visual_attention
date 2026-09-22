"""
init
Host-side interface to the NAS-GNN-KWS keyword spotter on the Opal Kelly board.

    from kws import KWSSource, make_backend

    backend = make_backend("frontpanel", bitfile="bitstreams/ok_top_wrapper.bit")
    src = KWSSource(backend, cmd_queue)
    src.start()
    ...
    src.close()

Layout:
    protocol.py  wire format -- endpoints, class order, frame decoding
    source.py    backends and the polling thread that emits commands
    probe.py     characterisation run:  python -m kws.probe
"""

from .protocol import (
    CLEARING, DIRECTIONS, UNKNOWN_INDEX, WORDS,
    decode_buffer, decode_frame, rank,
)
from .source import FrontPanelBackend, KWSSource, ReplayBackend, make_backend

__all__ = [
    "KWSSource", "make_backend", "FrontPanelBackend", "ReplayBackend",
    "WORDS", "DIRECTIONS", "CLEARING", "UNKNOWN_INDEX", "decode_frame", "decode_buffer", "rank",
]