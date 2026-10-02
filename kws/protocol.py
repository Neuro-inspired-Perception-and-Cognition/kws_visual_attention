"""
protocol.py
"""

import struct

# endpoints 
WIRE_IN_RESET = 0x00
WIRE_IN_ARM = 0x01
WIRE_OUT_FIFO_COUNT = 0x20
PIPE_OUT_PRED = 0xA0

# frame layout 
FRAME_BYTES = 16
WORDS_PER_FRAME = 4          # the fill count at 0x20 is in 32-bit words
N_CLASSES = 11
CONF_MAX = 255.0

# head output ordering -- must match the RTL, do not re-sort
# WORDS = ["yes", "no", "up", "down", "left",
#          "right", "on", "off", "stop", "go", "unknown"]
# for new implementation
WORDS = [
    "up", "down", "left", "right", "one", "two", "unknown", "None", "None", "None", "None"
]
UNKNOWN_INDEX = WORDS.index("unknown")

# how head classes map onto attention commands; everything else is a no-op
DIRECTIONS = {"up", "down", "left", "right"}
CLEARING = {"stop"}


SEQ_MOD = 256


def decode_frame(chunk):
    """16 bytes -> (conf, scores, seq), or None if the frame is FIFO padding.

    `seq` is the 8-bit frame counter; consecutive frames differ by 1 mod 256.
    """
    w1, w2, w3, w4 = struct.unpack("<IIII", chunk)
    if w1 == 0 and w2 == 0 and w3 == 0 and w4 == 0:
        return None
    conf = w4 & 0xFF
    scores = [
        (w4 >> 8) & 0xFF, (w4 >> 16) & 0xFF, (w4 >> 24) & 0xFF,
        w3 & 0xFF, (w3 >> 8) & 0xFF, (w3 >> 16) & 0xFF, (w3 >> 24) & 0xFF,
        w2 & 0xFF, (w2 >> 8) & 0xFF, (w2 >> 16) & 0xFF, (w2 >> 24) & 0xFF,
    ]
    # for new implementation
    scores[0] = scores[0] - 125
    scores[1] = scores[1] - 125
    scores[2] = scores[2] - 125
    scores[3] = scores[3] - 125
    scores[4] = -128
    scores[5] = -128
    scores[6] = scores[6] - 185
    scores[7] = -128
    scores[8] = -128
    scores[9] = -128
    scores[10] = -128
    # print(str(class_scores) + " | " + str(confidence))
    return conf, scores, w1 & 0xFF


def decode_buffer(buf, n_bytes=None):
    """Split a pipe read into frames, dropping padding. Returns a list."""
    end = len(buf) if n_bytes is None else min(n_bytes, len(buf))
    out = []
    for i in range(0, end - FRAME_BYTES + 1, FRAME_BYTES):
        f = decode_frame(buf[i:i + FRAME_BYTES])
        if f is not None:
            out.append(f)
    return out


def rank(scores):
    """Class indices sorted best-first."""
    return sorted(range(N_CLASSES), key=lambda i: scores[i], reverse=True)