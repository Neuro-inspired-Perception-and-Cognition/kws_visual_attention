"""
command_parser.py — turns a typed terminal line into a structured command.
Pure function, no cv2/torch/queue involved, so it's testable on its own.

Recognised forms:
    <direction>            e.g. "right"            -> conf defaults to 1.0
    <direction> <conf>     e.g. "right 0.8"         -> explicit confidence
    stop / none / clear    -> release the active command, freeze in place
    reset                  -> zero the membrane, re-fixate on next salmax
    mode pan / mode saccade -> switch panning mode live
    quit / exit / q        -> stop the session
    anything else          -> {"type": "unknown"}, printed and ignored
"""


def parse_command(line, dirs):
    line = (line or "").strip().lower()
    if not line:
        return None
    parts = line.split()
    head = parts[0]

    if head in ("quit", "exit", "q"):
        return {"type": "quit"}
    if head in ("stop", "none", "clear"):
        return {"type": "stop"}
    if head in ("reset",):
        return {"type": "reset"}
    if head == "mode" and len(parts) >= 2 and parts[1] in ("pan", "saccade"):
        return {"type": "mode", "mode": parts[1]}
    if head in dirs:
        conf = 1.0
        if len(parts) >= 2:
            try:
                conf = float(parts[1])
            except ValueError:
                pass
        return {"type": "word", "word": head, "conf": conf}
    return {"type": "unknown", "raw": line}