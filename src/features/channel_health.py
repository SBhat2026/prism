"""
channel_health.py — is a feature channel actually carrying information?

A channel that is constant across every pair in a complex cannot influence any
model, however the model is architected. Feeding it anyway is not neutral: it
looks like a working input in the UI and in feature lists, so a silent data
outage gets mistaken for a modelling result.

This module makes that distinction explicit and reusable:

  DEAD       constant off-diagonal — zero information, the model cannot use it
  SATURATED  varies, but the spread is tiny next to the reference channel, so it
             is effectively drowned out at the input scale the network sees
  OK         usable dynamic range

Two failures this was written to catch, both real and both live in this repo
before it existed:

  * `go_cos`  was identically zero for 100% of the corpus — the per-complex
    go.npy files were allocated but never populated.
  * `ppi`     was ~1e-3 everywhere because STRING scores were divided by 1000
    a second time on parse (see src/data_prep/string_api.py). Numerically
    invisible next to SASA, but it looked like a real channel.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

DEAD_STD = 1e-9          # below this the channel is constant
SATURATED_RATIO = 0.05   # <5% of the reference channel's spread → drowned out


@dataclass
class ChannelStat:
    name: str
    mean: float
    std: float
    vmin: float
    vmax: float
    status: str          # "ok" | "saturated" | "dead" | "missing"
    range_ratio: float   # std relative to the reference channel

    @property
    def usable(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict:
        return asdict(self)


def _offdiag(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] < 2:
        return m.ravel()
    return m[~np.eye(m.shape[0], dtype=bool)]


def channel_stats(
    matrices: dict[str, np.ndarray | None],
    reference: str = "sasa",
) -> dict[str, ChannelStat]:
    """Classify each named N×N channel. `reference` is the channel whose spread
    the others are judged against — normally SASA, the one known to work."""
    vals: dict[str, np.ndarray] = {}
    for name, m in matrices.items():
        vals[name] = _offdiag(m) if m is not None else np.array([])

    ref = vals.get(reference, np.array([]))
    ref_std = float(ref.std()) if ref.size else 0.0

    out: dict[str, ChannelStat] = {}
    for name, v in vals.items():
        if v.size == 0:
            out[name] = ChannelStat(name, 0.0, 0.0, 0.0, 0.0, "missing", 0.0)
            continue
        std = float(v.std())
        ratio = std / ref_std if ref_std > DEAD_STD else 0.0
        if std < DEAD_STD:
            status = "dead"
        elif name != reference and ref_std > DEAD_STD and ratio < SATURATED_RATIO:
            status = "saturated"
        else:
            status = "ok"
        out[name] = ChannelStat(name, float(v.mean()), std,
                                float(v.min()), float(v.max()), status, ratio)
    return out


def summarise(stats: dict[str, ChannelStat]) -> str:
    """One-line-per-channel table for logs."""
    rows = [f"{'channel':<10}{'mean':>9}{'std':>9}{'vs ref':>9}  status"]
    for s in stats.values():
        rows.append(f"{s.name:<10}{s.mean:>9.4f}{s.std:>9.4f}"
                    f"{s.range_ratio:>9.3f}  {s.status}")
    return "\n".join(rows)


def warn_if_degenerate(stats: dict[str, ChannelStat], context: str = "") -> list[str]:
    """Return human-readable warnings for channels that cannot contribute.

    Call this at data-load time. A dead channel is a data-pipeline bug, not a
    modelling outcome, and it should be surfaced loudly rather than absorbed.
    """
    msgs = []
    for s in stats.values():
        if s.status == "dead":
            msgs.append(f"[channel-health]{' ' + context if context else ''} "
                        f"'{s.name}' is CONSTANT ({s.mean:.4g}) — it carries no "
                        f"information and cannot affect any prediction.")
        elif s.status == "saturated":
            msgs.append(f"[channel-health]{' ' + context if context else ''} "
                        f"'{s.name}' spread is {s.range_ratio:.1%} of the "
                        f"reference channel — effectively drowned out.")
        elif s.status == "missing":
            msgs.append(f"[channel-health]{' ' + context if context else ''} "
                        f"'{s.name}' is absent.")
    return msgs
