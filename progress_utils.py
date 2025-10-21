# progress_utils.py
# -*- coding: utf-8 -*-
"""
Lightweight progress logging + SVG visualization utilities inspired by the
uploaded pipeline's heartbeat interface. Compatible with tqdm printing.

API (intentionally similar to the reference):
- Heartbeat(path)
    .write_beat(step:int, total:int, tag:str="")      # top-level progress
    .write_sub(cur:int, tot:int, tag:str="")          # sub-step progress
    .latest(n:int=50, sub:bool=False) -> List[dict]     # tail read

- render_progress(beats:list[dict], svg_path:str) -> str
- render_dashboard(beats:list[dict], steps:list[str], svg_path:str) -> str
- print_step(msg: str)  # tiny helper for console sections

Notes:
- JSONL written to `path` (top) and `path.replace('.jsonl','_sub.jsonl')` (sub)
- Functions avoid heavy deps; SVGs are tiny and self-contained.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Small console helper
# ──────────────────────────────────────────────────────────────────────────────

def print_step(msg: str):
    bar = "=" * max(10, min(80, len(msg) + 10))
    print(f"\n{bar}\n{msg}\n{bar}")


# ──────────────────────────────────────────────────────────────────────────────
# Heartbeat JSONL logger
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Heartbeat:
    path: str = "runs/heartbeat.jsonl"  # default compatible with reference

    def __post_init__(self):
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # reference-compatible sub path name
        self.sub_path = p.with_name("sub_heartbeat.jsonl")
        # also keep legacy pattern <stem>_sub.jsonl for readers that expect it
        self._legacy_sub = p.with_name(p.stem + "_sub.jsonl")
        for fp in (p, self.sub_path, self._legacy_sub):
            try:
                if not fp.exists():
                    fp.write_text("", encoding="utf-8")
            except Exception:
                pass

    @staticmethod
    def _sub_path(p: Path) -> Path:
        return p.with_name(p.stem + "_sub.jsonl")

    def _write_jsonl(self, p: Path, rec: dict):
        rec = dict(rec)
        rec["ts"] = time.time()
        try:
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "")
        except Exception:
            pass

    def write_beat(self, step: int, total: int, tag: str = "", extra: Optional[dict] = None):
        p = Path(self.path)
        pct = (float(step) / float(total)) if total else 0.0
        rec = {"step": int(step), "total": int(total), "tag": str(tag), "pct": float(pct), "extra": extra or {}}
        self._write_jsonl(p, rec)

    def write_sub(self, cur: int, tot: int, tag: str = "", extra: Optional[dict] = None):
        # write both reference and legacy sub paths
        rec = {"cur": int(cur), "tot": int(tot), "tag": str(tag), "pct": (float(cur)/float(tot)) if tot else 0.0, "extra": extra or {}}
        for sp in (self.sub_path, self._legacy_sub):
            self._write_jsonl(sp, rec)

    def latest(self, n: int = 50, sub: bool = False) -> List[dict]:
        p = self.sub_path if sub else Path(self.path)
        # if reference sub is empty, try legacy
        if sub and (not p.exists() or p.stat().st_size == 0):
            p = self._legacy_sub
        try:
            with p.open("r", encoding="utf-8") as f:
                lines = f.readlines()[-n:]
            out = []
            for ln in lines:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    continue
            return out
        except Exception:
            return []

    # legacy helper (kept for back-compat)
    def _sub_path(self, p: Path) -> Path:
        return p.with_name(p.stem + "_sub.jsonl")

    def old_latest(self, n: int = 50, sub: bool = False) -> List[dict]:
        # back-compat alias
        return self.latest(n=n, sub=sub)


# ──────────────────────────────────────────────────────────────────────────────
# Live log (append-only text log, reference-compatible)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class LiveLogger:
    path: str = "runs/live.log"

    def __post_init__(self):
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            try:
                p.write_text("", encoding="utf-8")
            except Exception:
                pass

    def _ts(self) -> str:
        try:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(time.time())

    def write(self, msg: str, level: str = "INFO", tag: str = ""):
        line = f"[{self._ts()}] {level.upper():5s} {f'[{tag}] ' if tag else ''}{msg}"
        try:
            with Path(self.path).open("a", encoding="utf-8") as f:
                f.write(line + "")
        except Exception:
            pass
        # mirror to stdout (non-intrusive)
        try:
            print(line)
        except Exception:
            pass

    def section(self, title: str):
        bar = "=" * max(10, min(80, len(title) + 10))
        self.write(bar, level="INFO")
        self.write(title, level="INFO")
        self.write(bar, level="INFO")


def live_log(msg: str, path: str = "runs/live.log", level: str = "INFO", tag: str = ""):
    """Convenience wrapper compatible with 'other version' imports."""
    logger = LiveLogger(path)
    logger.write(msg, level=level, tag=tag)


# ──────────────────────────────────────────────────────────────────────────────
# Tiny SVG renderers (no heavy deps)
# ──────────────────────────────────────────────────────────────────────────────
_SVG_TMPL = """<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>
  <style>
    .lab {{ font: 12px monospace; fill: #333; }}
    .small {{ font: 10px monospace; fill: #666; }}
  </style>
  {BODY}
</svg>"""


def _bar(x: int, n: int, x0: int, y: int, w: int, h: int, color: str = "#4C78A8") -> str:
    p = 0.0 if n <= 0 else max(0.0, min(1.0, float(x) / float(n)))
    fill_w = int(w * p)
    return (f"<rect x='{x0}' y='{y}' width='{w}' height='{h}' rx='6' ry='6' fill='#eee'/>"
            f"<rect x='{x0}' y='{y}' width='{fill_w}' height='{h}' rx='6' ry='6' fill='{color}'/>"
            f"<text class='lab' x='{x0+w+8}' y='{y+h-4}'> {int(p*100)}% ({x}/{n})</text>")


def render_progress(beats: List[dict], svg_path: str) -> str:
    """Render a single progress bar from top-level beats list to svg_path."""
    if beats:
        last = beats[-1]
        step = last.get("step", 0)
        total = last.get("total", 0)
        tag = last.get("tag", "")
        # prefer logged pct if present
        if "pct" in last and isinstance(last.get("pct"), (int, float)):
            pct = float(last.get("pct"))
            step = int(round(pct * (total or 100)))
    else:
        step, total, tag = 0, 0, ""
    body = [
        f"<text class='lab' x='8' y='18'>Progress: {tag}</text>",
        _bar(int(step), int(total), 8, 26, 320, 16),
    ]
    svg = _SVG_TMPL.format(W=380, H=60, BODY="".join(body))
    Path(svg_path).write_text(svg, encoding="utf-8")
    return svg_path


def render_dashboard(beats: List[dict], steps: List[str], svg_path: str) -> str:
    """Render a mini dashboard that shows which step out of `steps` we're on."""
    if not steps:
        return render_progress(beats, svg_path)
    step = beats[-1].get("step", 0) if beats else 0
    total = beats[-1].get("total", len(steps)) if beats else len(steps)
    y = 20
    rows = [f"<text class='lab' x='8' y='{y}'>Pipeline</text>"]
    y += 8
    rows.append(_bar(step, total, 8, y, 320, 14))
    y += 26
    for i, s in enumerate(steps, start=1):
        mark = "●" if i <= step else "○"
        rows.append(f"<text class='small' x='14' y='{y}'> {mark} {i}/{len(steps)}  {s}</text>")
        y += 16
    svg = _SVG_TMPL.format(W=420, H=y+8, BODY="\n  ".join(rows))
    Path(svg_path).write_text(svg, encoding="utf-8")
    return svg_path
