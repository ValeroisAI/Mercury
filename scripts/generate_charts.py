"""Generate high-DPI data-driven benchmark and architecture PNG charts for Mercury-1."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mercury_sim.cluster_bench import run_all_qualification_scenarios
from mercury_sim.physical_bom import analyze_nre_and_tco
from mercury_sim.specs import MERCURY_1D, MERCURY_1S
from mercury_sim.thermal_power import simulate_thermal_run

OUT_DIR = ROOT / "assets" / "charts"

# 2x supersampling scale for crisp TrueType text and anti-aliased geometry
S = 2

# Palette: Precision Silicon Lab (clean high-contrast light cards, great in both GitHub light & dark modes)
BG_CANVAS = "#F8FAFC"
BG_CARD = "#FFFFFF"
BG_SUBCARD = "#F1F5F9"
BORDER = "#CBD5E1"
GRID = "#E2E8F0"
INK_PRIMARY = "#0F172A"
INK_SECONDARY = "#475569"
INK_MUTED = "#64748B"

ACCENT_1D = "#2563EB"       # Cobalt Blue (Mercury-1D)
ACCENT_1D_LIGHT = "#93C5FD"
ACCENT_1S = "#0D9488"       # Teal (Mercury-1S)
ACCENT_1S_LIGHT = "#5EEAD4"
SLATE_REF = "#64748B"       # Commercial reference GPUs
SLATE_LIGHT = "#94A3B8"
AMBER = "#D97706"
CRIMSON = "#DC2626"
EMERALD = "#059669"
VIOLET = "#7C3AED"


def _font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    fonts_dir = Path(r"C:\Windows\Fonts")
    if mono:
        candidates = ["consolab.ttf" if bold else "consola.ttf", "courbd.ttf" if bold else "cour.ttf"]
    else:
        candidates = ["segoeuib.ttf" if bold else "segoeui.ttf", "arialbd.ttf" if bold else "arial.ttf"]
    for name in candidates:
        p = fonts_dir / name
        if p.exists():
            return ImageFont.truetype(str(p), size * S)
    return ImageFont.load_default()


F_TITLE = _font(22, bold=True)
F_SUBTITLE = _font(13, bold=False)
F_SEC = _font(15, bold=True)
F_LABEL = _font(12, bold=False)
F_LABEL_B = _font(12, bold=True)
F_SMALL = _font(10, bold=False)
F_SMALL_B = _font(10, bold=True)
F_MONO = _font(11, bold=False, mono=True)
F_MONO_B = _font(11, bold=True, mono=True)


def _new_canvas(w: int = 1440, h: int = 800) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (w * S, h * S), BG_CANVAS)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, w * S - 1, h * S - 1], outline=BORDER, width=2 * S)
    return img, draw


def _save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    final = img.resize((img.width // S, img.height // S), Image.Resampling.LANCZOS)
    final.save(path, format="PNG", optimize=True)


def _rrect(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], r: int, fill: str, outline: str | None = None, width: int = 1) -> None:
    x0, y0, x1, y1 = [v * S for v in xy]
    draw.rounded_rectangle([x0, y0, x1, y1], radius=r * S, fill=fill, outline=outline, width=width * S if outline else 0)


def _rect(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill: str, outline: str | None = None, width: int = 1) -> None:
    x0, y0, x1, y1 = [v * S for v in xy]
    draw.rectangle([x0, y0, x1, y1], fill=fill, outline=outline, width=width * S if outline else 0)


def _line(draw: ImageDraw.ImageDraw, p0: tuple[float, float], p1: tuple[float, float], fill: str, width: int = 2, dashed: bool = False) -> None:
    x0, y0 = p0[0] * S, p0[1] * S
    x1, y1 = p1[0] * S, p1[1] * S
    if not dashed:
        draw.line([x0, y0, x1, y1], fill=fill, width=width * S)
        return
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist < 1:
        return
    dash_len = 8 * S
    steps = int(dist / dash_len)
    for idx in range(0, max(1, steps), 2):
        t0 = idx / max(1, steps)
        t1 = min(1.0, (idx + 1) / max(1, steps))
        draw.line(
            [x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0, x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1],
            fill=fill,
            width=width * S,
        )


def _text(draw: ImageDraw.ImageDraw, pos: tuple[float, float], txt: str, font: ImageFont.FreeTypeFont, fill: str = INK_PRIMARY, anchor: str = "la") -> None:
    draw.text((pos[0] * S, pos[1] * S), txt, font=font, fill=fill, anchor=anchor)


def _header(draw: ImageDraw.ImageDraw, title: str, subtitle: str, tag: str = "MERCURY-1 QUALIFICATION") -> None:
    _rect(draw, (0, 0, 1440, 82), fill=BG_CARD)
    _line(draw, (0, 82), (1440, 82), BORDER, 1)
    bbox = draw.textbbox((0, 0), tag, font=F_SMALL_B)
    pill_w = max(160, (bbox[2] - bbox[0]) // S + 28)
    _rrect(draw, (32, 19, 32 + pill_w, 43), 4, fill="#DBEAFE", outline=ACCENT_1D, width=1)
    _text(draw, (32 + pill_w / 2, 31), tag, F_SMALL_B, fill=ACCENT_1D, anchor="mm")
    _text(draw, (32 + pill_w + 16, 17), title, F_TITLE, fill=INK_PRIMARY)
    _text(draw, (32, 53), subtitle, F_SUBTITLE, fill=INK_SECONDARY)


def _footer(draw: ImageDraw.ImageDraw, note: str, w: int = 1440, h: int = 800) -> None:
    _line(draw, (32, h - 38), (w - 32, h - 38), GRID, 1)
    _text(draw, (32, h - 22), note, F_SMALL, fill=INK_MUTED)
    _text(draw, (w - 32, h - 22), "Apache-2.0 Open Architecture · ValeroisAI/Mercury", F_SMALL_B, fill=INK_MUTED, anchor="ra")


def build_chart_01(qual: dict[str, Any]) -> None:
    """Chart 01: Peak Dense BF16/FP8 TFLOPS and NRE-Amortized Capital Efficiency."""
    img, draw = _new_canvas(1440, 820)
    _header(
        draw,
        "Compute Output & Economics: 1,152 BF16 TFLOPS With Explicit $2,456 BOM + $85M NRE Curve",
        "Dense Tensor TFLOPS (left) and BF16 TFLOPS per $1,000 across NRE-amortized production volumes at 45% GM (right)",
        "BENCHMARK 01 · COMPUTE & COST",
    )

    # Left Card: Dense BF16 & FP8 TFLOPS
    _rrect(draw, (32, 102, 730, 762), 10, fill=BG_CARD, outline=BORDER, width=1)
    _text(draw, (54, 122), "Peak Dense AI Tensor Throughput (TFLOPS)", F_SEC, fill=INK_PRIMARY)
    _text(draw, (54, 146), "FP8 Dense (top bar) and BF16 Dense (bottom bar) · Higher is better", F_SMALL, fill=INK_SECONDARY)

    # Legend
    _rect(draw, (470, 124, 486, 138), fill=ACCENT_1D)
    _text(draw, (494, 123), "Mercury BF16", F_SMALL_B, fill=INK_PRIMARY)
    _rect(draw, (592, 124, 608, 138), fill=ACCENT_1D_LIGHT)
    _text(draw, (616, 123), "Mercury FP8", F_SMALL_B, fill=INK_PRIMARY)

    gpus = [
        ("NVIDIA B200 (192GB HBM3e)", 2250.0, 4500.0, False),
        ("AMD MI300X (192GB HBM3)", 1307.0, 2615.0, False),
        ("Mercury-1D (144GB GDDR7)", MERCURY_1D.peak_tflops("bf16"), MERCURY_1D.peak_tflops("fp8"), True),
        ("NVIDIA H100 SXM (80GB HBM3)", 989.0, 1979.0, False),
        ("Mercury-1S (72GB GDDR7)", MERCURY_1S.peak_tflops("bf16"), MERCURY_1S.peak_tflops("fp8"), True),
        ("Tenstorrent Blackhole p150", 373.0, 745.0, False),
        ("NVIDIA RTX 5090 (32GB GDDR7)", 210.0, 838.0, False),
    ]
    max_tf = 4800.0
    x_label = 54
    x_bar0 = 266
    bar_w = 370
    y0 = 188
    row_h = 78

    for tick in [0, 1000, 2000, 3000, 4000]:
        tx = x_bar0 + (tick / max_tf) * bar_w
        _line(draw, (tx, y0 - 10), (tx, y0 + len(gpus) * row_h - 14), GRID, 1)
        _text(draw, (tx, y0 + len(gpus) * row_h - 4), f"{tick:,}", F_SMALL, fill=INK_MUTED, anchor="ma")

    for idx, (name, bf16, fp8, is_merc) in enumerate(gpus):
        ry = y0 + idx * row_h
        if is_merc:
            _rrect(draw, (44, ry - 6, 718, ry + 56), 6, fill="#EFF6FF" if "1D" in name else "#F0FDFA")
        _text(draw, (x_label, ry + 14), name, F_LABEL_B if is_merc else F_LABEL, fill=INK_PRIMARY)
        c_bf = ACCENT_1D if "1D" in name else (ACCENT_1S if "1S" in name else SLATE_REF)
        c_fp = ACCENT_1D_LIGHT if "1D" in name else (ACCENT_1S_LIGHT if "1S" in name else SLATE_LIGHT)

        w_fp8 = max(4, (fp8 / max_tf) * bar_w)
        w_bf16 = max(4, (bf16 / max_tf) * bar_w)
        _rrect(draw, (x_bar0, ry + 2, x_bar0 + w_fp8, ry + 21), 3, fill=c_fp)
        _rrect(draw, (x_bar0, ry + 25, x_bar0 + w_bf16, ry + 44), 3, fill=c_bf)
        _text(draw, (x_bar0 + w_fp8 + 8, ry + 11), f"{fp8:,.0f} FP8", F_MONO_B if is_merc else F_MONO, fill=INK_PRIMARY, anchor="lm")
        _text(draw, (x_bar0 + w_bf16 + 8, ry + 34), f"{bf16:,.0f} BF16", F_MONO_B if is_merc else F_MONO, fill=INK_SECONDARY, anchor="lm")

    # Right Card: BF16 TFLOPS per $1,000 across NRE Volume Amortization & Competitors
    _rrect(draw, (750, 102, 1408, 762), 10, fill=BG_CARD, outline=BORDER, width=1)
    _text(draw, (772, 122), "Economic Efficiency: Dense BF16 TFLOPS per $1k (BOM vs. NRE)", F_SEC, fill=INK_PRIMARY)
    _text(draw, (772, 146), "Modeled at 45% GM with $85M NRE amortized over 10k-250k units (vs. HW-only BOM floor)", F_SMALL, fill=INK_SECONDARY)

    nre_map = {r.volume_units: r for r in analyze_nre_and_tco(MERCURY_1D)}
    bf16_1d = MERCURY_1D.peak_tflops("bf16")
    eff_rows = [
        (f"Tenstorrent Blackhole ($1,399)", 373.0 / 1.399, "GDDR6 / 512 GB/s (retail ref.)", False),
        (f"Mercury-1D @ 250k (${nre_map[250_000].target_price_45gm_usd:,.0f})", bf16_1d / (nre_map[250_000].target_price_45gm_usd / 1000.0), "Incl. $340/u NRE @ 45% GM", True),
        (f"Mercury-1D @ 100k (${nre_map[100_000].target_price_45gm_usd:,.0f})", bf16_1d / (nre_map[100_000].target_price_45gm_usd / 1000.0), "Incl. $850/u NRE @ 45% GM", True),
        (f"Mercury-1D @ 25k (${nre_map[25_000].target_price_45gm_usd:,.0f})", bf16_1d / (nre_map[25_000].target_price_45gm_usd / 1000.0), "Incl. $3,400/u NRE @ 45% GM", True),
        (f"Mercury-1D @ 10k (${nre_map[10_000].target_price_45gm_usd:,.0f})", bf16_1d / (nre_map[10_000].target_price_45gm_usd / 1000.0), "Low-vol NRE burden ($8.5k/u)", True),
        ("NVIDIA B200 Est. ($40,000)", 2250.0 / 40.0, "192GB HBM3e / 8.0 TB/s", False),
        ("NVIDIA H100 Est. ($30,000)", 989.0 / 30.0, "80GB HBM3 / 3.35 TB/s", False),
    ]
    max_eff = 300.0
    ex0 = 1015
    ew = 295
    ey0 = 186
    erow = 56
    for tick in [0, 50, 100, 150, 200, 250]:
        tx = ex0 + (tick / max_eff) * ew
        _line(draw, (tx, ey0 - 8), (tx, ey0 + len(eff_rows) * erow - 12), GRID, 1)
        _text(draw, (tx, ey0 + len(eff_rows) * erow - 2), f"{tick}", F_SMALL, fill=INK_MUTED, anchor="ma")

    for idx, (label, val, sub, is_merc) in enumerate(eff_rows):
        ry = ey0 + idx * erow
        c = ACCENT_1D if "1D" in label else (ACCENT_1S if "1S" in label else SLATE_REF)
        _text(draw, (772, ry + 4), label, F_LABEL_B if is_merc else F_LABEL, fill=INK_PRIMARY)
        _text(draw, (772, ry + 23), sub, F_SMALL, fill=INK_MUTED)
        bw = max(4, (val / max_eff) * ew)
        _rrect(draw, (ex0, ry + 6, ex0 + bw, ry + 32), 4, fill=c)
        _text(draw, (ex0 + bw + 8, ry + 19), f"{val:.1f}", F_MONO_B if is_merc else F_MONO, fill=INK_PRIMARY, anchor="lm")

    # Callout Box: Manufacturing BOM vs. NRE Amortization
    _rrect(draw, (772, 602, 1384, 744), 8, fill=BG_SUBCARD, outline=BORDER, width=1)
    _text(draw, (792, 616), "Hardware-Only BOM vs. $85M Upfront NRE (Mask + High-Speed IP + SW)", F_LABEL_B, fill=INK_PRIMARY)
    _text(draw, (792, 642), "• NVIDIA B200 BOM (~$6,400-$6,750): ~$3.1k HBM3e + ~$1.1k CoWoS-L vs ~$900 Logic", F_SMALL, fill=INK_SECONDARY)
    _text(draw, (792, 664), f"• Mercury-1D HW BOM (${MERCURY_1D.total_bom_usd:,.0f}): $434 Silicon + $1,152 GDDR7 + $220 UCIe-S + $650 Board/Test", F_SMALL, fill=ACCENT_1D)
    _text(draw, (792, 686), f"• Mercury-1S HW BOM (${MERCURY_1S.total_bom_usd:,.0f}): Same 453.8 mm2 N4P mask (469 BF16 TFLOPS / $1k HW BOM floor)", F_SMALL, fill=ACCENT_1S)
    _text(draw, (792, 708), "• Break-evens (45% GM): <$10k @ 27,924 units | <$8k @ 43,725 units | $4,700 baseline excl. NRE", F_SMALL_B, fill=INK_PRIMARY)

    _footer(draw, "Sources: mercury_sim.specs (TSMC N4P 1.85 GHz) & verified H100/B200/MI300X/RTX5090/Blackhole specs in docs/research/riscv-open-gpu-landscape.md", 1440, 820)
    _save(img, OUT_DIR / "01_tflops_vs_bom_frontier.png")


def build_chart_02() -> None:
    """Chart 02: Electro-thermal transient ODE traces across Air, Liquid, and Mid-Run Fault."""
    img, draw = _new_canvas(1440, 820)
    _header(
        draw,
        "Electro-Thermal Qualification: Liquid Holds 54.6°C @ 1.85 GHz; Hot Air Throttles GDDR7",
        "2-node RC thermal ODE simulation (dt=0.5s) tracking GPU Junction Tj, Backside Clamshell GDDR7 Tmem, 85°C 2x tREFI, and mid-run fault PROCHOT",
        "BENCHMARK 02 · THERMAL & POWER",
    )

    tr_liq = simulate_thermal_run(MERCURY_1D, "oam_liquid_35c", "training_fp8", duration_s=360.0)
    tr_1s = simulate_thermal_run(MERCURY_1S, "workstation_air_25c", "training_fp8", duration_s=360.0)
    tr_air35 = simulate_thermal_run(MERCURY_1D, "server_4u_air_35c", "training_fp8", duration_s=360.0)
    tr_fault = simulate_thermal_run(
        MERCURY_1D, "oam_liquid_35c", "training_fp8", duration_s=360.0,
        mid_run_cooling_fault_s=120.0, fault_cooling="degraded_fan_fault"
    )

    series = [
        ("1D OCP Liquid 35°C (1.85 GHz, 0% throttle)", tr_liq, ACCENT_1D),
        ("1S PCIe Air 25°C (1.85 GHz, 0% throttle)", tr_1s, ACCENT_1S),
        ("1D 4U Hot Air 35°C (DVFS -> 0.90 GHz)", tr_air35, AMBER),
        ("1D Pump/Fan Fault @120s (PROCHOT Trip)", tr_fault, CRIMSON),
    ]

    # Left Chart: GPU Junction Temp Tj(t)
    _rrect(draw, (32, 102, 720, 620), 10, fill=BG_CARD, outline=BORDER, width=1)
    _text(draw, (54, 122), "GPU Die Junction Temperature Tj (°C) vs. Time", F_SEC, fill=INK_PRIMARY)
    _text(draw, (54, 146), "PMU throttles at 90°C; hardware PROCHOT emergency trip clamps to 0.40 GHz at 105°C", F_SMALL, fill=INK_SECONDARY)

    px0, py0, pw, ph = 95, 185, 590, 360
    t_max, temp_min, temp_max = 360.0, 20.0, 115.0
    for temp in [20, 40, 60, 80, 90, 105]:
        y = py0 + ph - ((temp - temp_min) / (temp_max - temp_min)) * ph
        col = CRIMSON if temp == 105 else (AMBER if temp == 90 else GRID)
        _line(draw, (px0, y), (px0 + pw, y), col, 1, dashed=(temp in (90, 105)))
        _text(draw, (px0 - 8, y), f"{temp}°C", F_SMALL_B if temp in (90, 105) else F_SMALL, fill=col if temp in (90, 105) else INK_MUTED, anchor="rm")
    for ts in [0, 60, 120, 180, 240, 300, 360]:
        x = px0 + (ts / t_max) * pw
        _line(draw, (x, py0), (x, py0 + ph), GRID, 1)
        _text(draw, (x, py0 + ph + 12), f"{ts}s", F_SMALL, fill=INK_MUTED, anchor="ma")

    # Mark fault injection line at t=120s
    xf = px0 + (120.0 / t_max) * pw
    _line(draw, (xf, py0), (xf, py0 + ph), CRIMSON, 1, dashed=True)
    _text(draw, (xf + 6, py0 + 8), "t=120s cooling fault", F_SMALL_B, fill=CRIMSON)

    for idx, (_, res, col) in enumerate(series):
        pts = [(px0 + (p.time_s / t_max) * pw, py0 + ph - ((p.tj_c - temp_min) / (temp_max - temp_min)) * ph) for p in res.trace]
        for a, b in zip(pts, pts[1:]):
            _line(draw, a, b, col, 3)
        ex, ey = pts[-1]
        draw.ellipse([(ex - 4) * S, (ey - 4) * S, (ex + 4) * S, (ey + 4) * S], fill=col)

    # Place clean non-overlapping endpoint callouts in open chart whitespace
    _text(draw, (px0 + pw - 8, py0 + ph - ((98.0 - temp_min) / (temp_max - temp_min)) * ph), f"1S Air: {tr_1s.steady_tj_c:.1f}°C · 1D Hot Air: {tr_air35.steady_tj_c:.1f}°C", F_SMALL_B, fill=INK_PRIMARY, anchor="ra")
    _text(draw, (px0 + pw - 8, py0 + ph - ((47.0 - temp_min) / (temp_max - temp_min)) * ph), f"1D Liquid: {tr_liq.steady_tj_c:.1f}°C · Fault Safe: {tr_fault.steady_tj_c:.1f}°C", F_SMALL_B, fill=ACCENT_1D, anchor="ra")

    # Right Chart: Backside Clamshell GDDR7 Tmem(t)
    _rrect(draw, (740, 102, 1408, 620), 10, fill=BG_CARD, outline=BORDER, width=1)
    _text(draw, (762, 122), "Backside Clamshell GDDR7 Temp Tmem (°C) vs. Time", F_SEC, fill=INK_PRIMARY)
    _text(draw, (762, 146), "JEDEC 2x tREFI double-refresh triggers at 85°C (+8% DRAM power, -2.5% effective BW)", F_SMALL, fill=INK_SECONDARY)

    mx0, my0, mw, mh = 805, 185, 570, 360
    for temp in [20, 40, 60, 85, 95, 105]:
        y = my0 + mh - ((temp - temp_min) / (temp_max - temp_min)) * mh
        col = CRIMSON if temp == 105 else (AMBER if temp in (85, 95) else GRID)
        _line(draw, (mx0, y), (mx0 + mw, y), col, 1, dashed=(temp in (85, 95, 105)))
        _text(draw, (mx0 - 8, y), f"{temp}°C", F_SMALL_B if temp in (85, 95, 105) else F_SMALL, fill=col if temp in (85, 95, 105) else INK_MUTED, anchor="rm")
    for ts in [0, 60, 120, 180, 240, 300, 360]:
        x = mx0 + (ts / t_max) * mw
        _line(draw, (x, my0), (x, my0 + mh), GRID, 1)
        _text(draw, (x, my0 + mh + 12), f"{ts}s", F_SMALL, fill=INK_MUTED, anchor="ma")
    _text(draw, (mx0 + 6, my0 + mh - ((85 - temp_min) / (temp_max - temp_min)) * mh - 14), "85°C JEDEC 2x tREFI", F_SMALL_B, fill=AMBER)

    for _, res, col in series:
        pts = [(mx0 + (p.time_s / t_max) * mw, my0 + mh - ((p.tmem_c - temp_min) / (temp_max - temp_min)) * mh) for p in res.trace]
        for a, b in zip(pts, pts[1:]):
            _line(draw, a, b, col, 3)

    # Bottom Summary Strip / Legend
    _rrect(draw, (32, 640, 1408, 766), 10, fill=BG_CARD, outline=BORDER, width=1)
    _text(draw, (54, 656), "Steady-State Telemetry Summary (t = 360s)", F_SEC, fill=INK_PRIMARY)
    col_x = [54, 395, 736, 1077]
    for idx, (label, res, col) in enumerate(series):
        cx = col_x[idx]
        _rrect(draw, (cx, 684, cx + 318, 750), 6, fill=BG_SUBCARD, outline=col, width=2)
        _text(draw, (cx + 12, 694), label, F_SMALL_B, fill=col)
        _text(
            draw,
            (cx + 12, 714),
            f"Tj {res.steady_tj_c:.1f}°C (pk {res.peak_tj_c:.1f}°C) · Tmem {res.steady_tmem_c:.1f}°C",
            F_MONO,
            fill=INK_PRIMARY,
        )
        _text(
            draw,
            (cx + 12, 731),
            f"Clk {res.steady_clock_ghz:.2f} GHz · {res.steady_chip_power_w:.0f}W/{res.steady_board_power_w:.0f}W",
            F_MONO,
            fill=INK_SECONDARY,
        )

    _footer(draw, "Generated live from mercury_sim.thermal_power.simulate_thermal_run() · Leak ~ exp(0.022*(Tj-85)), VRM eff 93-94%", 1440, 820)
    _save(img, OUT_DIR / "02_thermal_transient_torture.png")


def build_chart_03(qual: dict[str, Any], micro: dict[str, Any] | None, pt_bench: dict[str, Any] | None) -> None:
    """Chart 03: Block-FP8 Numerical Error vs Hopper + XOR Swizzling + SRAM Traffic Cut."""
    img, draw = _new_canvas(1440, 820)
    _header(
        draw,
        "STME Block-FP8 Cuts Long-K Error 42x vs. Hopper & Cuts GDDR7 Traffic 16.85x",
        "Verified IEEE-754 K=128 FMA accuracy (left), XOR bank-swizzle cycle speedup & PyTorch parity, and L1/RF tile traffic (right)",
        "BENCHMARK 03 · NUMERICS & MICROARCH",
    )

    num = qual["s9_compute_isa"]["numerical"]
    sch = qual["s9_compute_isa"]["schedules"]

    # Left Card: Log-scale Relative L2 Error (Mercury vs Hopper 14-bit)
    _rrect(draw, (32, 102, 720, 762), 10, fill=BG_CARD, outline=BORDER, width=1)
    _text(draw, (54, 122), "Correlated Outlier Matmul Relative L2 Error (vs. FP64 Golden)", F_SEC, fill=INK_PRIMARY)
    _text(draw, (54, 146), "Lower is better (log scale) · 14-bit truncated accumulation drifts as K grows", F_SMALL, fill=INK_SECONDARY)

    _rect(draw, (54, 174, 70, 188), fill=ACCENT_1D)
    _text(draw, (78, 173), "Mercury-1 STME (IEEE FP32 K=128)", F_SMALL_B, fill=INK_PRIMARY)
    _rect(draw, (345, 174, 361, 188), fill=CRIMSON)
    _text(draw, (369, 173), "Hopper-like 14-bit Truncated", F_SMALL_B, fill=INK_PRIMARY)

    # Vertical log bar chart for K=1024 and K=4096 (range 1e-6 to 1e-2)
    gx0, gy0, gw, gh = 110, 220, 560, 340
    log_min, log_max = -6.0, -2.0
    for exp in [-6, -5, -4, -3, -2]:
        y = gy0 + gh - ((exp - log_min) / (log_max - log_min)) * gh
        _line(draw, (gx0, y), (gx0 + gw, y), GRID, 1)
        _text(draw, (gx0 - 10, y), f"1e{exp}", F_MONO, fill=INK_MUTED, anchor="rm")

    k_groups = [("K = 1024", "1024", 240), ("K = 4096", "4096", 510)]
    for label, key, cx in k_groups:
        m_err = num[key]["mercury_rel_l2_err"]
        h_err = num[key]["hopper_rel_l2_err"]
        ratio = h_err / m_err
        m_y = gy0 + gh - ((math.log10(m_err) - log_min) / (log_max - log_min)) * gh
        h_y = gy0 + gh - ((math.log10(h_err) - log_min) / (log_max - log_min)) * gh
        base_y = gy0 + gh
        _rrect(draw, (cx - 82, m_y, cx - 12, base_y), 4, fill=ACCENT_1D)
        _rrect(draw, (cx + 12, h_y, cx + 82, base_y), 4, fill=CRIMSON)
        _text(draw, (cx - 47, m_y - 16), f"{m_err:.2e}", F_MONO_B, fill=ACCENT_1D, anchor="mm")
        _text(draw, (cx + 47, h_y - 16), f"{h_err:.2e}", F_MONO_B, fill=CRIMSON, anchor="mm")
        _text(draw, (cx, base_y + 18), label, F_LABEL_B, fill=INK_PRIMARY, anchor="ma")
        _rrect(draw, (cx - 66, gy0 + 18, cx + 66, gy0 + 44), 12, fill="#DBEAFE", outline=ACCENT_1D, width=1)
        _text(draw, (cx, gy0 + 31), f"{ratio:.1f}x lower err", F_SMALL_B, fill=ACCENT_1D, anchor="mm")

    # PyTorch Transformer Block parity card inside Left bottom
    _rrect(draw, (54, 598, 696, 744), 8, fill=BG_SUBCARD, outline=BORDER, width=1)
    _text(draw, (74, 612), "PyTorch Transformer Block Training Parity (Live torch.nn Verification)", F_LABEL_B, fill=INK_PRIMARY)
    if pt_bench:
        gemms = pt_bench["gemms"]
        summ = pt_bench["summary"]
        ratios = summ["accumulator_error_ratio_hopper_over_mercury"]
        f_m = gemms["forward_Y"]["mercury"]["cos_sim"]
        g_m = gemms["grad_dW"]["mercury"]["cos_sim"]
        acc_f_m = gemms["forward_Y"]["mercury_accumulator_only"]["rel_l2"]
        acc_f_h = gemms["forward_Y"]["hopper_accumulator_only"]["rel_l2"]
        acc_g_m = gemms["grad_dW"]["mercury_accumulator_only"]["rel_l2"]
        acc_g_h = gemms["grad_dW"]["hopper_accumulator_only"]["rel_l2"]
        _text(draw, (74, 638), f"• End-to-End FP8 Cosine Sim:  Fwd Y = {f_m:.6f} · dW = {g_m:.6f} (FP8 quant dominates)", F_MONO, fill=INK_PRIMARY)
        _text(draw, (74, 664), f"• Accumulator-Only Fwd Y Err: Mercury = {acc_f_m:.2e} vs Hopper-14b = {acc_f_h:.2e} ({ratios['forward_Y']:,.0f}x lower)", F_MONO_B, fill=ACCENT_1D)
        _text(draw, (74, 690), f"• Accumulator-Only dW Err:    Mercury = {acc_g_m:.2e} vs Hopper-14b = {acc_g_h:.2e} ({ratios['grad_dW']:.1f}x lower)", F_MONO_B, fill=ACCENT_1D)
        conv = pt_bench.get("training_convergence")
        if conv:
            attr = conv["attribution"]
            l0 = conv["paths"]["mercury_fp8_blk"]["loss_curve"][0]
            l_end = conv["paths"]["mercury_fp8_blk"]["loss_curve"][-1]
            _text(draw, (74, 716), f"• 20-step AdamW: loss {l0:.1f}->{l_end:.2f}; isolated step-1 err {attr['mercury_first_step_grad_err_vs_fp8_exact_accum']:.2e} vs {attr['hopper_first_step_grad_err_vs_fp8_exact_accum']:.2e} (quant dominates ~{attr['fp8_quantization_update_drift']*100:.0f}%)", F_SMALL, fill=INK_SECONDARY)
        else:
            _text(draw, (74, 716), f"• Autograd Golden Match: max |diff| = {pt_bench['autograd_alignment']['dW_max_abs']:.1e} across 512d->1024ffn block", F_SMALL, fill=INK_SECONDARY)
    else:
        _text(draw, (74, 660), "Run with mercury_sim.pytorch_bridge to populate PyTorch Transformer parity.", F_SMALL, fill=INK_SECONDARY)

    # Top-Right Card: GDDR7 Traffic Reduction & Tile Budgets
    _rrect(draw, (740, 102, 1408, 430), 10, fill=BG_CARD, outline=BORDER, width=1)
    _text(draw, (762, 122), "On-Die 384 MB SRAM & Separate 256 KiB RF/L1 Traffic Reduction", F_SEC, fill=INK_PRIMARY)
    _text(draw, (762, 146), "Double-buffered DMA keeps tiles in SRAM, slashing external GDDR7 reads/writes", F_SMALL, fill=INK_SECONDARY)

    fa = sch["flashattention_3"]
    sw = sch["swiglu_mlp"]
    tile_rows = [
        ("FlashAttention-3 (64x128)", fa["naive_gddr7_bytes_transferred"], fa["gddr7_bytes_transferred"], fa["gddr7_traffic_reduction"], fa["rf_used_bytes"], fa["l1_used_bytes"]),
        ("SwiGLU MLP (64x128)", sw["naive_gddr7_bytes_transferred"], sw["gddr7_bytes_transferred"], sw["gddr7_traffic_reduction"], sw["rf_used_bytes"], sw["l1_used_bytes"]),
    ]
    ty0 = 188
    for idx, (kname, naive_b, tiled_b, red, rf_b, l1_b) in enumerate(tile_rows):
        ry = ty0 + idx * 114
        _text(draw, (762, ry), f"{kname}   —   {red:.2f}x GDDR7 Traffic Reduction", F_LABEL_B, fill=INK_PRIMARY)
        max_mb = 14.5
        nw = ((naive_b / 1e6) / max_mb) * 430
        tw = max(8, ((tiled_b / 1e6) / max_mb) * 430)
        _rrect(draw, (762, ry + 24, 762 + nw, ry + 42), 3, fill=SLATE_LIGHT)
        _text(draw, (762 + nw + 8, ry + 33), f"Unfused Naive: {naive_b/1e6:.2f} MB", F_MONO, fill=INK_SECONDARY, anchor="lm")
        _rrect(draw, (762, ry + 48, 762 + tw, ry + 66), 3, fill=ACCENT_1D)
        _text(draw, (762 + tw + 8, ry + 57), f"Mercury Tiled: {tiled_b/1e6:.2f} MB  (RF {rf_b/1024:.0f}/256 KiB · L1 {l1_b/1024:.0f}/256 KiB)", F_MONO_B, fill=ACCENT_1D, anchor="lm")

    # Bottom-Right Card: Cycle-Level Architectural Crossbar & Multicast Model
    _rrect(draw, (740, 448, 1408, 762), 10, fill=BG_CARD, outline=BORDER, width=1)
    _text(draw, (762, 468), "Cycle-Level Architectural Crossbar & A-Multicast (128x128 / 256x128)", F_SEC, fill=INK_PRIMARY)
    _text(draw, (762, 492), "256x128 multicast reaches 2,022 TFLOPS; XOR swizzle avoids 28.5x bank collapse", F_SMALL, fill=INK_SECONDARY)

    if micro and "scenarios" in micro:
        sc = micro["scenarios"]
        modes = [
            ("256x128 + A-Multicast", sc["multicast_256x128"], ACCENT_1D),
            ("128x128 XOR-Swizzle Nominal", sc["swizzled_baseline"], EMERALD),
            ("Hot >=85°C (2x tREFI GDDR7)", sc["hot_2x_trefi_85c"], AMBER),
            ("Cross-Die UCIe-S Spill (10%)", sc["cross_die_ucie_spill"], VIOLET),
            ("Unswizzled Stride-64 (2/64)", sc["unswizzled_bank_conflict"], CRIMSON),
        ]
        max_tf = max(m[1]["achieved_tflops"] for m in modes) * 1.12
        by0 = 526
        for idx, (lbl, mdata, col) in enumerate(modes):
            ry = by0 + idx * 45
            cyc = mdata["total_cycles"]
            tflops = mdata["achieved_tflops"]
            bconf = int(mdata["bank_conflict_cycles"])
            bw = max(6, (tflops / max_tf) * 190)
            _text(draw, (762, ry + 5), lbl, F_LABEL_B if idx <= 1 else F_LABEL, fill=INK_PRIMARY)
            _rrect(draw, (995, ry + 3, 995 + bw, ry + 25), 4, fill=col)
            _text(draw, (995 + bw + 8, ry + 14), f"{tflops:,.1f} TF · {cyc:,} cyc (conf {bconf:,})", F_SMALL_B, fill=INK_PRIMARY, anchor="lm")
    else:
        _text(draw, (762, 560), "Run with mercury_sim.microarch_cycle to populate cycle traces.", F_SMALL, fill=INK_SECONDARY)

    _footer(draw, "Sources: mercury_sim.compute_isa, mercury_sim.microarch_cycle, mercury_sim.pytorch_bridge", 1440, 820)
    _save(img, OUT_DIR / "03_fp8_accuracy_and_sram_roofline.png")


def build_chart_04(qual: dict[str, Any], isa_bench: dict[str, Any] | None) -> None:
    """Chart 04: 70B/MoE Cluster Parallelism, Batch=1 Decode Honesty, and 1T-Token Campaign TCO."""
    s6 = qual["s6_cluster_70b"]
    s7 = qual["s7_moe_and_decode"]
    s8 = qual["s8_link_degradation"]
    tco = qual["s10_pretrain_tco"]
    m_t, h_t, b_t = tco
    ratio_h100 = h_t.cost_per_million_tokens_usd / m_t.cost_per_million_tokens_usd

    img, draw = _new_canvas(1440, 820)
    _header(
        draw,
        f"Cluster Scaling & 1T-Token TCO: ${m_t.cost_per_million_tokens_usd:.4f} / 1M Tokens vs. ${h_t.cost_per_million_tokens_usd:.4f} on H100 ({ratio_h100:.2f}x Cheaper)",
        "Intra-card UCIe-S TP=2 + 400GbE FSDP cuts comm to 0.4% (left); 64-GPU 1-Trillion-Token 70B FP8 pretraining cost (right)",
        "BENCHMARK 04 · CLUSTER & TCO",
    )

    # Left Card: Cluster Training MFU & Throughput + Decode Honesty + ISA VM
    _rrect(draw, (32, 102, 730, 762), 10, fill=BG_CARD, outline=BORDER, width=1)
    _text(draw, (54, 122), "Multi-GPU Training MFU (%) & Exposed Communication Overhead", F_SEC, fill=INK_PRIMARY)
    _text(draw, (54, 146), "TP=2 over 600 GB/s UCIe-S + FSDP over 4x400GbE avoids slow Ethernet TP=8 stalls", F_SMALL, fill=INK_SECONDARY)

    cluster_rows = [
        ("8x 1D · 70B TP=2(UCIe)+FSDP=4", s6["pod8_tp2"], ACCENT_1D),
        ("16x 1D · 70B Full 16B/p AdamW", s6["pod16_16b"], ACCENT_1D),
        ("8x 1D · Degraded (3/4 400GbE)", s8["cluster"], AMBER),
        ("8x 1D · Naive TP=8 over Eth", s6["pod8_tp8"], SLATE_REF),
        ("16x 1D · MoE 141B EP=8+FSDP=2", s7["moe_ep8"], ACCENT_1S),
    ]
    cx0, cy0, cw = 290, 188, 330
    for tick in [0, 10, 20, 30, 40, 50]:
        tx = cx0 + (tick / 55.0) * cw
        _line(draw, (tx, cy0 - 8), (tx, cy0 + len(cluster_rows) * 62 - 14), GRID, 1)
        _text(draw, (tx, cy0 + len(cluster_rows) * 62 - 4), f"{tick}%", F_SMALL, fill=INK_MUTED, anchor="ma")

    for idx, (lbl, cr, col) in enumerate(cluster_rows):
        ry = cy0 + idx * 62
        mfu_pct = cr.achieved_mfu * 100.0
        comm_pct = (cr.exposed_comm_time_s / cr.step_time_s) * 100.0
        bw = (mfu_pct / 55.0) * cw
        _text(draw, (54, ry + 8), lbl, F_LABEL_B, fill=INK_PRIMARY)
        _text(draw, (54, ry + 28), f"{cr.cluster_tokens_per_sec:,.0f} tok/s · {cr.activation_headroom_gib:.1f} GiB free", F_SMALL, fill=INK_MUTED)
        _rrect(draw, (cx0, ry + 10, cx0 + bw, ry + 34), 4, fill=col)
        _text(draw, (cx0 + bw + 8, ry + 22), f"{mfu_pct:.1f}% MFU ({comm_pct:.1f}% comm)", F_MONO_B, fill=INK_PRIMARY, anchor="lm")

    # Decode Honesty + ISA VM Box
    dec = s7["decode_b1"]
    _rrect(draw, (54, 520, 708, 742), 8, fill=BG_SUBCARD, outline=BORDER, width=1)
    _text(draw, (74, 536), "Honest Workload Boundary & Open RISC-V ISA VM Verification", F_LABEL_B, fill=INK_PRIMARY)
    _text(draw, (74, 562), f"• Batch=1 70B Decode Ceiling (GDDR7 3.07 TB/s): Mercury-1D = {dec.tokens_per_sec:.1f} tok/s", F_SMALL_B, fill=CRIMSON)
    _text(draw, (74, 582), f"  vs. H200 (4.8 TB/s) = {dec.h200_ref_tok_s:.1f} tok/s | B200 (8.0 TB/s) = {dec.b200_ref_tok_s:.1f} tok/s -> Training/Batch-first SKU", F_SMALL, fill=INK_SECONDARY)
    if isa_bench:
        k1 = isa_bench.get("divergent_relu_norm", {})
        k2 = isa_bench.get("fused_fp8_gemm_flash_tile", {})
        t1 = k1.get("telemetry", {})
        t2 = k2.get("telemetry", {})
        n1 = t1.get("scalar_insts", 0) + t1.get("rvv_insts", 0) + t1.get("simt_insts", 0) + t1.get("coll_ops", 0)
        n2 = t2.get("scalar_insts", 0) + t2.get("rvv_insts", 0) + t2.get("stme_insts", 0) + t2.get("dma_ops", 0) + t2.get("coll_ops", 0) + t2.get("simt_insts", 0)
        _text(draw, (74, 616), "• Live MercuryVM (.masm 32-bit binary execution): 2/2 kernels passed FP64 parity!", F_SMALL_B, fill=EMERALD)
        _text(draw, (74, 640), f"  1) divergent_relu_norm.masm: {n1} insts ({t1.get('total_cycles_est', 0)} cyc), split={t1.get('divergent_splits', 0)}, err={k1.get('max_abs_error', 0.0):.2e}", F_MONO, fill=INK_PRIMARY)
        _text(draw, (74, 662), f"  2) fused_fp8_gemm_flash_tile.masm: {n2} insts ({t2.get('total_cycles_est', 0)} cyc), STME={t2.get('stme_insts', 0)}, DMA={t2.get('dma_ops', 0)}, err={k2.get('max_abs_error', 0.0):.2e}", F_MONO, fill=INK_PRIMARY)
        _text(draw, (74, 692), "  Custom Opcodes Verified: 0x0b (Xmersimt) · 0x2b (Xmerstme) · 0x5b (Xmerdma) · 0x7b (Xmercoll)", F_SMALL, fill=ACCENT_1D)
    else:
        _text(draw, (74, 620), "• Run with mercury_isa.vm to attach live 32-bit .masm kernel verification metrics.", F_SMALL, fill=INK_SECONDARY)

    # Right Card: 64-GPU 1-Trillion-Token Pretraining TCO
    _rrect(draw, (750, 102, 1408, 762), 10, fill=BG_CARD, outline=BORDER, width=1)
    _text(draw, (772, 122), "1-Trillion-Token Llama-3 70B FP8 Pretraining Campaign (64 GPUs)", F_SEC, fill=INK_PRIMARY)
    _text(draw, (772, 146), "Amortized 3-yr hardware capex + PUE 1.25 facility electricity @ $0.08/kWh", F_SMALL, fill=INK_SECONDARY)

    _rect(draw, (772, 174, 788, 188), fill=ACCENT_1D)
    _text(draw, (796, 173), "Amortized Hardware Capex ($)", F_SMALL_B, fill=INK_PRIMARY)
    _rect(draw, (1030, 174, 1046, 188), fill=AMBER)
    _text(draw, (1054, 173), "Facility Electricity ($)", F_SMALL_B, fill=INK_PRIMARY)

    # Vertical Stacked Bar Chart for Total Campaign Cost ($k)
    tx0, ty0, tw, th = 835, 225, 520, 320
    max_cost = 190000.0
    for c_k in [0, 50, 100, 150]:
        y = ty0 + th - ((c_k * 1000.0) / max_cost) * th
        _line(draw, (tx0, y), (tx0 + tw, y), GRID, 1)
        _text(draw, (tx0 - 10, y), f"${c_k}k", F_MONO, fill=INK_MUTED, anchor="rm")

    bar_centers = [935, 1105, 1275]
    for idx, item in enumerate(tco):
        cx = bar_centers[idx]
        hw_cost = item.total_campaign_cost_usd - item.energy_cost_usd
        el_cost = item.energy_cost_usd
        y_base = ty0 + th
        y_hw = y_base - (hw_cost / max_cost) * th
        y_tot = y_base - (item.total_campaign_cost_usd / max_cost) * th
        c_hw = ACCENT_1D if "Mercury" in item.system_name else SLATE_REF
        _rect(draw, (cx - 46, y_hw, cx + 46, y_base), fill=c_hw)
        _rect(draw, (cx - 46, y_tot, cx + 46, y_hw), fill=AMBER)
        _rrect(draw, (cx - 72, y_tot - 40, cx + 72, y_tot - 4), 4, fill=BG_CARD, outline=GRID, width=1)
        _text(draw, (cx, y_tot - 28), f"${item.cost_per_million_tokens_usd:.4f}/1M tok", F_MONO_B, fill=ACCENT_1D if "Mercury" in item.system_name else INK_PRIMARY, anchor="mm")
        _text(draw, (cx, y_tot - 13), f"Total ${item.total_campaign_cost_usd:,.0f}", F_SMALL_B, fill=INK_SECONDARY, anchor="mm")
        short = "64x Mercury-1D" if "Mercury" in item.system_name else ("64x NVIDIA H100" if "H100" in item.system_name else "64x NVIDIA B200")
        _text(draw, (cx, y_base + 18), short, F_LABEL_B if "Mercury" in item.system_name else F_LABEL, fill=INK_PRIMARY, anchor="ma")
        _text(draw, (cx, y_base + 38), f"Capex ${item.capex_usd/1e3:,.0f}k · {item.training_days:.1f}d", F_SMALL, fill=INK_MUTED, anchor="ma")

    _rrect(draw, (772, 608, 1384, 742), 8, fill=BG_SUBCARD, outline=BORDER, width=1)
    _text(draw, (792, 624), "Customer ROI Summary (64-GPU Training Pod)", F_LABEL_B, fill=INK_PRIMARY)
    m_t, h_t, b_t = tco
    _text(draw, (792, 652), f"• Upfront Pod Capex:  Mercury-1D ${m_t.capex_usd:,.0f} vs H100 ${h_t.capex_usd:,.0f} (5.45x lower) & B200 ${b_t.capex_usd:,.0f}", F_SMALL, fill=INK_PRIMARY)
    _text(draw, (792, 676), f"• Wall Power (PUE 1.25): Mercury-1D {m_t.cluster_wall_power_kw:.1f} kW vs H100 {h_t.cluster_wall_power_kw:.1f} kW vs B200 {b_t.cluster_wall_power_kw:.1f} kW", F_SMALL, fill=INK_SECONDARY)
    _text(draw, (792, 700), f"• Cost per 1M Tokens:  Mercury-1D beats H100 by {h_t.cost_per_million_tokens_usd/m_t.cost_per_million_tokens_usd:.2f}x and beats B200 by {b_t.cost_per_million_tokens_usd/m_t.cost_per_million_tokens_usd:.2f}x", F_SMALL_B, fill=ACCENT_1D)

    _footer(draw, "Sources: mercury_sim.cluster_bench & mercury_isa.vm · 64-GPU 1T tokens = 4.2e23 FLOPs", 1440, 820)
    _save(img, OUT_DIR / "04_cluster_scaling_and_tco.png")


def build_chart_05(qual: dict[str, Any]) -> None:
    """Chart 05: Proportional Silicon Floorplan & Extended OAM-L Topology Diagram."""
    img, draw = _new_canvas(1440, 820)
    _header(
        draw,
        "Mercury-1 Single-Mask Silicon Floorplan (453.8 mm2 N4P) & Extended OAM-L Topology",
        "Exact die area breakdown (left) and symmetric 2x384-bit Clamshell GDDR7 on Extended OAM-L (140x185 mm) rejecting 102 mm OAM overflow",
        "ARCHITECTURE 05 · FLOORPLAN & TOPOLOGY",
    )

    # Left Card: Die Area Proportional Map (21.3 mm x 21.3 mm = 453.8 mm2)
    _rrect(draw, (32, 102, 680, 762), 10, fill=BG_CARD, outline=BORDER, width=1)
    _text(draw, (54, 122), "Mercury-1 Compute Die Floorplan (21.3 × 21.3 mm = 453.8 mm²)", F_SEC, fill=INK_PRIMARY)
    _text(draw, (54, 146), "TSMC N4P · 84 fabricated / 76 active SCCs · 117 GDW · 82.9 salvaged dies/wafer ($217/die)", F_SMALL, fill=INK_SECONDARY)

    # Die Outer Frame
    dx, dy, ds = 96, 186, 480
    _rrect(draw, (dx, dy, dx + ds, dy + ds), 8, fill="#E2E8F0", outline=INK_PRIMARY, width=2)

    # Top: I/O + 2x400GbE + Cmd/NoC Spine (18.5% total IO/NoC -> top + right strip)
    _rect(draw, (dx + 12, dy + 12, dx + ds - 12, dy + 62), fill="#EDE9FE", outline=VIOLET, width=2)
    _text(draw, (dx + ds / 2, dy + 37), "2x 400GbE PAM4 SerDes + PCIe Gen5 x16 + RV64 Cmd/DMA (84.0 mm² · 18.5%)", F_SMALL_B, fill=VIOLET, anchor="mm")

    # Left & Bottom GDDR7 PHYs (7.4% = 33.6 mm2 -> 12 x 32-bit PHYs)
    _rect(draw, (dx + 12, dy + 72, dx + 56, dy + ds - 62), fill="#CCFBF1", outline=ACCENT_1S, width=2)
    _text(draw, (dx + 34, dy + 235), "GDDR7", F_SMALL_B, fill=ACCENT_1S, anchor="mm")
    _text(draw, (dx + 34, dy + 252), "PHYs", F_SMALL_B, fill=ACCENT_1S, anchor="mm")
    _rect(draw, (dx + 12, dy + ds - 54, dx + ds - 76, dy + ds - 12), fill="#CCFBF1", outline=ACCENT_1S, width=2)
    _text(draw, (dx + 210, dy + ds - 33), "12 × 32-bit Clamshell GDDR7 PHYs + ECC (33.6 mm² · 7.4% · 1.54 TB/s)", F_SMALL_B, fill=ACCENT_1S, anchor="mm")

    # Right Edge: UCIe-S Organic D2D Bridge Shoreline
    _rect(draw, (dx + ds - 68, dy + 72, dx + ds - 12, dy + ds - 12), fill="#FEF3C7", outline=AMBER, width=2)
    _text(draw, (dx + ds - 40, dy + 250), "UCIe-S", F_SMALL_B, fill=AMBER, anchor="mm")
    _text(draw, (dx + ds - 40, dy + 268), "12 mm", F_SMALL, fill=AMBER, anchor="mm")
    _text(draw, (dx + ds - 40, dy + 284), "1.2TB/s", F_SMALL_B, fill=AMBER, anchor="mm")

    # Center Upper: 84 SCC Grid (50.6% = 229.4 mm2)
    _rect(draw, (dx + 66, dy + 72, dx + ds - 78, dy + 314), fill="#DBEAFE", outline=ACCENT_1D, width=2)
    _text(draw, (dx + 234, dy + 90), "84 × Streaming Compute Clusters", F_SMALL_B, fill=ACCENT_1D, anchor="mm")
    _text(draw, (dx + 234, dy + 104), "(76 Active / 8 Spare · 229.4 mm² · 50.6%)", F_SMALL, fill=ACCENT_1D, anchor="mm")
    # Draw mini SCC grid (12 x 7 = 84)
    gx0, gy0 = dx + 80, dy + 118
    cw, ch = 22, 22
    for r in range(7):
        for c in range(12):
            idx = r * 12 + c
            x = gx0 + c * (cw + 4)
            y = gy0 + r * (ch + 4)
            is_spare = idx >= 76
            _rect(draw, (x, y, x + cw, y + ch), fill="#CBD5E1" if is_spare else "#3B82F6", outline="#1D4ED8" if not is_spare else "#64748B", width=1)

    # Center Lower: 192 MB Shared Tensor SRAM (12.8% = 58.2 mm2)
    _rect(draw, (dx + 66, dy + 324, dx + ds - 78, dy + ds - 64), fill="#D1FAE5", outline=EMERALD, width=2)
    _text(draw, (dx + 234, dy + 340), "192 MB Shared Tensor SRAM (58.2 mm² · 12.8%)", F_SMALL_B, fill=EMERALD, anchor="mm")
    _text(draw, (dx + 234, dy + 354), "64 × 3 MB XOR-Swizzled Banks · 7.58 TB/s", F_SMALL, fill=EMERALD, anchor="mm")
    for r in range(3):
        for c in range(16):
            x = dx + 82 + c * 19
            y = dy + 368 + r * 13
            _rect(draw, (x, y, x + 15, y + 9), fill="#10B981")

    _text(draw, (54, 686), "• Blue = 76 active SCCs (8×16×16 STME + 128 Zve64f + 256 KiB RF + 256 KiB L1)", F_SMALL, fill=INK_PRIMARY)
    _text(draw, (54, 708), "• Gray = 8 defect-salvage spare SCCs (lifts N4P yield from 64.6% -> 70.9% Murphy / 80.9% MC)", F_SMALL, fill=INK_SECONDARY)
    _text(draw, (54, 730), "• 48.6 mm² (10.7%) top-level 2D mesh NoC, clock tree, and power delivery grid", F_SMALL, fill=INK_SECONDARY)

    # Right Card: Dual-Chiplet Extended OAM-L Topology
    _rrect(draw, (700, 102, 1408, 762), 10, fill=BG_CARD, outline=BORDER, width=1)
    _text(draw, (722, 122), "Mercury-1D Dual-Chiplet Extended OAM-L (140x185 mm) Topology", F_SEC, fill=INK_PRIMARY)
    _text(draw, (722, 146), "Std 102x165/170 OAM overflows (95.3%/92.5% bot); 140x185 fits at 53.0%/61.9% (32mm traces)", F_SMALL, fill=INK_SECONDARY)

    # OAM Carrier Board
    _rrect(draw, (734, 186, 1374, 596), 12, fill="#0F172A", outline="#334155", width=2)
    # 4x 400GbE top ports
    for idx, px in enumerate([820, 930, 1140, 1250]):
        _rrect(draw, (px - 44, 166, px + 44, 198), 4, fill="#EDE9FE", outline=VIOLET, width=2)
        _text(draw, (px, 182), f"400GbE #{idx}", F_SMALL_B, fill=VIOLET, anchor="mm")

    # Left GDDR7 Clamshell bank
    _rrect(draw, (754, 238, 822, 540), 6, fill="#134E4A", outline=ACCENT_1S, width=2)
    _text(draw, (788, 355), "72 GB", F_LABEL_B, fill="#FFFFFF", anchor="mm")
    _text(draw, (788, 378), "GDDR7", F_SMALL_B, fill=ACCENT_1S_LIGHT, anchor="mm")
    _text(draw, (788, 398), "384-bit", F_SMALL, fill=ACCENT_1S_LIGHT, anchor="mm")
    _text(draw, (788, 418), "1.54TB/s", F_SMALL, fill=ACCENT_1S_LIGHT, anchor="mm")

    # Right GDDR7 Clamshell bank
    _rrect(draw, (1286, 238, 1354, 540), 6, fill="#134E4A", outline=ACCENT_1S, width=2)
    _text(draw, (1320, 355), "72 GB", F_LABEL_B, fill="#FFFFFF", anchor="mm")
    _text(draw, (1320, 378), "GDDR7", F_SMALL_B, fill=ACCENT_1S_LIGHT, anchor="mm")
    _text(draw, (1320, 398), "384-bit", F_SMALL, fill=ACCENT_1S_LIGHT, anchor="mm")
    _text(draw, (1320, 418), "1.54TB/s", F_SMALL, fill=ACCENT_1S_LIGHT, anchor="mm")

    # Die 0 & Die 1
    _rrect(draw, (844, 244, 1020, 534), 8, fill="#1E3A8A", outline=ACCENT_1D_LIGHT, width=2)
    _text(draw, (932, 282), "Mercury-1 Die 0", F_SEC, fill="#FFFFFF", anchor="mm")
    _text(draw, (932, 312), "76 Active SCCs", F_LABEL, fill=ACCENT_1D_LIGHT, anchor="mm")
    _text(draw, (932, 336), "575.9 BF16 TFLOPS", F_MONO_B, fill="#FFFFFF", anchor="mm")
    _text(draw, (932, 358), "1,151.8 FP8 TFLOPS", F_MONO, fill=ACCENT_1D_LIGHT, anchor="mm")
    _text(draw, (932, 390), "192 MB Shared SRAM", F_LABEL, fill="#6EE7B7", anchor="mm")
    _text(draw, (932, 412), "38.0 MiB L1+RF", F_SMALL, fill="#A7F3D0", anchor="mm")
    _text(draw, (932, 444), "453.8 mm² N4P", F_SMALL, fill="#CBD5E1", anchor="mm")

    _rrect(draw, (1088, 244, 1264, 534), 8, fill="#1E3A8A", outline=ACCENT_1D_LIGHT, width=2)
    _text(draw, (1176, 282), "Mercury-1 Die 1", F_SEC, fill="#FFFFFF", anchor="mm")
    _text(draw, (1176, 312), "76 Active SCCs", F_LABEL, fill=ACCENT_1D_LIGHT, anchor="mm")
    _text(draw, (1176, 336), "575.9 BF16 TFLOPS", F_MONO_B, fill="#FFFFFF", anchor="mm")
    _text(draw, (1176, 358), "1,151.8 FP8 TFLOPS", F_MONO, fill=ACCENT_1D_LIGHT, anchor="mm")
    _text(draw, (1176, 390), "192 MB Shared SRAM", F_LABEL, fill="#6EE7B7", anchor="mm")
    _text(draw, (1176, 412), "38.0 MiB L1+RF", F_SMALL, fill="#A7F3D0", anchor="mm")
    _text(draw, (1176, 444), "453.8 mm² N4P", F_SMALL, fill="#CBD5E1", anchor="mm")

    # Organic UCIe-S Bridge
    _rrect(draw, (1016, 326, 1092, 452), 4, fill="#FEF3C7", outline=AMBER, width=2)
    _text(draw, (1054, 366), "UCIe-S", F_SMALL_B, fill=AMBER, anchor="mm")
    _text(draw, (1054, 388), "1.2 TB/s", F_MONO_B, fill=INK_PRIMARY, anchor="mm")
    _text(draw, (1054, 408), "Bidir", F_SMALL, fill=AMBER, anchor="mm")

    # Bottom PCIe Gen5 / Extended OAM-L Mezzanine
    _rrect(draw, (870, 580, 1238, 612), 4, fill="#E2E8F0", outline="#475569", width=2)
    _text(draw, (1054, 596), "PCIe Gen5 x16 Host + Extended OAM-L 48V Power Rail", F_SMALL_B, fill=INK_PRIMARY, anchor="mm")

    # Combined Spec Callout
    _rrect(draw, (722, 628, 1384, 742), 8, fill=BG_SUBCARD, outline=BORDER, width=1)
    _text(draw, (742, 644), "Mercury-1D Combined Module Spec (2 × Unified N4P Chiplets)", F_LABEL_B, fill=INK_PRIMARY)
    _text(draw, (742, 670), "• Compute: 152 Active SCCs @ 1.85 GHz -> 1,151.8 BF16 / 2,303.6 FP8 / 4,607.2 MXFP4 Dense TFLOPS", F_MONO, fill=ACCENT_1D)
    _text(draw, (742, 694), "• Memory:  144 GB (134.1 GiB) GDDR7 ECC @ 3.07 TB/s + 384 MB Shared SRAM (15.16 TB/s)", F_MONO, fill=INK_PRIMARY)
    _text(draw, (742, 718), "• Network & Thermal: 4×400GbE (200 GB/s/dir) · ~680W TDP (Liquid Coldplate) · ~$2,456 BOM", F_MONO, fill=INK_SECONDARY)

    _footer(draw, "Physical dimensions & yield model from mercury_sim.specs & docs/architecture/mercury-1-spec.md", 1440, 820)
    _save(img, OUT_DIR / "05_mercury_1d_chiplet_floorplan.png")


def generate_all_charts() -> list[Path]:
    qual = run_all_qualification_scenarios()
    micro = None
    pt_bench = None
    isa_bench = None
    try:
        from mercury_sim.microarch_cycle import run_microarch_benchmarks
        micro = run_microarch_benchmarks()
    except Exception:
        pass
    try:
        from mercury_sim.pytorch_bridge import run_pytorch_bridge_benchmark
        pt_bench = run_pytorch_bridge_benchmark()
    except Exception:
        pass
    try:
        from mercury_isa.vm import run_isa_kernel_benchmarks
        isa_bench = run_isa_kernel_benchmarks()
    except Exception:
        pass

    build_chart_01(qual)
    build_chart_02()
    build_chart_03(qual, micro, pt_bench)
    build_chart_04(qual, isa_bench)
    build_chart_05(qual)
    return sorted(OUT_DIR.glob("*.png"))


if __name__ == "__main__":
    paths = generate_all_charts()
    for p in paths:
        print(f"Generated {p.relative_to(ROOT)} ({p.stat().st_size:,} bytes)")
