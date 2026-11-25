"""Mercury-1 courtyard screening and NRE-adjusted campaign economics.

Known inputs, high confidence:
- OCP OAM r2.0 v1.0, p. 10, lists "Module Dimension | 102mm x 170mm".
  https://www.opencompute.org/documents/oai-oam-base-specification-r2-0-v1-0-20230919-pdf
  The requested 102 x 165 mm scenarios use the earlier v1.5 envelope
  (Figure 2, "102mm wide OAM Form Factor Dimensions, Bottom View").
  https://www.opencompute.org/documents/ocp-accelerator-module-design-specification-v1p5-final-20220223-docx-1-pdf/
- Micron lists a "266-ball FBGA, 12.0mm x 14.0mm x 1.1mm" package.
  https://assets.micron.com/adobe/assets/urn%3Aaaid%3Aaem%3A087330f6-6d71-4575-b622-cb10f20cdaf0/original/as/gddr7-product-brief.pdf

Inferred, medium confidence within the scenario: the task-brief courtyard,
PDN, NRE and trace allowances are engineering assumptions. Hardware BOM
and the training campaign inputs come from specs.py and cluster_bench.py.
Unknown: routing and electrical sign-off, actual purchase costs and yield.
Package ball maps, a real PCB stackup, SI/PI results and supplier quotations
would replace those assumptions.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import combinations
import math

from .specs import MERCURY_1D, MERCURY_1S, MercurySKU

STANDARD_OAM_48 = "standard_oam_2_0_48_chips"
EXTENDED_OAM_48 = "extended_oam_l_140x185_48_chips"
STANDARD_OAM_32 = "standard_oam_2_0_96gb_32chips"
PCIE_24 = "pcie_dual_slot_111x267_24_chips"
OAM_R2_48 = "ocp_oam_r2_102x170_48_chips"

DEFAULT_VOLUMES = (10_000, 25_000, 50_000, 100_000, 250_000)
NRE_MASK_AND_TAPE_OUT_USD = 35_000_000
NRE_HIGH_SPEED_IP_USD = 22_000_000
NRE_SOFTWARE_USD = 28_000_000
DEFAULT_NRE_USD = NRE_MASK_AND_TAPE_OUT_USD + NRE_HIGH_SPEED_IP_USD + NRE_SOFTWARE_USD
_HOST_FABRIC_USD_PER_GPU = 800.0  # Same allowance as cluster_bench.py.
_LIFETIME_DAYS = 3 * 365
_CAMPAIGN_MILLION_TOKENS = 1_000_000

_EDGE_CLEARANCE_MM = 3.0
_MAX_SIDE_FILL_PCT = 75.0
_MAX_PAM3_TRACE_MM = 35.0  # Project screening budget, conditional on stackup and ball map.
_GDDR_COURTYARD_MM = (17.0, 15.0)  # 15 x 17 mm envelope in the modeled orientation.
_PATH_ALLOWANCE_MM = 8.0  # Bundle escape, bends and length matching.
_BACKSIDE_VIA_ALLOWANCE_MM = 1.5
_VCORE_V = 0.78
_REFERENCE_DUAL_VRM_AREA_MM2 = 3_800.0
_REFERENCE_DUAL_CORE_CURRENT_A = 680.0 / _VCORE_V


@dataclass(frozen=True)
class PCBCourtyardResult:
    form_factor: str
    board_mm: tuple[int, int]
    top_used_pct: float
    bottom_used_pct: float
    max_gddr7_trace_mm: float
    fits_physically: bool
    blocking_reason: str | None


@dataclass(frozen=True)
class VolumeEconomicsRow:
    volume_units: int
    hardware_bom_usd: float
    nre_per_unit_usd: float
    fully_loaded_unit_cost_usd: float
    target_price_45gm_usd: float
    gpu_capex_64gpu_usd: float
    capex_64gpu_usd: float | None
    campaign_1t_cost_per_1m_tokens_usd: float | None
    h100_campaign_cost_per_1m_tokens_usd: float | None
    b200_campaign_cost_per_1m_tokens_usd: float | None


@dataclass(frozen=True)
class _Layout:
    board_mm: tuple[int, int]
    packages_per_side: int
    dies: int
    package_mm: tuple[int, int]
    oam: bool


_LAYOUTS = {
    STANDARD_OAM_48: _Layout((102, 165), 24, 2, (62, 42), True),
    EXTENDED_OAM_48: _Layout((140, 185), 24, 2, (62, 42), True),
    STANDARD_OAM_32: _Layout((102, 165), 16, 2, (62, 42), True),
    PCIE_24: _Layout((111, 267), 12, 1, (40, 40), False),
    OAM_R2_48: _Layout((102, 170), 24, 2, (62, 42), True),
}


@dataclass(frozen=True)
class _Rect:
    label: str
    x: float
    y: float
    width: float
    height: float
    edge_clearance_mm: float = _EDGE_CLEARANCE_MM

    @property
    def area(self) -> float:
        return self.width * self.height

    def within(self, board_mm: tuple[int, int]) -> bool:
        margin = self.edge_clearance_mm
        return (
            abs(self.x) + self.width / 2 + margin <= board_mm[0] / 2
            and abs(self.y) + self.height / 2 + margin <= board_mm[1] / 2
        )

    def overlaps(self, other: _Rect) -> bool:
        return (
            abs(self.x - other.x) < (self.width + other.width) / 2
            and abs(self.y - other.y) < (self.height + other.height) / 2
        )


def _memory_layout(layout: _Layout) -> list[tuple[_Rect, str]]:
    """Place each clamshell pair at identical XY centers on the two faces."""
    width, height = _GDDR_COURTYARD_MM
    half_w, half_h = (value / 2 for value in layout.package_mm)
    rows_per_end = 1 if layout.packages_per_side == 12 else 2
    slots = [
        ((column - 1.5) * width, end * (half_h + 1 + (row + 0.5) * height), "end")
        for end in (-1, 1)
        for row in range(rows_per_end)
        for column in range(4)
    ]
    side_count = (layout.packages_per_side - len(slots)) // 2
    slots.extend(
        (side * (half_w + 3 + width / 2), (index - (side_count - 1) / 2) * height, "side")
        for side in (-1, 1)
        for index in range(side_count)
    )
    return [(_Rect(f"GDDR7 {i + 1}", x, y, width, height), facing)
            for i, (x, y, facing) in enumerate(slots)]


def _trace_length(rect: _Rect, facing: str, layout: _Layout) -> float:
    """Estimate a Manhattan ball-escape path to opposed package PHY landing bands."""
    launch_x, launch_y = (value / 2 - 4 for value in layout.package_mm)
    dx, dy = abs(rect.x), abs(rect.y)
    if facing == "end":
        distance = max(0.0, dx - launch_x) + max(0.0, dy - 5 - launch_y)
    else:
        distance = max(0.0, dx - 5 - launch_x) + max(0.0, dy - launch_y)
    return distance + _PATH_ALLOWANCE_MM + _BACKSIDE_VIA_ALLOWANCE_MM


def _footprints(sku: MercurySKU, layout: _Layout) -> tuple[list[_Rect], list[_Rect], float]:
    memory = _memory_layout(layout)
    top = [rect for rect, _ in memory]
    bottom = top.copy()
    package = _Rect("Compute package", 0, 0, *layout.package_mm)
    top.append(package)
    bottom.append(_Rect("Core backside decoupling", 0, 0, *layout.package_mm))

    half_board_x, half_board_y = (value / 2 for value in layout.board_mm)
    # Assign the full board power target to Vcore for a conservative
    # current envelope. The real regulator loads are split across rails.
    core_current = sku.tdp_w / _VCORE_V
    vrm_width = 76.0
    bank_height = _REFERENCE_DUAL_VRM_AREA_MM2 * core_current / _REFERENCE_DUAL_CORE_CURRENT_A / (2 * vrm_width)
    for end in (-1, 1):
        top.append(_Rect("VRM bank", 0, end * (half_board_y - 4 - bank_height / 2), vrm_width, bank_height))
    if layout.oam:
        for end in (-1, 1):
            bottom.append(_Rect("Mirror Mezz and stiffener reserve", 0, end * (half_board_y - 15.5), 96, 25))
        if layout.packages_per_side == 24:
            for side in (-1, 1):
                # Outer strips reserve dense 768-bit escape and supporting
                # backside passives alongside the side memory groups.
                bottom.append(_Rect("Full-bus backside routing and decoupling", side * 59.5, 0, 15, 84))
        retimer_x = 60 if layout.packages_per_side == 24 else 41
        for side in (-1, 1):
            top.append(_Rect("Ethernet retimers and caps", side * retimer_x, 0, 12, 50))
    else:
        fingers = _Rect("PCIe edge fingers and keep-out", -half_board_x + 5, 0, 10, 90, 0)
        top.append(fingers)
        bottom.append(fingers)
        if sku.eth_400g_ports:
            top.append(_Rect("Ethernet retimers and caps", 0, 56, 30, 10 * sku.eth_400g_ports))
    return top, bottom, max(_trace_length(rect, facing, layout) for rect, facing in memory)


def _check_face(name: str, rects: list[_Rect], board_mm: tuple[int, int], used_pct: float) -> list[str]:
    reasons = []
    if used_pct >= _MAX_SIDE_FILL_PCT:
        reasons.append(f"{name} fill {used_pct:.2f}% exceeds the routing allowance (<{_MAX_SIDE_FILL_PCT:g}%)")
    outside = sorted({rect.label for rect in rects if not rect.within(board_mm)})
    if outside:
        reasons.append(f"{name} edge clearance: {', '.join(outside)}")
    overlaps = sorted({f"{a.label} / {b.label}" for a, b in combinations(rects, 2) if a.overlaps(b)})
    if overlaps:
        reasons.append(f"{name} courtyard overlap: {', '.join(overlaps)}")
    return reasons


def analyze_pcb_courtyard(
    sku: MercurySKU = MERCURY_1D, form_factor: str | None = None
) -> PCBCourtyardResult:
    """Screen a concrete placement on both PCB faces and its longest estimated trace.

    Defaults to the extended 1D mezzanine or full-length 1S PCIe card.
    The legacy standard key names are retained for callers. The 96 GB case
    models a 512-bit interface with the same dual-die power/package envelope.
    """
    key = form_factor if form_factor is not None else (PCIE_24 if sku.num_dies == 1 else EXTENDED_OAM_48)
    layout = _LAYOUTS.get(key)
    if layout is None:
        raise ValueError(f"Unknown PCB form factor: {key!r}")
    if sku.num_dies != layout.dies:
        raise ValueError(f"{key} requires a {layout.dies}-die SKU")
    top, bottom, max_trace = _footprints(sku, layout)
    area = math.prod(layout.board_mm)
    top_pct = 100 * sum(rect.area for rect in top) / area
    bottom_pct = 100 * sum(rect.area for rect in bottom) / area
    reasons = _check_face("top", top, layout.board_mm, top_pct)
    reasons += _check_face("bottom", bottom, layout.board_mm, bottom_pct)
    if max_trace > _MAX_PAM3_TRACE_MM:
        reasons.append(f"estimated PAM3 trace {max_trace:g} mm exceeds {_MAX_PAM3_TRACE_MM:g} mm")
    return PCBCourtyardResult(key, layout.board_mm, top_pct, bottom_pct, max_trace, not reasons,
                              "; ".join(reasons) or None)


def analyze_nre_and_tco(
    sku: MercurySKU = MERCURY_1D, volumes: Iterable[int] = DEFAULT_VOLUMES, *,
    nre_usd: float = DEFAULT_NRE_USD,
) -> tuple[VolumeEconomicsRow, ...]:
    """Reprice the existing 64-device 1D campaign at a 45% gross margin.

    Production volume is the total number of units sharing all NRE in the
    scenario. GPU CAPEX includes the modeled unit price. System CAPEX adds
    the existing $800 per GPU host/fabric allowance. Campaign cost is the
    share of system CAPEX depreciated over the modeled run plus electricity.
    System CAPEX and campaign columns apply to the benchmarked 1D SKU.
    """
    volumes = tuple(volumes)
    if not volumes or any(type(volume) is not int or volume <= 0 for volume in volumes):
        raise ValueError("Production volumes must be positive integers")
    if not math.isfinite(nre_usd) or nre_usd < 0:
        raise ValueError("NRE must be finite and nonnegative")
    if sku.num_dies not in (1, 2):
        raise ValueError("Expected a Mercury-1S or Mercury-1D SKU")

    has_1d_campaign = sku == MERCURY_1D
    if has_1d_campaign:
        from .cluster_bench import simulate_1t_token_pretraining_tco

        mercury, h100, b200 = simulate_1t_token_pretraining_tco()
        depreciation = mercury.training_days / _LIFETIME_DAYS

    rows = []
    for volume in volumes:
        nre_per_unit = nre_usd / volume
        unit_cost = sku.total_bom_usd + nre_per_unit
        price = unit_cost / (1 - 0.45)
        gpu_capex = 64 * price
        system_capex = campaign_cost = h100_cost = b200_cost = None
        if has_1d_campaign:
            system_capex = gpu_capex + 64 * _HOST_FABRIC_USD_PER_GPU
            campaign_cost = (system_capex * depreciation + mercury.energy_cost_usd) / _CAMPAIGN_MILLION_TOKENS
            h100_cost = h100.cost_per_million_tokens_usd
            b200_cost = b200.cost_per_million_tokens_usd
        rows.append(VolumeEconomicsRow(volume, sku.total_bom_usd, nre_per_unit, unit_cost, price,
                                      gpu_capex, system_capex, campaign_cost, h100_cost, b200_cost))
    return tuple(rows)


def run_physical_bom_analysis() -> dict[str, object]:
    """Return JSON-native PCB and economics diagnostics for the qualification runner."""
    layouts = {}
    for key, layout in _LAYOUTS.items():
        result = analyze_pcb_courtyard(MERCURY_1S if layout.dies == 1 else MERCURY_1D, key)
        layouts[key] = {**asdict(result), "board_mm": list(result.board_mm)}
    return {
        "pcb_layouts": layouts,
        "nre_volume_table": [asdict(row) for row in analyze_nre_and_tco()],
        "nre_breakdown_usd": {
            "mask_and_tape_out": NRE_MASK_AND_TAPE_OUT_USD,
            "high_speed_ip_and_verification": NRE_HIGH_SPEED_IP_USD,
            "software_bring_up": NRE_SOFTWARE_USD,
            "total": DEFAULT_NRE_USD,
        },
    }
