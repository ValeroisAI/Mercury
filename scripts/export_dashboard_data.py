"""Export verified simulation telemetry for the Mercury-1 interactive engineering workbench."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mercury_sim.cluster_bench import run_all_qualification_scenarios
from mercury_sim.specs import MERCURY_1D, MERCURY_1S
from mercury_sim.thermal_power import simulate_thermal_run


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


def build_dashboard_payload() -> dict[str, Any]:
    qual = run_all_qualification_scenarios()

    # Full 0-360s transient thermal waveforms for interactive plotting
    thermal_traces = {
        "oam_liquid_35c": _to_jsonable(
            simulate_thermal_run(MERCURY_1D, "oam_liquid_35c", "training_fp8", duration_s=360.0)
        ),
        "workstation_air_25c": _to_jsonable(
            simulate_thermal_run(MERCURY_1S, "workstation_air_25c", "training_fp8", duration_s=360.0)
        ),
        "server_4u_air_35c": _to_jsonable(
            simulate_thermal_run(MERCURY_1D, "server_4u_air_35c", "training_fp8", duration_s=360.0)
        ),
        "mid_run_fault_120s": _to_jsonable(
            simulate_thermal_run(
                MERCURY_1D,
                "oam_liquid_35c",
                "training_fp8",
                duration_s=360.0,
                mid_run_cooling_fault_s=120.0,
                fault_cooling="degraded_fan_fault",
            )
        ),
    }

    microarch = None
    try:
        from mercury_sim.microarch_cycle import run_microarch_benchmarks
        microarch = run_microarch_benchmarks()
    except Exception:
        pass

    pytorch_bridge = None
    try:
        from mercury_sim.pytorch_bridge import run_pytorch_bridge_benchmark
        pytorch_bridge = run_pytorch_bridge_benchmark()
    except Exception:
        pass

    isa_vm = None
    try:
        from mercury_isa.vm import run_isa_kernel_benchmarks
        isa_vm = run_isa_kernel_benchmarks()
    except Exception:
        pass

    physical_bom = None
    try:
        from mercury_sim.physical_bom import run_physical_bom_analysis
        physical_bom = run_physical_bom_analysis()
    except Exception:
        pass

    skus = {
        "Mercury-1S": {
            "name": MERCURY_1S.name,
            "form_factor": MERCURY_1S.form_factor,
            "dies": MERCURY_1S.num_dies,
            "active_sccs": MERCURY_1S.total_active_sccs,
            "fab_sccs": MERCURY_1S.fab_sccs_per_die * MERCURY_1S.num_dies,
            "clock_ghz": MERCURY_1S.nominal_clock_ghz,
            "bf16_tflops": round(MERCURY_1S.peak_tflops("bf16"), 1),
            "fp8_tflops": round(MERCURY_1S.peak_tflops("fp8"), 1),
            "mxfp4_tflops": round(MERCURY_1S.peak_tflops("mxfp4"), 1),
            "fp32_vec_tflops": round(MERCURY_1S.peak_tflops("fp32"), 1),
            "vram_gb": MERCURY_1S.vram_gb_total,
            "vram_gib": round(MERCURY_1S.vram_gib_total, 1),
            "vram_bw_tbs": round(MERCURY_1S.vram_bandwidth_tbs, 2),
            "shared_sram_mb": MERCURY_1S.shared_sram_mb_total,
            "shared_sram_bw_tbs": round(MERCURY_1S.shared_sram_bandwidth_tbs(), 2),
            "l1_rf_mib": round(MERCURY_1S.active_l1_mib_total + MERCURY_1S.active_rf_mib_total, 1),
            "eth_ports_400g": MERCURY_1S.eth_400g_ports,
            "tdp_w": MERCURY_1S.tdp_w,
            "bom_usd": round(MERCURY_1S.total_bom_usd, 0),
        },
        "Mercury-1D": {
            "name": MERCURY_1D.name,
            "form_factor": "Extended OAM-L Liquid Module (140 x 185 mm)",
            "dies": MERCURY_1D.num_dies,
            "active_sccs": MERCURY_1D.total_active_sccs,
            "fab_sccs": MERCURY_1D.fab_sccs_per_die * MERCURY_1D.num_dies,
            "clock_ghz": MERCURY_1D.nominal_clock_ghz,
            "bf16_tflops": round(MERCURY_1D.peak_tflops("bf16"), 1),
            "fp8_tflops": round(MERCURY_1D.peak_tflops("fp8"), 1),
            "mxfp4_tflops": round(MERCURY_1D.peak_tflops("mxfp4"), 1),
            "fp32_vec_tflops": round(MERCURY_1D.peak_tflops("fp32"), 1),
            "vram_gb": MERCURY_1D.vram_gb_total,
            "vram_gib": round(MERCURY_1D.vram_gib_total, 1),
            "vram_bw_tbs": round(MERCURY_1D.vram_bandwidth_tbs, 2),
            "shared_sram_mb": MERCURY_1D.shared_sram_mb_total,
            "shared_sram_bw_tbs": round(MERCURY_1D.shared_sram_bandwidth_tbs(), 2),
            "l1_rf_mib": round(MERCURY_1D.active_l1_mib_total + MERCURY_1D.active_rf_mib_total, 1),
            "eth_ports_400g": MERCURY_1D.eth_400g_ports,
            "tdp_w": MERCURY_1D.tdp_w,
            "bom_usd": round(MERCURY_1D.total_bom_usd, 0),
        },
    }

    return {
        "skus": skus,
        "qualification": _to_jsonable(qual),
        "thermal_traces": thermal_traces,
        "microarch": _to_jsonable(microarch),
        "pytorch_bridge": _to_jsonable(pytorch_bridge),
        "isa_vm": _to_jsonable(isa_vm),
        "physical_bom": _to_jsonable(physical_bom),
    }


def main() -> int:
    payload = build_dashboard_payload()
    out_path = ROOT / "dashboard" / "data" / "qualification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Exported {out_path.relative_to(ROOT)} ({out_path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
