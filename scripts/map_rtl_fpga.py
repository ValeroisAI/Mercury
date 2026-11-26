"""Report out-of-context UltraScale+ primitive counts for Mercury RTL blocks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
_DOT_DEPENDENCIES = (
    "mercury_fp8_product",
    "mercury_fp32_add_rne",
    "mercury_fp32_fma_rne",
    "mercury_stme_group_sum",
    "mercury_stme_boundary_accumulator",
)
_BLOCKS = {
    "mercury_fp8_product": (),
    "mercury_fp32_add_rne": (),
    "mercury_fp32_add_pipe": (),
    "mercury_fp32_fma_pipe": (),
    "mercury_stme_interleaved_sum": ("mercury_fp8_product", "mercury_fp32_add_pipe"),
    "mercury_stme_boundary_accumulator": ("mercury_fp32_fma_rne",),
    "mercury_stme_fp8_dot": _DOT_DEPENDENCIES,
    "mercury_stme_scaled_fp8_dot": (*_DOT_DEPENDENCIES, "mercury_stme_fp8_dot"),
    "mercury_dma_tokens": (),
    "mercury_sram_arbiter": (),
    "mercury_sram_bram": ("mercury_sram_arbiter", "mercury_sram_payload"),
}


def _map(executable: Path, top: str, env: dict[str, str], path: Path) -> dict[str, object]:
    sources = " ".join(f"rtl/{module}.sv" for module in (*_BLOCKS[top], top))
    relative = path.relative_to(ROOT).as_posix()
    script = (
        f"read_verilog -sv {sources}; "
        f"synth_xilinx -top {top} -family xcup -flatten -noiopad -noclkbuf; "
        f"check -assert; tee -o {relative} stat -json"
    )
    run = subprocess.run(
        [str(executable), "-Q", "-T", "-p", script],
        cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if run.returncode or not path.exists():
        raise RuntimeError(f"Yosys mapping failed for {top}:\n" + "\n".join(run.stdout.splitlines()[-30:]))
    data = json.loads(path.read_text(encoding="utf-8"))
    stats = data["modules"]["\\" + top]
    cells = stats["num_cells_by_type"]
    return {
        "tool": data["creator"],
        "lut_primitives": sum(n for cell, n in cells.items() if cell.startswith("LUT")),
        "flip_flops": sum(n for cell, n in cells.items() if cell.startswith("FD")),
        "dsp_primitives": sum(n for cell, n in cells.items() if cell.startswith("DSP")),
        "cells_by_type": cells,
    }


def main() -> int:
    """Map selected RTL modules and print machine-readable primitive counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yosys", default=shutil.which("yosys") or shutil.which("yowasp-yosys"))
    parser.add_argument(
        "--block", action="append", choices=list(_BLOCKS),
        help="Select a block. Repeat to select more. The default maps all listed blocks.",
    )
    args = parser.parse_args()
    executable = shutil.which(args.yosys) if args.yosys else None
    if executable is None:
        parser.error("Install Yosys or pass its path with --yosys")
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    env = {**os.environ, "YOWASP_CACHE_DIR": str(build / "rtl-cache")}
    with tempfile.TemporaryDirectory(prefix="fpga-map-", dir=build) as folder:
        result = {
            top: _map(Path(executable).resolve(), top, env, Path(folder) / f"{top}.json")
            for top in args.block or _BLOCKS
        }
    print(json.dumps({"family": "xcup", "out_of_context": True, "blocks": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
