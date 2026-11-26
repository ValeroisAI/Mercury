"""Verify a directed dual-bank read/write and reset sequence in 12 steps."""

from __future__ import annotations

from check_sram_rtl import run_yosys_jobs

_DEFAULTS = {
    "rst": (1, 0),
    "req_valid": (4, 0),
    "req_write": (4, 0),
    "req_line": (32, 0),
    "req_wdata": (128, 0x87654321_89ABCDEF_01234567_DEADBEEF),
    "req_wstrb": (16, 0xFA5D),
    "bank_ready": (4, 0xF),
    "bank_read_valid_in": (4, 0),
    "bank_rdata": (128, 0xCAFEFEED_C0FFEE01_01234567_89012345),
    "resp_ready": (4, 0),
}
_STEPS = {
    1: {"rst": 1},
    # Parallel bank 0 and bank 2 reads, and a masked bank 1 write.
    2: {"req_valid": 0x7, "req_write": 0x2, "req_line": 0x00000408},
    # The first macro returns and immediately takes another read from requester 3.
    3: {"req_valid": 0x9, "req_write": 0x1, "req_line": 0x24000004, "bank_ready": 0xD,
        "bank_read_valid_in": 0x1},
    # Bank 0 remains busy until step 6. Bank 2 returns independently.
    4: {"req_valid": 0x1, "req_write": 0x1, "req_line": 0x00000004, "bank_read_valid_in": 0x4,
        "bank_rdata": 0xCAFEFEED_FFFFFFFF_00000000_FEDCBA98},
    5: {"resp_ready": 0x4, "req_valid": 0x1, "req_write": 0x1, "req_line": 0x00000004},
    # Bank 0 returns as requester 0 consumes its response and issues its write.
    6: {"resp_ready": 0x1, "req_valid": 0x5, "req_write": 0x5, "req_line": 0x00240004,
        "bank_read_valid_in": 0x1},
    7: {"req_valid": 0x6, "req_write": 0x4, "req_line": 0x00240C00},
    # Reset with requester 1's read in flight, then submit a new read.
    8: {"rst": 1},
    9: {"req_valid": 0x2, "req_line": 0x00000800},
    10: {"bank_read_valid_in": 0x4},
    11: {"resp_ready": 0x2},
}
_PROBES = (
    "PROBE_TWO_BANKS", "PROBE_HELD_STALL", "PROBE_WRITE_AFTER_READ", "PROBE_BANK_REFILL",
    "PROBE_RESET_FLUSH",
)


def _literal(name: str, value: int) -> str:
    width, _ = _DEFAULTS[name]
    return f"{width}'h{value:x}"


def _job(probe: str = "") -> str:
    define = f"-D{probe} " if probe else ""
    verification = "-falsify" if probe else "-verify"
    defaults = " ".join(f"-set {name} {_literal(name, value)}" for name, (_, value) in _DEFAULTS.items())
    overrides = " ".join(
        f"-set-at {step} {name} {_literal(name, value)}"
        for step, fields in _STEPS.items() for name, value in fields.items()
    )
    return (
        f"read_verilog -defer -formal -sv {define}"
        "rtl/mercury_sram_arbiter.sv rtl/mercury_sram_payload.sv tests/rtl/sram_payload_tb.sv; "
        "prep -top sram_payload_tb -flatten; memory_map; opt; chformal -lower; "
        f"sat {verification} -prove-asserts -seq 12 -set-def-inputs {defaults} {overrides}"
    )


if __name__ == "__main__":
    raise SystemExit(run_yosys_jobs(__doc__ or "", (_job(), *(_job(probe) for probe in _PROBES))))
