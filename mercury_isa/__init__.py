"""Assemble and run Mercury's open, 32-bit RISC-V instruction subset.

Run the examples from the repository root:
    python -c "from mercury_isa import run_isa_kernel_benchmarks; print(run_isa_kernel_benchmarks())"

For an empty standalone launch, use MercuryVM().run(assemble("ret").words).
"""

from .assembler import AssembledProgram, AssemblyError, assemble, assemble_file, disassemble
from .encoding import ISA, Instruction, InstructionDecodeError, decode_instruction, encode_instruction
from .vm import (
    DMADescriptor,
    HazardViolationError,
    MemoryRef,
    MercuryVM,
    Telemetry,
    Tile,
    VMExecutionError,
    run_isa_kernel_benchmarks,
)

__all__ = [
    "ISA",
    "AssembledProgram",
    "AssemblyError",
    "DMADescriptor",
    "HazardViolationError",
    "Instruction",
    "InstructionDecodeError",
    "MemoryRef",
    "MercuryVM",
    "Telemetry",
    "Tile",
    "VMExecutionError",
    "assemble",
    "assemble_file",
    "decode_instruction",
    "disassemble",
    "encode_instruction",
    "run_isa_kernel_benchmarks",
]
