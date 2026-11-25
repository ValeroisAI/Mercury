"""Two-pass assembler for the Mercury reference VM."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re

from .encoding import ISA, decode_instruction, encode_instruction

_LABEL = re.compile(r"[A-Za-z_.$][A-Za-z0-9_.$]*")
_ADDRESS = re.compile(r"(?P<offset>[^\s()]*)\s*\(\s*(?P<base>x[0-9]+)\s*\)", re.IGNORECASE)
_RELATIVE = {"beq": (2,), "bne": (2,), "blt": (2,), "jal": (1,), "m.split": (1, 2), "m.wspawn": (1,)}


class AssemblyError(ValueError):
    """The input contains an invalid mnemonic, operand, or label."""


@dataclass(frozen=True)
class AssembledProgram:
    words: tuple[int, ...]
    labels: dict[str, int]
    source_lines: tuple[int, ...]
    isa: str = ISA

    def to_bytes(self) -> bytes:
        """Return a little-endian instruction image."""
        return b"".join(word.to_bytes(4, "little") for word in self.words)


def _parse_operands(mnemonic: str, operand_text: str) -> list[int | str]:
    operands: list[int | str] = [part.strip() for part in operand_text.split(",")] if operand_text else []
    if len(operands) >= 2:
        match = _ADDRESS.fullmatch(str(operands[1]))
        if match:
            offset = match["offset"]
            operands[1] = match["base"]
            if mnemonic in ("ld", "sd", "jalr"):
                operands.insert(2, offset or "0")
            elif offset or not mnemonic.startswith(("vle", "vse")):
                raise ValueError("Expected a unit-stride vector address (xN)")
    return operands


def assemble(source: str, *, filename: str = "<string>") -> AssembledProgram:
    """Assemble source into words, byte-addressed labels, and source line numbers."""
    labels: dict[str, int] = {}
    instructions: list[tuple[int, str]] = []
    for number, line in enumerate(source.splitlines(), 1):
        line = re.split(r"[#;]", line, maxsplit=1)[0].strip()
        while ":" in line:
            label, line = (part.strip() for part in line.split(":", 1))
            if not _LABEL.fullmatch(label) or label in labels:
                raise AssemblyError(f"{filename}:{number}: Invalid or duplicate label {label!r}")
            labels[label] = len(instructions) * 4
        if line:
            instructions.append((number, line))
    words = []
    for index, (number, line) in enumerate(instructions):
        mnemonic, *rest = line.split(maxsplit=1)
        mnemonic = mnemonic.lower()
        try:
            operands = _parse_operands(mnemonic, rest[0] if rest else "")
            for position in _RELATIVE.get(mnemonic, ()):
                if position < len(operands) and _LABEL.fullmatch(str(operands[position])):
                    target = str(operands[position])
                    if target not in labels:
                        raise ValueError(f"Undefined label {target!r}")
                    operands[position] = labels[target] - index * 4
            words.append(encode_instruction(mnemonic, *operands))
        except ValueError as exc:
            raise AssemblyError(f"{filename}:{number}: {exc}") from exc
    return AssembledProgram(tuple(words), labels, tuple(number for number, _ in instructions))


def assemble_file(path: str | Path) -> AssembledProgram:
    """Assemble a UTF-8 MASM source file."""
    path = Path(path)
    return assemble(path.read_text(encoding="utf-8-sig"), filename=str(path))


def disassemble(words: Sequence[int] | AssembledProgram) -> str:
    """Return canonical source whose reassembly preserves every word."""
    if isinstance(words, AssembledProgram):
        words = words.words
    return "\n".join(str(decode_instruction(word)) for word in words)
