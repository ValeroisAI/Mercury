"""The 32-bit Mercury VM encoding. Xmer subopcodes are a simulator ABI.

Standard words follow the RISC-V opcode tables and Zvfbfmin specification
(https://github.com/riscv/riscv-opcodes and
https://github.com/riscv/riscv-isa-manual/blob/main/src/unpriv/zvfbfmin.adoc).
All PC-relative operands are byte offsets from the instruction.

m.split takes a predicate register and two signed, word-aligned byte offsets
for the else and join labels. m.wspawn takes a count register and child
entry offset. m.allreduce accepts an optional scope (0 warp, 1 workgroup).
vfexp.v is a Mercury helper, and the shorter BF16 names are assembler aliases.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

ISA = "rv64imafdc_zve64f_zvfbfmin_xmercoll_xmerdma_xmersimt_xmerstme"


class InstructionDecodeError(ValueError):
    """The word does not encode an instruction in the supported subset."""


@dataclass(frozen=True)
class _Field:
    # Each tuple maps a run of operand bits to a run of instruction bits.
    parts: tuple[tuple[int, int, int], ...]
    kind: str = ""
    signed: bool = False
    alignment: int = 1

    @property
    def mask(self) -> int:
        return sum(((1 << n) - 1) << dest for dest, n, _ in self.parts)

    @property
    def width(self) -> int:
        return max(n + source for _, n, source in self.parts)

    def insert(self, value: int | str) -> int:
        if isinstance(value, str) and self.kind in ("x", "v", "t"):
            if not re.fullmatch(self.kind + r"(0|[1-9][0-9]*)", value, re.IGNORECASE):
                raise ValueError(f"Expected a {self.kind} register, got {value!r}")
            value = int(value[1:])
        if isinstance(value, str):
            value = int(value, 0)
        limit = 8 if self.kind == "t" else 1 << (self.width - self.signed)
        lower = -limit if self.signed else 0
        if type(value) is not int or not lower <= value < limit or value % self.alignment:
            detail = f", aligned to {self.alignment} bytes" if self.alignment > 1 else ""
            raise ValueError(f"Operand {value!r} must be in [{lower}, {limit - 1}]{detail}")
        return sum(((value >> source) & ((1 << n) - 1)) << dest for dest, n, source in self.parts)

    def extract(self, word: int) -> int:
        value = sum(((word >> dest) & ((1 << n) - 1)) << source for dest, n, source in self.parts)
        if self.signed and value & (1 << (self.width - 1)):
            value -= 1 << self.width
        self.insert(value)
        return value


def _reg(kind: str, position: int) -> _Field:
    return _Field(((position, 5, 0),), kind)


XD, X1, X2 = (_reg("x", pos) for pos in (7, 15, 20))
VD, V1, V2 = (_reg("v", pos) for pos in (7, 15, 20))
TD, T1, T2 = (_reg("t", pos) for pos in (7, 15, 20))
I12 = _Field(((20, 12, 0),), signed=True)
S12 = _Field(((7, 5, 0), (25, 7, 5)), signed=True)
B13 = _Field(((8, 4, 1), (25, 6, 5), (7, 1, 11), (31, 1, 12)), signed=True, alignment=2)
J21 = _Field(((21, 10, 1), (20, 1, 11), (12, 8, 12), (31, 1, 20)), signed=True, alignment=2)
SPLIT_ELSE = _Field(((16, 8, 2),), signed=True, alignment=4)
SPLIT_JOIN = _Field(((24, 8, 2),), signed=True, alignment=4)
SPAWN_PC = _Field(((15, 17, 2),), signed=True, alignment=4)
BARRIER_ID = _Field(((15, 5, 0),))
SCOPE = _Field(((25, 1, 0),))
VM = _Field(((25, 1, 0),), "vm")


@dataclass(frozen=True)
class _Form:
    fixed: int
    fields: tuple[_Field, ...]
    optional_last: int | None = None

    @property
    def mask(self) -> int:
        return 0xFFFFFFFF ^ sum(field.mask for field in self.fields)


def _r(opcode: int, funct3: int = 0, funct7: int = 0) -> int:
    return opcode | (funct3 << 12) | (funct7 << 25)


_FORMS = {
    "ret": _Form(0x00008067, ()),
    "addi": _Form(_r(0x13), (XD, X1, I12)),
    "add": _Form(_r(0x33), (XD, X1, X2)),
    "sub": _Form(_r(0x33, funct7=0x20), (XD, X1, X2)),
    "mul": _Form(_r(0x33, funct7=1), (XD, X1, X2)),
    "ld": _Form(_r(0x03, 3), (XD, X1, I12)),
    "sd": _Form(_r(0x23, 3), (X2, X1, S12)),
    "beq": _Form(_r(0x63), (X1, X2, B13)),
    "bne": _Form(_r(0x63, 1), (X1, X2, B13)),
    "blt": _Form(_r(0x63, 4), (X1, X2, B13)),
    "jal": _Form(_r(0x6F), (XD, J21)),
    "jalr": _Form(_r(0x67), (XD, X1, I12)),
    "m.wspawn": _Form(_r(0x0B), (XD, SPAWN_PC)),
    "m.tmc": _Form(_r(0x0B, 1), (X1,)),
    # pred occupies rd. Both signed word offsets fit in one word.
    "m.split": _Form(_r(0x0B, 2) | (1 << 15), (XD, SPLIT_ELSE, SPLIT_JOIN)),
    "m.join": _Form(_r(0x0B, 3), ()),
    "m.bar": _Form(_r(0x0B, 4), (BARRIER_ID,), 0),
    # Simulator exponential helper, with VM in bit 25 and a separate funct6.
    "vfexp.v": _Form(_r(0x0B, 7, 2), (VD, V2, VM), 1),
    "m.mma.bf16": _Form(_r(0x2B), (TD, T1, T2)),
    "m.mma.fp8.blk": _Form(_r(0x2B, 1), (TD, T1, T2)),
    "m.mma.wait": _Form(_r(0x2B, 2), (TD,)),
    "m.dma.copy": _Form(_r(0x5B), (XD, X1)),
    "m.dma.wait": _Form(_r(0x5B, 1), (X1,)),
    "m.allreduce.sum": _Form(_r(0x7B), (VD, V2, SCOPE), 0),
    "m.allreduce.max": _Form(_r(0x7B, 1), (VD, V2, SCOPE), 0),
}

for _name, _funct6 in (("vfadd", 0), ("vfsub", 2), ("vfmax", 6), ("vfdiv", 32), ("vfmul", 36)):
    _FORMS[f"{_name}.vv"] = _Form(_r(0x57, 1, _funct6 << 1), (VD, V2, V1, VM), 1)

for _name, _selector in (("vfcvt.f.bf16.v", 0b01101), ("vfcvt.bf16.f.v", 0b11101)):
    _FORMS[_name] = _Form(_r(0x57, 1, 0b010010 << 1) | (_selector << 15), (VD, V2, VM), 1)

for _width, _funct3 in ((16, 5), (32, 6)):
    _FORMS[f"vle{_width}.v"] = _Form(_r(0x07, _funct3), (VD, X1, VM), 1)
    _FORMS[f"vse{_width}.v"] = _Form(_r(0x27, _funct3), (VD, X1, VM), 1)

_BY_OPCODE: dict[int, list[tuple[str, _Form]]] = {}
for _name, _form in _FORMS.items():
    _BY_OPCODE.setdefault(_form.fixed & 0x7F, []).append((_name, _form))

_ALIASES = {"vfwcvtbf16.f.f.v": "vfcvt.f.bf16.v", "vfncvtbf16.f.f.w": "vfcvt.bf16.f.v"}


def _check_masked_destination(name: str, dest: int, mask_bit: int) -> None:
    if not name.startswith("vse") and dest == mask_bit == 0:
        raise ValueError("A masked vector destination cannot be v0")


@dataclass(frozen=True)
class Instruction:
    mnemonic: str
    operands: tuple[int, ...]

    def __str__(self) -> str:
        form = _FORMS[self.mnemonic]
        args = [f"{field.kind}{value}" for field, value in zip(form.fields, self.operands)]
        if form.fields and form.fields[-1] == VM:
            args.pop()
            if not self.operands[-1]:
                args.append("v0.t")
        if self.mnemonic in ("ld", "sd", "jalr"):
            args = [args[0], f"{args[2]}({args[1]})"]
        if self.mnemonic.startswith(("vle", "vse")):
            args[1] = f"({args[1]})"
        return self.mnemonic + (" " + ", ".join(args) if args else "")


def encode_instruction(
    mnemonic: str | Instruction, *operands: int | str | tuple[int | str, ...] | list[int | str]
) -> int:
    """Encode a mnemonic with operands, or re-encode a decoded Instruction."""
    if isinstance(mnemonic, Instruction):
        if operands:
            raise ValueError("A decoded instruction already contains operands")
        return encode_instruction(mnemonic.mnemonic, *mnemonic.operands)
    if len(operands) == 1 and isinstance(operands[0], (tuple, list)):
        operands = tuple(operands[0])
    name = mnemonic.lower()
    name = _ALIASES.get(name, name)
    form = _FORMS.get(name)
    if form is None:
        raise ValueError(f"Unsupported instruction: {mnemonic}")
    if form.optional_last is not None and len(operands) + 1 == len(form.fields):
        operands += (form.optional_last,)
    if len(operands) != len(form.fields):
        raise ValueError(f"{name} expects {len(form.fields)} operands")
    word = form.fixed
    for field, value in zip(form.fields, operands):
        if field.kind == "vm" and isinstance(value, str):
            if value.lower() != "v0.t":
                raise ValueError("Expected v0.t as the RVV mask operand")
            value = 0
        word |= field.insert(value)
    if form.fields[-1:] == (VM,):
        _check_masked_destination(name, (word >> 7) & 31, (word >> 25) & 1)
    return word


def decode_instruction(word: int) -> Instruction:
    """Decode and reject reserved bits, subopcodes, and tile handles."""
    if type(word) is not int or not 0 <= word <= 0xFFFFFFFF:
        raise InstructionDecodeError("Instruction must be an unsigned 32-bit word")
    for name, form in _BY_OPCODE.get(word & 0x7F, ()):
        if word & form.mask == form.fixed:
            try:
                operands = tuple(field.extract(word) for field in form.fields)
                if form.fields[-1:] == (VM,):
                    _check_masked_destination(name, operands[0], operands[-1])
                return Instruction(name, operands)
            except ValueError as exc:
                raise InstructionDecodeError(f"Illegal {name} encoding 0x{word:08x}: {exc}") from exc
    raise InstructionDecodeError(f"Unsupported or illegal word 0x{word:08x}")
