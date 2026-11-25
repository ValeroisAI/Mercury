"""Functional binary simulator for one Mercury SCC.

The host configures warp arguments, vector lengths, DMA descriptors and tile
bindings before launching a program. Two warp issue slots and simple async
latencies give a cycle estimate, without modeling physical bank or port timing.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
import math
import operator
from pathlib import Path
import struct
from typing import Any

from mercury_sim.compute_isa import _f32, _fma_f32, encode_fp8, mma_fp8_blk_mercury

from .assembler import AssembledProgram, assemble_file
from .encoding import Instruction, decode_instruction

LANES = 32
FULL_MASK = (1 << LANES) - 1
_U64 = (1 << 64) - 1
_SEGMENT = 1 << 40
_SPACES = ("gddr7", "shared_sram", "l1")
_CAPACITY = (72_000_000_000, 192_000_000, 256 * 1024)
_TILE_BYTES = {"e4m3": 1, "e5m2": 1, "bf16": 2, "f32": 4}
_FP8 = ("e4m3", "e5m2")
_U16_PACK = struct.Struct("<H")
_U32_PACK = struct.Struct("<I")
_F32_PACK = struct.Struct("<f")


class VMExecutionError(RuntimeError):
    """The launched program violated the VM execution contract."""


class HazardViolationError(VMExecutionError):
    """An asynchronous operation's destination or borrowed source is busy."""


@dataclass(frozen=True)
class MemoryRef:
    space: str
    offset: int

    @property
    def address(self) -> int:
        """Return the mapped RV64 byte address."""
        if self.space not in _SPACES or type(self.offset) is not int:
            raise ValueError(f"Invalid memory reference {self}")
        segment = _SPACES.index(self.space)
        if not 0 <= self.offset < _CAPACITY[segment]:
            raise ValueError(f"Memory offset outside {self.space}: {self.offset}")
        return segment * _SEGMENT + self.offset


@dataclass(frozen=True)
class DMADescriptor:
    src: MemoryRef
    dst: MemoryRef
    row_bytes: int
    rows: int = 1
    src_stride: int | None = None
    dst_stride: int | None = None
    valid_rows: int | None = None
    valid_row_bytes: int | None = None


@dataclass(frozen=True)
class Tile:
    rows: int
    cols: int
    dtype: str
    data: MemoryRef
    scales: MemoryRef | None = None


@dataclass
class Telemetry:
    total_cycles_est: int = 0
    scalar_insts: int = 0
    rvv_insts: int = 0
    stme_insts: int = 0
    dma_ops: int = 0
    coll_ops: int = 0
    divergent_splits: int = 0
    max_reconv_depth: int = 0
    simt_insts: int = 0


@dataclass
class _Frame:
    entry_mask: int
    control_mask: int
    pending_mask: int
    else_pc: int
    join_pc: int


@dataclass
class _Warp:
    pc: int = 0
    active: int = FULL_MASK
    control: int = FULL_MASK
    live: int = FULL_MASK
    vl: int = LANES
    scalar: list[list[int]] = field(default_factory=lambda: [[0] * LANES for _ in range(32)])
    vector: list[list[int]] = field(default_factory=lambda: [[0] * LANES for _ in range(32)])
    stack: list[_Frame] = field(default_factory=list)
    blocked: bool = False
    retired: bool = False


@dataclass(frozen=True)
class _Range:
    start: int
    size: int

    def overlaps(self, other: _Range) -> bool:
        return bool(self.size and other.size and self.start < other.start + other.size and other.start < self.start + self.size)


@dataclass
class _Pending:
    ready: int
    reads: tuple[_Range, ...]
    writes: tuple[tuple[_Range, bytes], ...]


@dataclass
class _Sync:
    participants: frozenset[int]
    arrivals: dict[int, tuple[int, tuple[int, ...], tuple[float, ...]]] = field(default_factory=dict)


class _Memory:
    # Lazily committed pages let the host use full architectural addresses.
    _PAGE = 4096

    def __init__(self) -> None:
        self.pages: dict[int, bytearray] = {}

    @staticmethod
    def check(address: int, size: int) -> _Range:
        segment, offset = divmod(address, _SEGMENT)
        if not 0 <= segment < len(_CAPACITY) or size < 0 or offset + size > _CAPACITY[segment]:
            raise VMExecutionError(f"Unmapped memory range 0x{address:x} + {size} bytes")
        return _Range(address, size)

    @classmethod
    def _chunks(cls, region: _Range) -> Iterator[tuple[int, int, int]]:
        address = region.start
        remaining = region.size
        while remaining:
            page, offset = divmod(address, cls._PAGE)
            count = min(remaining, cls._PAGE - offset)
            yield page, offset, count
            address += count
            remaining -= count

    def read(self, region: _Range) -> bytes:
        result = bytearray()
        for page, offset, count in self._chunks(region):
            data = self.pages.get(page)
            result.extend(data[offset:offset + count] if data is not None else bytes(count))
        return bytes(result)

    def write(self, region: _Range, data: bytes) -> None:
        cursor = 0
        for page, offset, count in self._chunks(region):
            if page not in self.pages:
                self.pages[page] = bytearray(self._PAGE)
            self.pages[page][offset:offset + count] = data[cursor:cursor + count]
            cursor += count


def _lanes(mask: int) -> tuple[int, ...]:
    return tuple(i for i in range(LANES) if mask & (1 << i))


def _signed(value: int) -> int:
    return value - (1 << 64) if value & (1 << 63) else value


def _float_bits(value: float) -> int:
    return _U32_PACK.unpack(_F32_PACK.pack(_f32(value)))[0]


def _bits_float(bits: int) -> float:
    return _F32_PACK.unpack(_U32_PACK.pack(bits))[0]


def _to_bf16(bits: int) -> int:
    if bits & 0x7FFFFFFF > 0x7F800000:
        return 0x7FC0
    return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16


def _max(a: float, b: float) -> float:
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    if a == b == 0:
        return -0.0 if math.copysign(1, a) == math.copysign(1, b) == -1 else 0.0
    return max(a, b)


def _divide(a: float, b: float) -> float:
    if b != 0:
        return a / b
    if a == 0 or math.isnan(a):
        return math.nan
    return math.copysign(math.inf, math.copysign(1, a) * math.copysign(1, b))


def _exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


_SCALAR_ARITHMETIC = {"add": operator.add, "sub": operator.sub, "mul": operator.mul}
_VECTOR_ARITHMETIC = {
    "vfadd.vv": operator.add, "vfsub.vv": operator.sub, "vfmul.vv": operator.mul,
    "vfmax.vv": _max, "vfdiv.vv": _divide,
}


class MercuryVM:
    """A single-use SCC launch with 1–16 warp slots and a shared scratchpad."""

    def __init__(self, num_warps: int = 1, *, initial_warps: int = 1) -> None:
        if not 1 <= initial_warps <= num_warps <= 16:
            raise ValueError("Expected 1 <= initial_warps <= num_warps <= 16")
        self.telemetry = Telemetry()
        self._warps = [_Warp() if i < initial_warps else None for i in range(num_warps)]
        self._memory = _Memory()
        self._tiles: dict[tuple[int, int], Tile] = {}
        self._descriptors: dict[int, tuple[DMADescriptor, ...]] = {}
        self._dma: dict[tuple[int, int], _Pending] = {}
        self._mma: dict[tuple[int, int], _Pending] = {}
        self._sync: dict[tuple[str, int], _Sync] = {}
        self._token = 0
        self._started = False
        self._end = 0
        self._cycle = 0

    def _warp(self, slot: int) -> _Warp:
        if not 0 <= slot < len(self._warps) or self._warps[slot] is None:
            raise ValueError(f"Warp slot {slot} is not initialized")
        warp = self._warps[slot]
        assert warp is not None
        return warp

    def set_scalar(self, register: int, value: int | Sequence[int], *, warp: int = 0) -> None:
        """Set scalar lanes. Launch initializes x1, x30 and x31 as return, warp and lane IDs."""
        if not 1 <= register < 32:
            raise ValueError("Writable scalar registers are x1 through x31")
        values = [value] * LANES if isinstance(value, int) else list(value)
        if len(values) != LANES:
            raise ValueError("Expected 32 scalar lanes")
        self._warp(warp).scalar[register] = [item & _U64 for item in values]

    def read_scalar(self, register: int, *, warp: int = 0) -> tuple[int, ...]:
        """Read all lanes as unsigned RV64 values."""
        if not 0 <= register < 32:
            raise ValueError("Expected x0 through x31")
        return tuple(self._warp(warp).scalar[register])

    def set_vector(self, register: int, values: Sequence[float], *, warp: int = 0) -> None:
        """Initialize one vector register with binary32 values."""
        if not 0 <= register < 32 or len(values) != LANES:
            raise ValueError("Expected a vector register v0 through v31 with 32 lanes")
        self._warp(warp).vector[register] = [_float_bits(value) for value in values]

    def read_vector(self, register: int, *, warp: int = 0) -> tuple[float, ...]:
        """Read one binary32 vector register."""
        if not 0 <= register < 32:
            raise ValueError("Expected v0 through v31")
        return tuple(map(_bits_float, self._warp(warp).vector[register]))

    def configure_warp(self, *, warp: int = 0, vl: int = LANES, live_mask: int = FULL_MASK) -> None:
        """Configure launch-time vector length and live lanes for a resident warp."""
        if self._started or not 0 <= vl <= LANES or not 0 < live_mask <= FULL_MASK:
            raise ValueError("Expected a pre-launch VL in [0, 32] and a nonzero live mask")
        state = self._warp(warp)
        state.vl = vl
        state.active = state.control = state.live = live_mask

    def set_vector_mask(self, mask: int, *, warp: int = 0) -> None:
        """Set the packed RVV v0 mask bits."""
        if not 0 <= mask <= FULL_MASK:
            raise ValueError("Expected a 32-bit mask")
        self._warp(warp).vector[0][0] = mask

    def register_dma(self, handle: int, *descriptors: DMADescriptor) -> None:
        """Register one or more planes transferred by the same DMA token."""
        if self._started or not 0 <= handle <= _U64 or not descriptors:
            raise ValueError("A DMA handle requires at least one pre-launch descriptor")
        self._descriptors[handle] = descriptors

    def bind_tile(self, slot: int, tile: Tile, *, warp: int = 0) -> None:
        """Bind a warp-local tile handle to its mapped data and scale storage."""
        if self._started or not 0 <= warp < len(self._warps) or not 0 <= slot < 8:
            raise ValueError("Expected a pre-launch t0 through t7 binding for a valid warp slot")
        if tile.dtype not in _TILE_BYTES or min(tile.rows, tile.cols) <= 0:
            raise ValueError("Tile shape and dtype must be valid")
        if (tile.dtype in _FP8) != (tile.scales is not None):
            raise ValueError("Exactly FP8 tiles require a scale plane")
        self._tile_data(tile)
        self._tiles[warp, slot] = tile

    def _access(self, address: int, size: int, *, write: bool = False, ignore: _Pending | None = None) -> _Range:
        region = self._memory.check(address, size)
        for op in (*self._dma.values(), *self._mma.values()):
            if op is ignore:
                continue
            busy = any(region.overlaps(destination) for destination, _ in op.writes)
            if busy or (write and any(region.overlaps(source) for source in op.reads)):
                mode = "write" if write else "read"
                raise HazardViolationError(f"Cannot {mode} 0x{address:x} + {size} bytes before async wait")
        return region

    def read_memory(self, ref: MemoryRef, size: int) -> bytes:
        """Read mapped memory subject to the asynchronous scoreboard."""
        return self._memory.read(self._access(ref.address, size))

    def write_memory(self, ref: MemoryRef, data: bytes) -> None:
        """Write mapped memory subject to the asynchronous scoreboard."""
        self._memory.write(self._access(ref.address, len(data), write=True), data)

    def _tile_data(self, tile: Tile) -> _Range:
        return self._memory.check(tile.data.address, tile.rows * tile.cols * _TILE_BYTES[tile.dtype])

    @staticmethod
    def _write_scalar(warp: _Warp, register: int, lane: int, value: int) -> None:
        if register:
            warp.scalar[register][lane] = value & _U64

    @staticmethod
    def _uniform(values: Sequence[int]) -> int:
        first = values[0]
        if any(value != first for value in values):
            raise VMExecutionError("Non-uniform warp control operand; use m.split for divergence")
        return first

    def _argument(self, warp: _Warp, register: int) -> int:
        return self._uniform([warp.scalar[register][i] for i in _lanes(warp.active)])

    def _target(self, address: int) -> int:
        if address % 4 or not 0 <= address <= self._end:
            raise VMExecutionError(f"Invalid instruction address {address:#x}")
        return address

    def _initialize_warp(self, warp: _Warp, wid: int, pc: int) -> None:
        warp.pc = pc
        warp.scalar[1] = [self._end] * LANES
        warp.scalar[30] = [wid] * LANES
        warp.scalar[31] = list(range(LANES))

    def _retire(self, wid: int, warp: _Warp) -> None:
        if any(owner == wid for owner, _ in (*self._dma.keys(), *self._mma.keys())):
            raise HazardViolationError(f"Warp {wid} retired before an async wait")
        if warp.stack:
            raise VMExecutionError(f"Warp {wid} exited before m.join")
        if any(wid in sync.participants for sync in self._sync.values()):
            raise VMExecutionError(f"Warp {wid} retired before participating in a workgroup operation")
        warp.retired = True

    def _converged(self, warp: _Warp) -> None:
        if warp.stack:
            raise VMExecutionError("Workgroup or matrix issue requires converged control flow")

    def _arrive(
        self, key: tuple[str, int], wid: int, warp: _Warp,
        data: tuple[int, tuple[int, ...], tuple[float, ...]] = (0, (), ()),
    ) -> None:
        sync = self._sync.get(key)
        if sync is None:
            sync = _Sync(frozenset(i for i, w in enumerate(self._warps) if w is not None and not w.retired))
            self._sync[key] = sync
        warp.blocked = True
        sync.arrivals[wid] = data
        if sync.participants != sync.arrivals.keys():
            return
        if key[0] != "m.bar":
            # Fixed slot and lane order keeps FP32 sums independent of arrival timing.
            values = [value for slot in sorted(sync.participants) for value in sync.arrivals[slot][2]]
            result = _float_bits(self._reduce(key[0], values))
            for slot, (dest, lanes, _) in sync.arrivals.items():
                for lane in lanes:
                    self._warp(slot).vector[dest][lane] = result
        for slot in sync.participants:
            self._warp(slot).blocked = False
        del self._sync[key]

    @staticmethod
    def _reduce(op: str, values: Sequence[float]) -> float:
        if op == "m.allreduce.sum":
            acc = 0.0
            for value in values:
                acc = _f32(acc + value)
            return acc
        acc = math.nan
        for value in values:
            acc = _max(acc, value)
        return acc

    def _simt(self, ins: Instruction, wid: int, warp: _Warp, pc: int) -> None:
        name, args = ins.mnemonic, ins.operands
        if name == "m.split":
            pred, else_offset, join_offset = args
            else_pc, join_pc = self._target(pc + else_offset), self._target(pc + join_offset)
            true_mask = sum(1 << lane for lane in _lanes(warp.active) if warp.scalar[pred][lane])
            false_mask = warp.active ^ true_mask
            warp.stack.append(_Frame(warp.active, warp.control, false_mask if true_mask else 0, else_pc, join_pc))
            self.telemetry.divergent_splits += bool(true_mask and false_mask)
            self.telemetry.max_reconv_depth = max(self.telemetry.max_reconv_depth, len(warp.stack))
            warp.control = warp.active = true_mask or false_mask
            if not true_mask:
                warp.pc = else_pc
        elif name == "m.join":
            if not warp.stack or warp.stack[-1].join_pc != pc:
                raise VMExecutionError("m.join does not match the current reconvergence PC")
            frame = warp.stack[-1]
            if frame.pending_mask:
                warp.control = warp.active = frame.pending_mask
                frame.pending_mask = 0
                warp.pc = frame.else_pc
            else:
                warp.active, warp.control = frame.entry_mask, frame.control_mask
                warp.stack.pop()
        elif name == "m.tmc":
            warp.active = self._argument(warp, args[0]) & warp.control & warp.live
            if not warp.active:
                if warp.stack:
                    warp.pc = warp.stack[-1].join_pc
                else:
                    self._retire(wid, warp)
        elif name == "m.wspawn":
            self._converged(warp)
            if self._sync:
                raise VMExecutionError("Cannot spawn while a workgroup operation is in progress")
            count = self._argument(warp, args[0])
            entry = self._target(pc + args[1])
            free = [slot for slot, w in enumerate(self._warps) if w is None or w.retired]
            if count > len(free):
                raise VMExecutionError("m.wspawn exceeds reserved warp slots")
            for slot in free[:count]:
                child = _Warp(active=warp.live, control=warp.live, live=warp.live, vl=warp.vl)
                child.scalar = [reg.copy() for reg in warp.scalar]
                child.vector = [reg.copy() for reg in warp.vector]
                self._initialize_warp(child, slot, entry)
                self._warps[slot] = child
        else:
            self._converged(warp)
            self._arrive((name, args[0]), wid, warp)
        self.telemetry.simt_insts += 1

    def _scalar(self, ins: Instruction, warp: _Warp, pc: int) -> None:
        name, args = ins.mnemonic, ins.operands
        lanes = _lanes(warp.active)
        regs = warp.scalar
        if name in ("beq", "bne", "blt"):
            src, rhs, relative = args
            tests = {
                "beq": lambda a, b: a == b,
                "bne": lambda a, b: a != b,
                "blt": lambda a, b: _signed(a) < _signed(b),
            }
            taken = self._uniform([tests[name](regs[src][i], regs[rhs][i]) for i in lanes])
            if taken:
                warp.pc = self._target(pc + relative)
        elif name in ("jal", "jalr", "ret"):
            if name == "jal":
                dest, relative = args
                target = pc + relative
            else:
                dest, base, offset = (0, 1, 0) if name == "ret" else args
                target = ((self._argument(warp, base) + offset) & _U64) & ~1
            warp.pc = self._target(target)
            for lane in lanes:
                self._write_scalar(warp, dest, lane, pc + 4)
        else:
            dest, src, rhs = args
            for lane in lanes:
                left = regs[src][lane]
                if name in ("ld", "sd"):
                    address = (left + rhs) & _U64
                    region = self._access(address, 8, write=name == "sd")
                    if name == "sd":
                        self._memory.write(region, regs[dest][lane].to_bytes(8, "little"))
                        continue
                    value = int.from_bytes(self._memory.read(region), "little")
                elif name == "addi":
                    value = left + rhs
                else:
                    right = regs[rhs][lane]
                    value = _SCALAR_ARITHMETIC[name](left, right)
                self._write_scalar(warp, dest, lane, value)
        self.telemetry.scalar_insts += 1

    def _vector(self, ins: Instruction, warp: _Warp) -> None:
        name, (dest, src, *rest) = ins.mnemonic, ins.operands
        enabled = warp.active & ((1 << warp.vl) - 1)
        if rest[-1] == 0:
            enabled &= warp.vector[0][0]
        lanes = _lanes(enabled)
        vectors = warp.vector
        if name.startswith(("vle", "vse")):
            width = 2 if "16" in name else 4
            if lanes:
                address = self._uniform([warp.scalar[src][i] for i in lanes])
                store = name.startswith("vse")
                regions = [self._access(address + lane * width, width, write=store) for lane in lanes]
                for lane, region in zip(lanes, regions):
                    if store:
                        mask = (1 << (width * 8)) - 1
                        self._memory.write(region, (vectors[dest][lane] & mask).to_bytes(width, "little"))
                    else:
                        vectors[dest][lane] = int.from_bytes(self._memory.read(region), "little")
        else:
            for lane in lanes:
                source = vectors[src][lane]
                if name == "vfcvt.bf16.f.v":
                    value = _to_bf16(source)
                elif name == "vfcvt.f.bf16.v":
                    value = (source & 0xFFFF) << 16
                    if value & 0x7FFFFFFF > 0x7F800000:
                        value = 0x7FC00000
                elif name == "vfexp.v":
                    value = _float_bits(_exp(_bits_float(source)))
                else:
                    a, b = _bits_float(source), _bits_float(vectors[rest[0]][lane])
                    value = _float_bits(_VECTOR_ARITHMETIC[name](a, b))
                vectors[dest][lane] = value
        self.telemetry.rvv_insts += 1

    def _collective(self, ins: Instruction, wid: int, warp: _Warp) -> None:
        dest, src, scope = ins.operands
        lanes = _lanes(warp.active & ((1 << warp.vl) - 1))
        values = tuple(_bits_float(warp.vector[src][lane]) for lane in lanes)
        if scope:
            self._converged(warp)
            self._arrive((ins.mnemonic, 0), wid, warp, (dest, lanes, values))
        else:
            result = _float_bits(self._reduce(ins.mnemonic, values))
            for lane in lanes:
                warp.vector[dest][lane] = result
        self.telemetry.coll_ops += 1

    def _dma_copy(self, wid: int, warp: _Warp, dest: int, desc_register: int) -> None:
        self._converged(warp)
        handle = self._argument(warp, desc_register)
        descriptors = self._descriptors.get(handle)
        if descriptors is None:
            raise VMExecutionError(f"No DMA descriptor for handle {handle}")
        if dest == 0:
            raise VMExecutionError("m.dma.copy requires a writable token register")
        reads = []
        writes = []
        for plane in descriptors:
            src_step = plane.row_bytes if plane.src_stride is None else plane.src_stride
            dst_step = plane.row_bytes if plane.dst_stride is None else plane.dst_stride
            rows = plane.rows if plane.valid_rows is None else plane.valid_rows
            width = plane.row_bytes if plane.valid_row_bytes is None else plane.valid_row_bytes
            if (
                plane.rows <= 0 or plane.row_bytes <= 0 or dst_step < plane.row_bytes
                or not 0 <= rows <= plane.rows or not 0 <= width <= plane.row_bytes
                or src_step < width
            ):
                raise VMExecutionError(f"Invalid DMA plane: {plane}")
            self._memory.check(plane.dst.address, (plane.rows - 1) * dst_step + plane.row_bytes)
            if rows and width:
                self._memory.check(plane.src.address, (rows - 1) * src_step + width)
            for row in range(plane.rows):
                target = self._access(plane.dst.address + row * dst_step, plane.row_bytes, write=True)
                if row < rows and width:
                    source = self._access(plane.src.address + row * src_step, width)
                    reads.append(source)
                    data = self._memory.read(source)
                else:
                    data = b""
                writes.append((target, data.ljust(plane.row_bytes, b"\0")))
        destinations = sorted((region for region, _ in writes), key=lambda region: region.start)
        if any(a.overlaps(b) for a, b in zip(destinations, destinations[1:])):
            raise VMExecutionError("Overlapping destinations within one DMA token")
        total = sum(region.size for region in destinations)
        self._token += 1
        self._dma[wid, self._token] = _Pending(self._cycle + 3 + (total + 63) // 64, tuple(reads), tuple(writes))
        for lane in _lanes(warp.active):
            self._write_scalar(warp, dest, lane, self._token)
        self.telemetry.dma_ops += 1

    def _wait(self, pending: dict[tuple[int, int], _Pending], key: tuple[int, int]) -> int | None:
        op = pending.get(key)
        if op is None:
            raise HazardViolationError(f"No matching outstanding async token {key[1]} for warp {key[0]}")
        if self._cycle < op.ready:
            return op.ready
        for region, data in op.writes:
            self._memory.write(region, data)
        del pending[key]
        return None

    def _get_tile(self, wid: int, slot: int) -> Tile:
        tile = self._tiles.get((wid, slot))
        if tile is None:
            raise VMExecutionError(f"No binding for warp {wid} tile t{slot}")
        return tile

    def _read_tile(self, tile: Tile, count: int) -> tuple[bytes, tuple[float, ...], tuple[_Range, ...]]:
        payload = self._tile_data(tile)
        data = self._memory.read(self._access(payload.start, payload.size))
        if tile.scales is None:
            return data, (), (payload,)
        region = self._access(tile.scales.address, count * 4)
        scales = tuple(value for (value,) in struct.iter_unpack("<f", self._memory.read(region)))
        return data, scales, (payload, region)

    @staticmethod
    def _fp8_mma(
        a: Tile, b: Tile, left: bytes, right: bytes,
        a_scales: tuple[float, ...], b_scales: tuple[float, ...], acc: list[float],
    ) -> list[float]:
        m, n, k = a.rows, b.cols, a.cols
        formats = {"a_format": a.dtype, "b_format": b.dtype}
        if all(value == 0 and math.copysign(1, value) > 0 for value in acc):
            return mma_fp8_blk_mercury(left, a_scales, right, b_scales, m, n, k, **formats)
        # Reuse the reference's K-ordered group sum, seeding its FP32
        # boundary FMA with the committed or forwarded accumulator value.
        groups, cols = (k + 127) // 128, (n + 127) // 128
        if any(not math.isfinite(x) or x <= 0 for x in (*a_scales, *b_scales)):
            raise ValueError("MMA scales must be finite positive FP32 values")
        for group, start in enumerate(range(0, k, 128)):
            width = min(k - start, 128)
            stop = start + width
            tile_a = b"".join(left[i * k + start:i * k + stop] for i in range(m))
            tile_b = right[start * n:stop * n]
            partial = mma_fp8_blk_mercury(tile_a, [1.0] * m, tile_b, [1.0] * cols, m, n, width, **formats)
            for i in range(m):
                for j in range(n):
                    scale = _f32(a_scales[i * groups + group] * b_scales[group * cols + j // 128])
                    idx = i * n + j
                    acc[idx] = _fma_f32(partial[idx], scale, acc[idx])
        return acc

    @staticmethod
    def _bf16_mma(a: Tile, b: Tile, left: bytes, right: bytes, acc: list[float]) -> list[float]:
        lhs = [_bits_float(bits << 16) for (bits,) in struct.iter_unpack("<H", left)]
        rhs = [_bits_float(bits << 16) for (bits,) in struct.iter_unpack("<H", right)]
        for i in range(a.rows):
            for j in range(b.cols):
                idx = i * b.cols + j
                for k in range(a.cols):
                    acc[idx] = _fma_f32(lhs[i * a.cols + k], rhs[k * b.cols + j], acc[idx])
        return acc

    def _issue_mma(self, ins: Instruction, wid: int, warp: _Warp) -> None:
        self._converged(warp)
        dest, src, rhs = ins.operands
        c, a, b = (self._get_tile(wid, slot) for slot in (dest, src, rhs))
        if a.cols != b.rows or (a.rows, b.cols) != (c.rows, c.cols) or c.dtype != "f32":
            raise VMExecutionError("Unmatched MMA tile shapes or accumulator dtype")
        fp8 = ins.mnemonic == "m.mma.fp8.blk"
        formats = _FP8 if fp8 else ("bf16",)
        if a.dtype not in formats or b.dtype not in formats:
            raise VMExecutionError("MMA operand dtype does not match the instruction")
        count = (a.cols + 127) // 128
        left, a_scales, a_reads = self._read_tile(a, count * a.rows)
        right, b_scales, b_reads = self._read_tile(b, count * ((b.cols + 127) // 128))
        key = (wid, dest)
        previous = self._mma.get(key)
        target = self._tile_data(c)
        self._access(target.start, target.size, write=True, ignore=previous)
        reads = a_reads + b_reads + (previous.reads if previous else ())
        if any(target.overlaps(region) for region in reads):
            raise HazardViolationError("MMA accumulator overlaps an active operand")
        old = previous.writes[0][1] if previous else self._memory.read(target)
        accumulator = [value for (value,) in struct.iter_unpack("<f", old)]
        if fp8:
            result = self._fp8_mma(a, b, left, right, a_scales, b_scales, accumulator)
        else:
            result = self._bf16_mma(a, b, left, right, accumulator)
        data = struct.pack(f"<{len(result)}f", *result)
        macs = a.rows * a.cols * b.cols
        rate = 4096 if fp8 else 2048
        start = max(self._cycle, previous.ready) if previous else self._cycle
        self._mma[key] = _Pending(start + 4 + (macs + rate - 1) // rate, reads, ((target, data),))

    def _step(self, word: int, wid: int, warp: _Warp) -> int | None:
        ins = decode_instruction(word)
        name, args = ins.mnemonic, ins.operands
        pc = warp.pc
        warp.pc += 4
        deadline = None
        if name == "m.dma.copy":
            self._dma_copy(wid, warp, *args)
        elif name == "m.dma.wait":
            deadline = self._wait(self._dma, (wid, self._argument(warp, args[0])))
        elif name.startswith("m.mma."):
            if name == "m.mma.wait":
                deadline = self._wait(self._mma, (wid, args[0]))
            else:
                self._issue_mma(ins, wid, warp)
            self.telemetry.stme_insts += deadline is None
        elif name.startswith("m.allreduce."):
            self._collective(ins, wid, warp)
        elif name.startswith("m."):
            self._simt(ins, wid, warp, pc)
        elif name.startswith("v"):
            self._vector(ins, warp)
        else:
            self._scalar(ins, warp, pc)
        if deadline is not None:
            warp.pc = pc
        return deadline

    def run(self, words: Sequence[int] | AssembledProgram, *, entry: int = 0, max_steps: int = 100_000) -> Telemetry:
        """Run binary words until all resident warps retire; fail on deadlock."""
        if self._started or max_steps <= 0:
            raise ValueError("Expected a new VM and a positive instruction budget")
        self._started = True
        if isinstance(words, AssembledProgram):
            words = words.words
        self._end = len(words) * 4
        for wid, warp in enumerate(self._warps):
            if warp is not None:
                self._initialize_warp(warp, wid, self._target(entry))
        cursor, steps = 0, 0
        while any(warp is not None and not warp.retired for warp in self._warps):
            issued = 0
            deadlines = []
            next_cursor = cursor
            for offset in range(len(self._warps)):
                wid = (cursor + offset) % len(self._warps)
                warp = self._warps[wid]
                if warp is None or warp.retired or warp.blocked:
                    continue
                if warp.pc == self._end:
                    self._retire(wid, warp)
                    continue
                deadline = self._step(words[warp.pc // 4], wid, warp)
                if deadline is not None:
                    deadlines.append(deadline)
                    continue
                steps += 1
                if steps > max_steps:
                    raise VMExecutionError("VM instruction budget exhausted")
                issued += 1
                next_cursor = (wid + 1) % len(self._warps)
                if issued == 2:
                    break
            cursor = next_cursor
            if issued:
                self._cycle += 1
            elif deadlines:
                self._cycle = min(deadlines)
            elif any(warp is not None and not warp.retired for warp in self._warps):
                raise VMExecutionError("Workgroup deadlock: no warp can reach the next synchronization")
        self.telemetry.total_cycles_est = self._cycle
        return self.telemetry


_KERNEL_DIR = Path(__file__).resolve().parent.parent / "examples" / "kernels"


def _relu_benchmark() -> tuple[Telemetry, tuple[float, ...], Sequence[float], float]:
    vm = MercuryVM()
    inputs = tuple(float((i * 7) % 19 - 9) for i in range(LANES))
    start, signs, output = (MemoryRef("gddr7", offset) for offset in (0, 256, 512))
    vm.write_memory(start, struct.pack("<32f", *inputs))
    vm.write_memory(signs, struct.pack("<32Q", *(value > 0 for value in inputs)))
    vm.set_scalar(2, start.address)
    vm.set_scalar(3, [signs.address + lane * 8 for lane in range(LANES)])
    vm.set_scalar(4, output.address)
    vm.set_vector(31, [2.0**-126] * LANES)
    telemetry = vm.run(assemble_file(_KERNEL_DIR / "divergent_relu_norm.masm"))
    positive = [max(value, 0.0) for value in inputs]
    total = math.fsum(positive)
    return telemetry, struct.unpack("<32f", vm.read_memory(output, 128)), [value / total for value in positive], 1e-7


def _attention_benchmark() -> tuple[Telemetry, tuple[float, ...], Sequence[float], float]:
    vm = MercuryVM()
    q = [float(i % 5 - 2) for i in range(128)]
    kt = [float((i * 3 + j * 7) % 9 - 4) for i in range(128) for j in range(LANES)]
    values = [(i % 7 - 3) / 4 for i in range(LANES)]
    q_scale, k_scale = 0.5 / math.sqrt(128), 0.25
    payloads = {
        0: bytes(map(encode_fp8, q)),
        256: bytes(map(encode_fp8, kt)),
        4608: struct.pack("<f", q_scale),
        4612: struct.pack("<f", k_scale),
        4672: struct.pack("<32H", *(_to_bf16(_float_bits(value)) for value in values)),
    }
    for offset, payload in payloads.items():
        vm.write_memory(MemoryRef("gddr7", offset), payload)
    for handle, src, dst in ((1, "gddr7", "shared_sram"), (2, "shared_sram", "l1")):
        planes = [DMADescriptor(MemoryRef(src, offset), MemoryRef(dst, offset), len(data))
                  for offset, data in payloads.items()]
        vm.register_dma(handle, *planes)
    score, probabilities, numerator = (MemoryRef("l1", offset) for offset in (4800, 4928, 4992))
    output = MemoryRef("gddr7", 8192)
    vm.bind_tile(0, Tile(1, LANES, "f32", score))
    vm.bind_tile(1, Tile(1, 128, "e4m3", MemoryRef("l1", 0), MemoryRef("l1", 4608)))
    vm.bind_tile(2, Tile(128, LANES, "e4m3", MemoryRef("l1", 256), MemoryRef("l1", 4612)))
    vm.bind_tile(3, Tile(1, 1, "f32", numerator))
    vm.bind_tile(4, Tile(1, LANES, "bf16", probabilities))
    vm.bind_tile(5, Tile(LANES, 1, "bf16", MemoryRef("l1", 4672)))
    for register, value in ((2, 1), (3, 2), (4, score.address), (5, probabilities.address),
                            (6, 1), (7, numerator.address), (9, output.address)):
        vm.set_scalar(register, value)
    telemetry = vm.run(assemble_file(_KERNEL_DIR / "fused_fp8_gemm_flash_tile.masm"))
    logits = [q_scale * k_scale * math.fsum(q[i] * kt[i * LANES + j] for i in range(128))
              for j in range(LANES)]
    peak = max(logits)
    weights = [math.exp(logit - peak) for logit in logits]
    golden = math.fsum(p * v for p, v in zip(weights, values)) / math.fsum(weights)
    return telemetry, struct.unpack("<f", vm.read_memory(output, 4)), [golden], 3e-4


def run_isa_kernel_benchmarks() -> dict[str, Any]:
    """Execute both shipped MASM kernels and check results against FP64 references."""
    results = {}
    for name, run in (("divergent_relu_norm", _relu_benchmark), ("fused_fp8_gemm_flash_tile", _attention_benchmark)):
        telemetry, actual, expected, tolerance = run()
        if len(actual) != len(expected) or any(not math.isfinite(value) for value in actual):
            raise AssertionError(f"{name} produced invalid values")
        error = max(abs(value - reference) for value, reference in zip(actual, expected))
        if error > tolerance:
            raise AssertionError(f"{name} exceeded FP64 parity tolerance: {error} > {tolerance}")
        results[name] = {
            "telemetry": asdict(telemetry),
            "fp64_parity": True,
            "max_abs_error": error,
            "abs_error_tolerance": tolerance,
        }
    return results
