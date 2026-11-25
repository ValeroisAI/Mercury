"""Binary encoding and functional verification for the Mercury SCC VM."""

import json
from pathlib import Path
import struct

import pytest

from mercury_isa import (
    ISA,
    AssemblyError,
    DMADescriptor,
    HazardViolationError,
    InstructionDecodeError,
    MemoryRef,
    MercuryVM,
    Tile,
    VMExecutionError,
    assemble,
    assemble_file,
    decode_instruction,
    disassemble,
    encode_instruction,
    run_isa_kernel_benchmarks,
)


@pytest.mark.parametrize(
    ("source", "word"),
    [
        ("m.wspawn x2, 8", 0x0001010B),
        ("m.tmc x2", 0x0001100B),
        ("m.split x2, 8, 20", 0x0502A10B),
        ("m.join", 0x0000300B),
        ("m.bar 3", 0x0001C00B),
        ("m.mma.fp8.blk t1, t2, t3", 0x003110AB),
        ("m.mma.bf16 t1, t2, t3", 0x003100AB),
        ("m.mma.wait t3", 0x000021AB),
        ("m.dma.copy x7, x3", 0x000183DB),
        ("m.dma.wait x7", 0x0003905B),
        ("m.allreduce.max v4, v5, 1", 0x0250127B),
        ("m.allreduce.sum v4, v5, 0", 0x0050027B),
        ("addi x3, x4, -1", 0xFFF20193),
        ("sub x3, x4, x5", 0x405201B3),
        ("mul x3, x4, x5", 0x025201B3),
        ("ld x5, 8(x6)", 0x00833283),
        ("sd x5, -8(x6)", 0xFE533C23),
        ("beq x1, x2, 8", 0x00208463),
        ("jal x0, -4", 0xFFDFF06F),
        ("ret", 0x00008067),
        ("vle32.v v1, (x2)", 0x02016087),
        ("vse32.v v1, (x2), v0.t", 0x000160A7),
        ("vfadd.vv v1, v3, v2", 0x023110D7),
        ("vfwcvtbf16.f.f.v v1, v2", 0x4A2690D7),
        ("vfncvtbf16.f.f.w v1, v2", 0x4A2E90D7),
    ],
)
def test_exact_word_and_round_trip(source, word):
    program = assemble(source)
    assert program.words == (word,)
    assert program.to_bytes() == struct.pack("<I", word)
    assert encode_instruction(decode_instruction(word)) == word
    assert assemble(disassemble(program)).words == program.words


@pytest.mark.parametrize(
    "word", [0, -1, 1 << 32, 0x8000300B, 0x0031142B, 0x020183DB, 0x0400127B, 0x00006087 | (1 << 20), 0x00311057]
)
def test_illegal_encoding(word):
    with pytest.raises(InstructionDecodeError):
        decode_instruction(word)


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("add", ("x32", "x1", "x2")),
        ("vfmul.vv", ("v1", "v2", "v32")),
        ("vfmul.vv", ("v0", "v1", "v2", "v0.t")),
        ("m.mma.bf16", ("t8", "t1", "t2")),
        ("m.dma.copy", ("x2", "v1")),
        ("m.split", ("x2", 512, 8)),
        ("m.split", ("x2", 2, 8)),
        ("addi", ("x1", "x2", -2049)),
        ("beq", ("x1", "x2", 4096)),
        ("jal", ("x1", "2", 4)),
    ],
)
def test_encoder_rejects_wrong_registers_and_immediates(name, args):
    with pytest.raises(ValueError):
        encode_instruction(name, *args)


def test_labels_and_source_metadata():
    program = assemble(
        """
        start: addi x2, x0, 3 # instruction with a label
        again: addi x2, x2, -1
        bne x2, x0, again     ; backward branch
        blt x2, x0, bad
        beq x0, x2, done
        bad: addi x8, x0, 1
        done: jal x0, exit
        addi x8, x0, 2
        exit: ret
        """
    )
    assert program.isa == ISA
    assert program.labels == {"start": 0, "again": 4, "bad": 20, "done": 24, "exit": 32}
    assert program.source_lines == tuple(range(2, 11))
    assert decode_instruction(program.words[2]).operands[-1] == -4
    vm = MercuryVM()
    vm.run(program.words)
    assert vm.read_scalar(2) == (0,) * 32
    assert vm.read_scalar(8) == (0,) * 32
    with pytest.raises(AssemblyError, match="<test>:1: Undefined label"):
        assemble("m.split x2, missing, missing", filename="<test>")
    with pytest.raises(AssemblyError, match="duplicate label"):
        assemble("double: ret\ndouble: ret")


def test_nested_splits_restore_masks_and_merge_both_paths():
    program = assemble(
        """
        m.split x2, outer_else, outer_join
        m.split x3, inner_else, inner_join
        addi x4, x0, 11
        jal x0, inner_join
        inner_else: addi x4, x0, 12
        inner_join: m.join
        jal x0, outer_join
        outer_else: addi x4, x0, 13
        outer_join: m.join
        addi x4, x4, 100
        ret
        """
    )
    vm = MercuryVM()
    vm.set_scalar(2, [int(i % 2 == 0) for i in range(32)])
    vm.set_scalar(3, [int(i % 4 == 0) for i in range(32)])
    telemetry = vm.run(program)
    assert vm.read_scalar(4) == tuple(113 if i % 2 else 112 if i % 4 else 111 for i in range(32))
    assert telemetry.divergent_splits == 2
    assert telemetry.max_reconv_depth == 2


def test_zero_thread_mask_in_a_branch_reconverges_and_retirement_is_final():
    vm = MercuryVM()
    vm.set_scalar(2, [i % 2 for i in range(32)])
    vm.run(assemble(
        """
        m.split x2, other, done
        m.tmc x0
        addi x3, x0, 99
        other: addi x3, x0, 5
        done: m.join
        addi x3, x3, 1
        m.tmc x0
        addi x3, x0, 255
        """
    ))
    assert vm.read_scalar(3) == (6, 1) * 16


def test_scalar_call_and_return_wrap_rv64_values():
    vm = MercuryVM()
    program = assemble(
        """
        addi x2, x0, -2
        jal x5, routine
        sub x3, x3, x2
        ret
        routine: add x3, x2, x2
        mul x3, x3, x2
        jalr x0, 0(x5)
        """
    )
    vm.run(program)
    assert vm.read_scalar(2)[0] == (1 << 64) - 2
    assert vm.read_scalar(3) == (10,) * 32


def test_rvv_respects_vl_live_mask_and_explicit_v0_mask():
    vm = MercuryVM()
    vm.configure_warp(vl=8, live_mask=0b01101111)
    vm.set_vector_mask(0b11010111)
    vm.set_vector(1, [3.0] * 32)
    vm.set_vector(2, [2.0] * 32)
    vm.set_vector(3, [-1.0] * 32)
    vm.run(assemble("vfmul.vv v3, v1, v2, v0.t\nret"))
    assert vm.read_vector(3) == tuple(6.0 if i in (0, 1, 2, 6) else -1.0 for i in range(32))


def test_bf16_rounding_and_widening():
    vm = MercuryVM()
    vm.set_vector(1, [1 + 2.0**-8, 1 + 3 * 2.0**-8, -1, float("inf")] * 8)
    vm.run(assemble("vfcvt.bf16.f.v v2, v1\nvfcvt.f.bf16.v v3, v2\nret"))
    assert vm.read_vector(3) == (1.0, 1.015625, -1.0, float("inf")) * 8


def test_spawn_barrier_reuse_and_group_collectives():
    vm = MercuryVM(num_warps=2)
    shared = MemoryRef("shared_sram", 0)
    vm.set_scalar(6, shared.address)
    vm.set_vector(1, [1.0] * 32)
    vm.run(assemble(
        """
        addi x2, x0, 1
        m.wspawn x2, child
        m.bar 3
        ld x7, 0(x6)
        m.allreduce.sum v2, v1, 1
        m.allreduce.max v3, v1, 1
        m.bar 3
        ret
        child: addi x7, x0, 42
        sd x7, 0(x6)
        m.bar 3
        vfadd.vv v1, v1, v1
        m.allreduce.sum v2, v1, 1
        m.allreduce.max v3, v1, 1
        m.bar 3
        m.tmc x0
        """
    ))
    assert vm.read_scalar(7) == (42,) * 32
    for warp in (0, 1):
        assert vm.read_vector(2, warp=warp) == (96.0,) * 32
        assert vm.read_vector(3, warp=warp) == (2.0,) * 32
    assert vm.read_scalar(30, warp=1) == (1,) * 32
    assert vm.telemetry.coll_ops == 4


def test_barrier_deadlock_is_reported():
    vm = MercuryVM(num_warps=2, initial_warps=2)
    with pytest.raises(VMExecutionError, match="deadlock"):
        vm.run(assemble(
            """
            beq x0, x30, first
            m.bar 1
            ret
            first: m.bar 0
            ret
            """
        ))


def _dma_vm():
    vm = MercuryVM()
    src = MemoryRef("gddr7", 0)
    dst = MemoryRef("shared_sram", 128)
    vm.write_memory(src, b"abcd0123EFGH4567")
    vm.set_scalar(2, 7)
    vm.set_scalar(4, dst.address)
    vm.set_scalar(5, src.address)
    vm.register_dma(7, DMADescriptor(src, dst, 4, rows=3, src_stride=8, dst_stride=8, valid_rows=2, valid_row_bytes=3))
    return vm, dst


def test_dma_strides_zero_fill_and_visibility():
    vm, dst = _dma_vm()
    vm.configure_warp(vl=1)
    vm.run(assemble("m.dma.copy x3, x2\nm.dma.wait x3\nvle32.v v3, (x4)\nret"))
    assert vm.read_memory(dst, 20) == b"abc\0\0\0\0\0EFG\0\0\0\0\0\0\0\0\0"
    assert vm.telemetry.dma_ops == 1
    assert vm.telemetry.total_cycles_est > 4


def test_barrier_does_not_publish_an_unwaited_dma_token():
    vm = MercuryVM(num_warps=2, initial_warps=2)
    src, dst = MemoryRef("gddr7", 0), MemoryRef("l1", 0)
    vm.register_dma(1, DMADescriptor(src, dst, 128))
    vm.set_scalar(2, 1)
    vm.set_scalar(4, dst.address, warp=1)
    with pytest.raises(HazardViolationError, match="read.*before async wait"):
        vm.run(assemble(
            """
            bne x30, x0, reader
            m.dma.copy x3, x2
            m.bar 0
            m.dma.wait x3
            ret
            reader: m.bar 0
            vle32.v v1, (x4)
            ret
            """
        ))


@pytest.mark.parametrize("next_instruction", ["vle32.v v3, (x4)", "sd x0, 0(x5)", "ret"])
def test_dma_hazard_on_unwaited_destination_or_live_source(next_instruction):
    vm, _ = _dma_vm()
    with pytest.raises(HazardViolationError, match="async wait"):
        vm.run(assemble(f"m.dma.copy x3, x2\n{next_instruction}\nret"))


def _mma_vm():
    vm = MercuryVM()
    acc, a, b, scale_a, scale_b = [MemoryRef("l1", offset) for offset in (0, 128, 256, 384, 448)]
    vm.bind_tile(0, Tile(1, 1, "f32", acc))
    vm.bind_tile(1, Tile(1, 2, "e4m3", a, scale_a))
    vm.bind_tile(2, Tile(2, 1, "e4m3", b, scale_b))
    vm.write_memory(acc, struct.pack("<f", 7.0))
    vm.write_memory(a, bytes([0x38, 0x40]))  # 1, 2
    vm.write_memory(b, bytes([0x40, 0x44]))  # 2, 3
    vm.write_memory(scale_a, struct.pack("<f", 0.5))
    vm.write_memory(scale_b, struct.pack("<f", 2.0))
    vm.set_scalar(4, acc.address)
    vm.set_scalar(5, a.address)
    vm.set_scalar(6, scale_a.address)
    vm.configure_warp(vl=1)
    return vm, acc


@pytest.mark.parametrize("next_instruction", ["vle32.v v3, (x4)", "sd x0, 0(x5)", "sd x0, 0(x6)", "ret"])
def test_mma_hazard_on_unwaited_accumulator_or_live_input(next_instruction):
    vm, _ = _mma_vm()
    with pytest.raises(HazardViolationError, match="async wait"):
        vm.run(assemble(f"m.mma.fp8.blk t0, t1, t2\n{next_instruction}\nret"))


def test_chained_mma_forwards_accumulator_then_releases_borrowed_operand():
    vm, acc = _mma_vm()
    vm.run(assemble(
        """
        m.mma.fp8.blk t0, t1, t2
        m.mma.fp8.blk t0, t1, t2
        m.mma.wait t0
        sd x0, 0(x5)
        ret
        """
    ))
    assert struct.unpack("<f", vm.read_memory(acc, 4)) == (23.0,)
    assert vm.telemetry.stme_insts == 3


def test_both_shipped_kernels_run_from_binary_and_match_fp64():
    root = Path(__file__).resolve().parent.parent / "examples" / "kernels"
    results = json.loads(json.dumps(run_isa_kernel_benchmarks(), allow_nan=False))
    assert set(results) == {"divergent_relu_norm", "fused_fp8_gemm_flash_tile"}
    for name, result in results.items():
        program = assemble_file(root / f"{name}.masm")
        assert assemble(disassemble(program.words)).words == program.words
        assert result["fp64_parity"]
        assert result["max_abs_error"] < result["abs_error_tolerance"]
        assert result["telemetry"]["total_cycles_est"] > 0
    assert results["divergent_relu_norm"]["telemetry"]["divergent_splits"] == 1
    assert results["fused_fp8_gemm_flash_tile"]["telemetry"]["stme_insts"] == 4
    assert results["fused_fp8_gemm_flash_tile"]["telemetry"]["dma_ops"] == 2
