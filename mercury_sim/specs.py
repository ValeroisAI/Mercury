"""Locked architectural, physical, thermal, and BOM specifications for Mercury-1."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DVFSState:
    """Single voltage/frequency operating point for Mercury-1 SCCs."""

    name: str
    clock_ghz: float
    vcore_v: float


@dataclass(frozen=True)
class CoolingProfile:
    """Thermal resistance & ambient boundary model for a real server environment."""

    name: str
    description: str
    ambient_c: float
    r_theta_jc_die: float  # Junction-to-case/coldplate thermal resistance per die (C/W)
    r_theta_ca: float      # Case/coldplate-to-ambient/coolant thermal resistance per card (C/W)
    c_th_die_j_per_c: float  # Effective die + spreader thermal capacitance (J/C)
    c_th_mem_j_per_c: float  # Effective GDDR7 clamshell PCB/DRAM thermal capacitance (J/C)
    mem_coupling_factor: float  # Fraction of GDDR7 power + PCB heat reflected into mem node


@dataclass(frozen=True)
class MercurySKU:
    """Immutable hardware specification for a Mercury-1 product SKU."""

    name: str
    form_factor: str
    num_dies: int
    die_area_mm2: float
    fab_sccs_per_die: int
    active_sccs_per_die: int
    systolic_arrays_per_scc: int
    systolic_m: int
    systolic_n: int
    rvv_fp32_lanes_per_scc: int
    rf_kib_per_scc: int
    l1_kib_per_scc: int
    shared_sram_mb_per_die: float
    shared_sram_banks_per_die: int
    shared_sram_bytes_per_cycle_per_bank: int
    gddr7_bus_bits_per_die: int
    gddr7_gbps_per_pin: float
    vram_gb_total: int
    ucie_s_bidir_tbs: float
    pcie_gen5_lanes: int
    eth_400g_ports: int
    nominal_clock_ghz: float
    nominal_vcore_v: float
    tdp_w: int
    board_power_limit_w: int
    vrm_efficiency: float
    tj_throttle_c: float
    tj_trip_c: float
    tmem_double_refresh_c: float
    tmem_throttle_c: float
    tmem_trip_c: float
    silicon_bom_usd: float
    mem_bom_usd: float
    pkg_bom_usd: float
    board_cool_bom_usd: float
    test_osat_bom_usd: float
    target_street_usd: float

    @property
    def total_active_sccs(self) -> int:
        return self.num_dies * self.active_sccs_per_die

    @property
    def bf16_macs_per_scc(self) -> int:
        return self.systolic_arrays_per_scc * self.systolic_m * self.systolic_n

    @property
    def vram_gib_total(self) -> float:
        return (self.vram_gb_total * 1e9) / (1024**3)

    @property
    def vram_bandwidth_gbs(self) -> float:
        return (self.num_dies * self.gddr7_bus_bits_per_die * self.gddr7_gbps_per_pin) / 8.0

    @property
    def vram_bandwidth_tbs(self) -> float:
        return self.vram_bandwidth_gbs / 1000.0

    @property
    def shared_sram_mb_total(self) -> float:
        return self.num_dies * self.shared_sram_mb_per_die

    @property
    def active_l1_mib_total(self) -> float:
        return (self.total_active_sccs * self.l1_kib_per_scc) / 1024.0

    @property
    def active_rf_mib_total(self) -> float:
        return (self.total_active_sccs * self.rf_kib_per_scc) / 1024.0

    @property
    def total_bom_usd(self) -> float:
        return (
            self.silicon_bom_usd
            + self.mem_bom_usd
            + self.pkg_bom_usd
            + self.board_cool_bom_usd
            + self.test_osat_bom_usd
        )

    @property
    def eth_unidir_gbs(self) -> float:
        return (self.eth_400g_ports * 400.0) / 8.0

    def peak_tflops(self, dtype: str, clock_ghz: float | None = None, active_sccs: int | None = None, sparse: bool = False) -> float:
        clk = self.nominal_clock_ghz if clock_ghz is None else clock_ghz
        sccs = self.total_active_sccs if active_sccs is None else active_sccs
        key = dtype.lower()
        if key == "bf16":
            base = sccs * self.bf16_macs_per_scc * 2.0 * clk / 1000.0
        elif key == "fp8":
            base = sccs * self.bf16_macs_per_scc * 2.0 * 2.0 * clk / 1000.0
        elif key in ("fp4", "mxfp4"):
            base = sccs * self.bf16_macs_per_scc * 4.0 * 2.0 * clk / 1000.0
        elif key == "tf32":
            base = sccs * self.bf16_macs_per_scc * 0.5 * 2.0 * clk / 1000.0
        elif key == "fp32":
            base = sccs * self.rvv_fp32_lanes_per_scc * 2.0 * clk / 1000.0
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        if sparse and key in ("bf16", "fp8", "fp4", "mxfp4"):
            return base * 2.0
        return base

    def shared_sram_bandwidth_tbs(self, clock_ghz: float | None = None) -> float:
        clk = self.nominal_clock_ghz if clock_ghz is None else clock_ghz
        per_die_gbs = (
            self.shared_sram_banks_per_die
            * self.shared_sram_bytes_per_cycle_per_bank
            * clk
        )
        return (self.num_dies * per_die_gbs) / 1000.0


DVFS_TABLE: tuple[DVFSState, ...] = (
    DVFSState("P0_BOOST", 1.85, 0.78),
    DVFSState("P1_SUSTAIN", 1.70, 0.74),
    DVFSState("P2_ECO", 1.50, 0.69),
    DVFSState("P3_THERMAL", 1.25, 0.64),
    DVFSState("P4_SAFE", 0.90, 0.59),
    DVFSState("P5_IDLE", 0.40, 0.55),
)


MERCURY_1S = MercurySKU(
    name="Mercury-1S",
    form_factor="PCIe Gen5 x16 Dual-Slot",
    num_dies=1,
    die_area_mm2=453.8,
    fab_sccs_per_die=84,
    active_sccs_per_die=76,
    systolic_arrays_per_scc=8,
    systolic_m=16,
    systolic_n=16,
    rvv_fp32_lanes_per_scc=128,
    rf_kib_per_scc=256,
    l1_kib_per_scc=256,
    shared_sram_mb_per_die=192.0,
    shared_sram_banks_per_die=64,
    shared_sram_bytes_per_cycle_per_bank=64,
    gddr7_bus_bits_per_die=384,
    gddr7_gbps_per_pin=32.0,
    vram_gb_total=72,
    ucie_s_bidir_tbs=0.0,
    pcie_gen5_lanes=16,
    eth_400g_ports=2,
    nominal_clock_ghz=1.85,
    nominal_vcore_v=0.78,
    tdp_w=340,
    board_power_limit_w=365,
    vrm_efficiency=0.93,
    tj_throttle_c=90.0,
    tj_trip_c=105.0,
    tmem_double_refresh_c=85.0,
    tmem_throttle_c=95.0,
    tmem_trip_c=105.0,
    silicon_bom_usd=217.0,
    mem_bom_usd=576.0,
    pkg_bom_usd=90.0,
    board_cool_bom_usd=240.0,
    test_osat_bom_usd=115.0,
    target_street_usd=2400.0,
)

MERCURY_1D = MercurySKU(
    name="Mercury-1D",
    form_factor="Extended OAM-L 48V Module (140x185 mm)",
    num_dies=2,
    die_area_mm2=453.8,
    fab_sccs_per_die=84,
    active_sccs_per_die=76,
    systolic_arrays_per_scc=8,
    systolic_m=16,
    systolic_n=16,
    rvv_fp32_lanes_per_scc=128,
    rf_kib_per_scc=256,
    l1_kib_per_scc=256,
    shared_sram_mb_per_die=192.0,
    shared_sram_banks_per_die=64,
    shared_sram_bytes_per_cycle_per_bank=64,
    gddr7_bus_bits_per_die=384,
    gddr7_gbps_per_pin=32.0,
    vram_gb_total=144,
    ucie_s_bidir_tbs=1.2,
    pcie_gen5_lanes=16,
    eth_400g_ports=4,
    nominal_clock_ghz=1.85,
    nominal_vcore_v=0.78,
    tdp_w=680,
    board_power_limit_w=730,
    vrm_efficiency=0.94,
    tj_throttle_c=90.0,
    tj_trip_c=105.0,
    tmem_double_refresh_c=85.0,
    tmem_throttle_c=95.0,
    tmem_trip_c=105.0,
    silicon_bom_usd=434.0,
    mem_bom_usd=1152.0,
    pkg_bom_usd=220.0,
    board_cool_bom_usd=420.0,
    test_osat_bom_usd=230.0,
    target_street_usd=4700.0,
)


COOLING_PROFILES: dict[str, CoolingProfile] = {
    "workstation_air_25c": CoolingProfile(
        name="workstation_air_25c",
        description="PCIe Dual-Slot Active Air Cooler @ 25C Room Ambient",
        ambient_c=25.0,
        r_theta_jc_die=0.055,
        r_theta_ca=0.115,
        c_th_die_j_per_c=45.0,
        c_th_mem_j_per_c=28.0,
        mem_coupling_factor=0.35,
    ),
    "workstation_air_35c": CoolingProfile(
        name="workstation_air_35c",
        description="PCIe Dual-Slot Active Air Cooler @ 35C Hot Chassis",
        ambient_c=35.0,
        r_theta_jc_die=0.055,
        r_theta_ca=0.115,
        c_th_die_j_per_c=45.0,
        c_th_mem_j_per_c=28.0,
        mem_coupling_factor=0.35,
    ),
    "server_4u_air_25c": CoolingProfile(
        name="server_4u_air_25c",
        description="4U High-CFM Air-Cooled OAM Chassis @ 25C Cold Aisle",
        ambient_c=25.0,
        r_theta_jc_die=0.045,
        r_theta_ca=0.068,
        c_th_die_j_per_c=85.0,
        c_th_mem_j_per_c=52.0,
        mem_coupling_factor=0.40,
    ),
    "server_4u_air_35c": CoolingProfile(
        name="server_4u_air_35c",
        description="4U Air-Cooled OAM Chassis @ 35C Elevated Inlet",
        ambient_c=35.0,
        r_theta_jc_die=0.045,
        r_theta_ca=0.068,
        c_th_die_j_per_c=85.0,
        c_th_mem_j_per_c=52.0,
        mem_coupling_factor=0.40,
    ),
    "oam_liquid_35c": CoolingProfile(
        name="oam_liquid_35c",
        description="Extended OAM-L Direct-to-Chip Liquid Coldplate @ 35C Coolant",
        ambient_c=35.0,
        r_theta_jc_die=0.025,
        r_theta_ca=0.022,
        c_th_die_j_per_c=75.0,
        c_th_mem_j_per_c=48.0,
        mem_coupling_factor=0.22,
    ),
    "degraded_fan_fault": CoolingProfile(
        name="degraded_fan_fault",
        description="Simulated Server Fan / Pump Stall Fault @ 30C Ambient",
        ambient_c=30.0,
        r_theta_jc_die=0.055,
        r_theta_ca=0.195,
        c_th_die_j_per_c=60.0,
        c_th_mem_j_per_c=35.0,
        mem_coupling_factor=0.45,
    ),
}
