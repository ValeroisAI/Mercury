(() => {
  "use strict";

  const state = {
    data: null,
    sku: "Mercury-1D",
    thermalKey: "oam_liquid_35c",
    nreVolume: 100000,
  };

  const fmtNum = (n, d = 1) => Number(n).toLocaleString("en-US", {
    minimumFractionDigits: d,
    maximumFractionDigits: d,
  });
  const fmtInt = (n) => Math.round(Number(n)).toLocaleString("en-US");

  function currentThemeIsDark() {
    const explicit = document.documentElement.getAttribute("data-theme");
    if (explicit === "dark") return true;
    if (explicit === "light") return false;
    return !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  }

  function syncThemeButton() {
    const btn = document.getElementById("theme-toggle");
    if (!btn) return;
    const dark = currentThemeIsDark();
    btn.setAttribute("aria-pressed", dark ? "true" : "false");
  }

  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    syncThemeButton();
  }

  function renderHero() {
    if (!state.data) return;
    const s = state.data.skus[state.sku];
    const ledger = document.getElementById("hero-ledger");
    ledger.innerHTML = `
      <div class="ledger-cell">
        <dt>Dense Tensor Compute</dt>
        <dd class="ledger-val">${fmtNum(s.bf16_tflops)} TF</dd>
        <dd class="ledger-sub">BF16 (${fmtNum(s.fp8_tflops)} TF FP8)</dd>
      </div>
      <div class="ledger-cell">
        <dt>GDDR7 &amp; Tensor SRAM</dt>
        <dd class="ledger-val">${s.vram_gb} GB</dd>
        <dd class="ledger-sub">${s.vram_gib} GiB · ${s.vram_bw_tbs} TB/s DRAM · ${s.shared_sram_mb} MB SRAM at ${s.shared_sram_bw_tbs} TB/s · ${s.l1_rf_mib} MiB L1/RF</dd>
      </div>
      <div class="ledger-cell">
        <dt>Est. Hardware BOM</dt>
        <dd class="ledger-val">$${fmtInt(s.bom_usd)}</dd>
        <dd class="ledger-sub">Hardware-only, excl. $85M NRE</dd>
      </div>
      <div class="ledger-cell">
        <dt>Thermal &amp; Networking</dt>
        <dd class="ledger-val">${s.tdp_w} W</dd>
        <dd class="ledger-sub">${s.eth_ports_400g}x 400GbE, ${s.active_sccs}/${s.fab_sccs} SCCs</dd>
      </div>
    `;
  }

  function renderThermalSvg() {
    const svg = document.getElementById("thermal-svg");
    if (!svg || !state.data) return;
    const run = state.data.thermal_traces[state.thermalKey];
    if (!run) return;
    const pts = run.trace;

    const w = 1160, h = 360;
    const padL = 56, padR = 24, padT = 38, padB = 34;
    const pw = w - padL - padR;
    const ph = h - padT - padB;
    const tMax = 360.0, yMin = 20.0, yMax = 115.0;
    const xOf = (t) => (padL + (t / tMax) * pw).toFixed(1);
    const yOf = (c) => (padT + ph - ((c - yMin) / (yMax - yMin)) * ph).toFixed(1);

    const hGrid = [20, 40, 60, 85, 90, 105].map((temp) => {
      const y = yOf(temp);
      const col = temp === 105 ? "var(--status-fail)" : (temp === 85 || temp === 90 ? "var(--status-warn)" : "var(--border-hairline)");
      const dash = temp === 85 || temp === 90 || temp === 105 ? 'stroke-dasharray="6 4"' : "";
      const wt = temp === 85 || temp === 90 || temp === 105 ? "1.4" : "1";
      return `<line x1="${padL}" y1="${y}" x2="${padL + pw}" y2="${y}" stroke="${col}" stroke-width="${wt}" ${dash}/>
              <text x="${padL - 8}" y="${y}" fill="${temp >= 85 ? col : "var(--ink-secondary)"}" font-family="Consolas,monospace" font-size="11" text-anchor="end" dominant-baseline="middle">${temp}°C</text>`;
    }).join("");

    const vGrid = [0, 60, 120, 180, 240, 300, 360].map((sec) => {
      const x = xOf(sec);
      return `<line x1="${x}" y1="${padT}" x2="${x}" y2="${padT + ph}" stroke="var(--border-hairline)" stroke-width="1"/>
              <text x="${x}" y="${padT + ph + 16}" fill="var(--ink-secondary)" font-family="Consolas,monospace" font-size="11" text-anchor="middle">${sec}s</text>`;
    }).join("");

    const faultLine = state.thermalKey === "mid_run_fault_120s"
      ? `<line x1="${xOf(120)}" y1="${padT}" x2="${xOf(120)}" y2="${padT + ph}" stroke="var(--status-fail)" stroke-width="1.5" stroke-dasharray="5 4"/>
         <text x="${Number(xOf(120)) + 8}" y="${padT + 14}" fill="var(--status-fail)" font-family="Segoe UI,sans-serif" font-size="12" font-weight="600">t=120s cooling fault</text>`
      : "";

    const poly = (getter, color) => {
      const d = pts.map((p, i) => `${i === 0 ? "M" : "L"}${xOf(p.time_s)},${yOf(getter(p))}`).join(" ");
      const last = pts[pts.length - 1];
      return `<path d="${d}" fill="none" stroke="${color}" stroke-width="2.8" stroke-linejoin="round" stroke-linecap="round"/>
              <circle cx="${xOf(last.time_s)}" cy="${yOf(getter(last))}" r="4" fill="${color}"/>`;
    };

    const thresholdNotes = `
      <text x="${padL + 8}" y="${Number(yOf(85)) + 12}" fill="var(--status-warn)" font-family="Segoe UI,sans-serif" font-size="11" font-weight="600">85°C JEDEC 2x tREFI</text>
      <text x="${padL + pw - 6}" y="${Number(yOf(105)) - 6}" fill="var(--status-fail)" font-family="Segoe UI,sans-serif" font-size="11" font-weight="600" text-anchor="end">105°C PROCHOT Trip</text>
    `;

    svg.innerHTML = `
      <rect x="0" y="0" width="${w}" height="${h}" rx="4" fill="var(--bg-surface)"/>
      ${hGrid}
      ${vGrid}
      ${faultLine}
      ${thresholdNotes}
      ${poly((p) => p.tcase_c, "var(--status-pass)")}
      ${poly((p) => p.tmem_c, "var(--status-warn)")}
      ${poly((p) => p.tj_c, "var(--accent)")}
      <g font-family="Segoe UI,sans-serif" font-size="12" font-weight="600">
        <rect x="${padL}" y="12" width="14" height="4" fill="var(--accent)"/><text x="${padL + 20}" y="18" fill="var(--ink-primary)">Die Tj (°C)</text>
        <rect x="${padL + 120}" y="12" width="14" height="4" fill="var(--status-warn)"/><text x="${padL + 140}" y="18" fill="var(--ink-primary)">Clamshell GDDR7 Tmem (°C)</text>
        <rect x="${padL + 335}" y="12" width="14" height="4" fill="var(--status-pass)"/><text x="${padL + 355}" y="18" fill="var(--ink-primary)">Case / Coldplate (°C)</text>
      </g>
    `;

    document.getElementById("thermal-readout").textContent =
      `Steady Tj ${fmtNum(run.steady_tj_c)}°C, Tmem ${fmtNum(run.steady_tmem_c)}°C at ${fmtNum(run.steady_clock_ghz, 2)} GHz (${fmtInt(run.steady_chip_power_w)}W chip)`;
  }

  function renderThermalTable() {
    const tbl = document.getElementById("thermal-table");
    const runs = [
      ["1D OCP Liquid 35°C", state.data.thermal_traces.oam_liquid_35c],
      ["1S Workstation Air 25°C", state.data.thermal_traces.workstation_air_25c],
      ["1D 4U Hot Air 35°C", state.data.thermal_traces.server_4u_air_35c],
      ["1D Mid-Run Fault @120s", state.data.thermal_traces.mid_run_fault_120s],
    ];
    tbl.innerHTML = `
      <thead><tr>
        <th>Scenario</th><th class="num">Steady Tj</th><th class="num col-hide-sm">Steady Tmem</th>
        <th class="num">Clock</th><th class="num col-hide-sm">Chip / Board</th><th class="num col-hide-sm">Throttle</th><th class="col-hide-sm">2x tREFI</th><th>Trip</th>
      </tr></thead>
      <tbody>
        ${runs.map(([name, r]) => `
          <tr>
            <td>${name}</td>
            <td class="num">${fmtNum(r.steady_tj_c)}°C<span class="col-hide-sm"> (pk ${fmtNum(r.peak_tj_c)}°C)</span></td>
            <td class="num col-hide-sm">${fmtNum(r.steady_tmem_c)}°C</td>
            <td class="num">${fmtNum(r.steady_clock_ghz, 2)} GHz</td>
            <td class="num col-hide-sm">${fmtInt(r.steady_chip_power_w)}W / ${fmtInt(r.steady_board_power_w)}W</td>
            <td class="num col-hide-sm">${fmtNum(r.throttle_fraction * 100)}%</td>
            <td class="col-hide-sm"><span class="badge ${r.double_refresh_triggered ? "badge-warn" : "badge-pass"}">${r.double_refresh_triggered ? "ACTIVE" : "NO"}</span></td>
            <td><span class="badge ${r.thermal_trip_triggered ? "badge-fail" : "badge-pass"}">${r.thermal_trip_triggered ? "PROCHOT" : "SAFE"}</span></td>
          </tr>
        `).join("")}
      </tbody>
    `;
  }

  function renderPcbAndNre() {
    const pb = state.data.physical_bom;
    const pcbTbl = document.getElementById("pcb-table");
    if (!pb || !pb.pcb_layouts) {
      pcbTbl.innerHTML = `<tbody><tr><td>Physical PCB courtyard module not loaded in JSON.</td></tr></tbody>`;
      return;
    }
    const pcbRows = Array.isArray(pb.pcb_layouts) ? pb.pcb_layouts : Object.values(pb.pcb_layouts);
    const labelMap = {
      standard_oam_2_0_48_chips: "Legacy OAM v1.5 (144 GB / 48 chips)",
      ocp_oam_r2_102x170_48_chips: "OCP OAM r2.0 (144 GB / 48 chips)",
      extended_oam_l_140x185_48_chips: "Extended OAM-L (144 GB / 48 chips)",
      standard_oam_2_0_96gb_32chips: "Legacy OAM v1.5 Fallback (96 GB / 32 chips)",
      pcie_dual_slot_111x267_24_chips: "Mercury-1S PCIe Card (72 GB / 24 chips)",
    };
    pcbTbl.innerHTML = `
      <thead><tr>
        <th>Form Factor</th><th class="num">Board</th><th class="num">Top / Bot Fill</th><th class="num col-hide-sm">Max Trace</th><th>Fit</th>
      </tr></thead>
      <tbody>
        ${pcbRows.map((row) => {
          const bStr = Array.isArray(row.board_mm) ? `${row.board_mm[0]}x${row.board_mm[1]} mm` : row.board_mm;
          const title = labelMap[row.form_factor] || row.form_factor;
          const reason = row.blocking_reason ? row.blocking_reason.split(";")[0] : "Within <75% routing allowance & <=35 mm PAM3 trace target";
          return `
          <tr>
            <td><strong>${title}</strong><br><span class="box-sub">${reason}</span></td>
            <td class="num">${bStr}</td>
            <td class="num">${fmtNum(row.top_used_pct)}% / ${fmtNum(row.bottom_used_pct)}%</td>
            <td class="num col-hide-sm">${fmtNum(row.max_gddr7_trace_mm)} mm</td>
            <td><span class="badge ${row.fits_physically ? "badge-pass" : "badge-fail"}">${row.fits_physically ? "PASS" : "OVERFLOW"}</span></td>
          </tr>
        `;
        }).join("")}
      </tbody>
    `;

    const nreRows = pb.nre_volume_table || [];
    const sel = nreRows.find((r) => r.volume_units === state.nreVolume) || nreRows[0];
    const callout = document.getElementById("nre-callout");
    if (sel) {
      const bom = sel.hardware_bom_usd ?? sel.bom_usd;
      callout.textContent = `At ${fmtInt(sel.volume_units)} units: BOM $${fmtInt(bom)} + NRE $${fmtInt(sel.nre_per_unit_usd)} = $${fmtInt(sel.fully_loaded_unit_cost_usd)} unit cost, 45% GM target $${fmtInt(sel.target_price_45gm_usd)}, $${fmtNum(sel.campaign_1t_cost_per_1m_tokens_usd, 4)} per 1M tokens`;
    }
    const nreTbl = document.getElementById("nre-table");
    nreTbl.innerHTML = `
      <thead><tr>
        <th class="num">Volume</th><th class="num">NRE / Unit</th><th class="num">Loaded Cost</th><th class="num">45% GM Price</th><th class="num">1T Tok $/1M</th>
      </tr></thead>
      <tbody>
        ${nreRows.map((r) => `
          <tr style="${r.volume_units === state.nreVolume ? "background:var(--accent-soft)" : ""}">
            <td class="num">${fmtInt(r.volume_units)}</td>
            <td class="num">$${fmtInt(r.nre_per_unit_usd)}</td>
            <td class="num">$${fmtInt(r.fully_loaded_unit_cost_usd)}</td>
            <td class="num">$${fmtInt(r.target_price_45gm_usd)}</td>
            <td class="num">$${fmtNum(r.campaign_1t_cost_per_1m_tokens_usd, 4)}</td>
          </tr>
        `).join("")}
      </tbody>
    `;
  }

  function renderConvergenceSvg() {
    const svg = document.getElementById("convergence-svg");
    if (!svg || !state.data || !state.data.pytorch_bridge) return;
    const conv = state.data.pytorch_bridge.training_convergence;
    if (!conv || !conv.paths) return;

    const w = 600, h = 220;
    const padL = 48, padR = 16, padT = 30, padB = 28;
    const pw = w - padL - padR;
    const ph = h - padT - padB;
    const fp64 = conv.paths.fp64_golden.loss_curve;
    const merc = conv.paths.mercury_fp8_blk.loss_curve;
    const hop = conv.paths.hopper_14bit.loss_curve;
    const all = [...fp64, ...merc, ...hop];
    const minL = Math.min(...all) * 0.96;
    const maxL = Math.max(...all) * 1.03;
    const xOf = (i) => (padL + (i / Math.max(1, fp64.length - 1)) * pw).toFixed(1);
    const yOf = (v) => (padT + ph - ((v - minL) / Math.max(1e-6, maxL - minL)) * ph).toFixed(1);

    const grid = [0, 1, 2, 3].map((k) => {
      const val = minL + (k / 3) * (maxL - minL);
      const y = yOf(val);
      return `<line x1="${padL}" y1="${y}" x2="${padL + pw}" y2="${y}" stroke="var(--border-hairline)" stroke-width="1"/>
              <text x="${padL - 6}" y="${y}" fill="var(--ink-secondary)" font-family="Consolas,monospace" font-size="11" text-anchor="end" dominant-baseline="middle">${val.toFixed(1)}</text>`;
    }).join("");

    const path = (arr, col, width, dash) => {
      const d = arr.map((v, i) => `${i === 0 ? "M" : "L"}${xOf(i)},${yOf(v)}`).join(" ");
      return `<path d="${d}" fill="none" stroke="${col}" stroke-width="${width}" stroke-linejoin="round" ${dash ? `stroke-dasharray="${dash}"` : ""}/>`;
    };
    const endFp64 = fp64[fp64.length - 1].toFixed(2);
    const endMerc = merc[merc.length - 1].toFixed(2);
    const endHop = hop[hop.length - 1].toFixed(2);

    svg.innerHTML = `
      <rect x="0" y="0" width="${w}" height="${h}" rx="4" fill="var(--bg-surface)"/>
      ${grid}
      ${path(fp64, "var(--status-pass)", "3.2", "")}
      ${path(hop, "var(--status-fail)", "2.2", "7 4")}
      ${path(merc, "var(--accent)", "1.8", "3 2")}
      <text x="${padL}" y="${padT + ph + 18}" fill="var(--ink-secondary)" font-family="Consolas,monospace" font-size="11">Step 1</text>
      <text x="${padL + pw}" y="${padT + ph + 18}" fill="var(--ink-secondary)" font-family="Consolas,monospace" font-size="11" text-anchor="end">Step ${fp64.length} (near-identical convergence)</text>
      <g font-family="Segoe UI,sans-serif" font-size="11" font-weight="600">
        <rect x="${padL}" y="10" width="12" height="3" fill="var(--status-pass)"/><text x="${padL + 16}" y="14" fill="var(--ink-primary)">FP64 (${endFp64})</text>
        <rect x="${padL + 118}" y="10" width="12" height="3" fill="var(--accent)"/><text x="${padL + 134}" y="14" fill="var(--ink-primary)">Mercury FP8 (${endMerc})</text>
        <rect x="${padL + 266}" y="10" width="12" height="3" fill="var(--status-fail)"/><text x="${padL + 282}" y="14" fill="var(--ink-primary)">Hopper-14b (${endHop})</text>
      </g>
    `;
  }

  function renderMicroarchAndTorch() {
    const m = state.data.microarch;
    const mTbl = document.getElementById("microarch-table");
    if (m && m.scenarios) {
      const order = [
        ["multicast_256x128", "256x128 + A-Multicast"],
        ["swizzled_baseline", "128x128 XOR-Swizzle Baseline"],
        ["hot_2x_trefi_85c", "Hot >=85°C (2x tREFI)"],
        ["cross_die_ucie_spill", "Cross-Die UCIe Spill (10%)"],
        ["unswizzled_bank_conflict", "Unswizzled Stride-64 (2/64 banks)"],
      ].filter(([k]) => m.scenarios[k]);
      mTbl.innerHTML = `
        <thead><tr>
          <th>Configuration</th><th class="num">Cycles</th><th class="num">FP8 TFLOPS</th><th class="num col-hide-sm">SRAM TB/s</th><th class="num col-hide-sm">Bank Conf</th>
        </tr></thead>
        <tbody>
          ${order.map(([k, lbl]) => {
            const s = m.scenarios[k];
            return `<tr>
              <td>${lbl}</td>
              <td class="num">${fmtInt(s.total_cycles)}</td>
              <td class="num">${fmtNum(s.achieved_tflops)}</td>
              <td class="num col-hide-sm">${fmtNum(s.achieved_sram_tbs, 2)}</td>
              <td class="num col-hide-sm">${fmtInt(s.bank_conflict_cycles)}</td>
            </tr>`;
          }).join("")}
        </tbody>
      `;
    }

    const pt = state.data.pytorch_bridge;
    const ptTbl = document.getElementById("pytorch-table");
    if (pt && pt.gemms) {
      const ratios = pt.summary.accumulator_error_ratio_hopper_over_mercury;
      const rows = [
        ["Forward Y", pt.gemms.forward_Y, ratios.forward_Y],
        ["Input Grad dX", pt.gemms.grad_dX, ratios.grad_dX],
        ["Weight Grad dW", pt.gemms.grad_dW, ratios.grad_dW],
      ];
      ptTbl.innerHTML = `
        <thead><tr>
          <th>GEMM</th><th class="num">E2E Cos Sim (Merc / Hop)</th><th class="num col-hide-sm">Acc Rel L2 (Merc vs Hop)</th><th class="num">Acc Ratio</th>
        </tr></thead>
        <tbody>
          ${rows.map(([lbl, g, r]) => `
            <tr>
              <td>${lbl}</td>
              <td class="num">${g.mercury.cos_sim.toFixed(6)} / ${g.hopper_14bit.cos_sim.toFixed(6)}</td>
              <td class="num col-hide-sm">${g.mercury_accumulator_only.rel_l2.toExponential(2)} vs ${g.hopper_accumulator_only.rel_l2.toExponential(2)}</td>
              <td class="num"><span class="badge badge-pass">${fmtNum(r)}x lower</span></td>
            </tr>
          `).join("")}
        </tbody>
      `;
    }
    renderConvergenceSvg();
  }

  function renderIsaAndCluster() {
    const isa = state.data.isa_vm;
    const isaTbl = document.getElementById("isa-table");
    if (isa) {
      isaTbl.innerHTML = `
        <thead><tr>
          <th>Kernel</th><th class="num">Cycles</th><th class="num col-hide-sm">STME/DMA/Coll/Split</th><th class="num">Max Err vs FP64</th><th>Parity</th>
        </tr></thead>
        <tbody>
          ${Object.entries(isa).map(([name, k]) => {
            const t = k.telemetry;
            return `<tr>
              <td><code class="mono">${name}.masm</code></td>
              <td class="num">${t.total_cycles_est}</td>
              <td class="num col-hide-sm">${t.stme_insts} / ${t.dma_ops} / ${t.coll_ops} / ${t.divergent_splits}</td>
              <td class="num">${k.max_abs_error.toExponential(2)}</td>
              <td><span class="badge ${k.fp64_parity ? "badge-pass" : "badge-fail"}">${k.fp64_parity ? "PASS" : "FAIL"}</span></td>
            </tr>`;
          }).join("")}
        </tbody>
      `;
    }

    const q = state.data.qualification;
    const s6 = q.s6_cluster_70b;
    const s7 = q.s7_moe_and_decode;
    const cTbl = document.getElementById("cluster-table");
    const clusters = [
      ["8x 1D 70B TP=2(UCIe)+FSDP=4", s6.pod8_tp2],
      ["16x 1D 70B Full 16B/p AdamW", s6.pod16_16b],
      ["8x 1D Naive TP=8 over Eth", s6.pod8_tp8],
      ["16x 1D MoE 141B EP=8+FSDP=2", s7.moe_ep8],
    ];
    cTbl.innerHTML = `
      <thead><tr>
        <th>Cluster Workload</th><th class="num col-hide-sm">Free GiB</th><th class="num col-hide-sm">Comm %</th><th class="num">MFU</th><th class="num">Cluster tok/s</th>
      </tr></thead>
      <tbody>
        ${clusters.map(([lbl, c]) => `
          <tr>
            <td>${lbl}</td>
            <td class="num col-hide-sm">${fmtNum(c.activation_headroom_gib)} GiB</td>
            <td class="num col-hide-sm">${fmtNum((c.exposed_comm_time_s / c.step_time_s) * 100)}%</td>
            <td class="num">${fmtNum(c.achieved_mfu * 100)}%</td>
            <td class="num">${fmtInt(c.cluster_tokens_per_sec)}</td>
          </tr>
        `).join("")}
        <tr>
          <td><strong>Batch=1 70B FP8 Decode (Honesty)</strong></td>
          <td class="num col-hide-sm">—</td>
          <td class="num col-hide-sm">BW bound</td>
          <td class="num">3.07 TB/s</td>
          <td class="num"><span class="badge badge-warn">${fmtNum(s7.decode_b1.tokens_per_sec)} tok/s</span></td>
        </tr>
      </tbody>
    `;
  }

  function wireEvents() {
    if (window.matchMedia) {
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", syncThemeButton);
    }
    document.getElementById("theme-toggle").addEventListener("click", () => {
      setTheme(currentThemeIsDark() ? "light" : "dark");
    });

    document.querySelectorAll(".sku-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.sku = btn.dataset.sku;
        document.querySelectorAll(".sku-btn").forEach((b) => {
          const on = b === btn;
          b.classList.toggle("is-active", on);
          b.setAttribute("aria-pressed", on ? "true" : "false");
        });
        renderHero();
      });
    });

    const tabs = [...document.querySelectorAll(".tab-btn")];
    function activateTab(selected) {
      tabs.forEach((tab) => {
        const on = tab === selected;
        tab.classList.toggle("is-active", on);
        tab.setAttribute("aria-selected", String(on));
        tab.tabIndex = on ? 0 : -1;
        const panel = document.getElementById(tab.getAttribute("aria-controls"));
        panel.classList.toggle("is-active", on);
        panel.hidden = !on;
      });
    }
    tabs.forEach((tab, index) => {
      tab.addEventListener("click", () => activateTab(tab));
      tab.addEventListener("keydown", (ev) => {
        const offsets = {
          ArrowLeft: (index - 1 + tabs.length) % tabs.length,
          ArrowRight: (index + 1) % tabs.length,
          Home: 0,
          End: tabs.length - 1,
        };
        if (!Object.hasOwn(offsets, ev.key)) return;
        ev.preventDefault();
        const target = tabs[offsets[ev.key]];
        activateTab(target);
        target.focus();
      });
    });

    document.querySelectorAll("#thermal-buttons .pill-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.thermalKey = btn.dataset.thermal;
        document.querySelectorAll("#thermal-buttons .pill-btn").forEach((b) => {
          const on = b === btn;
          b.classList.toggle("is-active", on);
          b.setAttribute("aria-pressed", on ? "true" : "false");
        });
        renderThermalSvg();
      });
    });

    document.getElementById("nre-volume-select").addEventListener("change", (ev) => {
      state.nreVolume = Number(ev.target.value);
      renderPcbAndNre();
    });
  }

  async function init() {
    syncThemeButton();
    wireEvents();
    const banner = document.getElementById("load-state");
    try {
      const resp = await fetch("./data/qualification.json");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      state.data = await resp.json();
      renderHero();
      renderThermalSvg();
      renderThermalTable();
      renderPcbAndNre();
      renderMicroarchAndTorch();
      renderIsaAndCluster();
      banner.hidden = true;
    } catch (err) {
      banner.hidden = false;
      banner.dataset.error = "";
      banner.textContent = `Could not load dashboard/data/qualification.json (${err.message}). Run "python scripts/export_dashboard_data.py" and serve via HTTP.`;
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
