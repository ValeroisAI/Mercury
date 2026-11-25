# Mercury-1 Workbench Design Contract (`DESIGN.md`)

> Single source of truth for visual tokens, typography, spacing, layout rules, and interactive component behavior for the `Mercury-1` Engineering Qualification Workbench (`dashboard/`).

## 1. Design Direction
- **Audience & Mood:** Silicon architects, ML infrastructure leads, and hardware engineers evaluating pre-silicon engineering evidence. Calm, precision semiconductor datasheet and oscilloscope lab mood with zero marketing hype.
- **Signature Idea:** Asymmetric split-die hardware workbench pairing live simulation traces (2-node thermal RC ODEs, PCB courtyard clearance, NRE volume amortization, crossbar cycles, PyTorch convergence, and RISC-V `Xmer*` binary execution) with explicit pre-silicon boundaries.
- **Ownership:** Single owner for shared CSS and tokens (`dashboard/styles.css` strictly consumes CSS variables defined below).

## 2. Color Tokens

| Role | Light Theme (`[data-theme="light"]`) | Dark Theme (`[data-theme="dark"]`) | Usage |
| :--- | :--- | :--- | :--- |
| `--bg-canvas` | `#F4F6F9` | `#0B111A` | Page background |
| `--bg-surface` | `#FFFFFF` | `#121B28` | Primary analytical panels & tables |
| `--bg-inset` | `#EAEEF4` | `#192434` | Code blocks, secondary sub-panels, controls |
| `--ink-primary` | `#0D1522` | `#EDF2F7` | Headings, table figures, primary labels |
| `--ink-secondary` | `#4A5568` | `#94A3B8` | Supporting explanations, axis ticks |
| `--border-hairline` | `#D5DCE6` | `#243244` | 1px structural dividers & panel borders |
| `--accent` | `#0F52BA` | `#3B82F6` | Single primary focal accent (Mercury-1, active tabs) |
| `--accent-soft` | `#E6EEF9` | `#172A46` | Active pill / subtle highlight fill |
| `--status-pass` | `#047857` | `#10B981` | Passing qualification gates & XOR swizzle |
| `--status-warn` | `#B45309` | `#F59E0B` | `85°C 2x tREFI` warning & DVFS throttling |
| `--status-fail` | `#B91C1C` | `#EF4444` | Physical overflow, Hopper 14b error, `PROCHOT` |

## 3. Typography & Tabular Figures
- **Display & Body:** `"Segoe UI", system-ui, -apple-system, sans-serif` (offline-safe system stack; crisp weight contrast between `600/700` headings and `400` prose).
- **Numeric & Code:** `"Consolas", ui-monospace, monospace` with `font-variant-numeric: tabular-nums` enforced on **every** numeric readout, table cell, and chart axis.
- **Type Scale:**
  - Hero Headline: `clamp(1.75rem, 2.8vw, 2.5rem)` (`line-height: 1.15`, `letter-spacing: -0.02em`)
  - Section Title: `1.25rem` (`600`)
  - Panel Title: `1rem` (`600`)
  - Body Copy: `0.9375rem` (`line-height: 1.55`, max line measure `72ch`)
  - Micro / Table Label: `0.8125rem` (sentence case; never forced all-caps eyebrow clutter)

## 4. Spacing, Radii, Elevation & Motion
- **Spacing Scale:** `4px`, `8px`, `12px`, `16px`, `24px`, `32px`, `48px`.
- **Radii:** `4px` for buttons/badges/inputs, `8px` for panels and media frames (no over-rounded SaaS pills on cards).
- **Elevation:** Flat hairline borders (`1px solid var(--border-hairline)`) only — zero soft gray drop-shadows or glassmorphism blur.
- **Motion:** Single easing `cubic-bezier(0.2, 0.8, 0.2, 1)` at `150ms` for interactive state switches (tabs, SKU toggle, theme switch); zero gratuitous entrance animations; `@media (prefers-reduced-motion: reduce)` disables transitions.

## 5. Rules & Prohibitions
- **Never:** Purple gradients, frosted glass cards, emoji as icons, uniform rows of identical KPI cards as the entire layout, grey text over colored fills, or `->` / `→` appended to buttons.
- **Interactive States:** Every button, tab, and slider must render distinct default, `:hover`, `:focus-visible` (`2px solid var(--accent)` outline), active/selected, and empty/loading fallback states.
- **Added Components:** `.die-budget-card` (proportional N4P die area bar and yield figure), inline `<svg>` telemetry panels (`#thermal-svg` 2-node thermal chart, `#convergence-svg` 20-step PyTorch loss curves), `.grid-two` dual-panel workbench grid, and `.col-hide-sm` mobile table column collapsing without clipping.
- **Responsive Breakpoints:** Verified at desktop (`1440px`) and mobile portrait (`390px`) in both `light` and `dark` themes without horizontal overflow.
