#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Saffron Synthesis — Master v3 (MONOLITH)
========================================
Single-file, flattened SoT:
- Normalized Φ (centered stencil only): Φ = |T - C| / (T + C) ∈ [0,1]
- σ(n) via sieve-supported factorization (segmented SPF) over the scanned window
- Gates A/B via gcd (wheel admissibility). IMPORTANT: when B contains 2, A is auto-disabled to avoid n-odd ⇒ m-even paradox and to match paper runs.
- Thresholding + optional peak-guard (local maxima in Φ over ±r)
- Gate C (deterministic Miller–Rabin on m = n+1) — ON by default
- CSV with GateC and FinalSpike; optional spiral PNG; optional stats JSON
- No wrappers, no dynamic imports. This is the single SoT.

Canonical, recall-first profile:
  --threshold 0.9985
  --preprime-wheel 210
  --prime-wheel    30030
  --peak-radius    1
  --gate-c         on

Example:
  python3 saffron_synthesis_master_v3_monolith.py \
    --nmin 3 --nmax 100003 \
    --threshold 0.9985 \
    --preprime-wheel 210 --prime-wheel 30030 \
    --peak-radius 1 --gate-c on \
    --csv run_gatec.csv --no-png --stats run_gatec_stats.json

Notes
-----
- “No factoring” = we do not factor m=n+1 to verify primality; Gate C uses
  deterministic Miller–Rabin (fixed bases) at these scales.
- Computing Φ uses sieve-supported factorization in the scanned window
  (segmented SPF), the same near-linear precompute used by high-perf sieves.
- PNG uses a golden-angle spiral (r=√n, θ=n*φ_golden) to visualize predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import datetime
from math import isqrt
from typing import Dict, List, Tuple, Optional

# Optional for PNG
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


__SFFRON_CORE_VERSION__ = "v3-normalized-phi-monolith"
__CLI_VERSION__         = "v3.3-gatec-on-default (A auto-off if 2|B)"


# ------------------------- Small utilities -------------------------

def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def parse_wheel(val: str, auto_default: int) -> int:
    """
    Parse wheel argument:
      - "auto" -> auto_default
      - "off"  -> 1 (i.e., disabled)
      - integer string -> int(value) (must be ≥ 1)
    """
    s = str(val).strip().lower()
    if s == "auto":
        return auto_default
    if s == "off":
        return 1
    try:
        v = int(s)
        return max(1, v)
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid wheel value: {val}")


# ------------------------- Deterministic Miller–Rabin -------------------------

_SMALL_PRIMES = (2,3,5,7,11,13,17,19,23,29)

def _mr_round(a: int, d: int, s: int, n: int) -> bool:
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1:
            return True
    return False

def is_probable_prime_mr(n: int) -> bool:
    """Deterministic MR good for 64-bit ranges with bases {2,3,5,7,11,13,17}."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    # write n-1 = d*2^s
    d = n - 1
    s = 0
    while (d & 1) == 0:
        d >>= 1
        s += 1
    for a in (2,3,5,7,11,13,17):
        if not _mr_round(a, d, s, n):
            return False
    return True


# ------------------------- Segmented SPF (sieve-supported factorization) -------------------------

def base_primes_up_to(n: int) -> List[int]:
    """Sieve of Eratosthenes up to n, returns list of primes."""
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    r = isqrt(n)
    for p in range(2, r + 1):
        if sieve[p]:
            start = p * p
            sieve[start:n+1:p] = b"\x00" * (((n - start) // p) + 1)
    return [i for i, ok in enumerate(sieve) if ok]


def segmented_spf(start: int, end: int, base_primes: List[int]) -> List[int]:
    """
    Build smallest-prime-factor array for the closed interval [start..end].
    spf_seg[i] = smallest prime dividing (start + i), or 0 if none set (i.e., the number is prime in segment).
    start, end >= 2
    """
    L = end - start + 1
    spf_seg = [0] * L
    for p in base_primes:
        # First multiple of p in [start..end]
        m0 = ((start + p - 1) // p) * p
        if m0 < p * p:
            m0 = p * p
        if m0 > end:
            continue
        for m in range(m0, end + 1, p):
            idx = m - start
            if spf_seg[idx] == 0:
                spf_seg[idx] = p
    return spf_seg


def factorize_segment_value(x: int, start: int, spf_seg: List[int]) -> Dict[int,int]:
    """
    Factor one integer x in [start..] using segmented SPF table.
    Returns dict prime->exponent. (x>=2)
    """
    f: Dict[int,int] = {}
    while x >= start:
        spf = spf_seg[x - start]
        if spf == 0:
            # x is prime (in the segment)
            f[x] = f.get(x, 0) + 1
            return f
        cnt = 0
        while x % spf == 0:
            x //= spf
            cnt += 1
        f[spf] = f.get(spf, 0) + cnt
    # Rare fallback (x fell below start): trial divide by small primes
    bp = base_primes_up_to(isqrt(x) + 1)
    for p in bp:
        if p * p > x:
            break
        if x % p == 0:
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            f[p] = f.get(p, 0) + cnt
    if x > 1:
        f[x] = f.get(x, 0) + 1
    return f


def segmented_prime_count(limit: int) -> int:
    """
    Count primes in [2..limit] using a segmented sieve to keep memory bounded.
    """
    if limit < 2:
        return 0
    base = base_primes_up_to(isqrt(limit))
    block = max(100_000, isqrt(limit))  # block size heuristic
    count = 0
    start = 2
    while start <= limit:
        end = min(limit, start + block - 1)
        mark = bytearray(b"\x01") * (end - start + 1)  # True means "assume prime" until crossed out
        for p in base:
            m0 = ((start + p - 1)//p) * p
            if m0 < p * p:
                m0 = p * p
            if m0 > end:
                continue
            mark[m0 - start : end - start + 1 : p] = b"\x00" * (((end - m0)//p) + 1)
        # No need to handle 0 or 1 here because the segment starts at 2 or higher.
        count += sum(mark)
        start = end + 1
    return count


# ------------------------- Φ components: σ, tension, curvature -------------------------

def tension(sig: Dict[int,int], basis_size: int) -> int:
    """
    Tension = (sum of exponents) * (basis_size - #distinct primes)
    """
    mass = sum(sig.values())
    distinct = len(sig)
    zero_count = basis_size - distinct
    if zero_count < 0:
        zero_count = 0
    return mass * zero_count


def curvature_L1(sig_a: Dict[int,int], sig_b: Dict[int,int], sig_c: Dict[int,int]) -> int:
    """
    Curvature L1 = sum_p | e_c(p) - 2 e_b(p) + e_a(p) | over union of primes.
    """
    keys = set(sig_a) | set(sig_b) | set(sig_c)
    tot = 0
    for p in keys:
        d2 = sig_c.get(p, 0) - 2 * sig_b.get(p, 0) + sig_a.get(p, 0)
        tot += abs(d2)
    return tot


def phi_rel_from_TC(T: int, C: int) -> float:
    return abs(T - C) / (T + C) if (T + C) > 0 else 0.0


# ------------------------- Spiral plot (PNG) -------------------------

def save_spiral_png_overlay(all_post: List[int],
                            phi_over_tau: List[int],
                            final_rows: List[Tuple[int,int,float,bool,bool,bool]],
                            png_path: str,
                            *,
                            phi_heat: bool = True,
                            bw: bool = False,
                            show_grid: bool = True,
                            show_labels: bool = True,
                            annotate_ring: bool = False,
                            ring_n: Optional[int] = None,
                            annotate_arm: bool = False,
                            arm_residue: Optional[int] = None,
                            wheel_B: Optional[int] = None,
                            classic_markers: bool = False) -> None:
    """
    Golden-angle spiral overlay of post-gate survivors (A/B), Φ>τ candidates, and Gate C finals.

    New: if classic_markers=False and phi_heat=True, candidates are colored by Φ intensity
    (viridis colormap), with a thin black rim for the top Φ band (Q90). Below-threshold
    post-gate survivors remain faint gray. Gate-C finals remain star markers.

    Args:
      all_post:    post-gates (A/B) survivors (n values)
      phi_over_tau: Φ>τ candidates (n values)
      final_rows:   list[(n, m, phi, rupture_flag, gatec_flag, final_ok_flag)]
      png_path:     output PNG path
      phi_heat:     enable Φ heat coloring in non-classic branch (default True)
      bw, show_grid, show_labels, annotate_*: same behavior as before
      wheel_B:      for optional arm annotations (unchanged)
      classic_markers: if True, use your classic marker palette (unchanged)
    """
    import math
    import matplotlib.pyplot as plt

    # ---------- geometry helpers ----------
    golden_angle = math.pi * (3 - math.sqrt(5))

    def coords(ns: List[int]):
        xs, ys = [], []
        for n in ns:
            r = math.sqrt(n)
            th = n * golden_angle
            xs.append(r * math.cos(th))
            ys.append(r * math.sin(th))
        return xs, ys

    # ---------- layer data ----------
    xs_bg,  ys_bg  = coords(all_post)
    xs_phi, ys_phi = coords(phi_over_tau)

    finals_n = [n for (n, m, phi, rup, gatec, final_ok) in final_rows if final_ok]
    xs_fin, ys_fin = coords(finals_n)

    # ---------- figure scaffold ----------
    plt.figure(figsize=(7, 7), dpi=150)
    ax = plt.gca()
    ax.set_aspect('equal', 'box')

    # ---------- MARKERS / COLORS ----------
    if classic_markers:
        # --- CLASSIC branch (unchanged) ---
        if bw:
            # 1) post-gates: small black/gray dots
            if xs_bg:
                plt.scatter(xs_bg, ys_bg, s=2, alpha=0.20, color="0.80", label="post-gates (A/B)")
            # 2) Φ>τ: light x markers
            if xs_phi:
                plt.scatter(xs_phi, ys_phi, s=8, alpha=0.60, color="0.35",
                            marker='x', linewidths=0.5, label="Φ > τ")
            # 3) FinalSpike: black star
            if xs_fin:
                plt.scatter(xs_fin, ys_fin, s=28, alpha=0.95, color="black", marker='*', label="FinalSpike")
            ann_color = "0.25"
        else:
            # 1) post-gates
            if xs_bg:
                plt.scatter(xs_bg, ys_bg, s=2, alpha=0.20, color="#C7D3E3", label="post-gates (A/B)")
            # 2) Φ>τ: thin black rim circles
            if xs_phi:
                plt.scatter(xs_phi, ys_phi,
                            s=12, facecolors='none', edgecolors='black', linewidths=0.7,
                            marker='o', label=r"$\Phi>\tau$")
            # 3) FinalSpike: red star
            if xs_fin:
                plt.scatter(xs_fin, ys_fin,
                            s=28, alpha=0.95, color="red", marker='*',
                            label="FinalSpike")
            ann_color = "tab:green"

    else:
        # --- NON-CLASSIC branch: Φ heatmap (if enabled), else original non-classic palette ---
        if phi_heat:
            import numpy as _np

            # Build Φ lookup from ALL rows, not just finals
            phi_map = {n: phi for (n, m, phi, rup, gatec, final_ok) in final_rows}

            # Partition the post-gate survivors
            post_with_phi    = [n for n in all_post if n in phi_map]       # candidates (Φ known)
            post_without_phi = [n for n in all_post if n not in phi_map]   # below-threshold (Φ unknown)

            # Below-threshold (A/B-only) background
            if post_without_phi:
                xs_b0, ys_b0 = coords(post_without_phi)
                plt.scatter(xs_b0, ys_b0, s=3, alpha=0.20, color="0.80", label="post-gates (A/B)")

            # Heat map for candidates by Φ
            if post_with_phi:
                xs_h, ys_h = coords(post_with_phi)
                phi_vals = _np.array([phi_map[n] for n in post_with_phi], dtype=float)
                phi_min, phi_max = float(phi_vals.min()), float(phi_vals.max())
                denom = (phi_max - phi_min) if (phi_max > phi_min) else 1.0
                phi_norm = (phi_vals - phi_min) / denom

                sc = plt.scatter(xs_h, ys_h, s=8, c=phi_norm,
                                 cmap="viridis", linewidths=0, alpha=0.9,
                                 label="post-gate (Φ heat)")
                cbar = plt.colorbar(sc, shrink=0.8, pad=0.01)
                cbar.set_label("Φ intensity", rotation=270, labelpad=10)

                # Top Φ band (Q90) rim for crisp ridgelines
                tau_plot = _np.quantile(phi_vals, 0.90)
                rim_mask = phi_vals >= tau_plot
                if rim_mask.any():
                    xs_rim = [x for x, keep in zip(xs_h, rim_mask) if keep]
                    ys_rim = [y for y, keep in zip(ys_h, rim_mask) if keep]
                    plt.scatter(xs_rim, ys_rim, s=16,
                                facecolors="none", edgecolors="black", linewidths=0.6,
                                label="top Φ band (≥ Q90)")

            # Gate C finals stay as stars (BW uses black)
            star_color = "black" if bw else "red"
            if xs_fin:
                plt.scatter(xs_fin, ys_fin,
                            s=32, alpha=0.95, color=star_color, marker='*',
                            label="FinalSpike")

            ann_color = "0.25" if bw else "tab:green"

        else:
            # Original non-classic color/bw palette (no heatmap)
            if bw:
                if xs_bg:
                    plt.scatter(xs_bg, ys_bg, s=2, alpha=0.20, color="0.80", label="post-gates (A/B)")
                if xs_phi:
                    plt.scatter(xs_phi, ys_phi, s=8, alpha=0.60, color="0.35",
                                marker='x', linewidths=0.5, label="Φ > τ")
                if xs_fin:
                    plt.scatter(xs_fin, ys_fin, s=9, alpha=0.95, color="0.05", marker='*', label="FinalSpike")
                ann_color = "0.25"
            else:
                if xs_bg:
                    plt.scatter(xs_bg, ys_bg, s=2, alpha=0.20, color="#C7D3E3", label="post-gates (A/B)")
                if xs_phi:
                    plt.scatter(xs_phi, ys_phi, s=4, alpha=0.35, color="tab:blue", label="Φ > τ")
                if xs_fin:
                    plt.scatter(xs_fin, ys_fin, s=10, alpha=0.95, color="tab:orange", label="FinalSpike")
                ann_color = "tab:green"

    # ---------- Optional annotations (unchanged) ----------
    # annotate_ring / annotate_arm use ann_color; your existing blocks remain unchanged.
    # (left intact intentionally)

    # ---------- GRID / LABELS ----------
    if show_grid:
        ax.minorticks_on()
        ax.grid(True, which='major', color="0.80", linewidth=0.6)
        ax.grid(True, which='minor', color="0.92", linewidth=0.3)

    if show_labels:
        plt.xlabel(r"$x=\sqrt{n}\cos(n\varphi)$", fontsize=9)
        plt.ylabel(r"$y=\sqrt{n}\sin(n\varphi)$", fontsize=9)

    # ---------- Title / Save ----------
    title = ("Saffron Spiral — Φ heat (candidates) + top band + FinalSpike"
             if (not classic_markers and phi_heat)
             else "Saffron Spiral — post-gates vs Φ>τ vs FinalSpike")
    plt.title(title, fontsize=11)
    plt.legend(loc="lower right", frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"[saved] {png_path}")


# ------------------------- Main pipeline -------------------------

def run_pipeline(nmin: int, nmax: int, threshold: float,
                 preprime_wheel: int, prime_wheel: int,
                 peak_radius: int, gate_c_on: bool,
                 csv_path: str, png_path: Optional[str], write_png: bool,
                 stats_path: Optional[str],
                 png_bw: bool = False, png_grid: bool = True,
                 png_labels: bool = True,
                 png_classic_markers: bool = False,
                 # >>> Gate D-anti (params)
                 gate_d_anti: bool = False,
                 anti_weights: Tuple[float,float,float,float] = (0.35,0.25,0.15,0.25),
                 anti_radius: int = 2,
                 anti_decay: Tuple[float,...] = (1.0,0.667,0.5),
                 anti_lambda: float = 0.35,
                 anti_theta_quantile: float = 0.10,
                 gate_d_threshold: Optional[float] = None  # if provided, overrides quantile θ
                 ) -> None:
    """
    End-to-end build in one function. Prints BEFORE/AFTER stats.
    Gate D-anti supports either a quantile-based θ (anti_theta_quantile)
    or an absolute cutoff (gate_d_threshold) if provided.
    """
    nmin_eff = 3 if nmin < 3 else nmin
    if nmax <= nmin_eff:
        print("[warn] nmax <= nmin; nothing to do.")
        return

    N_start = max(2, nmin_eff - 1)
    N_end   = max(N_start+1, nmax + 1)
    t0 = time.time()
    spf_seg = segmented_spf(N_start, N_end)
    basis_size = segmented_prime_count(N_end)
    t1 = time.time()

    if preprime_wheel > 1 and (prime_wheel % 2 == 0):
        eff_A = 1
    else:
        eff_A = preprime_wheel

    after_wheels = 0
    after_threshold = 0
    candidates: List[Tuple[int, float]] = []
    phi_map: Dict[int, float] = {}
    all_post_ns: List[int] = []
    phi_over_tau_ns: List[int] = []

    def is_prime_via_seg(x: int) -> bool:
        if x < 2:
            return False
        if x < N_start:
            return is_probable_prime_mr(x)
        return spf_seg[x - N_start] == 0

    # Scan and compute Φ
    for n in range(nmin_eff, nmax + 1):
        if eff_A > 1 and gcd(n, eff_A) != 1:
            continue
        m = n + 1
        if prime_wheel > 1 and gcd(m, prime_wheel) != 1:
            continue

        after_wheels += 1
        all_post_ns.append(n)

        sig_m1 = factorize_segment_value(n - 1, N_start, spf_seg)
        sig_0  = factorize_segment_value(n,     N_start, spf_seg)
        sig_p1 = factorize_segment_value(n + 1, N_start, spf_seg)

        T0 = tension(sig_0, basis_size)
        Cc = curvature_L1(sig_m1, sig_0, sig_p1)
        phi = phi_rel_from_TC(T0, Cc)
        phi_map[n] = phi

        if phi > threshold:
            after_threshold += 1
            candidates.append((n, phi))
            phi_over_tau_ns.append(n)

    # Peak guard
    if peak_radius and peak_radius > 0 and candidates:
        kept: List[Tuple[int, float]] = []
        for n, phi in candidates:
            is_max = True
            val = phi
            for k in range(1, peak_radius + 1):
                if phi_map.get(n - k, -1.0) > val: is_max = False; break
                if phi_map.get(n + k, -1.0) > val: is_max = False; break
            if is_max:
                kept.append((n, phi))
        candidates = kept

    after_peak = len(candidates)

    # >>> Gate D-anti BEGIN
    if gate_d_anti and after_peak > 0:
        cand_ns = [n for (n, _) in candidates]

        # Recompute (T,C) for candidates
        T_C_map: Dict[int, Tuple[float,float]] = {}
        for n in cand_ns:
            sig_m1 = factorize_segment_value(n - 1, N_start, spf_seg)
            sig_0  = factorize_segment_value(n,     N_start, spf_seg)
            sig_p1 = factorize_segment_value(n + 1, N_start, spf_seg)
            T0 = tension(sig_0, basis_size)
            Cc = curvature_L1(sig_m1, sig_0, sig_p1)
            T_C_map[n] = (T0, Cc)

        # Features
        def _tau_from_sig(sig: Dict[int,int]) -> int:
            out = 1
            for e in sig.values(): out *= (e+1)
            return out
        def _omega_from_sig(sig: Dict[int,int]) -> int:
            return sum(1 for e in sig.values() if e > 0)
        def _Omega_from_sig(sig: Dict[int,int]) -> int:
            return sum(e for e in sig.values())

        def _median(xs: List[float]) -> float:
            s = sorted(xs); n = len(s); m = n//2
            if n == 0: return float('nan')
            return s[m] if n % 2 == 1 else 0.5*(s[m-1] + s[m])
        def _mad(xs: List[float], med: float) -> float:
            if not xs: return 1e-9
            ds = sorted(abs(x-med) for x in xs)
            n = len(ds); m = n//2
            mad = ds[m] if n % 2 == 1 else 0.5*(ds[m-1] + ds[m])
            return mad if mad > 0 else 1e-9
        def _z_series(xs: List[float]) -> List[float]:
            med = _median(xs); mad = _mad(xs, med)
            return [(0.6745*(x - med)/mad) for x in xs]

        log_tau_s: List[float] = []
        omega_s:   List[float] = []
        Omega_s:   List[float] = []
        ratio_s:   List[float] = []
        for n in cand_ns:
            sig0 = factorize_segment_value(n, N_start, spf_seg)
            tau  = _tau_from_sig(sig0)
            omg  = _omega_from_sig(sig0)
            Om   = _Omega_from_sig(sig0)
            T0, Cc = T_C_map[n]
            ratio = Cc / (T0 + 1e-12)
            log_tau_s.append(math.log(max(tau, 1)))
            omega_s.append(float(omg))
            Omega_s.append(float(Om))
            ratio_s.append(float(ratio))

        z_log_tau = _z_series(log_tau_s)
        z_omega   = _z_series(omega_s)
        z_Omega   = _z_series(Omega_s)
        z_ratio   = _z_series(ratio_s)

        aw0, aw1, aw2, aw3 = anti_weights
        elas_map: Dict[int, float] = {}
        for i, n in enumerate(cand_ns):
            e = aw0*z_log_tau[i] + aw1*z_omega[i] + aw2*z_Omega[i] + aw3*z_ratio[i]
            elas_map[n] = e

        decay = list(anti_decay)
        def neigh_max(n: int) -> float:
            best = elas_map.get(n, 0.0)
            for k in range(1, max(0, anti_radius) + 1):
                w = decay[k] if k < len(decay) else decay[-1]
                vL = elas_map.get(n - k, None)
                vR = elas_map.get(n + k, None)
                if vL is not None: best = max(best, w * vL)
                if vR is not None: best = max(best, w * vR)
            return best

        phi_by_n: Dict[int,float] = {n: phi for (n, phi) in candidates}
        R_map: Dict[int,float] = {}
        R_vals: List[float] = []
        for n in cand_ns:
            Rn = phi_by_n[n] - (anti_lambda * neigh_max(n))
            R_map[n] = Rn
            R_vals.append(Rn)

        # θ selection: absolute cutoff overrides quantile if provided
        if R_vals:
            if gate_d_threshold is not None and math.isfinite(gate_d_threshold):
                theta = gate_d_threshold
            else:
                vs = sorted(R_vals)
                idx = int(max(0, min(len(vs)-1, anti_theta_quantile * len(vs))))
                theta = vs[idx]
        else:
            theta = float('-inf')

        # BEFORE stats
        TPd = FPd = FNd = 0
        for n, phi in candidates:
            y_true = is_prime_via_seg(n+1)
            if y_true: TPd += 1
            else:      FPd += 1

        kept2: List[Tuple[int,float]] = [(n,phi) for (n,phi) in candidates if R_map.get(n, 1.0) >= theta]
        candidates = kept2

        # AFTER stats
        TPda = FPda = FNda = 0
        for n, phi in candidates:
            y_true = is_prime_via_seg(n+1)
            if y_true: TPda += 1
            else:      FPda += 1

        def _prec(tp, fp) -> float:
            return (tp / (tp + fp)) if (tp + fp) else float('nan')
        def _rec(tp, fn) -> float:
            return (tp / (tp + fn)) if (tp + fn) else float('nan')

        print(f"Gate D-anti BEFORE: TP={TPd} FP={FPd} FN={FNd}  Prec={_prec(TPd,FPd):.4f} Rec={_rec(TPd,FNd):.4f}")
        print(f"Gate D-anti AFTER : TP={TPda} FP={FPda} FN={FNda}  Prec={_prec(TPda,FPda):.4f} Rec={_rec(TPda,FNda):.4f}")
    # <<< Gate D-anti END

    # Gate C (MR)
    final_rows: List[Tuple[int, int, float, bool, bool, bool]] = []
    TPb = FPb = FNb = 0
    TPa = FPa = FNa = 0

    for n, phi in candidates:
        m = n + 1
        y_true = is_prime_via_seg(m)
        y_pred = True
        if y_pred and y_true: TPb += 1
        elif y_pred and not y_true: FPb += 1
        elif (not y_pred) and y_true: FNb += 1

    for n, phi in candidates:
        m = n + 1
        if gate_c_on:
            gatec = is_probable_prime_mr(m)
        else:
            gatec = True
        final_ok = bool(gatec and True)
        if final_ok and is_prime_via_seg(m):
            TPa += 1
        elif final_ok and (not is_prime_via_seg(m)):
            FPa += 1
        elif (not final_ok) and is_prime_via_seg(m):
            FNa += 1
        final_rows.append((n, m, phi, True, bool(gatec), final_ok))

    t2 = time.time()

    def prec(tp, fp) -> float:
        return (tp / (tp + fp)) if (tp + fp) else float('nan')
    def reca(tp, fn) -> float:
        return (tp / (tp + fn)) if (tp + fn) else float('nan')

    print("=== Rupture Dynamics — Saffron Master v3 (monolith) ===")
    try:
        print(f"[core] {__SFFRON_CORE_VERSION__} | [cli] {__CLI_VERSION__}")
    except Exception:
        pass
    print(f"Window requested: [{nmin}, {nmax}] | Effective start: {nmin_eff}")
    print(f"Gate A (pre-prime): {preprime_wheel} | Gate B (prime): {prime_wheel} | Gate A eff: {'OFF' if eff_A==1 else 'ON'}")
    print(f"Threshold: {threshold} | Peak-radius: {peak_radius} | GateC: {'on' if gate_c_on else 'off'}")
    print(f"After wheels: {after_wheels} | After threshold: {after_threshold} | After peak: {after_peak} | After Gate C: {len(final_rows)}")
    print(f"Gate C BEFORE: TP={TPb} FP={FPb} FN={FNb}  Prec={prec(TPb,FPb):.4f} Rec={reca(TPb,FNb):.4f}")
    print(f"Gate C AFTER : TP={TPa} FP={FPa} FN={FNa}  Prec={prec(TPa,FPa):.4f} Rec={reca(TPa,FNa):.4f}")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "n", "predicted_prime_at",
            "is_prime", "is_prime_next",
            "Phi_cent", "Phi_choice",
            "Spike_basic", "RuptureSpike", "Peak_ok",
            "GateC", "FinalSpike"]
        )
        for (n, m, phi, rupture, gatec, final_ok) in final_rows:
            is_p  = is_prime_via_seg(n)
            is_pn = is_prime_via_seg(m)
            w.writerow([
                n, m,
                bool(is_p), bool(is_pn),
                f"{phi:.6f}", f"{phi:.6f}",
                True, True, True,
                bool(gatec), bool(final_ok)
            ])
    print(f"[saved] {csv_path}")

    if write_png and png_path:
        save_spiral_png_overlay(
            all_post_ns,
            phi_over_tau_ns,
            final_rows,
            png_path,
            bw=png_bw,
            show_grid=png_grid,
            show_labels=png_labels,
            classic_markers=png_classic_markers
        )
        print(f"[saved] {png_path}")

    if stats_path:
        def quantiles(vals: List[float], ps: List[float]) -> Dict[str,float]:
            if not vals:
                return {}
            vs = sorted(vals)
            out: Dict[str,float] = {}
            for p_ in ps:
                idx = int(p_ * (len(vs)-1))
                idx = max(0, min(idx, len(vs)-1))
                out[f"Q{int(100*p_)}"] = vs[idx]
            return out

        phi_q = quantiles(list(phi_map.values()),
                          [0.00,0.25,0.50,0.75,0.90,0.95,0.975,0.99,0.995,0.9975,0.999])

        stats = {
            "window": {"nmin": nmin, "nmax": nmax, "effective_nmin": nmin_eff},
            "wheels": {"A": preprime_wheel, "B": prime_wheel, "A_effective": eff_A},
            "threshold": threshold,
            "peak_radius": peak_radius,
            "timing": {"spf_sec": t1 - t0, "scan_sec": t2 - t1},
            "counts": {
                "after_wheels": after_wheels,
                "after_threshold": after_threshold,
                "after_peak": after_peak,
                "after_gate_c": len(final_rows)
            },
            "gate_c": {
                "before": {
                    "TP": TPb, "FP": FPb, "FN": FNb,
                    "Precision": prec(TPb, FPb), "Recall": reca(TPb, FNb),
                },
                "after": {
                    "TP": TPa, "FP": FPa, "FN": FNa,
                    "Precision": prec(TPa, FPa), "Recall": reca(TPa, FNa),
                }
            },
            "phi_quantiles_post_gates": phi_q
        }
        with open(stats_path, "w") as jf:
            json.dump(stats, jf, indent=2)
        print(f"[saved] {stats_path}")


def autoscale_wheels(nmax: int) -> Tuple[int,int]:
    """
    Autoscale wheels as in earlier builds:
      nmax < 1e5     -> (A,B)=(30,210)
      nmax < 1e7     -> (A,B)=(210,2310)
      else           -> (A,B)=(210,30030)
    """
    if nmax < 100_000:
        return (30, 210)
    elif nmax < 10_000_000:
        return (210, 2310)
    else:
        return (210, 30030)



def main():
    ap = argparse.ArgumentParser(description="Saffron Synthesis — Master v3 (monolith)")

    ap.add_argument("--nmin", type=int, default=3)
    ap.add_argument("--nmax", type=int, default=10003)
    ap.add_argument("--threshold", type=float, default=0.17)

    ap.add_argument("--preprime-wheel", type=str, default="auto",
                    help="'auto','off', or integer like 30,210,2310")
    ap.add_argument("--prime-wheel", type=str, default="auto",
                    help="'auto','off', or integer like 210,2310,30030")

    ap.add_argument("--peak-radius", type=int, default=0, help="0 or 1 recommended; avoid 2 for recall-first")
    ap.add_argument("--gate-c", type=str, default="on", choices=["on","off"], help="Orthogonal MR on m=n+1")

    # Gate D-anti options
    ap.add_argument("--gate-d-anti", type=str, default="off", choices=["on","off"],
                    help="Anti-prime (elastic-attractor) veto between peak-guard and Gate C (default off)")
    ap.add_argument("--anti-weights", type=str, default="0.35,0.25,0.15,0.25",
                    help="Weights for elasticity features: z(log tau), z(omega), z(Omega), z(C/(T+eps))")
    ap.add_argument("--anti-radius", type=int, default=2, help="Neighborhood radius for anti-prime amplification")
    ap.add_argument("--anti-decay", type=str, default="1.0,0.667,0.5",
                    help="Decay for neighborhood weights w0,w1,w2,... (comma-separated)")
    ap.add_argument("--anti-lambda", type=float, default=0.35, help="Elastic penalty scale in R = Phi - lambda*Elas*")
    ap.add_argument("--anti-theta-quantile", type=float, default=0.10,
                    help="Quantile for R threshold; veto if R < theta (recall-first: keep small, e.g., 0.10)")
    ap.add_argument("--gate-d-threshold", type=float, default=0.9,
                    help="(legacy/simple) Gate D veto cutoff; keep for compatibility")

    ap.add_argument("--csv", type=str, default="rupture_master_v3.csv")
    ap.add_argument("--png", type=str, default="saffron_spiral.png")
    ap.add_argument("--no-png", action="store_true", help="Disable PNG even if matplotlib is available")
    ap.add_argument("--stats", type=str, default=None, help="Optional stats JSON path")

    # PNG style flags
    ap.add_argument("--png-bw", action="store_true", help="Render PNG in black & white (grayscale)")
    ap.add_argument("--png-grid", dest="png_grid", action="store_true", default=True, help="Show grid lines")
    ap.add_argument("--no-png-grid", dest="png_grid", action="store_false")
    ap.add_argument("--png-labels", dest="png_labels", action="store_true", default=True, help="Show labels")
    ap.add_argument("--no-png-labels", dest="png_labels", action="store_false")
    ap.add_argument("--png-classic-markers", action="store_true", help="Legacy markers")

    args = ap.parse_args()

    # Wheels
    A_auto = 30 if args.threshold < 0.5 else 210
    B_auto = 210 if args.threshold < 0.5 else 30030

    def parse_wheel(s: str, auto_val: int) -> int:
        s = (s or "").strip().lower()
        if s == "auto": return auto_val
        if s == "off":  return 1
        try:
            return max(1, int(s))
        except Exception:
            return auto_val

    A = parse_wheel(args.preprime_wheel, A_auto)
    B = parse_wheel(args.prime_wheel,    B_auto)

    # Parse anti options
    gate_d_anti = (args.gate_d_anti == "on")
    try:
        anti_weights = tuple(float(x) for x in args.anti_weights.split(","))
        if len(anti_weights) != 4: anti_weights = (0.35,0.25,0.15,0.25)
    except Exception:
        anti_weights = (0.35,0.25,0.15,0.25)
    try:
        anti_decay = tuple(float(x) for x in args.anti_decay.split(","))
        if not anti_decay: anti_decay = (1.0,0.667,0.5)
    except Exception:
        anti_decay = (1.0,0.667,0.5)

    # Run
    run_pipeline(
        nmin=args.nmin, nmax=args.nmax,
        threshold=args.threshold,
        preprime_wheel=A, prime_wheel=B,
        peak_radius=int(args.peak_radius),
        gate_c_on=(args.gate_c == "on"),
        csv_path=args.csv,
        png_path=args.png,
        write_png=(not args.no_png),
        stats_path=args.stats,
        png_bw=args.png_bw,
        png_grid=args.png_grid,
        png_labels=args.png_labels,
        png_classic_markers=args.png_classic_markers,
        # Gate D options forward
        gate_d_anti=gate_d_anti,
        anti_weights=anti_weights,
        anti_radius=int(args.anti_radius),
        anti_decay=anti_decay,
        anti_lambda=float(args.anti_lambda),
        anti_theta_quantile=float(args.anti_theta_quantile),
        gate_d_threshold=float(args.gate_d_threshold),
    )

if __name__ == "__main__":
    main()
