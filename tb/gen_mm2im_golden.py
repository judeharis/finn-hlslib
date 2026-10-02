# Jude: Created
"""Goldens and a bit-true Python model for the MM2IM HLS kernel (mm2im.hpp).

Write a test configuration (golden.h for tb/mm2im_tb.cpp and tb/mm2im_top.cpp):

    python3 gen_mm2im_golden.py --cfg K S P H W CI CO PE SIMD [--idt UINT4] [--wdt INT8]
                                [--seed 0] [--out DIR]

Self-test (no files written): on random configurations, the numpy reference must equal
PyTorch ConvTranspose2d (float64, exact for these ranges), and the kernel model, which
replays mm2im.hpp's loop order, accumulator banks and drain order, must equal the
reference with its iteration count equal to the closed form in mm2im_sched.hpp:

    python3 gen_mm2im_golden.py --selftest [N]

Layouts: input NHWC, weights W[CO][K][K][CI] (FINN's Deconvolution layout, no kernel
flip), output NHWC cropped by P on every edge.
"""
import argparse
import os
import sys

import numpy as np

DTYPES = {
    "UINT2": (0, 3, "ap_uint<2>"),
    "UINT4": (0, 15, "ap_uint<4>"),
    "UINT8": (0, 255, "ap_uint<8>"),
    "INT2": (-2, 1, "ap_int<2>"),
    "INT4": (-8, 7, "ap_int<4>"),
    "INT8": (-128, 127, "ap_int<8>"),
}


# ---------------------------------------------------------------------------------------
# geometry, mirroring mm2im_sched.hpp
# ---------------------------------------------------------------------------------------
class Geom:
    def __init__(self, K, S, P, H, W):
        assert K > 0 and S > 0 and H > 0 and W > 0 and P < K
        self.K, self.S, self.P, self.H, self.W = K, S, P, H, W
        self.HF, self.WF = (H - 1) * S + K, (W - 1) * S + K
        assert 2 * P < self.HF and 2 * P < self.WF
        self.HO, self.WO = self.HF - 2 * P, self.WF - 2 * P
        self.KB = (K - 1) // S + 1
        self.BANDS = H + self.KB - 1

    def tap_lo(self, i):
        return 0 if i * self.S >= self.P else self.P - i * self.S

    def tap_hi(self, i, O):
        if self.P + O <= i * self.S:
            return 0
        return min(self.K, self.P + O - i * self.S)

    def row_lo(self, b):
        return 0 if b * self.S >= self.P else min(self.S, self.P - b * self.S)

    def row_hi(self, b):
        if self.P + self.HO <= b * self.S:
            return 0
        return min(self.S, self.P + self.HO - b * self.S)

    def rows_kept(self, b):
        return max(0, self.row_hi(b) - self.row_lo(b))

    def taps_x(self, skip):
        return sum((self.tap_hi(i, self.WO) - self.tap_lo(i)) if skip else self.K for i in range(self.W))

    def taps_y(self, i, skip):
        return (self.tap_hi(i, self.HO) - self.tap_lo(i)) if skip else self.K

    def iterations(self, CF, SF, skip):
        n = sum(self.taps_y(i, skip) * self.taps_x(skip) * CF * SF for i in range(self.H))
        n += sum(self.rows_kept(b) * self.WO * CF for b in range(self.BANDS))
        return n


# ---------------------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------------------
def convtranspose_ref(x, w, S, P):
    """Scatter definition. x [H][W][CI], w [CO][K][K][CI] -> int64 [HO][WO][CO]."""
    x = np.asarray(x, dtype=np.int64)
    w = np.asarray(w, dtype=np.int64)
    H, W, _ = x.shape
    CO, K, _, _ = w.shape
    full = np.zeros(((H - 1) * S + K, (W - 1) * S + K, CO), dtype=np.int64)
    for ky in range(K):
        for kx in range(K):
            full[ky:ky + (H - 1) * S + 1:S, kx:kx + (W - 1) * S + 1:S, :] += np.einsum(
                "hwi,oi->hwo", x, w[:, ky, kx, :]
            )
    HO, WO = full.shape[0] - 2 * P, full.shape[1] - 2 * P
    return full[P:P + HO, P:P + WO, :]


def torch_ref(x, w, S, P):
    import torch

    CO, K, _, CI = w.shape
    layer = torch.nn.ConvTranspose2d(CI, CO, K, stride=S, padding=P, bias=False).double()
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(np.asarray(w, dtype=np.float64).transpose(3, 0, 1, 2)))
        y = layer(torch.from_numpy(np.asarray(x, dtype=np.float64).transpose(2, 0, 1)[None]))
    y = y[0].numpy().transpose(1, 2, 0)
    assert np.array_equal(y, np.round(y)), "torch output is not integer-valued"
    return np.round(y).astype(np.int64)


def rom_order(w, PE, SIMD):
    """W[CO][K][K][CI] -> kernel[t][PE][SIMD], t = ((ky*K + kx)*CF + cf)*SF + sf."""
    CO, K, _, CI = w.shape
    CF, SF = CO // PE, CI // SIMD
    return np.asarray(w).reshape(CF, PE, K, K, SF, SIMD).transpose(2, 3, 0, 4, 1, 5).reshape(-1, PE, SIMD)


def kernel_model(x, rom, K, S, P, CO, CI, PE, SIMD, skip):
    """Replay mm2im.hpp: returns (output words [n][PE] in stream order, iterations)."""
    H, W, _ = x.shape
    g = Geom(K, S, P, H, W)
    CF, SF, NB = CO // PE, CI // SIMD, g.KB
    xin = np.asarray(x, dtype=np.int64).reshape(H, W, SF, SIMD)
    rom = np.asarray(rom, dtype=np.int64)
    acc = np.zeros((NB, S, g.WO, CF, PE), dtype=np.int64)
    out, its = [], 0

    def drain(bank, b):
        nonlocal its
        for rr in range(g.row_lo(b), g.row_hi(b)):
            for ow in range(g.WO):
                for cf in range(CF):
                    out.append(acc[bank, rr, ow, cf].copy())
                    acc[bank, rr, ow, cf] = 0
                    its += 1

    bank0 = 0
    for ih in range(H):
        ky_lo, ky_hi = (g.tap_lo(ih), g.tap_hi(ih, g.HO)) if skip else (0, K)
        for iw in range(W):
            kx_lo, kx_hi = (g.tap_lo(iw), g.tap_hi(iw, g.WO)) if skip else (0, K)
            for ky in range(ky_lo, ky_hi):
                for kx in range(kx_lo, kx_hi):
                    for cf in range(CF):
                        psum = np.zeros(PE, dtype=np.int64)
                        for sf in range(SF):
                            t = ((ky * K + kx) * CF + cf) * SF + sf
                            psum += rom[t] @ xin[ih, iw, sf]
                            its += 1
                        r, c = ih * S + ky, iw * S + kx
                        if skip or (P <= r < P + g.HO and P <= c < P + g.WO):
                            bank = (bank0 + ky // S) % NB
                            acc[bank, ky % S, c - P, cf] += psum
        drain(bank0, ih)
        bank0 = (bank0 + 1) % NB
    for b in range(H, g.BANDS):
        drain(bank0, b)
        bank0 = (bank0 + 1) % NB
    return np.array(out, dtype=np.int64).reshape(-1, PE), its


# ---------------------------------------------------------------------------------------
# golden.h
# ---------------------------------------------------------------------------------------
def c_array(a):
    return ", ".join(str(int(v)) for v in np.asarray(a).ravel())


def c_nested(rom):
    rows = []
    for t in rom:
        rows.append("{" + ", ".join("{" + ", ".join(str(int(v)) for v in pe) + "}" for pe in t) + "}")
    return ",\n\t".join(rows)


def write_golden(out, K, S, P, H, W, CI, CO, PE, SIMD, idt, wdt, seed):
    rng = np.random.default_rng(seed)
    ilo, ihi, ityp = DTYPES[idt]
    wlo, whi, wtyp = DTYPES[wdt]
    x = rng.integers(ilo, ihi + 1, size=(H, W, CI))
    w = rng.integers(wlo, whi + 1, size=(CO, K, K, CI))
    y = convtranspose_ref(x, w, S, P)
    assert np.array_equal(y, torch_ref(x, w, S, P)), "numpy and torch references disagree"
    for skip in (True, False):
        words, _ = kernel_model(x, rom_order(w, PE, SIMD), K, S, P, CO, CI, PE, SIMD, skip)
        assert np.array_equal(words.reshape(y.shape), y), "kernel model disagrees with the reference"
    g = Geom(K, S, P, H, W)
    rom = rom_order(w, PE, SIMD)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "golden.h"), "w") as f:
        f.write(f"""// generated by tb/gen_mm2im_golden.py -- do not edit
// --cfg {K} {S} {P} {H} {W} {CI} {CO} {PE} {SIMD} --idt {idt} --wdt {wdt} --seed {seed}
#ifndef MM2IM_GOLDEN_H
#define MM2IM_GOLDEN_H

#include <ap_int.h>

constexpr unsigned  K = {K};
constexpr unsigned  S = {S};
constexpr unsigned  P = {P};
constexpr unsigned  H = {H};
constexpr unsigned  W = {W};
constexpr unsigned  CI = {CI};
constexpr unsigned  CO = {CO};
constexpr unsigned  PE = {PE};
constexpr unsigned  SIMD = {SIMD};
constexpr unsigned  HO = {g.HO};
constexpr unsigned  WO = {g.WO};
constexpr unsigned long long  ITER_SKIP = {g.iterations(CO // PE, CI // SIMD, True)}ULL;
constexpr unsigned long long  ITER_NOSKIP = {g.iterations(CO // PE, CI // SIMD, False)}ULL;

using  TI = {ityp};
using  TW = {wtyp};
using  TO = ap_int<32>;

static TW const  KERNEL[{len(rom)}][{PE}][{SIMD}] = {{
\t{c_nested(rom)}
}};

static int const  INPUT[{H * W * CI}] = {{ {c_array(x)} }};	// NHWC
static long long const  EXPECTED[{g.HO * g.WO * CO}] = {{ {c_array(y)} }};	// NHWC

#endif
""")
    return y


# ---------------------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------------------
def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def selftest(n, seed=1):
    rng = np.random.default_rng(seed)
    try:
        import torch  # noqa: F401

        have_torch = True
    except ImportError:
        have_torch = False
        print("torch not available: skipping the torch cross-check")
    done = kS = KltS = 0
    while done < n:
        K = int(rng.integers(1, 7))
        S = int(rng.integers(1, 5))
        P = int(rng.integers(0, K))
        H, W = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        if 2 * P >= (H - 1) * S + K or 2 * P >= (W - 1) * S + K:
            continue
        CI, CO = int(rng.choice([1, 2, 3, 4, 6, 8])), int(rng.choice([1, 2, 3, 4, 6]))
        PE, SIMD = int(rng.choice(divisors(CO))), int(rng.choice(divisors(CI)))
        idt, wdt = rng.choice(list(DTYPES)), rng.choice(["INT4", "INT8", "INT2"])
        ilo, ihi, _ = DTYPES[idt]
        wlo, whi, _ = DTYPES[wdt]
        x = rng.integers(ilo, ihi + 1, size=(H, W, CI))
        w = rng.integers(wlo, whi + 1, size=(CO, K, K, CI))
        y = convtranspose_ref(x, w, S, P)
        if have_torch:
            assert np.array_equal(y, torch_ref(x, w, S, P)), ("torch", K, S, P, H, W)
        g = Geom(K, S, P, H, W)
        for skip in (True, False):
            words, its = kernel_model(x, rom_order(w, PE, SIMD), K, S, P, CO, CI, PE, SIMD, skip)
            assert np.array_equal(words.reshape(y.shape), y), ("model", K, S, P, H, W, PE, SIMD, skip)
            assert its == g.iterations(CO // PE, CI // SIMD, skip), ("iterations", K, S, P, H, W, skip)
        done += 1
        kS += K % S != 0
        KltS += K < S
    print(f"selftest: {n} random configs OK ({kS} with K%S!=0, {KltS} with K<S); "
          f"numpy == {'torch' if have_torch else '(torch skipped)'}, model == reference "
          f"(SKIP on and off), iterations == closed form")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", nargs=9, type=int, metavar=("K", "S", "P", "H", "W", "CI", "CO", "PE", "SIMD"))
    ap.add_argument("--idt", default="UINT4", choices=list(DTYPES))
    ap.add_argument("--wdt", default="INT8", choices=list(DTYPES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--selftest", nargs="?", const=200, type=int)
    a = ap.parse_args()
    if a.selftest:
        selftest(a.selftest)
        return 0
    if not a.cfg:
        ap.error("--cfg or --selftest required")
    K, S, P, H, W, CI, CO, PE, SIMD = a.cfg
    assert CO % PE == 0 and CI % SIMD == 0, "PE must divide CO and SIMD must divide CI"
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mm2im",
                                f"k{K}s{S}p{P}_h{H}w{W}_ci{CI}co{CO}_pe{PE}simd{SIMD}_{a.idt}x{a.wdt}")
    write_golden(out, K, S, P, H, W, CI, CO, PE, SIMD, a.idt, a.wdt, a.seed)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
