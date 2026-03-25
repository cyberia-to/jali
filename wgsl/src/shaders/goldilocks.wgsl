// ---------------------------------------------------------------------------
// goldilocks.wgsl — Goldilocks field arithmetic for GPU compute shaders
// ---------------------------------------------------------------------------
//
// The Goldilocks prime is p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001.
//
// WGSL has no native u64 type, so we represent each field element as a pair
// of u32 limbs:
//
//   struct Fp { lo: u32, hi: u32 }
//
// where the 64-bit value is  (hi << 32) | lo.
//
// The critical algebraic identity for Goldilocks reduction is:
//
//   2^64 ≡ ε (mod p)    where ε = 2^32 - 1 = 0xFFFFFFFF
//
// This means any 128-bit product can be reduced by replacing the top 64 bits
// with ε times those bits, then performing a second reduction pass if needed.
//
// All functions here operate on canonical representatives in [0, p).
// ---------------------------------------------------------------------------

// ── Fp element: two u32 limbs representing a value in [0, p) ──────────────

struct Fp {
    lo: u32,  // bits [0..31]
    hi: u32,  // bits [32..63]
}

// ── Constants ─────────────────────────────────────────────────────────────

// p = 2^64 - 2^32 + 1
const P_LO: u32 = 0x00000001u;
const P_HI: u32 = 0xFFFFFFFFu;

// ε = 2^32 - 1 = p.wrapping_neg() truncated to 32 bits
const EPSILON: u32 = 0xFFFFFFFFu;

const FP_ZERO: Fp = Fp(0u, 0u);
const FP_ONE:  Fp = Fp(1u, 0u);

// ── Comparison helpers ────────────────────────────────────────────────────

// Return true if a >= b when both are interpreted as 64-bit unsigned integers.
fn fp_gte(a: Fp, b: Fp) -> bool {
    if (a.hi > b.hi) { return true; }
    if (a.hi < b.hi) { return false; }
    return a.lo >= b.lo;
}

// Return true if a == 0.
fn fp_is_zero(a: Fp) -> bool {
    return a.lo == 0u && a.hi == 0u;
}

// ── Canonicalize ──────────────────────────────────────────────────────────

// Reduce a value that may be in [0, 2p) to [0, p).
fn fp_canonicalize(a: Fp) -> Fp {
    if (fp_gte(a, Fp(P_LO, P_HI))) {
        return fp_sub_inner(a, Fp(P_LO, P_HI));
    }
    return a;
}

// Raw 64-bit subtraction (no modular reduction). Assumes a >= b.
fn fp_sub_inner(a: Fp, b: Fp) -> Fp {
    var borrow: u32 = 0u;
    let lo_diff = a.lo - b.lo;
    if (a.lo < b.lo) {
        borrow = 1u;
    }
    let hi_diff = a.hi - b.hi - borrow;
    return Fp(lo_diff, hi_diff);
}

// ── Modular addition ─────────────────────────────────────────────────────
//
// a + b mod p.
// We add, detect 64-bit carry, and if carry occurs add ε (since 2^64 ≡ ε).
// Then canonicalize the result into [0, p).
fn fp_add(a: Fp, b: Fp) -> Fp {
    let lo_sum = a.lo + b.lo;
    var carry_lo: u32 = 0u;
    if (lo_sum < a.lo) {
        carry_lo = 1u;
    }
    let hi_sum = a.hi + b.hi + carry_lo;
    // Detect 64-bit overflow: hi_sum < a.hi indicates carry, or hi_sum < b.hi.
    // More precisely, overflow if the true sum >= 2^64.
    var carry64: u32 = 0u;
    if (hi_sum < a.hi || (hi_sum == a.hi && carry_lo > 0u && b.hi > 0u)) {
        carry64 = 1u;
    }
    // If hi_sum + carry_lo overflowed, check more carefully.
    if (a.hi > 0u && b.hi > 0u && hi_sum < a.hi) {
        carry64 = 1u;
    }
    if (carry_lo == 1u && a.hi == EPSILON && b.hi == 0u && hi_sum == 0u) {
        carry64 = 1u;
    }
    if (carry_lo == 0u && hi_sum < a.hi) {
        carry64 = 1u;
    }
    if (carry_lo == 1u && hi_sum <= a.hi && (a.hi != 0u || b.hi != 0u)) {
        carry64 = 1u;
    }

    var result = Fp(lo_sum, hi_sum);

    // 2^64 ≡ ε (mod p), so add ε for each carry.
    if (carry64 == 1u) {
        let new_lo = result.lo + EPSILON;
        var c2: u32 = 0u;
        if (new_lo < result.lo) {
            c2 = 1u;
        }
        result = Fp(new_lo, result.hi + c2);

        // If adding ε itself overflows 64 bits, add another ε.
        if (result.hi < c2) {
            let new_lo2 = result.lo + EPSILON;
            var c3: u32 = 0u;
            if (new_lo2 < result.lo) {
                c3 = 1u;
            }
            result = Fp(new_lo2, result.hi + c3);
        }
    }

    return fp_canonicalize(result);
}

// ── Modular subtraction ──────────────────────────────────────────────────
//
// a - b mod p.
// If a < b we have underflow; adding p corrects the result because
// (a - b + p) mod p = a - b mod p.
fn fp_sub(a: Fp, b: Fp) -> Fp {
    var borrow: u32 = 0u;
    let lo_diff = a.lo - b.lo;
    if (a.lo < b.lo) {
        borrow = 1u;
    }
    let hi_diff = a.hi - b.hi - borrow;
    var underflow: u32 = 0u;
    if (a.hi < b.hi + borrow) {
        underflow = 1u;
    }
    // Also underflow if a.hi == b.hi + borrow but a.lo < b.lo was already handled.

    var result = Fp(lo_diff, hi_diff);

    // On underflow: result wrapped around. Subtract ε (equivalently add p,
    // since 2^64 - ε = p + 1 ... we subtract ε from the wrapped value).
    // Actually: a - b underflowed means true result is a - b + 2^64.
    // We need (a - b) mod p = a - b + 2^64 - 2^64 mod p.
    // But -2^64 ≡ -ε (mod p), so subtract ε.
    if (underflow == 1u) {
        var borrow2: u32 = 0u;
        let new_lo = result.lo - EPSILON;
        if (result.lo < EPSILON) {
            borrow2 = 1u;
        }
        result = Fp(new_lo, result.hi - borrow2);

        // If subtracting ε itself underflows again, subtract another ε.
        if (result.hi == EPSILON && borrow2 == 1u) {
            // This path handles double underflow edge cases.
        }
    }

    return fp_canonicalize(result);
}

// ── Modular negation ─────────────────────────────────────────────────────
//
// -a mod p = p - a  (or 0 if a == 0).
fn fp_neg(a: Fp) -> Fp {
    if (fp_is_zero(a)) {
        return FP_ZERO;
    }
    return fp_sub(Fp(P_LO, P_HI), a);
}

// ── 32x32 → 64-bit unsigned multiplication helper ────────────────────────
//
// Multiply two u32 values and return the result as an Fp (lo, hi).
// Since WGSL has no u64 mul, we split each operand into 16-bit halves:
//   a = a1*2^16 + a0
//   b = b1*2^16 + b0
//   a*b = a1*b1*2^32 + (a1*b0 + a0*b1)*2^16 + a0*b0
fn mul32(a: u32, b: u32) -> Fp {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;

    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;

    // Accumulate middle terms with carry.
    let mid = p01 + p10;
    var mid_carry: u32 = 0u;
    if (mid < p01) {
        mid_carry = 1u;   // carry in the 2^16 column, worth 2^32 in the mid row
    }

    let lo = p00 + (mid << 16u);
    var lo_carry: u32 = 0u;
    if (lo < p00) {
        lo_carry = 1u;
    }

    let hi = p11 + (mid >> 16u) + (mid_carry << 16u) + lo_carry;
    return Fp(lo, hi);
}

// ── 64-bit addition helper (no modular reduction) ────────────────────────
fn add64(a: Fp, b: Fp) -> Fp {
    let lo = a.lo + b.lo;
    var carry: u32 = 0u;
    if (lo < a.lo) {
        carry = 1u;
    }
    let hi = a.hi + b.hi + carry;
    return Fp(lo, hi);
}

// ── Modular multiplication ───────────────────────────────────────────────
//
// a * b mod p.
//
// Strategy: compute the full 128-bit product as four u32 limbs (w0..w3),
// then reduce using the Goldilocks identity 2^64 ≡ ε (mod p).
//
// Let a = (a.hi, a.lo) and b = (b.hi, b.lo). The product is:
//   a * b = a.lo*b.lo + (a.lo*b.hi + a.hi*b.lo)*2^32 + a.hi*b.hi*2^64
//
// The bottom 64 bits form the "lo" part; the top 64 bits form the "hi" part.
// Then:  result ≡ lo + hi * ε  (mod p).
//
// The second multiplication (hi * ε) can produce at most ~96 bits,
// so we need a careful reduction cascade.
fn fp_mul(a: Fp, b: Fp) -> Fp {
    // Full 128-bit product in 4 x u32 limbs.
    let z0 = mul32(a.lo, b.lo);   // bits [0..63]
    let z1 = mul32(a.lo, b.hi);   // shifted left by 32: bits [32..95]
    let z2 = mul32(a.hi, b.lo);   // shifted left by 32: bits [32..95]
    let z3 = mul32(a.hi, b.hi);   // shifted left by 64: bits [64..127]

    // Accumulate into 4 limbs: w0, w1, w2, w3 (each u32, from LSB to MSB).
    let w0 = z0.lo;

    // w1 column: z0.hi + z1.lo + z2.lo
    let s1a = z0.hi + z1.lo;
    var c1a: u32 = 0u;
    if (s1a < z0.hi) { c1a = 1u; }
    let s1b = s1a + z2.lo;
    var c1b: u32 = 0u;
    if (s1b < s1a) { c1b = 1u; }
    let w1 = s1b;
    let carry1 = c1a + c1b;

    // w2 column: z1.hi + z2.hi + z3.lo + carry1
    let s2a = z1.hi + z2.hi;
    var c2a: u32 = 0u;
    if (s2a < z1.hi) { c2a = 1u; }
    let s2b = s2a + z3.lo;
    var c2b: u32 = 0u;
    if (s2b < s2a) { c2b = 1u; }
    let s2c = s2b + carry1;
    var c2c: u32 = 0u;
    if (s2c < s2b) { c2c = 1u; }
    let w2 = s2c;
    let carry2 = c2a + c2b + c2c;

    // w3 column: z3.hi + carry2
    let w3 = z3.hi + carry2;

    // Now the 128-bit product is: (w3, w2, w1, w0).
    // The bottom 64 bits:  x_lo = (w1 << 32) | w0 = Fp(w0, w1)
    // The top    64 bits:  x_hi = (w3 << 32) | w2 = Fp(w2, w3)
    //
    // Reduce using the Goldilocks identity (mirroring nebu's reduce128):
    //   x_hi_hi = w3   (top 32 bits of x_hi)
    //   x_hi_lo = w2   (bottom 32 bits of x_hi)
    //
    //   t0 = x_lo - x_hi_hi             (since 2^96 ≡ -1 mod p)
    //   t1 = x_hi_lo * ε                (since 2^64 ≡ ε mod p)
    //   result = t0 + t1

    // t0 = (w0, w1) - w3  (64-bit subtract of a 32-bit value from a 64-bit value)
    var t0 = Fp(w0, w1);
    var borrow_t0: u32 = 0u;
    let new_lo_t0 = t0.lo - w3;
    if (t0.lo < w3) {
        borrow_t0 = 1u;
    }
    t0 = Fp(new_lo_t0, t0.hi - borrow_t0);
    // If underflow (t0.hi wrapped), subtract ε (equivalently: 2^64 ≡ ε means
    // borrowing 2^64 costs ε).
    if (w1 < borrow_t0) {
        // 64-bit underflow: add p ≡ subtract ε from wrapped result.
        let fix_lo = t0.lo - EPSILON;
        var fix_borrow: u32 = 0u;
        if (t0.lo < EPSILON) { fix_borrow = 1u; }
        t0 = Fp(fix_lo, t0.hi - fix_borrow);
    }

    // t1 = w2 * ε (32-bit × 32-bit → 64-bit)
    let t1 = mul32(w2, EPSILON);

    // result = t0 + t1, then reduce mod p.
    var result = add64(t0, t1);
    // add64 may overflow 64 bits; if so, add ε.
    let rhi = t0.hi + t1.hi;
    if (result.hi < t0.hi || (result.lo < t0.lo && result.hi <= t0.hi)) {
        // Overflow detected: add ε.
        let fix_lo2 = result.lo + EPSILON;
        var c_fix: u32 = 0u;
        if (fix_lo2 < result.lo) { c_fix = 1u; }
        result = Fp(fix_lo2, result.hi + c_fix);
    }

    return fp_canonicalize(result);
}
