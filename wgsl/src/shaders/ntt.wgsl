// ---------------------------------------------------------------------------
// ntt.wgsl — Negacyclic NTT butterfly stages on GPU
// ---------------------------------------------------------------------------
//
// This shader implements the Number Theoretic Transform for the ring
// R_q = F_p[x]/(x^n+1) over the Goldilocks prime field.
//
// === GPU NTT dispatch model ===
//
// The NTT is decomposed into log2(n) butterfly stages, each dispatched as
// a separate compute shader invocation. The host (Rust) code calls:
//
//   for stage in 0..log2(n) {
//       set params.stage = stage;
//       dispatch(ceil(n/2 / 256));     // n/2 butterfly pairs per stage
//   }
//
// Each thread handles one butterfly pair. For n=1024 and workgroup_size=256,
// that is 2 workgroups per stage, 10 stages total — 20 dispatches.
//
// This stage-by-stage model avoids the need for cross-workgroup barriers
// (which WGSL does not support). The host synchronizes between stages via
// successive dispatches.
//
// === Memory layout ===
//
// Ring elements are stored as contiguous arrays of Fp structs in a storage
// buffer. Each Fp occupies 8 bytes (2 x u32). A polynomial of degree n has
// n Fp entries = 8n bytes.
//
// The twiddle factors (precomputed roots of unity) are stored in a separate
// read-only storage buffer. For forward NTT stage s with butterfly size
// m = 2^(s+1), there are m/2 twiddle factors. The total across all stages
// is n-1 factors, laid out sequentially: stage 0 factors, then stage 1, etc.
//
// === Forward vs inverse ===
//
// Forward NTT (direction == 0):
//   Cooley-Tukey decimation-in-time (DIT). Data is bit-reverse permuted
//   before the first stage (done by a separate kernel or on the host).
//   Butterfly:  t = w * data[j]
//               data[j] = data[i] - t
//               data[i] = data[i] + t
//
// Inverse NTT (direction == 1):
//   Gentleman-Sande decimation-in-frequency (DIF). Stages run in reverse
//   order. Bit-reverse permutation is applied after the last stage.
//   Butterfly:  u = data[i]
//               v = data[j]
//               data[i] = u + v
//               data[j] = w * (u - v)
//   The N^(-1) scaling factor is applied in a separate pass or folded
//   into the final untwist.
//
// ---------------------------------------------------------------------------

// Include Goldilocks field arithmetic (Fp struct, fp_add, fp_sub, fp_mul, etc.)
// In the host pipeline, goldilocks.wgsl is prepended to this source before
// shader compilation:
//   let source = format!("{}\n{}", GOLDILOCKS, NTT);

// ── Bindings ──────────────────────────────────────────────────────────────

// Polynomial coefficients (read/write). Layout: n consecutive Fp values.
@group(0) @binding(0) var<storage, read_write> data: array<Fp>;

// Precomputed twiddle factors (read-only). Layout: n-1 consecutive Fp values,
// arranged stage-by-stage. For stage s, the twiddle offset is sum_{k=0}^{s-1} 2^k.
@group(0) @binding(1) var<storage, read> twiddles: array<Fp>;

// NTT parameters passed as a uniform buffer.
struct NttParams {
    n: u32,           // polynomial degree (must be power of 2)
    log_n: u32,       // log2(n)
    stage: u32,       // current butterfly stage (0..log_n-1)
    direction: u32,   // 0 = forward (DIT), 1 = inverse (DIF)
}
@group(0) @binding(2) var<uniform> params: NttParams;

// ── Twiddle offset helper ─────────────────────────────────────────────────
//
// For stage s, the twiddle factors begin at index 2^s - 1 in the twiddles
// array. (Stage 0 has 1 factor starting at offset 0; stage 1 has 2 factors
// starting at offset 1; stage 2 has 4 factors starting at offset 3; etc.)
fn twiddle_offset(stage: u32) -> u32 {
    // offset = 2^stage - 1
    return (1u << stage) - 1u;
}

// ── Forward NTT butterfly (Cooley-Tukey DIT) ──────────────────────────────
//
// For stage s with butterfly size m = 2^(s+1), half_m = 2^s:
//   - Thread k handles the butterfly pair at positions (i, j) where
//     i = (k / half_m) * m + (k % half_m)
//     j = i + half_m
//   - Twiddle factor: twiddles[twiddle_offset(s) + (k % half_m)]
//
// The butterfly operation is:
//   t = w * data[j]
//   data[j] = data[i] - t
//   data[i] = data[i] + t
fn forward_butterfly(thread_id: u32) {
    let s = params.stage;
    let half_m = 1u << s;         // 2^s
    let m = half_m << 1u;         // 2^(s+1)

    let k = thread_id;            // butterfly index within this stage
    let group = k / half_m;       // which butterfly group
    let pos = k % half_m;         // position within group

    let i = group * m + pos;
    let j = i + half_m;

    // Bounds check: each stage has n/2 butterflies.
    if (j >= params.n) {
        return;
    }

    let w = twiddles[twiddle_offset(s) + pos];

    let a = data[i];
    let t = fp_mul(data[j], w);

    data[i] = fp_add(a, t);
    data[j] = fp_sub(a, t);
}

// ── Inverse NTT butterfly (Gentleman-Sande DIF) ──────────────────────────
//
// For the inverse NTT, stages run from log_n-1 down to 0. The host sets
// params.stage to the appropriate value for each dispatch.
//
// For stage s:
//   u = data[i]
//   v = data[j]
//   data[i] = u + v
//   data[j] = w * (u - v)
//
// The twiddle factors for the inverse NTT are the inverses of the forward
// twiddles. The host can either precompute inverse twiddles or the same
// table can be used if omega^(-1) is supplied.
fn inverse_butterfly(thread_id: u32) {
    let s = params.stage;
    let half_m = 1u << s;
    let m = half_m << 1u;

    let k = thread_id;
    let group = k / half_m;
    let pos = k % half_m;

    let i = group * m + pos;
    let j = i + half_m;

    if (j >= params.n) {
        return;
    }

    let w = twiddles[twiddle_offset(s) + pos];

    let u = data[i];
    let v = data[j];

    data[i] = fp_add(u, v);
    data[j] = fp_mul(w, fp_sub(u, v));
}

// ── Main entry point ──────────────────────────────────────────────────────
//
// 256 threads per workgroup. The host dispatches ceil(n/2 / 256) workgroups.
// Each thread computes one butterfly pair for the current stage.

@compute @workgroup_size(256)
fn ntt_butterfly(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_id = gid.x;

    // n/2 butterflies per stage.
    let num_butterflies = params.n >> 1u;
    if (thread_id >= num_butterflies) {
        return;
    }

    if (params.direction == 0u) {
        forward_butterfly(thread_id);
    } else {
        inverse_butterfly(thread_id);
    }
}

// ── Bit-reverse permutation kernel ────────────────────────────────────────
//
// A separate entry point for the bit-reverse permutation step required by
// the Cooley-Tukey forward NTT (applied before butterflies) and the
// Gentleman-Sande inverse NTT (applied after butterflies).
//
// Each thread swaps data[i] with data[bit_reverse(i)] if i < bit_reverse(i).

fn bit_reverse_u32(val: u32, bits: u32) -> u32 {
    var x = val;
    var result = 0u;
    for (var b = 0u; b < bits; b = b + 1u) {
        result = result | ((x & 1u) << (bits - 1u - b));
        x = x >> 1u;
    }
    return result;
}

@compute @workgroup_size(256)
fn bit_reverse_permute(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) {
        return;
    }

    let j = bit_reverse_u32(i, params.log_n);
    if (i < j) {
        let tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}

// ── N-inverse scaling kernel ──────────────────────────────────────────────
//
// After inverse NTT, multiply every element by N^(-1) mod p.
// The host precomputes n_inv and passes it as twiddles[0] (or a dedicated
// uniform). Here we use twiddles[0] as the scaling factor.

@compute @workgroup_size(256)
fn scale_by_n_inv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) {
        return;
    }

    let n_inv = twiddles[0];
    data[i] = fp_mul(data[i], n_inv);
}
