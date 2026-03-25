// ---
// tags: jali, rust
// crystal-type: source
// crystal-domain: comp
// ---
//! jali CLI — polynomial ring arithmetic tool.

use nebu::Goldilocks;
use jali::ring::RingElement;
use jali::ntt;
use jali::sample;
use jali::encoding;

fn print_usage() {
    eprintln!("jali — polynomial ring arithmetic R_q = F_p[x]/(x^n+1)");
    eprintln!();
    eprintln!("usage:");
    eprintln!("  jali calc add <n> <a0,a1,...> <b0,b1,...>  — add two polynomials");
    eprintln!("  jali calc mul <n> <a0,a1,...> <b0,b1,...>  — multiply two polynomials");
    eprintln!("  jali ntt forward <n> <a0,a1,...>           — forward NTT");
    eprintln!("  jali ntt inverse <n> <a0,a1,...>           — inverse NTT");
    eprintln!("  jali sample uniform <seed> <n>             — sample uniform polynomial");
    eprintln!("  jali sample ternary <seed> <n>             — sample ternary polynomial");
    eprintln!("  jali sample cbd <seed> <n> <eta>           — sample CBD polynomial");
    eprintln!("  jali bench <n> <iters>                     — benchmark ring ops");
    eprintln!("  jali help                                  — show this message");
}

fn parse_poly(s: &str, n: usize) -> RingElement {
    let mut elem = RingElement::new(n);
    for (i, tok) in s.split(',').enumerate() {
        if i >= n { break; }
        let v: u64 = tok.trim().parse().unwrap_or(0);
        elem.coeffs[i] = Goldilocks::new(v);
    }
    elem
}

fn print_poly(elem: &RingElement) {
    let n = elem.n;
    let mut first = true;
    for i in 0..n {
        let v = elem.coeffs[i].as_u64();
        if !first { print!(","); }
        print!("{}", v);
        first = false;
    }
    println!();
}

fn cmd_calc(args: &[String]) {
    if args.len() < 4 {
        eprintln!("error: calc requires: <add|mul> <n> <poly_a> <poly_b>");
        return;
    }
    let op = args[0].as_str();
    let n: usize = args[1].parse().unwrap_or(0);
    if !n.is_power_of_two() || n > 4096 {
        eprintln!("error: n must be a power of 2 <= 4096");
        return;
    }
    let a = parse_poly(&args[2], n);
    let b = parse_poly(&args[3], n);
    let result = match op {
        "add" => a.add(&b),
        "mul" => a.mul(&b),
        _ => { eprintln!("error: unknown calc op '{}'", op); return; }
    };
    print_poly(&result);
}

fn cmd_ntt(args: &[String]) {
    if args.len() < 3 {
        eprintln!("error: ntt requires: <forward|inverse> <n> <coeffs>");
        return;
    }
    let direction = args[0].as_str();
    let n: usize = args[1].parse().unwrap_or(0);
    if !n.is_power_of_two() || n > 4096 {
        eprintln!("error: n must be a power of 2 <= 4096");
        return;
    }
    let mut elem = parse_poly(&args[2], n);
    match direction {
        "forward" => {
            ntt::to_ntt(&mut elem);
            print_poly(&elem);
        }
        "inverse" => {
            elem.is_ntt = true;
            ntt::from_ntt(&mut elem);
            print_poly(&elem);
        }
        _ => { eprintln!("error: unknown ntt direction '{}'", direction); }
    }
}

fn cmd_sample(args: &[String]) {
    if args.len() < 3 {
        eprintln!("error: sample requires: <uniform|ternary|cbd> <seed> <n> [eta]");
        return;
    }
    let kind = args[0].as_str();
    let seed: u64 = args[1].parse().unwrap_or(0);
    let n: usize = args[2].parse().unwrap_or(0);
    if !n.is_power_of_two() || n > 4096 {
        eprintln!("error: n must be a power of 2 <= 4096");
        return;
    }
    let elem = match kind {
        "uniform" => sample::sample_uniform(seed, n),
        "ternary" => sample::sample_ternary(seed, n),
        "cbd" => {
            if args.len() < 4 {
                eprintln!("error: cbd requires eta parameter");
                return;
            }
            let eta: usize = args[3].parse().unwrap_or(2);
            sample::sample_cbd(seed, n, eta)
        }
        _ => { eprintln!("error: unknown sample kind '{}'", kind); return; }
    };
    print_poly(&elem);
}

fn cmd_bench(args: &[String]) {
    let n: usize = if args.is_empty() { 1024 } else { args[0].parse().unwrap_or(1024) };
    let iters: u64 = if args.len() < 2 { 1000 } else { args[1].parse().unwrap_or(1000) };

    if !n.is_power_of_two() || n > 4096 {
        eprintln!("error: n must be a power of 2 <= 4096");
        return;
    }

    let a = sample::sample_uniform(1, n);
    let b = sample::sample_uniform(2, n);

    // Benchmark add
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = a.add(&b);
    }
    let elapsed = start.elapsed();
    eprintln!("ring_add  n={}: {:.1} us/op ({} iters)",
        n, elapsed.as_micros() as f64 / iters as f64, iters);

    // Benchmark mul
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = a.mul(&b);
    }
    let elapsed = start.elapsed();
    eprintln!("ring_mul  n={}: {:.1} us/op ({} iters)",
        n, elapsed.as_micros() as f64 / iters as f64, iters);

    // Benchmark NTT forward
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let mut c = a.clone();
        ntt::to_ntt(&mut c);
    }
    let elapsed = start.elapsed();
    eprintln!("ntt_fwd   n={}: {:.1} us/op ({} iters)",
        n, elapsed.as_micros() as f64 / iters as f64, iters);

    // Benchmark encoding roundtrip
    let mut buf = vec![0u8; n * 8];
    let start = std::time::Instant::now();
    for _ in 0..iters {
        encoding::encode_ring(&a, &mut buf);
        let _ = encoding::decode_ring(&buf, n);
    }
    let elapsed = start.elapsed();
    eprintln!("enc_rt    n={}: {:.1} us/op ({} iters)",
        n, elapsed.as_micros() as f64 / iters as f64, iters);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "calc" => cmd_calc(&args[2..]),
        "ntt" => cmd_ntt(&args[2..]),
        "sample" => cmd_sample(&args[2..]),
        "bench" => cmd_bench(&args[2..]),
        "help" | "--help" | "-h" => print_usage(),
        _ => {
            eprintln!("error: unknown command '{}'\n", args[1]);
            print_usage();
        }
    }
}
