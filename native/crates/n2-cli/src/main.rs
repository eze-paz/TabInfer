//! `n2` — the command line for the N2 inference engine.
//!
//!   n2 cpu                 what this machine can do, and which kernel wins
//!   n2 selftest            kernel correctness against the dequantise oracle
//!   n2 bench [N] [K]       ternary GEMV throughput, with a control run
//!   n2 info <file.cpu.bin> read a packed container and describe it

use n2_kernels::{cpu, timing, tern};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("help");

    let r = match cmd {
        "cpu" => {
            cmd_cpu();
            Ok(())
        }
        "selftest" => cmd_selftest(),
        "bench" => {
            let n = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2048);
            let k = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2048);
            cmd_bench(n, k)
        }
        "info" => match args.get(1) {
            Some(p) => cmd_info(p),
            None => Err("usage: n2 info <file.cpu.bin>".to_string()),
        },
        "run" => cmd_run(&args[1..]),
        "help" | "-h" | "--help" => {
            help();
            Ok(())
        }
        other => Err(format!("unknown command `{other}` (try `n2 help`)")),
    };

    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn help() {
    println!(
        "n2 {} — native ternary inference\n\n\
         USAGE\n  \
           n2 cpu                    report ISA features and the selected kernel\n  \
           n2 selftest               validate kernels against the dequantise oracle\n  \
           n2 bench [N] [K]          ternary GEMV throughput (default 2048x2048)\n  \
           n2 info <file.cpu.bin>    describe a packed weight container\n  \
           n2 run <file.cpu.bin>     generate, reporting prefill and decode tok/s\n\n\
         n2 run OPTIONS\n  \
           --ids <a,b,c>             prompt as token ids (default: the built-in\n                             \
                                     'capital of France' probe from ref_forward.py)\n  \
           --max-new <n>             tokens to generate (default 16)\n  \
           --verify                  check the first prediction is Paris (12095)\n                             \
                                     and print layer-0 RMS for comparison with\n                             \
                                     tools/ref_forward.py\n",
        env!("CARGO_PKG_VERSION")
    );
}

fn cmd_cpu() {
    println!("arch            {}", std::env::consts::ARCH);
    println!("os              {}", std::env::consts::OS);
    println!("logical cpus    {}", cpu::logical_cpus());
    println!("kernel backend  {}", tern::backend());
    println!();
    println!("{:<14} {:<9} {}", "feature", "present", "why it matters");
    for f in cpu::features() {
        println!(
            "{:<14} {:<9} {}",
            f.name,
            if f.present { "yes" } else { "-" },
            f.why
        );
    }
}

/// Synthetic ternary weights + activations in the real packed layout.
fn fixture(n: usize, k: usize) -> (Vec<u8>, Vec<f32>, Vec<i8>, Vec<i32>) {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 32) as u32
    };
    let w: Vec<f32> = (0..n * k)
        .map(|_| match next() % 5 {
            0 => -1.0,
            1 => 1.0,
            2 => 0.0,
            3 => 0.2,
            _ => -0.9,
        })
        .collect();
    let (codes, scales) = tern::pack_ternary(&w, n, k);
    let x: Vec<f32> = (0..k).map(|_| (next() % 2001) as f32 / 1000.0 - 1.0).collect();
    let (act, xsum, _) = tern::quantize_act(&x);
    (codes, scales, act, xsum)
}

fn cmd_selftest() -> Result<(), String> {
    let mut failures = 0;
    println!("{:<18} {:<12} {}", "case", "max rel err", "verdict");

    for &(n, k) in &[(1usize, 64usize), (3, 192), (17, 256), (64, 2048), (128, 4096)] {
        let (codes, scales, act, xsum) = fixture(n, k);
        let mut got = vec![0f32; n];
        let mut want = vec![0f32; n];

        // Dispatching kernel (whatever this CPU selected) vs the independent
        // dequantise-and-multiply oracle. This is the check that catches a
        // misread of the interleaved 2-bit layout, which no backend-vs-backend
        // comparison can see.
        tern::gemv_tern(&mut got, &codes, &scales, &act, &xsum, n, k);
        tern::gemv_dequant_reference(&mut want, &codes, &scales, &act, n, k);

        let rel = got
            .iter()
            .zip(&want)
            .map(|(&a, &b)| (a - b).abs() / a.abs().max(b.abs()).max(1e-3))
            .fold(0.0f32, f32::max);

        let ok = rel < 1e-5;
        if !ok {
            failures += 1;
        }
        println!(
            "{:<18} {:<12.3e} {}",
            format!("{}x{} vs oracle", n, k),
            rel,
            if ok { "PASS" } else { "FAIL" }
        );
    }

    // Scalar vs the selected backend, so a broken SIMD path is named explicitly.
    let backend = tern::backend();
    if backend != tern::Backend::Scalar {
        let (n, k) = (64usize, 2048usize);
        let (codes, scales, act, xsum) = fixture(n, k);
        let mut a = vec![0f32; n];
        let mut b = vec![0f32; n];
        tern::gemv_tern_scalar(&mut a, &codes, &scales, &act, &xsum, n, k);
        tern::gemv_tern(&mut b, &codes, &scales, &act, &xsum, n, k);
        let rel = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| (x - y).abs() / x.abs().max(y.abs()).max(1e-3))
            .fold(0.0f32, f32::max);
        let ok = rel < 1e-5;
        if !ok {
            failures += 1;
        }
        println!(
            "{:<18} {:<12.3e} {}",
            format!("{backend} vs scalar"),
            rel,
            if ok { "PASS" } else { "FAIL" }
        );
    }

    println!();
    if failures == 0 {
        println!("all kernel self-tests passed");
        Ok(())
    } else {
        Err(format!("{failures} self-test(s) failed"))
    }
}

fn cmd_bench(n: usize, k: usize) -> Result<(), String> {
    if k % n2_kernels::G != 0 {
        return Err(format!("K must be a multiple of {}", n2_kernels::G));
    }
    let (codes, scales, act, xsum) = fixture(n, k);
    let mut out = vec![0f32; n];
    let macs = (n * k) as f64;

    println!("ternary GEMV  N={n} K={k}  backend={}", tern::backend());
    println!(
        "weights {:.1} MiB codes + {:.2} MiB scales\n",
        (n * k / 4) as f64 / (1 << 20) as f64,
        (n * (k / n2_kernels::G) * 4) as f64 / (1 << 20) as f64
    );

    // Rule 3 first: if the control does not read ~1.000x, nothing below means
    // anything and we say so instead of printing numbers that look like data.
    let ctrl = {
        let mut o = vec![0f32; n];
        timing::control(
            "scalar",
            || {
                tern::gemv_tern_scalar(&mut o, &codes, &scales, &act, &xsum, n, k);
                o[0]
            },
            9,
            2,
        )
    };
    println!("{ctrl}");
    let clean = timing::control_is_clean(&ctrl, 0.15);
    if !clean {
        println!(
            "\n  ^ control is {:.3}x, not ~1.000x: this box is too noisy right now.\n    \
             Treat the A/B below as unusable and re-run when it settles.",
            ctrl.speedup()
        );
    }
    println!();

    let backend = tern::backend();
    if backend == tern::Backend::Scalar {
        println!("only the scalar backend is available here; no A/B to run");
    } else {
        let mut o1 = vec![0f32; n];
        let mut o2 = vec![0f32; n];
        let r = timing::ab(
            "scalar",
            || {
                tern::gemv_tern_scalar(&mut o1, &codes, &scales, &act, &xsum, n, k);
                o1[0]
            },
            &backend.to_string(),
            || {
                tern::gemv_tern(&mut o2, &codes, &scales, &act, &xsum, n, k);
                o2[0]
            },
            31,
            3,
        );
        println!("{r}");
        println!(
            "\n{:<10} {:>8.2} GMAC/s\n{:<10} {:>8.2} GMAC/s",
            "scalar",
            macs / r.a.as_secs_f64() / 1e9,
            backend.to_string(),
            macs / r.b.as_secs_f64() / 1e9,
        );
    }

    // Keep `out` alive so the compiler cannot elide the fixture.
    tern::gemv_tern(&mut out, &codes, &scales, &act, &xsum, n, k);
    std::hint::black_box(&out);

    if clean {
        Ok(())
    } else {
        Err("control run was not clean; numbers above are not trustworthy".into())
    }
}

/// The ground-truth probe from `tools/ref_forward.py`: ChatML-wrapped
/// "The capital of France is".
const PROBE_IDS: [u32; 13] = [
    151644, 872, 198, 785, 6722, 315, 9625, 374, 151645, 198, 151644, 77091, 198,
];

/// Both spellings of the answer. `ref_forward.py` probes 12095 (" Paris", with a
/// leading space); Bonsai actually emits a `<think>` block first and then writes
/// "**Paris**", where the token is 59604 ("Paris", no leading space). Checking only
/// 12095 as the *first* prediction was a wrong assertion about a reasoning model,
/// not a wrong model.
const PARIS_TOKENS: [u32; 2] = [12095, 59604];
const THINK_OPEN: u32 = 151667;

fn cmd_run(args: &[String]) -> Result<(), String> {
    let path = args
        .first()
        .filter(|a| !a.starts_with("--"))
        .ok_or("usage: n2 run <file.cpu.bin> [--ids a,b,c] [--max-new n] [--verify]")?;

    let mut ids: Vec<u32> = PROBE_IDS.to_vec();
    let mut max_new = 16usize;
    let mut verify = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--ids" => {
                let raw = args.get(i + 1).ok_or("--ids needs a comma-separated list")?;
                ids = raw
                    .split(',')
                    .map(|s| s.trim().parse::<u32>().map_err(|_| format!("bad token id `{s}`")))
                    .collect::<Result<_, _>>()?;
                if ids.is_empty() {
                    return Err("--ids was empty".into());
                }
                i += 2;
            }
            "--max-new" => {
                max_new = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .ok_or("--max-new needs a number")?;
                i += 2;
            }
            "--verify" => {
                verify = true;
                i += 1;
            }
            other => return Err(format!("unknown option `{other}`")),
        }
    }

    let cfg = n2_core::Config::bonsai_1_7b();
    let capacity = ids.len() + max_new + 8;

    let t0 = std::time::Instant::now();
    let container = n2_core::Container::open(path)?;
    let mut model = n2_core::Model::load(container, cfg, capacity)?;
    let load = t0.elapsed();

    let mut cache = n2_core::KvCache::new(&cfg, capacity);

    println!(
        "model   {} layers, hidden {}, {}Q/{}KV heads, vocab {}, rope_theta {:.0e}",
        cfg.layers, cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.vocab, cfg.rope_theta
    );
    println!("backend {}  (single-threaded)", n2_kernels::tern::backend());
    println!("loaded  {:.2} s", load.as_secs_f64());
    println!("prompt  {} tokens\n", ids.len());

    // ── Prefill: every prompt token, logits only on the last ──────────────────
    let t_pre = std::time::Instant::now();
    let mut first_pred = 0usize;
    for (n, &tok) in ids.iter().enumerate() {
        let last = n == ids.len() - 1;
        if let Some(logits) = model.forward(tok, &mut cache, last)? {
            first_pred = n2_core::ops::argmax(logits);
        }
    }
    let prefill = t_pre.elapsed();
    let l0 = model.l0;

    // ── Decode: greedy, one token at a time ───────────────────────────────────
    let mut generated = Vec::with_capacity(max_new);
    let mut next = first_pred as u32;
    let t_dec = std::time::Instant::now();
    for _ in 0..max_new {
        generated.push(next);
        // 151643 is <|endoftext|>, 151645 is <|im_end|>.
        if next == 151643 || next == 151645 {
            break;
        }
        let logits = model
            .forward(next, &mut cache, true)?
            .ok_or("decode asked for logits and got none")?;
        next = n2_core::ops::argmax(logits) as u32;
    }
    let decode = t_dec.elapsed();

    println!("generated ids: {generated:?}");
    println!();

    let pre_tps = ids.len() as f64 / prefill.as_secs_f64();
    let dec_n = generated.len().saturating_sub(1).max(1);
    let dec_tps = dec_n as f64 / decode.as_secs_f64();

    println!("{:<10} {:>8}   {:>10}   {:>12}", "phase", "tokens", "seconds", "tok/s");
    println!(
        "{:<10} {:>8}   {:>10.3}   {:>12.2}",
        "prefill",
        ids.len(),
        prefill.as_secs_f64(),
        pre_tps
    );
    println!(
        "{:<10} {:>8}   {:>10.3}   {:>12.2}",
        "decode",
        dec_n,
        decode.as_secs_f64(),
        dec_tps
    );
    println!(
        "\nprefill is batch-1 (one forward per prompt token), so it is expected to\n\
         track decode closely. Batched prefill is P3 on the roadmap."
    );

    if verify {
        println!("── verification (smoke test, NOT a correctness proof) ──");
        println!(
            "layer-0 rms: xn={:.3} o={:.3} after_attn={:.3} after_mlp={:.3}",
            l0.xn, l0.o, l0.after_attn, l0.after_mlp
        );
        println!("first prediction: {first_pred}{}", if first_pred as u32 == THINK_OPEN {
            "  (<think> — expected, Bonsai is a reasoning model)"
        } else {
            ""
        });

        let found = generated.iter().find(|t| PARIS_TOKENS.contains(t));
        match found {
            Some(t) => println!("answer token:     {t} (Paris) -- PASS"),
            None => {
                return Err(format!(
                    "no Paris token ({PARIS_TOKENS:?}) in {generated:?}. \
                     The arch or the weight mapping is wrong; do not trust the text."
                ))
            }
        }

        println!(
            "\nThis only shows the model answers correctly. It does NOT verify the\n\
             graph numerically — a subtly wrong RoPE or GQA mapping can still pass.\n\
             For that, run the numpy oracle and compare the layer-0 RMS above:\n\
             \n  N2_BONSAI_DIR=/path/to/_bonsai17 python tools/ref_forward.py\n\
             \nIt needs numpy and roughly 11 GiB of RAM (it caches every dequantised\n\
             f32 weight), so it will not run on a loaded machine."
        );
    }

    Ok(())
}

fn cmd_info(path: &str) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let c = n2_core::Container::open(path)?;
    let load = t0.elapsed();

    let mut tern_bytes = 0usize;
    let mut embed_bytes = 0usize;
    let mut f32_bytes = 0usize;
    let mut tern_params = 0usize;

    for t in c.tensors.values() {
        match t {
            n2_core::Tensor::Ternary { n, k, .. } => {
                tern_bytes += t.bytes();
                tern_params += n * k;
            }
            n2_core::Tensor::EmbedF16 { .. } => embed_bytes += t.bytes(),
            n2_core::Tensor::F32 { .. } => f32_bytes += t.bytes(),
        }
    }

    let mib = |b: usize| b as f64 / (1 << 20) as f64;
    println!("file            {path}");
    println!("read in         {:.2} s", load.as_secs_f64());
    println!("group size      {}", c.group_size);
    println!("tensors         {}", c.tensors.len());
    println!("blob            {:.1} MiB", mib(c.blob_bytes()));
    println!();
    println!("ternary         {:.1} MiB  ({} params, {:.2} bits/param)",
        mib(tern_bytes),
        tern_params,
        if tern_params > 0 { tern_bytes as f64 * 8.0 / tern_params as f64 } else { 0.0 });
    println!("embed f16       {:.1} MiB", mib(embed_bytes));
    println!("f32 (norms)     {:.1} MiB", mib(f32_bytes));

    // A few named tensors, so a wrong container is obvious at a glance.
    println!("\n{:<44} {:<10} shape", "tensor", "kind");
    for (name, t) in c.tensors.iter().take(6) {
        let shape = match t {
            n2_core::Tensor::Ternary { n, k, .. } => format!("[{n}, {k}]"),
            n2_core::Tensor::EmbedF16 { shape, .. } | n2_core::Tensor::F32 { shape, .. } => {
                format!("{shape:?}")
            }
        };
        println!("{name:<44} {:<10} {shape}", t.kind());
    }
    if c.tensors.len() > 6 {
        println!("... and {} more", c.tensors.len() - 6);
    }
    Ok(())
}
