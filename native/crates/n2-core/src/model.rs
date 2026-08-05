//! The Bonsai-1.7B forward pass.
//!
//! Architecture is Qwen3-shaped: 28 dense layers, RMSNorm, GQA with per-head
//! qk-norm, split-half RoPE, SwiGLU, untied `lm_head`. Every constant here comes
//! from `tools/ref_forward.py`, which is the oracle this is validated against —
//! including `rope_theta = 1e6`, which is *not* the 10000 that most configs use
//! and which would produce fluent, wrong output if assumed.

use crate::binfmt::{Container, Tensor};
use crate::ops;
use n2_kernels::tern;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub layers: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub rope_theta: f32,
    pub eps: f32,
}

impl Config {
    /// Bonsai-1.7B. Matches `ref_forward.py`'s
    /// `L, H, NH, NKV, HD, I, THETA, EPS, VOCAB`.
    pub fn bonsai_1_7b() -> Self {
        Config {
            layers: 28,
            hidden: 2048,
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 128,
            intermediate: 6144,
            vocab: 151936,
            rope_theta: 1e6,
            eps: 1e-6,
        }
    }

    pub fn q_per_kv(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

/// One layer's tensors, resolved once at load.
struct Layer {
    input_ln: Tensor,
    post_ln: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

/// Per-layer KV cache. Stores post-RoPE keys and raw values, appended per position.
pub struct KvCache {
    /// `[layer][pos * kv_dim .. +kv_dim]`
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    len: usize,
}

impl KvCache {
    pub fn new(cfg: &Config, capacity: usize) -> Self {
        let kv = cfg.kv_dim();
        KvCache {
            k: (0..cfg.layers).map(|_| Vec::with_capacity(capacity * kv)).collect(),
            v: (0..cfg.layers).map(|_| Vec::with_capacity(capacity * kv)).collect(),
            len: 0,
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn clear(&mut self) {
        for l in self.k.iter_mut() {
            l.clear();
        }
        for l in self.v.iter_mut() {
            l.clear();
        }
        self.len = 0;
    }
}

/// Reusable buffers, so a token's forward pass allocates nothing.
struct Scratch {
    x: Vec<f32>,
    xn: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    scores: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    swi: Vec<f32>,
    tmp: Vec<f32>,
    logits: Vec<f32>,
    act: Vec<i8>,
    xsum: Vec<i32>,
}

impl Scratch {
    fn new(cfg: &Config, capacity: usize) -> Self {
        // The widest activation fed to a matmul is the MLP's `intermediate`.
        let widest = cfg.intermediate.max(cfg.hidden);
        Scratch {
            x: vec![0.0; cfg.hidden],
            xn: vec![0.0; cfg.hidden],
            q: vec![0.0; cfg.q_dim()],
            k: vec![0.0; cfg.kv_dim()],
            v: vec![0.0; cfg.kv_dim()],
            attn: vec![0.0; cfg.q_dim()],
            scores: vec![0.0; capacity],
            gate: vec![0.0; cfg.intermediate],
            up: vec![0.0; cfg.intermediate],
            swi: vec![0.0; cfg.intermediate],
            tmp: vec![0.0; widest],
            logits: vec![0.0; cfg.vocab],
            act: vec![0; widest],
            xsum: vec![0; widest / n2_kernels::G],
        }
    }
}

/// Per-layer diagnostics for layer 0, mirroring the reference's printout. This is
/// how the forward pass is checked, rather than by reading the generated text.
#[derive(Debug, Default, Clone, Copy)]
pub struct Layer0Rms {
    pub xn: f32,
    pub o: f32,
    pub after_attn: f32,
    pub after_mlp: f32,
}

pub struct Model {
    pub cfg: Config,
    c: Container,
    embed: Tensor,
    final_norm: Tensor,
    lm_head: Tensor,
    layers: Vec<Layer>,
    s: Scratch,
    /// Captured on the most recent forward pass, for validation.
    pub l0: Layer0Rms,
}

impl Model {
    /// Resolve every tensor up front. A missing or wrongly-typed tensor is an
    /// error here rather than a surprise mid-generation.
    pub fn load(c: Container, cfg: Config, capacity: usize) -> Result<Self, String> {
        let need = |name: &str| -> Result<Tensor, String> {
            c.get(name).cloned().ok_or_else(|| format!("missing tensor `{name}`"))
        };

        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = format!("model.layers.{l}.");
            layers.push(Layer {
                input_ln: need(&format!("{p}input_layernorm.weight"))?,
                post_ln: need(&format!("{p}post_attention_layernorm.weight"))?,
                q_norm: need(&format!("{p}self_attn.q_norm.weight"))?,
                k_norm: need(&format!("{p}self_attn.k_norm.weight"))?,
                q: need(&format!("{p}self_attn.q_proj.weight"))?,
                k: need(&format!("{p}self_attn.k_proj.weight"))?,
                v: need(&format!("{p}self_attn.v_proj.weight"))?,
                o: need(&format!("{p}self_attn.o_proj.weight"))?,
                gate: need(&format!("{p}mlp.gate_proj.weight"))?,
                up: need(&format!("{p}mlp.up_proj.weight"))?,
                down: need(&format!("{p}mlp.down_proj.weight"))?,
            });
        }

        let m = Model {
            embed: need("model.embed_tokens.weight")?,
            final_norm: need("model.norm.weight")?,
            lm_head: need("lm_head.weight")?,
            layers,
            s: Scratch::new(&cfg, capacity),
            cfg,
            c,
            l0: Layer0Rms::default(),
        };
        m.check_shapes()?;
        Ok(m)
    }

    /// Cross-check the container against the config, so a mismatched model fails
    /// at load rather than producing plausible garbage.
    fn check_shapes(&self) -> Result<(), String> {
        let cfg = &self.cfg;
        let expect = |t: &Tensor, n: usize, k: usize, what: &str| -> Result<(), String> {
            match t {
                Tensor::Ternary { n: tn, k: tk, .. } if *tn == n && *tk == k => Ok(()),
                Tensor::Ternary { n: tn, k: tk, .. } => Err(format!(
                    "{what}: container says [{tn}, {tk}], config implies [{n}, {k}]"
                )),
                other => Err(format!("{what}: expected `tern`, found `{}`", other.kind())),
            }
        };
        let l = &self.layers[0];
        expect(&l.q, cfg.q_dim(), cfg.hidden, "q_proj")?;
        expect(&l.k, cfg.kv_dim(), cfg.hidden, "k_proj")?;
        expect(&l.v, cfg.kv_dim(), cfg.hidden, "v_proj")?;
        expect(&l.o, cfg.hidden, cfg.q_dim(), "o_proj")?;
        expect(&l.gate, cfg.intermediate, cfg.hidden, "gate_proj")?;
        expect(&l.up, cfg.intermediate, cfg.hidden, "up_proj")?;
        expect(&l.down, cfg.hidden, cfg.intermediate, "down_proj")?;
        expect(&self.lm_head, cfg.vocab, cfg.hidden, "lm_head")?;
        Ok(())
    }

    /// `out = W · x`, with `x` quantised to i8 and the result scaled back.
    ///
    /// This is where the engine differs from the f32 reference: activations go
    /// through int8, so outputs will not match numpy bit-for-bit. They should
    /// match closely in RMS and agree on the argmax.
    fn linear(
        c: &Container,
        w: &Tensor,
        x: &[f32],
        out: &mut [f32],
        act: &mut [i8],
        xsum: &mut [i32],
    ) -> Result<(), String> {
        let view = c.view_tern(w)?;
        debug_assert_eq!(view.k, x.len());
        let k = x.len();
        let act = &mut act[..k];
        let xsum = &mut xsum[..k / n2_kernels::G];
        let scale = tern::quantize_act_into(x, act, xsum);

        let out = &mut out[..view.n];
        tern::gemv_tern(out, view.codes, view.scales, act, xsum, view.n, k);

        let inv = 1.0 / scale;
        for v in out.iter_mut() {
            *v *= inv;
        }
        Ok(())
    }

    /// Run one token. Returns logits only when `want_logits`, since the lm_head
    /// projection over a 151936 vocabulary is ~18% of a token's work and is pure
    /// waste on every prompt token but the last.
    pub fn forward(
        &mut self,
        token: u32,
        cache: &mut KvCache,
        want_logits: bool,
    ) -> Result<Option<&[f32]>, String> {
        let cfg = self.cfg;
        let pos = cache.len;
        let hd = cfg.head_dim;
        let kv_dim = cfg.kv_dim();

        self.c.embed_row("model.embed_tokens.weight", token as usize, &mut self.s.x)?;

        let rope = ops::rope_table(hd, pos, cfg.rope_theta);
        let scale = 1.0 / (hd as f32).sqrt();

        for li in 0..cfg.layers {
            let l = &self.layers[li];

            let w = self.c.view_f32(&l.input_ln)?;
            ops::rmsnorm(&mut self.s.xn, &self.s.x, w, cfg.eps);

            Self::linear(&self.c, &l.q, &self.s.xn, &mut self.s.q, &mut self.s.act, &mut self.s.xsum)?;
            Self::linear(&self.c, &l.k, &self.s.xn, &mut self.s.k, &mut self.s.act, &mut self.s.xsum)?;
            Self::linear(&self.c, &l.v, &self.s.xn, &mut self.s.v, &mut self.s.act, &mut self.s.xsum)?;

            let qnw = self.c.view_f32(&l.q_norm)?;
            ops::rmsnorm_heads(&mut self.s.q, qnw, cfg.n_heads, hd, cfg.eps);
            let knw = self.c.view_f32(&l.k_norm)?;
            ops::rmsnorm_heads(&mut self.s.k, knw, cfg.n_kv_heads, hd, cfg.eps);

            ops::rope_with_table(&mut self.s.q, cfg.n_heads, hd, &rope);
            ops::rope_with_table(&mut self.s.k, cfg.n_kv_heads, hd, &rope);

            cache.k[li].extend_from_slice(&self.s.k);
            cache.v[li].extend_from_slice(&self.s.v);
            let t = cache.k[li].len() / kv_dim;

            if self.s.scores.len() < t {
                self.s.scores.resize(t, 0.0);
            }

            // GQA: query head h reads KV head h / (n_heads / n_kv_heads).
            let qpk = cfg.q_per_kv();
            for h in 0..cfg.n_heads {
                let kvh = h / qpk;
                let qh = &self.s.q[h * hd..][..hd];

                let scores = &mut self.s.scores[..t];
                for (p, sc) in scores.iter_mut().enumerate() {
                    let kh = &cache.k[li][p * kv_dim + kvh * hd..][..hd];
                    let mut dot = 0.0f32;
                    for i in 0..hd {
                        dot += kh[i] * qh[i];
                    }
                    *sc = dot * scale;
                }
                ops::softmax(scores);

                let out = &mut self.s.attn[h * hd..][..hd];
                out.fill(0.0);
                for (p, &wgt) in scores.iter().enumerate() {
                    let vh = &cache.v[li][p * kv_dim + kvh * hd..][..hd];
                    for i in 0..hd {
                        out[i] += wgt * vh[i];
                    }
                }
            }

            Self::linear(&self.c, &l.o, &self.s.attn, &mut self.s.tmp, &mut self.s.act, &mut self.s.xsum)?;
            if li == 0 {
                self.l0.xn = ops::rms(&self.s.xn);
                self.l0.o = ops::rms(&self.s.tmp[..cfg.hidden]);
            }
            for i in 0..cfg.hidden {
                self.s.x[i] += self.s.tmp[i];
            }
            if li == 0 {
                self.l0.after_attn = ops::rms(&self.s.x);
            }

            let w = self.c.view_f32(&l.post_ln)?;
            ops::rmsnorm(&mut self.s.xn, &self.s.x, w, cfg.eps);

            Self::linear(&self.c, &l.gate, &self.s.xn, &mut self.s.gate, &mut self.s.act, &mut self.s.xsum)?;
            Self::linear(&self.c, &l.up, &self.s.xn, &mut self.s.up, &mut self.s.act, &mut self.s.xsum)?;
            ops::swiglu(&mut self.s.swi, &self.s.gate, &self.s.up);

            Self::linear(&self.c, &l.down, &self.s.swi, &mut self.s.tmp, &mut self.s.act, &mut self.s.xsum)?;
            for i in 0..cfg.hidden {
                self.s.x[i] += self.s.tmp[i];
            }
            if li == 0 {
                self.l0.after_mlp = ops::rms(&self.s.x);
            }
        }

        cache.len = pos + 1;

        if !want_logits {
            return Ok(None);
        }

        let w = self.c.view_f32(&self.final_norm)?;
        ops::rmsnorm(&mut self.s.xn, &self.s.x, w, cfg.eps);
        let lm = self.lm_head.clone();
        Self::linear(&self.c, &lm, &self.s.xn, &mut self.s.logits, &mut self.s.act, &mut self.s.xsum)?;
        Ok(Some(&self.s.logits))
    }
}
