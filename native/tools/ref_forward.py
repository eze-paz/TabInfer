"""Full-precision numpy reference forward for Bonsai-1.7B (Qwen3 arch).
Reads _bonsai17 safetensors, dequantizes exactly, runs the prompt, reports the
top predicted tokens. Ground truth: if THIS picks 'Paris', the weights + arch are
right and the JS engine has a bug; if not, the repack/arch is wrong.
Fast (numpy), runs from the shell — no browser, no 70s reload.
"""
import json, struct, os, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..', '_bonsai17')
L, H, NH, NKV, HD, I, THETA, EPS, VOCAB = 28, 2048, 16, 8, 128, 6144, 1e6, 1e-6, 151936
IDS = [151644,872,198,785,6722,315,9625,374,151645,198,151644,77091,198]
PARIS = 12095

def load_all():
    idx = json.load(open(os.path.join(ROOT, 'model.safetensors.index.json')))
    shards = sorted(set(idx['weight_map'].values()))
    W = {}
    for sh in shards:
        with open(os.path.join(ROOT, sh), 'rb') as f:
            hlen = struct.unpack('<Q', f.read(8))[0]
            hdr = json.loads(f.read(hlen))
            base = 8 + hlen
            for name, info in hdr.items():
                if name == '__metadata__': continue
                b0, b1 = info['data_offsets']
                f.seek(base + b0); raw = f.read(b1 - b0)
                u16 = np.frombuffer(raw, dtype=np.uint16).copy()
                W[name] = (info['dtype'], info['shape'], u16)
        print('loaded', sh, flush=True)
    return W

def bf16(u16):  # uint16 bf16 bits -> f32
    return (u16.astype(np.uint32) << 16).view(np.float32)
def f16(u16):
    return u16.view(np.float16).astype(np.float32)

def deq(W, name):
    dt, shape, u16 = W[name]
    a = bf16(u16) if dt == 'BF16' else f16(u16)
    return a.reshape(shape)

def rmsnorm(x, w):
    v = x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + EPS)
    return v * w

def main():
    W = load_all()
    embed = W['model.embed_tokens.weight']  # (dtype,shape,u16) F16
    def emb_row(t):
        _, shape, u16 = embed
        return f16(u16.reshape(shape)[t])
    # matmul cache: dequant weight lazily
    wc = {}
    def lin(name, x):
        if name not in wc: wc[name] = deq(W, name)  # (out,in)
        return wc[name] @ x
    norms = {k: deq(W, k) for k in W if k.endswith('norm.weight')}

    kv = [([], []) for _ in range(L)]
    def forward(tid, pos, want_logits):
        x = emb_row(tid).astype(np.float32)
        for l in range(L):
            P = f'model.layers.{l}.'
            xn = rmsnorm(x, norms[P+'input_layernorm.weight'])
            q = lin(P+'self_attn.q_proj.weight', xn)
            k = lin(P+'self_attn.k_proj.weight', xn)
            v = lin(P+'self_attn.v_proj.weight', xn)
            qnw = norms[P+'self_attn.q_norm.weight']; knw = norms[P+'self_attn.k_norm.weight']
            q = q.reshape(NH, HD); k = k.reshape(NKV, HD); v = v.reshape(NKV, HD)
            q = rmsnorm(q, qnw); k = rmsnorm(k, knw)
            # rope
            half = HD // 2
            inv = THETA ** (-(2*np.arange(half))/HD)
            ang = pos * inv; c = np.cos(ang); s = np.sin(ang)
            def rope(t):
                x0 = t[:, :half].copy(); x1 = t[:, half:].copy()
                t[:, :half] = x0*c - x1*s; t[:, half:] = x1*c + x0*s
            rope(q); rope(k)
            kv[l][0].append(k); kv[l][1].append(v)
            K = np.stack(kv[l][0]); V = np.stack(kv[l][1])  # (T,NKV,HD)
            attn = np.zeros((NH, HD), np.float32); scale = 1/np.sqrt(HD); qpk = NH//NKV
            for h in range(NH):
                kvh = h // qpk
                sc = (K[:, kvh, :] @ q[h]) * scale  # (T,)
                sc = np.exp(sc - sc.max()); sc /= sc.sum()
                attn[h] = (sc[:, None] * V[:, kvh, :]).sum(0)
            o = lin(P+'self_attn.o_proj.weight', attn.reshape(-1))
            x = x + o
            xn2 = rmsnorm(x, norms[P+'post_attention_layernorm.weight'])
            g = lin(P+'mlp.gate_proj.weight', xn2); u = lin(P+'mlp.up_proj.weight', xn2)
            swi = (g / (1+np.exp(-g))) * u
            x = x + lin(P+'mlp.down_proj.weight', swi)
            if l == 0 and want_logits is None:
                pass
            if l == 0:
                print(f'  l0 rms: xn={np.sqrt((xn**2).mean()):.3f} o={np.sqrt((o**2).mean()):.3f} x_afterAttn={np.sqrt((x**2).mean()):.3f} afterMLP={np.sqrt((x**2).mean()):.3f}', flush=True)
        if not want_logits: return None
        xf = rmsnorm(x, norms['model.norm.weight'])
        return lin('lm_head.weight', xf)

    lg = None
    for i, t in enumerate(IDS):
        lg = forward(t, i, i == len(IDS)-1)
    order = np.argsort(-lg)
    rank = int((lg > lg[PARIS]).sum()) + 1
    print(f'\nParis(12095) logit={lg[PARIS]:.2f} rank={rank}/{VOCAB}')
    print('top10 ids:', order[:10].tolist())
    print('top10 logits:', [round(float(lg[i]),2) for i in order[:10]])

if __name__ == '__main__':
    main()
