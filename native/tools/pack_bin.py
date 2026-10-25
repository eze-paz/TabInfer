"""Pre-pack Bonsai-1.7B into a single binary the CPU engine loads in seconds
(instead of 70s of in-browser JS packing). Layout matches cpuengine.js exactly:
  - proj/lm_head: ternary codes (interleaved 2-bit, G=64) + f32 per-group scales
  - embed: raw f16 (u16)
  - norms: f32
Output: _cpukern/bonsai17.cpu.bin  = [u32 headerLen][json header][data blob]
"""
import json, struct, os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..', '_bonsai17')
OUT = os.path.join(os.path.dirname(__file__), '..', 'bonsai17.cpu.bin')
G = 64

def bf16(u16): return (u16.astype(np.uint32) << 16).view(np.float32)
def f16(u16): return u16.view(np.float16).astype(np.float32)
def is_tern(n): return n.endswith('_proj.weight') or n == 'lm_head.weight'

def pack_ternary(w):  # w: (N,K) f32 ternary -> codes u8 (N,K/4), scales f32 (N,ng)
    N, K = w.shape; ng = K // G
    g = w.reshape(N, ng, G)
    s = np.abs(g).max(2)                       # (N,ng)
    thr = (s * 0.5)[:, :, None]
    code = np.where(np.abs(g) < thr, 1, np.where(g < 0, 0, 2)).astype(np.uint8)  # (N,ng,64)
    codes = np.zeros((N, ng, 16), np.uint8)    # 16 bytes per 64-block
    for p in range(4):
        for i in range(16):
            codes[:, :, i] |= code[:, :, p * 16 + i] << (2 * p)
    return codes.reshape(N, K // 4), s.astype(np.float32)

def main():
    idx = json.load(open(os.path.join(ROOT, 'model.safetensors.index.json')))
    shards = sorted(set(idx['weight_map'].values()))
    # collect raw tensors
    raw = {}
    for sh in shards:
        with open(os.path.join(ROOT, sh), 'rb') as f:
            hlen = struct.unpack('<Q', f.read(8))[0]
            hdr = json.loads(f.read(hlen)); base = 8 + hlen
            for name, info in hdr.items():
                if name == '__metadata__': continue
                b0, b1 = info['data_offsets']; f.seek(base + b0)
                raw[name] = (info['dtype'], info['shape'], np.frombuffer(f.read(b1 - b0), np.uint16).copy())
        print('read', sh, flush=True)

    header = {'tensors': {}, 'G': G}
    blob = bytearray()
    def put(a):
        off = len(blob); b = a.tobytes(); blob.extend(b)
        # 16-align
        pad = (-len(blob)) % 16
        blob.extend(b'\x00' * pad)
        return off, len(b)

    for name, (dt, shape, u16) in raw.items():
        if name == 'model.embed_tokens.weight':
            off, ln = put(u16)  # keep f16 bits
            header['tensors'][name] = {'kind': 'embed_f16', 'shape': shape, 'off': off, 'len': ln}
        elif is_tern(name):
            w = bf16(u16).reshape(shape)
            codes, scales = pack_ternary(w)
            co, cl = put(codes.reshape(-1)); so, sl = put(scales.reshape(-1))
            header['tensors'][name] = {'kind': 'tern', 'N': shape[0], 'K': shape[1],
                                       'codesOff': co, 'codesLen': cl, 'scalesOff': so, 'scalesLen': sl}
            print('packed', name, shape, flush=True)
        else:  # norms -> f32
            a = f16(u16)
            off, ln = put(a)
            header['tensors'][name] = {'kind': 'f32', 'shape': shape, 'off': off, 'len': ln}

    hjson = json.dumps(header).encode()
    with open(OUT, 'wb') as f:
        f.write(struct.pack('<I', len(hjson))); f.write(hjson); f.write(blob)
    print(f'wrote {OUT}: header {len(hjson)}B, blob {len(blob)/1e6:.0f}MB', flush=True)

if __name__ == '__main__':
    main()
