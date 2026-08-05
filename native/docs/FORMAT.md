# The `.cpu.bin` weight format

This is the authoritative spec. `tools/pack_bin.py` writes it and
`crates/n2-core/src/binfmt.rs` reads it; if they ever disagree, this document is
the tie-breaker and one of them is a bug.

It is written out in full deliberately. This project has already lost time to a
*stale control spec* — workers grinding the wrong matrices and emitting plausible
garbage, with every self-test green — and the fix for that class of bug is a
written spec plus an independent oracle, not more assertions.

## Container

```
offset 0   u32 le    header_len
offset 4   bytes     header_len bytes of UTF-8 JSON
offset 4+header_len  the blob
```

All offsets inside the header are **blob-relative**: absolute file position is
`4 + header_len + off`. Every blob entry is padded to a **16-byte** boundary,
which is what makes the `f32` ranges safe to reinterpret in place.

### Header

```json
{
  "G": 64,
  "tensors": {
    "<tensor name>": { "kind": "...", ... }
  }
}
```

`G` is the scale-group size and must equal the kernels' `G`. The reader refuses a
container with any other value rather than silently mismatching.

Three `kind`s:

| kind | fields | meaning |
|---|---|---|
| `tern` | `N`, `K`, `codesOff`, `codesLen`, `scalesOff`, `scalesLen` | packed ternary matrix |
| `embed_f16` | `shape`, `off`, `len` | raw IEEE binary16 bit patterns (u16 le) |
| `f32` | `shape`, `off`, `len` | f32 le |

The reader cross-checks `codesLen == N*K/4` and `scalesLen == N*(K/G)*4` against
the declared shape. A packer/reader disagreement therefore fails at load instead
of at inference.

## Ternary encoding (`Q2_0_g64`)

A row of `K` weights is divided into `K/64` groups of 64. Per group:

- **scale** `s = max(|w|)` over the group, one f32.
- **code** per weight, from a threshold at half the scale:

  ```
  |w| < 0.5*s  ->  1     (zero)
  w < 0        ->  0     (negative)
  otherwise    ->  2     (positive)
  ```

So the reconstructed weight is `(code - 1) * s`, and codes are `{0,1,2}` rather
than `{-1,0,+1}`.

`scales` is `N x (K/64)` f32, row-major. `codes` is `N x (K/4)` bytes, row-major,
16 bytes per group.

### The interleaved bit layout

**This is the part that is easy to get wrong.** Codes within a 64-weight group are
*not* stored consecutively. For code index `c` in `0..64`:

```
byte     = c % 16          (i.e. c & 15)
bit pos  = 2 * (c / 16)    (i.e. 2 * (c >> 4))
```

Equivalently: byte `i` of the group holds codes `i`, `16+i`, `32+i`, `48+i` at bit
positions 0, 2, 4, 6.

The payoff is that one 16-byte load plus four shift-and-mask operations yields
four vectors of 16 codes each, with no shuffle and no scalar work. "Plane `j`"
means the codes at bit position `2j`, which are exactly codes `j*16 .. j*16+15`,
and they pair with activations `act[group*64 + j*16 .. +16]` — contiguous.

`crates/n2-kernels/src/tern.rs::pack_ternary` is a Rust transcription of the
Python packer, so the tests generate weights in this exact layout. The test
`packing_places_codes_in_the_expected_plane` pins the mapping directly.

## The kernel identity

Because codes are unsigned, the GEMV computes:

```
out[n] = Σ_b s[n,b] · dot[n,b]  -  Σ_b s[n,b] · xsum[b]

  dot[n,b] = Σ_{k∈b} code[n,k] · act[k]
  xsum[b]  = Σ_{k∈b} act[k]
```

This is BitNet's no-multiply identity. Subtracting `xsum` is what turns unsigned
codes back into signed weights — it is **not** an optional optimisation, and the
two terms must be accumulated separately and subtracted once per row so that
backends agree on rounding.

Note also, from the WASM engine's investigation: the dot instruction is a fused
int8 MAC and is already optimal. "Add-based ternary" needs *more* SIMD
instructions, not fewer; BitNet's no-multiply claim is about ASIC energy, not x86
throughput.

## Activation quantisation

Activations are quantised per-vector (not per-group) to i8:

```
scale = 127 / max(|x|)        (or 1.0 when max is 0)
act   = round(x * scale) clamped to [-127, 127]
```

`xsum[b]` is the i32 sum of `act` over group `b`. Kernel output is in units of
`1/scale`; the caller multiplies back.

The `max == 0` case returns an all-zero activation vector and a scale of 1.0
rather than a NaN — there is a regression test for it.

## Verified against the real pack

`n2 info` on the shipped Bonsai-1.7B container reads:

```
group size      64
tensors         311
blob            1106.7 MiB
ternary         512.7 MiB  (1720451072 params, 2.50 bits/param)
embed f16       593.5 MiB
f32 (norms)     0.5 MiB
```

2.50 bits/param is exactly right: 2 bits of code plus one f32 scale per 64
weights (`32/64 = 0.5`).

Note that the **f16 embedding table is 53% of the file** while holding ~18% of the
parameters. That is the single biggest size win available — see the roadmap.

## Open question, not yet resolved

`tools/pack_bin.py` decodes ternary weights with a **bf16** helper but the norm
weights with an **f16** helper, from the same checkpoint:

```python
def bf16(u16): return (u16.astype(np.uint32) << 16).view(np.float32)
def f16(u16):  return u16.view(np.float16).astype(np.float32)
...
else:  # norms -> f32
    a = f16(u16)
```

If that checkpoint is bf16 throughout, the norm weights are being misdecoded. The
packed model demonstrably works, so either the norms genuinely are f16 in that
snapshot or the error is small enough to be invisible in output quality — which
is precisely the kind of thing that hides. **Resolve by cross-checking a norm
tensor against `tools/ref_forward.py` on torch CPU before trusting the forward
pass**, and do not "fix" it by inspection; the packed `.cpu.bin` is verified
byte-identical to a known-good build, so changing the packer invalidates that.
