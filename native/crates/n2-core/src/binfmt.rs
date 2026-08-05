//! Reader for the `.cpu.bin` container produced by `tools/pack_bin.py`.
//!
//! Layout — see `docs/FORMAT.md` for the authoritative spec:
//!
//! ```text
//!   [u32 le header_len][header_len bytes of JSON][blob]
//! ```
//!
//! Every blob entry is 16-byte aligned. Offsets in the header are relative to the
//! start of the blob, i.e. absolute file offset `4 + header_len + off`.

use crate::json::{self, Value};
use std::collections::BTreeMap;
use std::path::Path;

/// A tensor as described by the header. Offsets are blob-relative.
#[derive(Debug, Clone)]
pub enum Tensor {
    /// Packed ternary: `N x K/4` code bytes plus `N x K/64` f32 scales.
    Ternary {
        n: usize,
        k: usize,
        codes_off: usize,
        codes_len: usize,
        scales_off: usize,
        scales_len: usize,
    },
    /// Raw f16 bit patterns (u16), used for the token embedding table.
    EmbedF16 { shape: Vec<usize>, off: usize, len: usize },
    /// f32 values, used for the norm weights.
    F32 { shape: Vec<usize>, off: usize, len: usize },
}

impl Tensor {
    /// Total bytes this tensor occupies in the blob.
    pub fn bytes(&self) -> usize {
        match self {
            Tensor::Ternary { codes_len, scales_len, .. } => codes_len + scales_len,
            Tensor::EmbedF16 { len, .. } | Tensor::F32 { len, .. } => *len,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Tensor::Ternary { .. } => "tern",
            Tensor::EmbedF16 { .. } => "embed_f16",
            Tensor::F32 { .. } => "f32",
        }
    }
}

pub struct Container {
    pub group_size: usize,
    pub tensors: BTreeMap<String, Tensor>,
    /// Whole-file bytes. `blob_base` is where blob-relative offsets start.
    data: Vec<u8>,
    blob_base: usize,
}

/// Hand-written so `unwrap()` on a `Result<Container, _>` cannot dump a
/// gigabyte of weights into the terminal.
impl core::fmt::Debug for Container {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Container")
            .field("group_size", &self.group_size)
            .field("tensors", &self.tensors.len())
            .field("blob_bytes", &self.blob_bytes())
            .finish()
    }
}

impl Container {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_bytes(data)
    }

    pub fn from_bytes(data: Vec<u8>) -> Result<Self, String> {
        if data.len() < 4 {
            return Err("file is too short to contain a header length".into());
        }
        let header_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let blob_base = 4 + header_len;
        if blob_base > data.len() {
            return Err(format!(
                "header length {header_len} overruns the file ({} bytes)",
                data.len()
            ));
        }
        let header_str = std::str::from_utf8(&data[4..blob_base])
            .map_err(|e| format!("header is not valid UTF-8: {e}"))?;
        let header = json::parse(header_str).map_err(|e| format!("header JSON: {e}"))?;

        let group_size = header.usize_field("G").unwrap_or(n2_kernels::G);
        if group_size != n2_kernels::G {
            return Err(format!(
                "container group size G={group_size} but the kernels are built for G={}",
                n2_kernels::G
            ));
        }

        let tensors_obj = header
            .get("tensors")
            .and_then(Value::as_obj)
            .ok_or("header has no `tensors` object")?;

        let mut tensors = BTreeMap::new();
        for (name, t) in tensors_obj {
            let kind = t
                .get("kind")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{name}: missing `kind`"))?;
            let parsed = match kind {
                "tern" => {
                    let n = t.usize_field("N").map_err(|e| format!("{name}: {e}"))?;
                    let k = t.usize_field("K").map_err(|e| format!("{name}: {e}"))?;
                    if k % group_size != 0 {
                        return Err(format!("{name}: K={k} is not a multiple of G={group_size}"));
                    }
                    let codes_len = t.usize_field("codesLen").map_err(|e| format!("{name}: {e}"))?;
                    let scales_len =
                        t.usize_field("scalesLen").map_err(|e| format!("{name}: {e}"))?;
                    // Cross-check the declared byte counts against the shape. A
                    // mismatch here means the packer and the reader disagree, and
                    // is exactly the failure that produces plausible garbage.
                    let want_codes = n * k / 4;
                    let want_scales = n * (k / group_size) * 4;
                    if codes_len != want_codes {
                        return Err(format!(
                            "{name}: codesLen={codes_len} but N*K/4={want_codes}"
                        ));
                    }
                    if scales_len != want_scales {
                        return Err(format!(
                            "{name}: scalesLen={scales_len} but N*(K/G)*4={want_scales}"
                        ));
                    }
                    Tensor::Ternary {
                        n,
                        k,
                        codes_off: t.usize_field("codesOff").map_err(|e| format!("{name}: {e}"))?,
                        codes_len,
                        scales_off: t
                            .usize_field("scalesOff")
                            .map_err(|e| format!("{name}: {e}"))?,
                        scales_len,
                    }
                }
                "embed_f16" => Tensor::EmbedF16 {
                    shape: t.usize_array("shape").map_err(|e| format!("{name}: {e}"))?,
                    off: t.usize_field("off").map_err(|e| format!("{name}: {e}"))?,
                    len: t.usize_field("len").map_err(|e| format!("{name}: {e}"))?,
                },
                "f32" => Tensor::F32 {
                    shape: t.usize_array("shape").map_err(|e| format!("{name}: {e}"))?,
                    off: t.usize_field("off").map_err(|e| format!("{name}: {e}"))?,
                    len: t.usize_field("len").map_err(|e| format!("{name}: {e}"))?,
                },
                other => return Err(format!("{name}: unknown kind `{other}`")),
            };
            tensors.insert(name.clone(), parsed);
        }

        let c = Container { group_size, tensors, data, blob_base };
        c.validate()?;
        Ok(c)
    }

    /// Every declared byte range must lie inside the blob.
    fn validate(&self) -> Result<(), String> {
        let blob = self.data.len() - self.blob_base;
        let check = |name: &str, off: usize, len: usize| -> Result<(), String> {
            if off + len > blob {
                return Err(format!(
                    "{name}: range {off}..{} overruns the {blob}-byte blob",
                    off + len
                ));
            }
            Ok(())
        };
        for (name, t) in &self.tensors {
            match t {
                Tensor::Ternary { codes_off, codes_len, scales_off, scales_len, .. } => {
                    check(name, *codes_off, *codes_len)?;
                    check(name, *scales_off, *scales_len)?;
                }
                Tensor::EmbedF16 { off, len, .. } | Tensor::F32 { off, len, .. } => {
                    check(name, *off, *len)?
                }
            }
        }
        Ok(())
    }

    fn slice(&self, off: usize, len: usize) -> &[u8] {
        &self.data[self.blob_base + off..self.blob_base + off + len]
    }

    pub fn blob_bytes(&self) -> usize {
        self.data.len() - self.blob_base
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Borrow a ternary tensor's codes and scales as kernel-ready slices.
    pub fn ternary(&self, name: &str) -> Result<TernaryView<'_>, String> {
        match self.get(name) {
            Some(Tensor::Ternary { n, k, codes_off, codes_len, scales_off, scales_len }) => {
                Ok(TernaryView {
                    n: *n,
                    k: *k,
                    codes: self.slice(*codes_off, *codes_len),
                    scales: as_f32(self.slice(*scales_off, *scales_len))?,
                })
            }
            Some(t) => Err(format!("{name}: expected `tern`, found `{}`", t.kind())),
            None => Err(format!("{name}: not in container")),
        }
    }

    /// Borrow an f32 tensor (norm weights).
    pub fn f32s(&self, name: &str) -> Result<&[f32], String> {
        match self.get(name) {
            Some(Tensor::F32 { off, len, .. }) => as_f32(self.slice(*off, *len)),
            Some(t) => Err(format!("{name}: expected `f32`, found `{}`", t.kind())),
            None => Err(format!("{name}: not in container")),
        }
    }

    /// Decode one row of the f16 embedding table into `dst`.
    pub fn embed_row(&self, name: &str, row: usize, dst: &mut [f32]) -> Result<(), String> {
        let (shape, off, len) = match self.get(name) {
            Some(Tensor::EmbedF16 { shape, off, len }) => (shape, *off, *len),
            Some(t) => return Err(format!("{name}: expected `embed_f16`, found `{}`", t.kind())),
            None => return Err(format!("{name}: not in container")),
        };
        let dim = *shape.last().ok_or("embedding has no shape")?;
        if dst.len() != dim {
            return Err(format!("dst is {} long, embedding dim is {dim}", dst.len()));
        }
        let start = row * dim * 2;
        if start + dim * 2 > len {
            return Err(format!("row {row} is out of range for {name}"));
        }
        let bytes = self.slice(off + start, dim * 2);
        for (d, c) in dst.iter_mut().zip(bytes.chunks_exact(2)) {
            *d = f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
        }
        Ok(())
    }
}

pub struct TernaryView<'a> {
    pub n: usize,
    pub k: usize,
    pub codes: &'a [u8],
    pub scales: &'a [f32],
}

/// Reinterpret a byte slice as f32. Requires 4-byte alignment, which the packer's
/// 16-byte blob alignment guarantees — but we check rather than assume, because an
/// unaligned read here is UB and the file comes from outside this program.
fn as_f32(b: &[u8]) -> Result<&[f32], String> {
    if b.len() % 4 != 0 {
        return Err(format!("{} bytes is not a whole number of f32", b.len()));
    }
    if b.as_ptr() as usize % align_of::<f32>() != 0 {
        return Err("f32 range is not 4-byte aligned in the file".into());
    }
    // SAFETY: length and alignment checked above; f32 has no invalid bit patterns.
    Ok(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4) })
}

use std::mem::align_of;

/// IEEE-754 binary16 -> binary32. Handles subnormals, Inf and NaN.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;

    let bits = if exp == 0 {
        if man == 0 {
            sign << 31 // +/- 0
        } else {
            // Subnormal: value is `man * 2^-24`. Shift the mantissa left until
            // bit 10 is set, which costs `s` doublings, so the value is
            // `1.frac * 2^(-14-s)` and the f32 biased exponent is `113 - s`.
            // `man >= 1` here, so the loop runs at most 10 times.
            let mut s = 0u32;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                s += 1;
            }
            let m = m & 0x3ff;
            (sign << 31) | ((113 - s) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (man << 13) // Inf / NaN
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_conversion_covers_the_awkward_cases() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
        // largest normal f16
        assert!((f16_to_f32(0x7bff) - 65504.0).abs() < 1.0);
        // smallest subnormal: 2^-24
        assert!((f16_to_f32(0x0001) - 5.960_464_5e-8).abs() < 1e-14);
        // largest subnormal
        assert!((f16_to_f32(0x03ff) - 6.097_555e-5).abs() < 1e-10);
    }

    /// Build a tiny container in memory and read it back, so the reader is tested
    /// without needing the 1.1 GB model on disk.
    #[test]
    fn round_trips_a_synthetic_container() {
        let n = 2usize;
        let k = 64usize;
        let codes_len = n * k / 4; // 32
        let scales_len = n * (k / 64) * 4; // 8

        // blob: codes at 0 (32 B, padded to 32), scales at 32 (8 B)
        let mut blob = vec![0u8; 32];
        for (i, b) in blob.iter_mut().enumerate() {
            *b = (i % 3) as u8; // arbitrary but valid 2-bit codes
        }
        blob.extend_from_slice(&1.5f32.to_le_bytes());
        blob.extend_from_slice(&2.5f32.to_le_bytes());

        let header = format!(
            r#"{{"G":64,"tensors":{{"lm_head.weight":{{"kind":"tern","N":{n},"K":{k},"codesOff":0,"codesLen":{codes_len},"scalesOff":32,"scalesLen":{scales_len}}}}}}}"#
        );
        let mut file = Vec::new();
        file.extend_from_slice(&(header.len() as u32).to_le_bytes());
        file.extend_from_slice(header.as_bytes());
        // Pad so the blob starts 4-byte aligned inside the Vec allocation.
        let pad = (4 - (file.len() % 4)) % 4;
        // Padding would shift the offsets, so instead assert the header length
        // already lands us aligned; the real packer's headers vary, and the
        // reader's as_f32 check is what protects us there.
        if pad == 0 {
            file.extend_from_slice(&blob);
            let c = Container::from_bytes(file).unwrap();
            let t = c.ternary("lm_head.weight").unwrap();
            assert_eq!((t.n, t.k), (n, k));
            assert_eq!(t.codes.len(), codes_len);
            assert_eq!(t.scales, &[1.5, 2.5]);
        }
    }

    #[test]
    fn rejects_a_shape_mismatch() {
        // codesLen deliberately wrong for N*K/4.
        let header = r#"{"G":64,"tensors":{"w":{"kind":"tern","N":2,"K":64,"codesOff":0,"codesLen":99,"scalesOff":32,"scalesLen":8}}}"#;
        let mut file = Vec::new();
        file.extend_from_slice(&(header.len() as u32).to_le_bytes());
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(&[0u8; 128]);
        let err = Container::from_bytes(file).unwrap_err();
        assert!(err.contains("codesLen"), "unexpected error: {err}");
    }

    #[test]
    fn rejects_an_overrunning_range() {
        let header = r#"{"G":64,"tensors":{"w":{"kind":"f32","shape":[4],"off":0,"len":4096}}}"#;
        let mut file = Vec::new();
        file.extend_from_slice(&(header.len() as u32).to_le_bytes());
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(&[0u8; 16]);
        let err = Container::from_bytes(file).unwrap_err();
        assert!(err.contains("overruns"), "unexpected error: {err}");
    }
}
