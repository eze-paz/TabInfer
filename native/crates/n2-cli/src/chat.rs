//! Tokenisation and the chat template.
//!
//! Kept in the CLI so the engine crates stay dependency-free. See the note in
//! `Cargo.toml` for why this is the one place a dependency is allowed.

use tokenizers::Tokenizer;

/// `<|im_end|>` and `<|endoftext|>`: generation stops on either.
pub const IM_END: u32 = 151645;
pub const END_OF_TEXT: u32 = 151643;

pub struct Chat {
    tok: Tokenizer,
}

impl Chat {
    pub fn open(path: &str) -> Result<Self, String> {
        let tok = Tokenizer::from_file(path)
            .map_err(|e| format!("{path}: could not load tokenizer: {e}"))?;
        Ok(Chat { tok })
    }

    /// Wrap a user message in ChatML and encode it.
    ///
    /// The template is spelled out rather than read from `chat_template.jinja`
    /// because it is one line and pulling in a Jinja engine to render it would be
    /// absurd. It is validated by [`Chat::self_check`], which requires that this
    /// exact template round-trips to the probe ids in `ref_forward.py`.
    pub fn encode_user(&self, msg: &str) -> Result<Vec<u32>, String> {
        let text = format!(
            "<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
        );
        self.encode_raw(&text)
    }

    pub fn encode_raw(&self, text: &str) -> Result<Vec<u32>, String> {
        let enc = self
            .tok
            .encode(text, false)
            .map_err(|e| format!("encode failed: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode ids to text, keeping special tokens so `<think>` blocks stay visible.
    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.tok
            .decode(ids, false)
            .map_err(|e| format!("decode failed: {e}"))
    }

    /// Decode dropping special tokens, for the clean answer text.
    pub fn decode_clean(&self, ids: &[u32]) -> Result<String, String> {
        self.tok
            .decode(ids, true)
            .map_err(|e| format!("decode failed: {e}"))
    }

    /// Verify the tokenizer against a known-good encoding.
    ///
    /// `ref_forward.py` hardcodes the ids for ChatML-wrapped "The capital of France
    /// is". If this tokenizer plus this chat template reproduce those ids exactly,
    /// then the byte-level alphabet, the merge ranks, the special tokens and the
    /// template are all right. That is a real check, not a smoke test — and it is
    /// why the probe ids were worth keeping around.
    pub fn self_check(&self, expected: &[u32]) -> Result<(), String> {
        let got = self.encode_user("The capital of France is")?;
        if got == expected {
            return Ok(());
        }
        Err(format!(
            "tokenizer self-check failed.\n  expected {expected:?}\n  got      {got:?}\n\
             The byte-level alphabet, merge ranks, special tokens or the chat\n\
             template disagree with tools/ref_forward.py. Do not trust generated text."
        ))
    }
}
