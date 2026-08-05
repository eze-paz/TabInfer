//! A minimal JSON reader, just enough for a `.cpu.bin` header.
//!
//! Hand-rolled to keep the workspace dependency-free (see the note in the root
//! `Cargo.toml`). It is not a general-purpose parser: no escape sequences beyond
//! the common ones, no big-number edge cases. It is, however, strict — it errors
//! rather than guessing, because a silently mis-parsed header would show up as
//! plausible garbage at inference time, which is the single hardest class of bug
//! to find in this codebase's history.

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Value>),
    Obj(BTreeMap<String, Value>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Obj(m) => m.get(key),
            _ => None,
        }
    }

    pub fn as_obj(&self) -> Option<&BTreeMap<String, Value>> {
        match self {
            Value::Obj(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Value::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some(*n as usize),
            _ => None,
        }
    }

    /// `self[key]` as a usize, with a message naming the key on failure.
    pub fn usize_field(&self, key: &str) -> Result<usize, String> {
        self.get(key)
            .ok_or_else(|| format!("missing field `{key}`"))?
            .as_usize()
            .ok_or_else(|| format!("field `{key}` is not a non-negative integer"))
    }

    pub fn usize_array(&self, key: &str) -> Result<Vec<usize>, String> {
        match self.get(key) {
            Some(Value::Arr(a)) => a
                .iter()
                .map(|v| v.as_usize().ok_or_else(|| format!("`{key}` has a non-integer entry")))
                .collect(),
            Some(_) => Err(format!("field `{key}` is not an array")),
            None => Err(format!("missing field `{key}`")),
        }
    }
}

pub fn parse(s: &str) -> Result<Value, String> {
    let b = s.as_bytes();
    let mut p = Parser { b, i: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != b.len() {
        return Err(format!("trailing input at byte {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn peek(&self) -> Result<u8, String> {
        self.b.get(self.i).copied().ok_or_else(|| "unexpected end of input".to_string())
    }

    fn eat(&mut self, c: u8) -> Result<(), String> {
        if self.peek()? != c {
            return Err(format!(
                "expected `{}` at byte {}, found `{}`",
                c as char, self.i, self.b[self.i] as char
            ));
        }
        self.i += 1;
        Ok(())
    }

    fn lit(&mut self, word: &str, v: Value) -> Result<Value, String> {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(format!("invalid literal at byte {}", self.i))
        }
    }

    fn value(&mut self) -> Result<Value, String> {
        match self.peek()? {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(Value::Str(self.string()?)),
            b't' => self.lit("true", Value::Bool(true)),
            b'f' => self.lit("false", Value::Bool(false)),
            b'n' => self.lit("null", Value::Null),
            _ => self.number(),
        }
    }

    fn object(&mut self) -> Result<Value, String> {
        self.eat(b'{')?;
        let mut m = BTreeMap::new();
        self.ws();
        if self.peek()? == b'}' {
            self.i += 1;
            return Ok(Value::Obj(m));
        }
        loop {
            self.ws();
            let k = self.string()?;
            self.ws();
            self.eat(b':')?;
            self.ws();
            let v = self.value()?;
            m.insert(k, v);
            self.ws();
            match self.peek()? {
                b',' => self.i += 1,
                b'}' => {
                    self.i += 1;
                    return Ok(Value::Obj(m));
                }
                c => return Err(format!("expected `,` or `}}` at byte {}, found `{}`", self.i, c as char)),
            }
        }
    }

    fn array(&mut self) -> Result<Value, String> {
        self.eat(b'[')?;
        let mut a = Vec::new();
        self.ws();
        if self.peek()? == b']' {
            self.i += 1;
            return Ok(Value::Arr(a));
        }
        loop {
            self.ws();
            a.push(self.value()?);
            self.ws();
            match self.peek()? {
                b',' => self.i += 1,
                b']' => {
                    self.i += 1;
                    return Ok(Value::Arr(a));
                }
                c => return Err(format!("expected `,` or `]` at byte {}, found `{}`", self.i, c as char)),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut s = String::new();
        loop {
            let c = self.peek()?;
            self.i += 1;
            match c {
                b'"' => return Ok(s),
                b'\\' => {
                    let e = self.peek()?;
                    self.i += 1;
                    match e {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'n' => s.push('\n'),
                        b't' => s.push('\t'),
                        b'r' => s.push('\r'),
                        b'b' => s.push('\u{8}'),
                        b'f' => s.push('\u{c}'),
                        b'u' => {
                            let h = self
                                .b
                                .get(self.i..self.i + 4)
                                .ok_or_else(|| "truncated \\u escape".to_string())?;
                            let n = u32::from_str_radix(
                                std::str::from_utf8(h).map_err(|_| "bad \\u escape")?,
                                16,
                            )
                            .map_err(|_| "bad \\u escape")?;
                            self.i += 4;
                            // Surrogate pairs are not expected in these headers
                            // (tensor names are ASCII); refuse rather than mangle.
                            s.push(
                                char::from_u32(n)
                                    .ok_or_else(|| "unsupported \\u code point".to_string())?,
                            );
                        }
                        _ => return Err(format!("bad escape `\\{}`", e as char)),
                    }
                }
                _ => s.push(c as char),
            }
        }
    }

    fn number(&mut self) -> Result<Value, String> {
        let start = self.i;
        if self.peek()? == b'-' {
            self.i += 1;
        }
        while self.i < self.b.len()
            && matches!(self.b[self.i], b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')
        {
            self.i += 1;
        }
        let t = std::str::from_utf8(&self.b[start..self.i]).map_err(|_| "bad number")?;
        t.parse::<f64>()
            .map(Value::Num)
            .map_err(|_| format!("bad number `{t}` at byte {start}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_header_shaped_document() {
        let v = parse(
            r#"{"G":64,"tensors":{"model.embed_tokens.weight":{"kind":"embed_f16","shape":[151936,2048],"off":0,"len":622329856},
               "lm_head.weight":{"kind":"tern","N":151936,"K":2048,"codesOff":16,"codesLen":32,"scalesOff":64,"scalesLen":8}}}"#,
        )
        .unwrap();
        assert_eq!(v.usize_field("G").unwrap(), 64);
        let t = v.get("tensors").unwrap().as_obj().unwrap();
        assert_eq!(t.len(), 2);
        let e = &t["model.embed_tokens.weight"];
        assert_eq!(e.get("kind").unwrap().as_str().unwrap(), "embed_f16");
        assert_eq!(e.usize_array("shape").unwrap(), vec![151936, 2048]);
        assert_eq!(t["lm_head.weight"].usize_field("K").unwrap(), 2048);
    }

    #[test]
    fn rejects_rather_than_guesses() {
        assert!(parse("{\"a\":}").is_err());
        assert!(parse("{\"a\":1,}").is_err());
        assert!(parse("{\"a\":1} junk").is_err());
        assert!(parse("").is_err());
    }

    #[test]
    fn negative_and_fractional_are_not_usize() {
        let v = parse(r#"{"a":-1,"b":1.5,"c":3}"#).unwrap();
        assert!(v.usize_field("a").is_err());
        assert!(v.usize_field("b").is_err());
        assert_eq!(v.usize_field("c").unwrap(), 3);
    }
}
