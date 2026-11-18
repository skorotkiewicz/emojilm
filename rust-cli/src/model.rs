use anyhow::{Context, Result, ensure};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::ops;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
pub struct Config {
    pub dim: usize,
    pub ctx: usize,
    pub heads: usize,
    pub layers: usize,
    pub vocab: usize,
}
pub struct Model {
    pub cfg: Config,
    pub weights: HashMap<String, Tensor>,
}

impl Model {
    pub fn weight(&self, key: &str) -> Result<&Tensor> {
        self.weights
            .get(key)
            .with_context(|| format!("missing tensor {key}"))
    }
    fn linear(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        Ok(x.matmul(&self.weight(&format!("{name}.weight"))?.t()?)?)
    }
    fn norm(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        Ok(ops::layer_norm(
            x,
            self.weight(&format!("{name}.weight"))?,
            self.weight(&format!("{name}.bias"))?,
            1e-5,
        )?)
    }
    pub fn hidden(&self, ids: &[u32]) -> Result<Tensor> {
        let n = ids.len();
        let c = &self.cfg;
        ensure!(
            n > 0 && n <= c.ctx,
            "prompt needs {n} tokens; context is {}",
            c.ctx
        );
        ensure!(
            c.heads > 0 && c.dim.is_multiple_of(c.heads),
            "invalid attention dimensions"
        );
        let width = c.dim / c.heads;
        let mut x = self
            .weight("model.tok_emb.weight")?
            .index_select(&Tensor::new(ids, &Device::Cpu)?, 0)?
            .add(&self.weight("model.pos_emb.weight")?.narrow(0, 0, n)?)?;
        let mask = Tensor::tril2(n, DType::U8, &Device::Cpu)?.broadcast_as((c.heads, n, n))?;
        let neg = Tensor::full(f32::NEG_INFINITY, (c.heads, n, n), &Device::Cpu)?;
        for i in 0..c.layers {
            let b = format!("model.blocks.{i}");
            let qkv = self.linear(
                &self.norm(&x, &format!("{b}.ln1"))?,
                &format!("{b}.attn.qkv"),
            )?;
            let heads = |offset| {
                qkv.narrow(1, offset, c.dim)?
                    .reshape((n, c.heads, width))?
                    .transpose(0, 1)?
                    .contiguous()
            };
            let (q, k, v) = (heads(0)?, heads(c.dim)?, heads(2 * c.dim)?);
            let scores = q
                .matmul(&k.transpose(1, 2)?)?
                .affine(1.0 / (width as f64).sqrt(), 0.)?;
            let probs = ops::softmax_last_dim(&mask.where_cond(&scores, &neg)?)?;
            let y = probs
                .matmul(&v)?
                .transpose(0, 1)?
                .contiguous()?
                .reshape((n, c.dim))?;
            x = x.add(&self.linear(&y, &format!("{b}.attn.proj"))?)?;
            let y = self
                .linear(&self.norm(&x, &format!("{b}.ln2"))?, &format!("{b}.mlp.fc"))?
                .gelu_erf()?;
            x = x.add(&self.linear(&y, &format!("{b}.mlp.proj"))?)?;
        }
        self.norm(&x, "model.ln_f")
    }
    pub fn logits(&self, h: &Tensor) -> Result<Tensor> {
        self.linear(h, "model.tok_emb")
    }

    pub fn score(&self, prefix: &[u32], target: &[u32]) -> Result<f32> {
        let mut ids = prefix.to_vec();
        ids.extend_from_slice(&target[..target.len() - 1]);
        let hidden = self
            .hidden(&ids)?
            .narrow(0, prefix.len() - 1, target.len())?;
        let logp = ops::log_softmax(&self.logits(&hidden)?, D::Minus1)?.to_vec2::<f32>()?;
        Ok(logp
            .iter()
            .zip(target)
            .map(|(row, &id)| row[id as usize])
            .sum())
    }
}
