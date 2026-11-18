use anyhow::{Context, Result, bail, ensure};
use candle_core::{
    DType, Tensor,
    pickle::{Object, Stack},
};
use serde_json::{Map, Value};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read},
    path::Path,
};
use zip::ZipArchive;

pub struct Checkpoint {
    pub meta: Value,
    pub tensors: HashMap<String, Tensor>,
}

impl Checkpoint {
    pub fn open(path: &Path) -> Result<Self> {
        let mut zip = ZipArchive::new(BufReader::new(File::open(path)?))?;
        let name = zip
            .file_names()
            .find(|n| n.ends_with("/data.pkl"))
            .context("missing PyTorch metadata")?
            .to_owned();
        let base = name
            .strip_suffix("data.pkl")
            .context("invalid archive layout")?;
        if let Ok(mut file) = zip.by_name(&format!("{base}byteorder")) {
            let mut endian = String::new();
            file.read_to_string(&mut endian)?;
            ensure!(endian == "little", "expected little-endian weights");
        }
        let mut stack = Stack::empty();
        stack.read_loop(&mut BufReader::new(zip.by_name(&name)?))?;
        let root = stack.finalize()?;
        let mut tensors = HashMap::new();
        let meta = unpack(root, "", &mut zip, &format!("{base}data"), &mut tensors)?;
        ensure!(
            meta["format"] == "emojilm-1",
            "expected an EmojiLM checkpoint"
        );
        Ok(Self { meta, tensors })
    }
}

// Candle parses pickle as data. Only tensor descriptions are materialized from storage.
fn unpack(
    obj: Object,
    key: &str,
    zip: &mut ZipArchive<BufReader<File>>,
    dir: &str,
    tensors: &mut HashMap<String, Tensor>,
) -> Result<Value> {
    Ok(match obj {
        Object::Dict(entries) => {
            let mut map = Map::new();
            for (k, v) in entries {
                let Object::Unicode(k) = k else {
                    bail!("unexpected dictionary key in {key}")
                };
                let full = if key.is_empty() {
                    k.clone()
                } else {
                    format!("{key}.{k}")
                };
                map.insert(k, unpack(v, &full, zip, dir, tensors)?);
            }
            Value::Object(map)
        }
        Object::Tuple(v) | Object::List(v) => Value::Array(
            v.into_iter()
                .map(|v| unpack(v, key, zip, dir, tensors))
                .collect::<Result<_>>()?,
        ),
        Object::Int(i) => i.into(),
        Object::Float(f) => f.into(),
        Object::Bool(b) => b.into(),
        Object::Unicode(s) => s.into(),
        Object::None => Value::Null,
        Object::Reduce { .. } => {
            let info = obj
                .into_tensor_info(Object::Unicode(key.into()), Path::new(dir))?
                .context("unrecognized tensor metadata")?;
            ensure!(
                info.dtype == DType::F32
                    && info.layout.is_contiguous()
                    && info.layout.start_offset() == 0,
                "unsupported tensor layout or dtype for {key}"
            );
            let mut file = zip.by_name(&info.path)?;
            let count = info
                .layout
                .dims()
                .iter()
                .try_fold(1usize, |a, &b| a.checked_mul(b))
                .context("tensor shape overflow")?;
            ensure!(
                count.checked_mul(4).map(|n| n as u64) == Some(file.size()),
                "tensor size mismatch for {key}"
            );
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)?;
            tensors.insert(
                key.into(),
                Tensor::from_raw_buffer(
                    &bytes,
                    info.dtype,
                    info.layout.dims(),
                    &candle_core::Device::Cpu,
                )?,
            );
            Value::Null
        }
        _ => bail!("unsupported checkpoint metadata at {key}"),
    })
}
