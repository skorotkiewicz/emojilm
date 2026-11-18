"""Export an EmojiLM checkpoint for ONNX Runtime Web and verify it against PyTorch."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emojilm.config import Config
from emojilm.model import EmojiLM


class BrowserModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids):
        hidden = self.model.hidden(ids)
        return hidden, F.linear(hidden, self.model.tok_emb.weight)


def tensor_to_list(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    raise TypeError(f"Unsupported metadata: {type(value).__name__}")


def quantize_weights(path):
    """Store each matrix as signed int8 with one floating-point scale per row."""
    graph_model = onnx.load(str(path))
    graph = graph_model.graph
    initializers = []
    dequantize = []
    for weight in graph.initializer:
        if len(weight.dims) != 2 or weight.data_type != onnx.TensorProto.FLOAT:
            initializers.append(weight)
            continue
        matrix = onnx.numpy_helper.to_array(weight)
        scale = np.maximum(np.abs(matrix).max(axis=1) / 127, 1e-30)
        quantized = np.rint(matrix / scale[:, None]).clip(-127, 127).astype(np.int8)
        zero = np.zeros_like(scale, dtype=np.int8)
        inputs = []
        for suffix, values in (("int8", quantized), ("scale", scale), ("zero", zero)):
            name = f"{weight.name}.{suffix}"
            initializers.append(onnx.numpy_helper.from_array(values, name))
            inputs.append(name)
        dequantize.append(
            onnx.helper.make_node("DequantizeLinear", inputs, [weight.name], axis=0)
        )
    if not dequantize:
        raise ValueError("ONNX graph contains no float32 weight matrices")
    # Keep the original tensor names so tied weights remain shared.
    nodes = dequantize + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    onnx.checker.check_model(graph_model)
    onnx.save(graph_model, str(path))


def verify(path, browser, cfg, int8=False):
    onnx.checker.check_model(str(path))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(21)
    for length in (1, 17, cfg.ctx):
        ids = rng.integers(0, cfg.vocab, size=(1, length), dtype=np.int64)
        with torch.no_grad():
            expected = browser(torch.from_numpy(ids))
        actual = session.run(["hidden", "logits"], {"ids": ids})
        for name, array, tensor in zip(("hidden", "logits"), actual, expected):
            reference = tensor.numpy()
            if array.shape != reference.shape or not np.isfinite(array).all():
                raise ValueError(f"Invalid {name} output at input length {length}")
            if int8:
                relative_error = np.linalg.norm(array - reference) / np.linalg.norm(
                    reference
                )
                if relative_error > 0.10:
                    raise ValueError(
                        f"{name} relative L2 error is {relative_error:.3f}"
                    )
                print(
                    f"length={length} {name} int8_relative_error={relative_error:.3g}"
                )
            else:
                np.testing.assert_allclose(array, reference, rtol=2e-4, atol=2e-4)
                error = np.max(np.abs(array - reference))
                print(f"length={length} {name} max_abs_error={error:.2g}")


def export(checkpoint, output, int8=False):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format") != "emojilm-1":
        raise ValueError("Expected an EmojiLM checkpoint")
    cfg = Config.from_dict(saved["cfg"])
    model = EmojiLM(cfg)
    model.load_state_dict(saved["model"])
    browser = BrowserModel(model.eval()).eval()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as directory:
        temp = Path(directory)
        path = temp / "model.onnx"
        with torch.no_grad():
            torch.onnx.export(
                browser,
                torch.tensor([[256, 259, 262]], dtype=torch.int64),
                path,
                dynamo=False,
                opset_version=17,
                do_constant_folding=False,
                input_names=["ids"],
                output_names=["hidden", "logits"],
                dynamic_axes={
                    name: {1: "sequence"} for name in ("ids", "hidden", "logits")
                },
            )
        verify(path, browser, cfg)
        if int8:
            quantize_weights(path)
            verify(path, browser, cfg, int8=True)
        bundle = {
            key: saved[key]
            for key in (
                "cfg",
                "tokenizer",
                "edit_candidates",
                "decode_readings",
                "sense_heads",
                "noun_verb_gate",
            )
            if key in saved
        }
        metadata = temp / "bundle.json"
        metadata.write_text(
            json.dumps(
                bundle,
                ensure_ascii=False,
                separators=(",", ":"),
                default=tensor_to_list,
            ),
            encoding="utf-8",
        )
        path.replace(output / path.name)
        metadata.replace(output / metadata.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("ckpt/final/model.pt"))
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "model"
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Quantize matrix weights for a smaller download",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    export(args.ckpt, args.out, args.int8)
