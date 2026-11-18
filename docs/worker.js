import * as ort from "./vendor/ort.wasm.min.mjs";
import { Translator } from "./translator.js";

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
let translator, session;
async function forward(ids, output) {
  self.postMessage({ status: "Translating on your device…" });
  const input = new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [
    1,
    ids.length,
  ]);
  const result = await session.run({ ids: input }, [output]);
  return result[output].data;
}
self.onmessage = async ({ data: { type, mode, text } }) => {
  try {
    if (type === "load") {
      if (!translator) {
        const res = await fetch(
          new URL("./model/bundle.json", import.meta.url),
        );
        if (!res.ok) throw Error("Could not load the translation bundle.");
        const bundle = await res.json();
        session = await ort.InferenceSession.create(
          new URL("./model/model.onnx", import.meta.url).href,
          { executionProviders: ["wasm"] },
        );
        translator = new Translator(bundle, forward);
      }
      self.postMessage({ ready: true });
      return;
    }
    self.postMessage({ result: await translator.translate(mode, text) });
  } catch (e) {
    self.postMessage({ error: e.message || String(e) });
  }
};
