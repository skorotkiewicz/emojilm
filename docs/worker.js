import * as ort from "./vendor/ort.wasm.min.mjs";
import { Translator } from "./translator.js";

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
let translator, session;
async function forward(ids, output) {
  if (!session) {
    self.postMessage({ status: "Loading neural weights…" });
    session = await ort.InferenceSession.create(
      new URL("./model/model.onnx", import.meta.url).href,
      { executionProviders: ["wasm"] },
    );
  }
  self.postMessage({ status: "Translating on your device…" });
  const input = new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [
    1,
    ids.length,
  ]);
  const result = await session.run({ ids: input }, [output]);
  return result[output].data;
}
self.onmessage = async ({ data: { mode, text } }) => {
  try {
    if (!translator) {
      const res = await fetch(new URL("./model/bundle.json", import.meta.url));
      if (!res.ok) throw Error("Could not load the translation bundle.");
      translator = new Translator(await res.json(), forward);
    }
    self.postMessage({ result: await translator.translate(mode, text) });
  } catch (e) {
    self.postMessage({ error: e.message || String(e) });
  }
};
