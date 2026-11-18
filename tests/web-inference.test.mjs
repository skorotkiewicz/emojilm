import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import * as ort from "../docs/vendor/ort.wasm.min.mjs";
import { Translator } from "../docs/translator.js";

test("the shipped ONNX and browser bundle translate together", async () => {
  ort.env.wasm.numThreads = 1;
  const path = (name) => new URL("../docs/model/" + name, import.meta.url);
  const session = await ort.InferenceSession.create(
    await readFile(path("model.onnx")),
  );
  try {
    const bundle = JSON.parse(await readFile(path("bundle.json"), "utf8"));
    const translator = new Translator(bundle, async (ids, output) => {
      const input = new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [
        1,
        ids.length,
      ]);
      return (await session.run({ ids: input }, [output]))[output].data;
    });
    for (
      const [mode, message, expected] of [
        [
          "encode",
          "I usually [train] before I catch the [train].",
          "I usually [train] before I catch the 🚆.",
        ],
        [
          "encode",
          "I paired the [mouse] with my new laptop.",
          "I paired the 🖱 with my new laptop.",
        ],
        ["decode", "I baked 🍎🥧 yesterday.", "I baked apple pie yesterday."],
        ["rebus", "🐝🍃", "believe"],
        ["emojify", "I love pizza and my dog", "🍕❤️"],
      ]
    ) {
      assert.equal(
        await translator.translate(mode, message),
        expected,
        message,
      );
    }
  } finally {
    await session.release();
  }
});
