const $ = (selector) => document.querySelector(selector);
const input = $("#message"),
  result = $("#result"),
  status = $("#status"),
  copy = $("#copy"),
  send = $("#send");
const examples = [
  {
    mode: "encode",
    icon: "🚆",
    name: "Train of thought",
    text: "I usually [train] before I catch the [train].",
  },
  {
    mode: "decode",
    icon: "🥧",
    name: "Pie, decoded",
    text: "I baked 🍎🥧 yesterday.",
  },
  { mode: "rebus", icon: "🐝", name: "Bee serious", text: "🐝🍃" },
  {
    mode: "encode",
    icon: "🖱️",
    name: "A mouse at work",
    text: "I paired the [mouse] with my new laptop.",
  },
  {
    mode: "emojify",
    icon: "🍕",
    name: "Dinner plans",
    text: "I love pizza and my dog",
  },
  {
    mode: "decode",
    icon: "🦋",
    name: "Butter fingers",
    text: "A 🧈🪰 landed on my hand.",
  },
];
const hints = {
  encode: "Wrap the words you want to translate in [brackets].",
  decode: "Mix emoji and ordinary text in the same sentence.",
  rebus: "Try two emoji that sound like a word when read aloud.",
  emojify: "Write a short sentence for the model to summarize in emoji.",
};
const drafts = {
  encode: "I baked an [apple] [pie].",
  decode: examples[1].text,
  rebus: "🐝🍃",
  emojify: examples[4].text,
};
let mode = "encode", busy = false, worker, offset = 0;
function clearResult() {
  result.textContent = "Your translation will arrive here.";
  copy.disabled = true;
  status.textContent = "";
}
function changeMode(next, text) {
  drafts[mode] = input.value;
  mode = next;
  input.value = text ?? drafts[mode];
  $("#hint").textContent = hints[mode];
  document.querySelectorAll("[data-mode]").forEach((b) =>
    b.setAttribute("aria-pressed", String(b.dataset.mode === mode))
  );
  clearResult();
}
function lock(value) {
  busy = value;
  for (const b of document.querySelectorAll("[data-mode],.card,#random")) {
    b.disabled = value;
  }
  send.disabled = value || !input.value.trim();
  input.readOnly = value;
  $(".delivery").setAttribute("aria-busy", String(value));
}
function fail(message) {
  lock(false);
  clearResult();
  status.textContent = message;
  result.textContent = "Delivery interrupted. Try again.";
}
function translate() {
  if (busy || !input.value.trim()) return;
  lock(true);
  copy.disabled = true;
  result.textContent = "Opening the mail…";
  status.textContent = "Reading your message…";
  try {
    if (!worker) {
      worker = new Worker(new URL("./worker.js", import.meta.url), {
        type: "module",
      });
      worker.onmessage = ({ data }) => {
        if (data.status) {
          status.textContent = data.status;
          return;
        }
        if (data.error) {
          fail(data.error);
          return;
        }
        lock(false);
        result.textContent = data.result ||
          "The model returned an empty reply.";
        copy.disabled = !data.result;
        status.textContent = "Delivered on this device.";
      };
      worker.onerror = () => {
        worker.terminate();
        worker = null;
        fail(
          "Could not start the translator. Check the local model and runtime files.",
        );
      };
    }
    worker.postMessage({ mode, text: input.value });
  } catch (error) {
    fail(error.message);
  }
}
function cards() {
  $("#cards").replaceChildren();
  for (let i = 0; i < 3; i++) {
    const sample = examples[(offset + i) % examples.length],
      b = document.createElement("button");
    b.className = "card";
    const icon = document.createElement("span");
    icon.className = "pic";
    icon.textContent = sample.icon;
    icon.ariaHidden = "true";
    const label = document.createElement("span"),
      title = document.createElement("strong"),
      kind = document.createElement("small");
    title.textContent = sample.name;
    kind.textContent = sample.mode;
    label.append(title, kind);
    const arrow = document.createElement("span");
    arrow.className = "arrow";
    arrow.textContent = "↗";
    arrow.ariaHidden = "true";
    b.append(icon, label, arrow);
    b.onclick = () => {
      changeMode(sample.mode, sample.text);
      translate();
    };
    $("#cards").append(b);
  }
}
document.querySelectorAll("[data-mode]").forEach((b) =>
  b.onclick = () => {
    changeMode(b.dataset.mode);
    lock(false);
    input.focus();
  }
);
send.onclick = translate;
input.oninput = () => {
  clearResult();
  lock(false);
};
input.onkeydown = (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    translate();
  }
};
copy.onclick = async () => {
  try {
    await navigator.clipboard.writeText(result.textContent);
    status.textContent = "Copied to clipboard.";
  } catch {
    const r = document.createRange();
    r.selectNodeContents(result);
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(r);
    status.textContent = "Result selected. Use your device’s copy command.";
  }
};
$("#random").onclick = () => {
  offset = (offset + 3) % examples.length;
  cards();
};
input.value = drafts.encode;
cards();
translate();
