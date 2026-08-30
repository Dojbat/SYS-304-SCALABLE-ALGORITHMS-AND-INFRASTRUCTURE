const MAX_CHARS = 280;

const input = document.getElementById("tweet-input");
const button = document.getElementById("classify-btn");
const charCount = document.getElementById("char-count");
const resultBox = document.getElementById("result");
const tweetEcho = document.getElementById("tweet-echo");
const labelEl = document.getElementById("label");
const confidenceEl = document.getElementById("confidence");
const errorBox = document.getElementById("error");

function updateCharCount() {
  const remaining = MAX_CHARS - input.value.length;
  charCount.textContent = remaining;
  charCount.classList.toggle("warn", remaining <= 20 && remaining >= 0);
  charCount.classList.toggle("over", remaining < 0);
  button.disabled = input.value.trim().length === 0 || remaining < 0;
}

async function classify() {
  const text = input.value.trim();
  if (!text) return;

  button.disabled = true;
  hide(resultBox);
  hide(errorBox);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      const detail = await safeJson(response);
      throw new Error(detail?.detail ? JSON.stringify(detail.detail) : `Request failed (${response.status})`);
    }

    const data = await response.json();
    renderResult(data);
    input.value = "";
    updateCharCount();
  } catch (err) {
    showError(err.message || "Couldn't reach the classifier. Is the backend running?");
  } finally {
    updateCharCount();
  }
}

function renderResult(data) {
  tweetEcho.textContent = data.text;
  const isDisaster = data.prediction === 1;
  labelEl.textContent = isDisaster ? "🚨 Disaster" : "✅ Not a disaster";
  labelEl.className = "verdict-chip " + (isDisaster ? "disaster" : "safe");
  labelEl.dataset.label = data.label;
  confidenceEl.textContent = (data.confidence * 100).toFixed(1) + "%";
  show(resultBox);
}

function showError(message) {
  errorBox.textContent = message;
  show(errorBox);
}

function show(el) {
  el.hidden = false;
}

function hide(el) {
  el.hidden = true;
}

async function safeJson(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

button.addEventListener("click", classify);
input.addEventListener("input", updateCharCount);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    classify();
  }
});

updateCharCount();
