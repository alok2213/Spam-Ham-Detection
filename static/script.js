/*
  Ye file browser me chalti hai. Kaam:
  1. User ka message lena
  2. Flask ke /predict route ko bhejna (fetch)
  3. Flask ka JSON jawab screen par dikhana
*/

// Shortcut: $("id") likhne se element mil jaata hai
const $ = (id) => document.getElementById(id);

const message = $("message");
const checkBtn = $("check");

// ---------- Page ke buttons ----------

// Character counter
message.addEventListener("input", () => {
  $("count").textContent = message.value.length + " / 2000";
});

// Ctrl+Enter se bhi check ho jaye
message.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) checkMessage();
});

checkBtn.addEventListener("click", checkMessage);

// Clear button: sab reset
$("clear").addEventListener("click", () => {
  message.value = "";
  $("count").textContent = "0 / 2000";
  $("result").hidden = true;
  $("empty").hidden = false;
  $("error").hidden = true;
  message.focus();
});

// Example buttons: text daalo aur check karo
document.querySelectorAll(".example").forEach((btn) => {
  btn.addEventListener("click", () => {
    message.value = btn.dataset.text;
    $("count").textContent = message.value.length + " / 2000";
    checkMessage();
  });
});

// ---------- Flask ko call karna ----------

async function checkMessage() {
  $("error").hidden = true;
  checkBtn.disabled = true;
  checkBtn.textContent = "Checking...";

  try {
    // POST request -> app.py ka predict() function chalta hai
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: message.value }),
    });

    const data = await response.json();   // Flask ka jsonify(...) yaha milta hai

    if (!response.ok) {
      throw new Error(data.error || "Something went wrong.");
    }
    showResult(data);
  } catch (err) {
    $("error").textContent = err.message;
    $("error").hidden = false;
  } finally {
    checkBtn.disabled = false;
    checkBtn.textContent = "Check message";
  }
}

// ---------- Result screen par dikhana ----------
// data = { label, spam_probability, confidence, top_words, processed_text }

function showResult(data) {
  const isSpam = data.label === "spam";

  $("empty").hidden = true;
  $("result").hidden = false;

  // Verdict
  const verdict = $("verdict");
  verdict.textContent = isSpam ? "Spam" : "Not spam";
  verdict.className = "verdict " + (isSpam ? "spam" : "ham");
  $("confidence").textContent = "Model confidence: " + data.confidence.toFixed(1) + "%";

  // Probability bar
  $("percent").textContent = data.spam_probability.toFixed(1) + "%";
  const fill = $("fill");
  fill.className = "fill " + (isSpam ? "spam" : "ham");
  fill.style.width = data.spam_probability + "%";

  // Influencing words (weight > 0 = spam ki taraf, < 0 = not spam ki taraf)
  const words = data.top_words.filter((w) => w.weight !== 0);
  $("wordsBox").hidden = words.length === 0;
  $("words").innerHTML = "";
  words.forEach((w) => {
    const tag = document.createElement("span");
    tag.className = "word " + (w.weight > 0 ? "spam" : "ham");
    tag.textContent = w.word;
    $("words").appendChild(tag);
  });
}
