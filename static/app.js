const btn = document.getElementById("translateBtn");
const emojiInput = document.getElementById("emojiInput");
const instructions = document.getElementById("instructions");
const resultWrap = document.getElementById("resultWrap");
const resultEl = document.getElementById("result");

async function doTranslate() {
  const emoji_sequence = (emojiInput.value || "").trim();
  const user_instructions = (instructions.value || "").trim();

  resultWrap.classList.remove("hidden");
  resultEl.textContent = "Translating…";
  btn.disabled = true;

  try {
    const res = await fetch("/api/translate", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        emoji_sequence,
        instructions: user_instructions
      })
    });

    const data = await res.json();
    resultEl.textContent = data.ok ? data.result : (data.result || "Error");
  } catch (e) {
    resultEl.textContent = "Error: could not reach backend.";
  } finally {
    btn.disabled = false;
  }
}

btn.addEventListener("click", doTranslate);
emojiInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") doTranslate();
});
