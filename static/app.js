document.addEventListener("DOMContentLoaded", () => {
  console.log("✅ app.js loaded");

  const btn = document.getElementById("translateBtn");
  const emojiInput = document.getElementById("emojiInput");
  const resultWrap = document.getElementById("resultWrap");
  const resultEl = document.getElementById("result");

  if (!btn || !emojiInput || !resultWrap || !resultEl) {
    console.error("❌ Missing element(s). Check IDs in index.html.", {
      btn, emojiInput, resultWrap, resultEl
    });
    return;
  }

  async function doTranslate() {
    const emoji_sequence = (emojiInput.value || "").trim();

    resultWrap.classList.remove("hidden");
    resultEl.textContent = "Translating…";
    btn.disabled = true;

    try {
      const res = await fetch("/api/translate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ emoji_sequence })
      });

      const data = await res.json();
      resultEl.textContent = data.ok ? data.result : (data.result || "Error");
    } catch (e) {
      console.error("❌ Fetch failed:", e);
      resultEl.textContent = "Error: could not reach backend.";
    } finally {
      btn.disabled = false;
    }
  }

  btn.addEventListener("click", doTranslate);
  emojiInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") doTranslate();
  });

  console.log("✅ Handlers attached");
});
