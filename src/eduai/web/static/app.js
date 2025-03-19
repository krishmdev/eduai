// Keyboard shortcuts on the question card: A-D choose, Enter submits (native), N = next question.
document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "select" || tag === "textarea" || (tag === "input" && e.target.type === "text")) return;
  const key = e.key.toUpperCase();
  if ("ABCD".includes(key) && key.length === 1) {
    const input = document.querySelector(`#question input[name="choice"][value="${key}"]`);
    if (input) { input.checked = true; input.focus(); e.preventDefault(); }
  } else if (key === "N") {
    const next = document.querySelector("#question button[hx-get]");
    if (next) { next.click(); e.preventDefault(); }
  }
});
document.addEventListener("htmx:afterSwap", (e) => {
  if (e.target.id === "stage") {
    const focus = document.querySelector("#question [autofocus], #question input[name='choice']");
    if (focus) focus.focus({ preventScroll: true });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
});
