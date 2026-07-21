// Keyboard: A-D choose an option, Enter submits (native form behavior), N or Enter continues.
function focusQuestion() {
  const target = document.querySelector("#question #next-action, #question input[name='choice']");
  if (target) target.focus({ preventScroll: true });
}

document.addEventListener("DOMContentLoaded", focusQuestion);

document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "select" || tag === "textarea" || (tag === "input" && e.target.type === "text")) return;
  const key = e.key.length === 1 ? e.key.toUpperCase() : e.key;
  const next = document.querySelector("#question #next-action");
  if (next && (key === "N" || (key === "Enter" && e.target !== next))) {
    // The button may have just been swapped in; make sure htmx has wired it up first.
    if (window.htmx && next.hasAttribute("hx-get")) window.htmx.process(next);
    next.click();
    e.preventDefault();
    return;
  }
  if ("ABCD".includes(key) && key.length === 1) {
    const input = document.querySelector(`#question input[name="choice"][value="${key}"]`);
    if (input) {
      input.checked = true;
      input.focus();
      e.preventDefault();
    }
  }
});

document.addEventListener("htmx:afterSwap", (e) => {
  if (e.target.id === "stage") {
    focusQuestion();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
});

// Network and server errors: say so above the question and let the student try again.
function showError(evt, message) {
  const slot = document.getElementById("notice");
  if (slot) {
    slot.innerHTML = "";
    const p = document.createElement("p");
    p.className = "notice notice-error";
    p.textContent = message;
    slot.appendChild(p);
  }
  document.querySelectorAll("#question button, #question input").forEach((el) => {
    el.disabled = false;
  });
  const form = evt.detail && evt.detail.elt && evt.detail.elt.closest("form");
  if (form) form.removeAttribute("aria-busy");
}

document.addEventListener("htmx:sendError", (e) =>
  showError(e, "Couldn't reach the server. Your answer wasn't saved. Try again."));
document.addEventListener("htmx:responseError", (e) =>
  showError(e, "Something went wrong on the server. Your answer wasn't saved. Try again."));
document.addEventListener("htmx:beforeRequest", (e) => {
  const slot = document.getElementById("notice");
  if (slot) slot.innerHTML = "";
  const form = e.detail.elt.closest && e.detail.elt.closest("form");
  if (form) form.setAttribute("aria-busy", "true");
});

// Chart points: one tab stop, arrow keys move between points (roving tabindex).
document.addEventListener("keydown", (e) => {
  const pt = e.target.closest && e.target.closest(".chart .pt");
  if (!pt || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
  const pts = [...pt.parentNode.querySelectorAll(".pt")];
  let i = pts.indexOf(pt);
  i = e.key === "Home" ? 0 : e.key === "End" ? pts.length - 1 : i + (e.key === "ArrowRight" ? 1 : -1);
  i = Math.max(0, Math.min(pts.length - 1, i));
  pts.forEach((p, k) => p.setAttribute("tabindex", k === i ? "0" : "-1"));
  pts[i].focus();
  e.preventDefault();
});

// Small screens: the mastery panel starts collapsed so the question stays in view. The sidebar is
// re-rendered after every answer, so the student's choice is remembered here.
let masteryOpen = null;
function compactMastery() {
  if (!window.matchMedia("(max-width: 900px)").matches) return;
  document.querySelectorAll("details.mastery").forEach((d) => {
    d.open = masteryOpen === true;
  });
}
document.addEventListener("DOMContentLoaded", compactMastery);
document.addEventListener("htmx:afterSettle", compactMastery);
document.addEventListener("click", (e) => {
  const summary = e.target.closest && e.target.closest("details.mastery > summary");
  if (summary) masteryOpen = !summary.parentElement.open;
});
