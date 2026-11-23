// options.js - Sign-O settings page. Every change is saved immediately.

import { DEFAULT_SETTINGS, loadSettings, onSettingsChanged, saveSettings } from "./src/settings.js";

const $ = (id) => document.getElementById(id);
const MODE_HELP = {
  auto: "Hold a sign steady and it is typed automatically — no clicking needed.",
  pinch: "Sign with one hand and pinch with the other to type.",
  mouse: "Letters are typed only when you click “Type letter” in the panel.",
};

let toastTimer = null;
function saved() {
  const toast = $("toast");
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 1200);
}

async function save(partial) {
  await saveSettings(partial);
  saved();
}

function render(s) {
  document.querySelectorAll("#modeGroup .segment").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.mode === s.currentMode)));
  document.querySelectorAll("#handGroup .segment").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.hand === s.clickHand)));
  $("modeHelp").textContent = MODE_HELP[s.currentMode];
  $("holdRow").classList.toggle("hidden", s.currentMode !== "auto");
  $("pinchRows").classList.toggle("hidden", s.currentMode !== "pinch");

  $("holdTime").value = s.holdTimeMs;
  $("holdTimeOut").textContent = `${(s.holdTimeMs / 1000).toFixed(1)} s`;
  $("minConfidence").value = Math.round(s.minConfidence * 100);
  $("minConfidenceOut").textContent = `${Math.round(s.minConfidence * 100)}%`;
  $("pinchThreshold").value = Math.round(s.pinchThreshold * 100);
  $("pinchThresholdOut").textContent = (s.pinchThreshold).toFixed(2);

  $("soundEnabled").checked = s.soundEnabled;
  $("showOverlay").checked = s.showOverlay;
  $("universalTyping").checked = s.universalTyping;
}

async function renderShortcut() {
  const commands = await chrome.commands.getAll();
  const command = commands.find((c) => c.name === "toggle-recognition");
  $("shortcutText").textContent = command?.shortcut
    ? `${command.shortcut} opens Sign-O or pauses / resumes recognition.`
    : "No shortcut is set. Choose one to open Sign-O or pause / resume recognition.";
}

function bindSlider(id, toValue, key) {
  const input = $(id);
  // Update the label while dragging; save when released.
  input.addEventListener("input", () => {
    const value = toValue(Number(input.value));
    $(`${id}Out`).textContent =
      key === "holdTimeMs" ? `${(value / 1000).toFixed(1)} s` : key === "minConfidence" ? `${Math.round(value * 100)}%` : value.toFixed(2);
  });
  input.addEventListener("change", () => save({ [key]: toValue(Number(input.value)) }));
}

async function init() {
  const settings = await loadSettings();
  render(settings);
  renderShortcut();
  onSettingsChanged(render);

  $("modeGroup").addEventListener("click", (e) => {
    const btn = e.target.closest(".segment");
    if (btn) save({ currentMode: btn.dataset.mode });
  });
  $("handGroup").addEventListener("click", (e) => {
    const btn = e.target.closest(".segment");
    if (btn) save({ clickHand: btn.dataset.hand });
  });
  bindSlider("holdTime", (v) => v, "holdTimeMs");
  bindSlider("minConfidence", (v) => v / 100, "minConfidence");
  bindSlider("pinchThreshold", (v) => v / 100, "pinchThreshold");
  $("soundEnabled").addEventListener("change", (e) => save({ soundEnabled: e.target.checked }));
  $("showOverlay").addEventListener("change", (e) => save({ showOverlay: e.target.checked }));
  $("universalTyping").addEventListener("change", (e) => save({ universalTyping: e.target.checked }));
  $("shortcutBtn").addEventListener("click", () => chrome.tabs.create({ url: "chrome://extensions/shortcuts" }));
  $("cameraBtn").addEventListener("click", () => chrome.tabs.create({ url: chrome.runtime.getURL("camera.html") }));
  $("resetBtn").addEventListener("click", () => save({ ...DEFAULT_SETTINGS }));
  window.addEventListener("focus", renderShortcut);
}

init();
