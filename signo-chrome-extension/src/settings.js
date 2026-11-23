// Shared settings for the service worker, side panel and options page.

export const DEFAULT_SETTINGS = Object.freeze({
  currentMode: "auto", // auto (hold to type) | pinch (two-hand) | mouse
  holdTimeMs: 1000, // how long a letter must stay stable before auto-typing
  minConfidence: 0.6, // smoothed top-1 probability needed to type
  pinchThreshold: 0.06, // normalized thumb-index distance that counts as a pinch
  clickHand: "left", // pinch mode: the hand that pinches; sign with the other
  soundEnabled: true,
  showOverlay: true, // floating letter badge on the web page
  universalTyping: true, // type through Chrome's debugger so it works on every site (shows a banner)
});

export const LIMITS = Object.freeze({
  holdTimeMs: [500, 3000],
  minConfidence: [0.3, 0.95],
  pinchThreshold: [0.03, 0.12],
});

// Bump when a migration must run once on existing installs.
const SETTINGS_VERSION = 4;

const MODES = new Set(["auto", "pinch", "mouse"]);
const HANDS = new Set(["left", "right"]);

function clamp(value, [lo, hi]) {
  return Math.min(hi, Math.max(lo, value));
}

/** Validate raw stored values, falling back to defaults for anything unusable. */
export function normalizeSettings(raw = {}) {
  const s = { ...DEFAULT_SETTINGS };
  if (MODES.has(raw.currentMode)) s.currentMode = raw.currentMode;
  if (HANDS.has(raw.clickHand)) s.clickHand = raw.clickHand;
  for (const key of Object.keys(LIMITS)) {
    const n = Number(raw[key]);
    if (raw[key] !== undefined && Number.isFinite(n)) s[key] = clamp(n, LIMITS[key]);
  }
  for (const key of ["soundEnabled", "showOverlay", "universalTyping"]) {
    if (typeof raw[key] === "boolean") s[key] = raw[key];
  }
  return s;
}

export async function loadSettings() {
  return normalizeSettings(await chrome.storage.sync.get(null));
}

export function saveSettings(partial) {
  return chrome.storage.sync.set(partial);
}

/** Call `callback(settings)` with the full, normalized settings whenever they change. */
export function onSettingsChanged(callback) {
  chrome.storage.onChanged.addListener(async (_changes, area) => {
    if (area === "sync") callback(await loadSettings());
  });
}

/**
 * Upgrade settings from older versions (v0.2 dwell mode/selectedHand, v0.3 typing default) and fill in
 * missing defaults without overwriting the user's choices.
 */
export async function migrateSettings() {
  const raw = await chrome.storage.sync.get(null);
  const updates = {};
  if (raw.currentMode === "dwell") updates.currentMode = "auto";
  if (raw.holdTimeMs === undefined && raw.dwellTimeMs !== undefined) {
    updates.holdTimeMs = clamp(Number(raw.dwellTimeMs) || DEFAULT_SETTINGS.holdTimeMs, LIMITS.holdTimeMs);
  }
  if (raw.clickHand === undefined && HANDS.has(raw.selectedHand)) updates.clickHand = raw.selectedHand;
  if ((raw.settingsVersion || 0) < 4) updates.universalTyping = true; // v0.3 defaulted to off
  if (raw.settingsVersion !== SETTINGS_VERSION) updates.settingsVersion = SETTINGS_VERSION;
  for (const [key, value] of Object.entries(DEFAULT_SETTINGS)) {
    if (raw[key] === undefined && updates[key] === undefined) updates[key] = value;
  }
  if (Object.keys(updates).length) await chrome.storage.sync.set(updates);
  const obsolete = ["dwellTimeMs", "selectedHand"].filter((key) => key in raw);
  if (obsolete.length) await chrome.storage.sync.remove(obsolete);
}
