// background.js - Sign-O Chrome Extension service worker
// Seeds default settings and opens the camera-permission page on first install.

const DEFAULT_SETTINGS = {
  currentMode: "pinch", // pinch | dwell | mouse
  dwellTimeMs: 1500,
  pinchThreshold: 0.06,
  clickHand: "left", // the hand that pinches to confirm; sign with the other hand
  minConfidence: 0.5,
};

chrome.runtime.onInstalled.addListener(async (details) => {
  // Only fill in missing keys so an update never overwrites the user's choices.
  const stored = await chrome.storage.sync.get(Object.keys(DEFAULT_SETTINGS));
  const missing = {};
  for (const [key, value] of Object.entries(DEFAULT_SETTINGS)) {
    if (stored[key] === undefined) missing[key] = value;
  }
  if (Object.keys(missing).length) await chrome.storage.sync.set(missing);

  // Extension popups cannot show Chrome's camera prompt, so ask once in a tab.
  if (details.reason === "install") {
    chrome.tabs.create({ url: chrome.runtime.getURL("camera.html") });
  }
});
