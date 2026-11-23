// background.js - Sign-O service worker (ES module)
// - Toolbar icon opens the side panel.
// - Alt+Shift+S opens the side panel, or pauses/resumes recognition if it is already open.
// - Migrates settings from older versions and opens the camera-permission page on install.

import { migrateSettings } from "./src/settings.js";

chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch((err) => console.error(err));

// Open side panels report their window id over a long-lived port.
const panelPorts = new Map(); // windowId -> Port

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== "sidepanel") return;
  let windowId = null;
  port.onMessage.addListener((message) => {
    if (message.type === "hello") {
      windowId = message.windowId;
      panelPorts.set(windowId, port);
    }
  });
  port.onDisconnect.addListener(() => {
    if (windowId !== null && panelPorts.get(windowId) === port) panelPorts.delete(windowId);
  });
});

chrome.commands.onCommand.addListener((command, tab) => {
  if (command !== "toggle-recognition" || !tab) return;
  const port = panelPorts.get(tab.windowId);
  if (port) {
    port.postMessage({ type: "toggle" });
  } else {
    // Must be called synchronously inside the shortcut's user gesture.
    chrome.sidePanel.open({ windowId: tab.windowId }).catch((err) => console.error(err));
  }
});

chrome.runtime.onInstalled.addListener(async (details) => {
  await migrateSettings();
  // Extension pages cannot always show Chrome's camera prompt, so ask once in a tab.
  if (details.reason === "install") {
    chrome.tabs.create({ url: chrome.runtime.getURL("camera.html") });
  }
});
