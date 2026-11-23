// Sends edits and the overlay badge state to the active tab.
//
// Typing back-ends:
//  - Universal (default): Chrome debugger Input.insertText, which behaves like a real
//    keyboard, so it works on every website: plain inputs, rich editors, Google Docs,
//    cross-origin iframes and shadow DOM. Chrome shows a "Sign-O started debugging this
//    browser" bar while attached. Before typing, the content script puts focus back in the
//    text box the user last clicked (the side panel has focus while they sign).
//  - Standard: the content script edits the last-focused input / textarea / contenteditable.
//    Used when universal typing is off, or automatically when the debugger cannot attach.
//
// No extension can type into chrome:// pages, the Chrome Web Store or other extensions;
// Chrome blocks both back-ends there.

const TYPABLE_URL = /^(https?|file):/;
const WEB_STORE_URL = /^https:\/\/(chromewebstore\.google\.com|chrome\.google\.com\/webstore)/;

/** Why Chrome forbids typing on this URL, or null if typing is possible. */
function restrictionReason(url = "") {
  if (WEB_STORE_URL.test(url)) return "Chrome blocks all extensions on the Chrome Web Store.";
  if (!TYPABLE_URL.test(url)) {
    return "Chrome blocks extensions on its own pages (chrome://, new tab, settings, other extensions). Open a website to type.";
  }
  return null;
}
const UNDO_LIMIT = 50;
const IS_MAC = /Mac/i.test(navigator.userAgentData?.platform || navigator.platform || "");

export class PageTyper {
  constructor({ getSettings, onDebuggerDetached }) {
    this.getSettings = getSettings;
    this.onDebuggerDetached = onDebuggerDetached;
    this.debuggerTab = null;
    this.overlayTab = null;
    this.undoStack = []; // [{tabId, length}]
    this.detachListenerAdded = false;
    this.injectedTabs = new Set(); // tabs we already tried to inject for the overlay
  }

  async activeTab() {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    return tab || null;
  }

  /** Insert text (or " "). Returns {ok, message}. */
  async insert(text) {
    const result = await this.edit("insert", { text });
    if (result.ok) {
      this.undoStack.push({ tabId: result.tabId, length: [...text].length });
      if (this.undoStack.length > UNDO_LIMIT) this.undoStack.shift();
    }
    return result;
  }

  async backspace() {
    const result = await this.edit("backspace", { count: 1 });
    if (result.ok) this.undoStack = []; // manual edits make the undo history unreliable
    return result;
  }

  async clear() {
    const result = await this.edit("clear");
    if (result.ok) this.undoStack = [];
    return result;
  }

  canUndo() {
    return this.undoStack.length > 0;
  }

  /** Remove the most recent insertion made by Sign-O. */
  async undo() {
    const last = this.undoStack[this.undoStack.length - 1];
    if (!last) return { ok: false, message: "Nothing to undo." };
    const tab = await this.activeTab();
    if (!tab || tab.id !== last.tabId) return { ok: false, message: "Switch back to the tab you were typing in to undo." };
    const result = await this.edit("backspace", { count: last.length });
    if (result.ok) this.undoStack.pop();
    return result;
  }

  async edit(action, { text = "", count = 1 } = {}) {
    let tab;
    try {
      tab = await this.activeTab();
    } catch (err) {
      return { ok: false, message: `No active tab: ${err.message}` };
    }
    if (!tab) return { ok: false, message: "No active tab." };
    const restricted = restrictionReason(tab.url);
    if (restricted) return { ok: false, message: restricted, tabId: tab.id };

    let result;
    if (this.getSettings().universalTyping) {
      result = await this.editUniversal(tab.id, action, text, count);
    } else {
      result = await this.editWithContentScript(tab.id, action, text, count);
    }
    return { ...result, tabId: tab.id };
  }

  // ---------- universal (debugger) back-end ----------
  async editUniversal(tabId, action, text, count) {
    // Put the caret back in the user's text box. Skipped silently if the content script
    // cannot run (e.g. a PDF); the debugger still types wherever the page focus is.
    const focus = await this.sendToTab(tabId, { type: "SIGNO_FOCUS" });
    if (focus && focus.focused === "none") {
      return { ok: false, message: "Click into a text box on the page first." };
    }

    try {
      await this.attach(tabId);
    } catch (err) {
      // e.g. a page the debugger may not inspect, or DevTools policy. Fall back.
      console.warn("Debugger attach failed, using standard typing:", err);
      return this.editWithContentScript(tabId, action, text, count);
    }

    try {
      const send = (method, params) => chrome.debugger.sendCommand({ tabId }, method, params);
      if (action === "insert") {
        await send("Input.insertText", { text });
      } else if (action === "backspace") {
        for (let i = 0; i < count; i++) await this.pressKey(send, "Backspace", 8);
      } else if (action === "clear") {
        await this.pressKey(send, "a", 65, IS_MAC ? 4 : 2); // Cmd/Ctrl+A inside the focused field
        await this.pressKey(send, "Backspace", 8);
      }
      return { ok: true };
    } catch (err) {
      this.debuggerTab = null;
      return { ok: false, message: `Typing failed: ${err.message}` };
    }
  }

  // ---------- standard (content-script) back-end ----------
  async editWithContentScript(tabId, action, text, count) {
    const message = { type: "SIGNO_EDIT", action, text, count };
    const response = await this.sendToTab(tabId, message);
    if (!response) return { ok: false, message: "Could not reach the page. Reload the tab and try again." };
    if (response.ok) return { ok: true };
    if (response.reason === "no-field") {
      return { ok: false, message: "Click into a text box on the page first." };
    }
    return { ok: false, message: `Typing failed: ${response.reason}` };
  }

  /** Message the tab's content script, injecting it first if the tab predates the extension. */
  async sendToTab(tabId, message) {
    try {
      return await chrome.tabs.sendMessage(tabId, message);
    } catch {
      try {
        await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
        return await chrome.tabs.sendMessage(tabId, message);
      } catch {
        return null;
      }
    }
  }

  async pressKey(send, key, keyCode, modifiers = 0) {
    const code = key.length === 1 ? `Key${key.toUpperCase()}` : key;
    const base = { key, code, windowsVirtualKeyCode: keyCode, nativeVirtualKeyCode: keyCode, modifiers };
    const commands = modifiers && key.length === 1 ? { commands: ["selectAll"] } : {};
    await send("Input.dispatchKeyEvent", { type: "rawKeyDown", ...base, ...commands });
    await send("Input.dispatchKeyEvent", { type: "keyUp", ...base });
  }

  async attach(tabId) {
    if (this.debuggerTab === tabId) return;
    await this.detach();
    if (!this.detachListenerAdded) {
      this.detachListenerAdded = true;
      chrome.debugger.onDetach.addListener((source, reason) => {
        if (source.tabId !== this.debuggerTab) return;
        this.debuggerTab = null;
        this.onDebuggerDetached?.(reason);
      });
    }
    await chrome.debugger.attach({ tabId }, "1.3");
    this.debuggerTab = tabId;
  }

  async detach() {
    if (this.debuggerTab === null) return;
    const tabId = this.debuggerTab;
    this.debuggerTab = null;
    try {
      await chrome.debugger.detach({ tabId });
    } catch {
      // tab already closed or detached by the user
    }
  }

  // ---------- on-page overlay badge ----------
  /** Show/update the floating badge on the active tab (fire-and-forget). */
  async overlay(state) {
    let tab;
    try {
      tab = await this.activeTab();
    } catch {
      return;
    }
    const tabId = tab && !restrictionReason(tab.url) ? tab.id : null;
    if (this.overlayTab !== null && this.overlayTab !== tabId) {
      chrome.tabs.sendMessage(this.overlayTab, { type: "SIGNO_OVERLAY", state: { visible: false } }).catch(() => {});
    }
    this.overlayTab = tabId;
    if (tabId === null) return;
    try {
      await chrome.tabs.sendMessage(tabId, { type: "SIGNO_OVERLAY", state });
    } catch {
      // Tab predates the extension: inject the content script once, then the next update lands.
      if (this.injectedTabs.has(tabId)) return;
      this.injectedTabs.add(tabId);
      chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] }).catch(() => {});
    }
  }

  hideOverlay() {
    if (this.overlayTab === null) return;
    chrome.tabs.sendMessage(this.overlayTab, { type: "SIGNO_OVERLAY", state: { visible: false } }).catch(() => {});
    this.overlayTab = null;
  }
}
