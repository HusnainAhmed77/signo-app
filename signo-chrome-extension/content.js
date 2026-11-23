// content.js - Sign-O Chrome Extension
// 1. Types into the text field the user last focused on this page (the side panel keeps
//    focus while the user signs, so we remember the field and its caret).
// 2. Shows a small floating badge with the current letter and hold-to-type progress.
//
// Messages from the side panel:
//   { type: "SIGNO_FOCUS" }  → refocus the remembered field before debugger typing
//   { type: "SIGNO_EDIT", action: "insert" | "backspace" | "clear", text?, count? }
//   { type: "SIGNO_OVERLAY", state: { visible, character, progress, locked, typed } }

(() => {
  // The side panel may inject this file again into tabs opened before install.
  if (window.__signoContentLoaded) return;
  window.__signoContentLoaded = true;

  const TEXT_INPUT_TYPES = new Set(["text", "search", "url", "tel", "email", "password", ""]);

  // ------------------------------------------------------------ focus tracking

  let lastEditable = null; // last focused input/textarea/contenteditable on this page
  let savedRange = null; // caret inside lastEditable when it is contenteditable

  function isTextField(el) {
    if (!el || el.disabled || el.readOnly) return false;
    if (el.tagName === "TEXTAREA") return true;
    return el.tagName === "INPUT" && TEXT_INPUT_TYPES.has((el.type || "").toLowerCase());
  }

  function editableRoot(el) {
    if (!el || !el.isContentEditable) return null;
    let root = el;
    while (root.parentElement && root.parentElement.isContentEditable) root = root.parentElement;
    return root;
  }

  function asEditable(el) {
    if (isTextField(el)) return el;
    return editableRoot(el);
  }

  document.addEventListener(
    "focusin",
    (event) => {
      const el = asEditable(event.composedPath()[0]);
      if (el) {
        lastEditable = el;
        savedRange = null;
      }
    },
    true,
  );

  document.addEventListener("selectionchange", () => {
    if (!lastEditable || isTextField(lastEditable)) return;
    const sel = document.getSelection();
    if (sel && sel.rangeCount && lastEditable.contains(sel.anchorNode)) {
      savedRange = sel.getRangeAt(0).cloneRange();
    }
  });

  // Follow focus through open shadow roots and same-origin iframes.
  function deepActiveElement() {
    let el = document.activeElement;
    while (el) {
      if (el.shadowRoot && el.shadowRoot.activeElement) {
        el = el.shadowRoot.activeElement;
      } else if (el.tagName === "IFRAME") {
        let inner = null;
        try {
          inner = el.contentDocument && el.contentDocument.activeElement;
        } catch {
          break; // cross-origin frame
        }
        if (!inner || (inner === el.contentDocument.body && !inner.isContentEditable)) break;
        el = inner;
      } else {
        break;
      }
    }
    return el;
  }

  function findTarget() {
    const active = asEditable(deepActiveElement());
    if (active) return active;
    if (lastEditable && lastEditable.isConnected && asEditable(lastEditable)) return lastEditable;
    return null;
  }

  // ------------------------------------------------------------ editing

  function notifyInput(el, inputType, data = null) {
    el.dispatchEvent(new InputEvent("input", { bubbles: true, composed: true, inputType, data }));
  }

  function editField(el, action, text, count) {
    const value = el.value;
    let start = el.selectionStart;
    let end = el.selectionEnd;
    if (start === null || end === null) start = end = value.length; // e.g. type=email
    if (action === "insert") {
      el.setRangeText(text, start, end, "end");
      notifyInput(el, "insertText", text);
    } else if (action === "backspace") {
      let from = start;
      if (start === end) {
        // Step back by whole code points so surrogate pairs are never split.
        const before = Array.from(value.slice(0, start));
        from = before.slice(0, Math.max(0, before.length - count)).join("").length;
      }
      if (from === end) return;
      el.setRangeText("", from, end, "end");
      notifyInput(el, "deleteContentBackward");
    } else if (action === "clear") {
      el.value = "";
      notifyInput(el, "deleteContent");
    }
  }

  function restoreCaret(el) {
    const doc = el.ownerDocument;
    const sel = doc.getSelection();
    if (sel.rangeCount && el.contains(sel.anchorNode)) return;
    const range = savedRange && el.contains(savedRange.startContainer) ? savedRange : endOf(el);
    sel.removeAllRanges();
    sel.addRange(range);
  }

  function endOf(el) {
    const range = el.ownerDocument.createRange();
    range.selectNodeContents(el);
    range.collapse(false);
    return range;
  }

  function editContentEditable(el, action, text, count) {
    const doc = el.ownerDocument;
    el.focus({ preventScroll: true });
    restoreCaret(el);
    if (action === "insert") {
      // execCommand keeps the page's undo stack and editor frameworks in sync.
      if (!doc.execCommand("insertText", false, text)) insertWithRange(doc, text);
    } else if (action === "backspace") {
      for (let i = 0; i < count; i++) doc.execCommand("delete", false);
    } else if (action === "clear") {
      const sel = doc.getSelection();
      const range = doc.createRange();
      range.selectNodeContents(el);
      sel.removeAllRanges();
      sel.addRange(range);
      doc.execCommand("delete", false);
    }
    const sel = doc.getSelection();
    if (sel.rangeCount) savedRange = sel.getRangeAt(0).cloneRange();
  }

  function insertWithRange(doc, text) {
    const sel = doc.getSelection();
    if (!sel || !sel.rangeCount) return;
    const range = sel.getRangeAt(0);
    range.deleteContents();
    const node = doc.createTextNode(text);
    range.insertNode(node);
    range.setStartAfter(node);
    range.collapse(true);
    sel.removeAllRanges();
    sel.addRange(range);
  }

  function handleEdit(message) {
    const el = findTarget();
    if (!el) return { ok: false, reason: "no-field" };
    const count = Math.max(1, Number(message.count) || 1);
    if (isTextField(el)) {
      el.focus({ preventScroll: true });
      editField(el, message.action, message.text || "", count);
    } else {
      editContentEditable(el, message.action, message.text || "", count);
    }
    return { ok: true };
  }

  /**
   * Before universal (debugger) typing: make sure keyboard focus is in a text box.
   * Returns "field" (focus restored or already there), "other" (something else holds focus,
   * e.g. a cross-origin iframe or closed shadow DOM editor — let the debugger try), or
   * "none" (nothing typable is focused or remembered).
   */
  function prepareFocus() {
    const active = deepActiveElement();
    if (asEditable(active)) return "field";
    const remembered = lastEditable && lastEditable.isConnected && asEditable(lastEditable);
    if (remembered) {
      lastEditable.focus({ preventScroll: true });
      if (!isTextField(lastEditable)) restoreCaret(lastEditable);
      return "field";
    }
    if (active && active !== document.body && active !== document.documentElement) return "other";
    return "none";
  }

  // ------------------------------------------------------------ overlay badge

  const RING = 2 * Math.PI * 30;
  let overlay = null;
  let hideTimer = null;

  function buildOverlay() {
    const host = document.createElement("signo-overlay");
    host.style.cssText = "all: initial; position: fixed; right: 20px; bottom: 20px; z-index: 2147483647; pointer-events: none;";
    const root = host.attachShadow({ mode: "closed" });
    root.innerHTML = `
      <style>
        .badge {
          width: 76px; height: 76px; border-radius: 50%; position: relative;
          background: rgba(29, 27, 24, 0.88); box-shadow: 0 6px 24px rgba(0,0,0,0.25);
          display: grid; place-items: center; opacity: 0; transform: translateY(8px) scale(0.92);
          transition: opacity 0.18s, transform 0.18s;
        }
        .badge.visible { opacity: 1; transform: none; }
        svg { position: absolute; inset: 4px; width: 68px; height: 68px; transform: rotate(-90deg); }
        circle { fill: none; stroke-width: 5; }
        .track { stroke: rgba(255,255,255,0.15); }
        .progress { stroke: #d4af37; stroke-linecap: round; stroke-dasharray: ${RING}; stroke-dashoffset: ${RING};
          transition: stroke-dashoffset 0.1s linear, stroke 0.2s; }
        .progress.locked { stroke: #66bb6a; }
        .letter { position: relative; color: #f1ece3; font: 600 30px "Noto Nastaliq Urdu", "Noto Naskh Arabic", "Geeza Pro", Tahoma, sans-serif; }
        .badge.typed .letter { color: #66bb6a; animation: pop 0.35s ease-out; }
        @keyframes pop { from { transform: scale(1.4); } to { transform: scale(1); } }
        .tag { position: absolute; bottom: -18px; left: 50%; transform: translateX(-50%);
          font: 600 10px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #6b6458;
          background: rgba(245,242,236,0.92); padding: 1px 6px; border-radius: 6px; white-space: nowrap; }
      </style>
      <div class="badge">
        <svg viewBox="0 0 68 68"><circle class="track" cx="34" cy="34" r="30"></circle><circle class="progress" cx="34" cy="34" r="30"></circle></svg>
        <span class="letter" dir="rtl"></span>
        <span class="tag">Sign-O</span>
      </div>`;
    document.documentElement.appendChild(host);
    return {
      host,
      badge: root.querySelector(".badge"),
      letter: root.querySelector(".letter"),
      progress: root.querySelector(".progress"),
      wasTyped: false,
    };
  }

  function updateOverlay(state) {
    if (!state.visible) {
      if (overlay) overlay.badge.classList.remove("visible");
      return;
    }
    if (!overlay || !overlay.host.isConnected) overlay = buildOverlay();
    overlay.letter.textContent = state.character || "…";
    overlay.progress.style.strokeDashoffset = String(RING * (1 - Math.min(1, state.progress || 0)));
    overlay.progress.classList.toggle("locked", Boolean(state.locked));
    if (state.typed && !overlay.wasTyped) overlay.badge.classList.add("typed");
    if (!state.typed) overlay.badge.classList.remove("typed");
    overlay.wasTyped = Boolean(state.typed);
    // Force a frame so the fade-in transition runs the first time.
    requestAnimationFrame(() => overlay && overlay.badge.classList.add("visible"));
    // If the side panel closes abruptly, hide the badge on our own.
    clearTimeout(hideTimer);
    hideTimer = setTimeout(() => overlay && overlay.badge.classList.remove("visible"), 1500);
  }

  // ------------------------------------------------------------ messaging

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (!message) return false;
    if (message.type === "SIGNO_EDIT") {
      try {
        sendResponse(handleEdit(message));
      } catch (err) {
        sendResponse({ ok: false, reason: String(err && err.message ? err.message : err) });
      }
    } else if (message.type === "SIGNO_FOCUS") {
      try {
        sendResponse({ focused: prepareFocus() });
      } catch {
        sendResponse({ focused: "other" });
      }
    } else if (message.type === "SIGNO_OVERLAY") {
      updateOverlay(message.state || {});
      sendResponse({ ok: true });
    }
    return false;
  });
})();
