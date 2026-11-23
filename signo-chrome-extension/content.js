// content.js - Sign-O Chrome Extension
// Edits the text field that was focused on the page before the popup opened.
// Popup messages: { type: "SIGNO_EDIT", action: "insert" | "backspace" | "clear", text? }

(() => {
  // The popup may inject this file again into tabs opened before install.
  if (window.__signoContentLoaded) return;
  window.__signoContentLoaded = true;

  const TEXT_INPUT_TYPES = new Set(["text", "search", "url", "tel", "email", "password", ""]);

  // Follow focus through open shadow roots and same-origin iframes.
  function deepActiveElement(doc = document) {
    let el = doc.activeElement;
    while (el) {
      if (el.shadowRoot && el.shadowRoot.activeElement) {
        el = el.shadowRoot.activeElement;
      } else if (el.tagName === "IFRAME") {
        try {
          const inner = el.contentDocument && el.contentDocument.activeElement;
          if (!inner || inner === el.contentDocument.body) break;
          el = inner;
        } catch {
          break; // cross-origin frame
        }
      } else {
        break;
      }
    }
    return el;
  }

  function isTextField(el) {
    if (!el || el.disabled || el.readOnly) return false;
    if (el.tagName === "TEXTAREA") return true;
    return el.tagName === "INPUT" && TEXT_INPUT_TYPES.has((el.type || "").toLowerCase());
  }

  function findTarget() {
    const el = deepActiveElement();
    if (isTextField(el)) return { kind: "field", el };
    if (el && el.isContentEditable) return { kind: "editable", el };
    return null;
  }

  function notifyInput(el, inputType, data = null) {
    el.dispatchEvent(new InputEvent("input", { bubbles: true, composed: true, inputType, data }));
  }

  function editField(el, action, text) {
    const value = el.value;
    let start = el.selectionStart;
    let end = el.selectionEnd;
    if (start === null || end === null) {
      start = end = value.length; // e.g. type=email exposes no selection
    }
    if (action === "insert") {
      el.setRangeText(text, start, end, "end");
      notifyInput(el, "insertText", text);
    } else if (action === "backspace") {
      if (start === end && start === 0) return;
      const from = start === end ? start - 1 : start;
      el.setRangeText("", from, end, "end");
      notifyInput(el, "deleteContentBackward");
    } else if (action === "clear") {
      el.value = "";
      notifyInput(el, "deleteContent");
    }
  }

  function editContentEditable(el, action, text) {
    const doc = el.ownerDocument;
    el.focus();
    if (action === "insert") {
      // execCommand keeps the page's undo stack and editor frameworks in sync.
      if (!doc.execCommand("insertText", false, text)) insertWithRange(doc, text);
    } else if (action === "backspace") {
      doc.execCommand("delete", false);
    } else if (action === "clear") {
      const sel = doc.getSelection();
      const range = doc.createRange();
      range.selectNodeContents(el);
      sel.removeAllRanges();
      sel.addRange(range);
      doc.execCommand("delete", false);
    }
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

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (!message || message.type !== "SIGNO_EDIT") return false;
    const target = findTarget();
    if (!target) {
      sendResponse({ ok: false, reason: "no-field" });
      return false;
    }
    try {
      if (target.kind === "field") editField(target.el, message.action, message.text);
      else editContentEditable(target.el, message.action, message.text);
      sendResponse({ ok: true });
    } catch (err) {
      sendResponse({ ok: false, reason: String(err && err.message ? err.message : err) });
    }
    return false;
  });
})();
