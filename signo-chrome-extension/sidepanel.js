// Sign-O side panel controller.
// Camera → MediaPipe HandLandmarker → hand-cropped MobileNetV3 (ONNX) → smoothed
// hold-to-type engine → typing into the page. The side panel stays open while the user
// works in the page, so recognition and auto-typing keep running.

import { FilesetResolver, HandLandmarker } from "./lib/mediapipe/vision_bundle.mjs";
import { SignClassifier } from "./src/classifier.js";
import { HoldToType } from "./src/holdToType.js";
import { PageTyper } from "./src/pageTyper.js";
import { playTypedSound } from "./src/feedback.js";
import { loadSettings, onSettingsChanged, saveSettings, DEFAULT_SETTINGS } from "./src/settings.js";

const PREDICTION_INTERVAL_MS = 150;
const PINCH_COOLDOWN_MS = 700;
const OVERLAY_INTERVAL_MS = 100;
const RING_CIRCUMFERENCE = 2 * Math.PI * 44;
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

let settings = { ...DEFAULT_SETTINGS };
const engine = new HoldToType();
const typer = new PageTyper({
  getSettings: () => settings,
  onDebuggerDetached: (reason) => {
    if (reason === "canceled_by_user") {
      showStatus("Debugger bar was cancelled. Sign-O reattaches on the next letter; turn off Universal typing in Settings to stop that.", "error");
    }
  },
});

const state = {
  classifier: null,
  landmarker: null,
  stream: null,
  running: false,
  loopId: 0,
  lastVideoTime: -1,
  predicting: false,
  lastPredictionAt: 0,
  signHand: null,
  snapshot: engine.snapshot(),
  pinchWasDown: false,
  lastPinchAt: 0,
  lastOverlayAt: 0,
  typedFlash: null, // {character, until}
  typing: false,
};

const $ = (id) => document.getElementById(id);
const ui = {
  video: $("cameraFeed"),
  overlay: $("landmarksOverlay"),
  cameraMessage: $("cameraMessage"),
  cameraPermissionBtn: $("cameraPermissionBtn"),
  pauseBtn: $("pauseBtn"),
  segments: document.querySelectorAll(".segment"),
  modeHint: $("modeHint"),
  ring: $("ringProgress"),
  letter: $("predictionLetter"),
  label: $("predictionLabel"),
  confidenceFill: $("confidenceFill"),
  confidenceText: $("confidenceText"),
  phase: $("phaseText"),
  top3: $("top3"),
  typeBtn: $("typeBtn"),
  spaceBtn: $("spaceBtn"),
  backspaceBtn: $("backspaceBtn"),
  undoBtn: $("undoBtn"),
  clearBtn: $("clearBtn"),
  status: $("statusText"),
  settingsBtn: $("settingsBtn"),
};
const overlayCtx = ui.overlay.getContext("2d");

// ============================================================ startup

init();

async function init() {
  bindUi();
  connectToBackground();
  settings = await loadSettings();
  onSettingsChanged((next) => {
    const modeChanged = next.currentMode !== settings.currentMode;
    settings = next;
    if (modeChanged) engine.reset();
    if (!settings.showOverlay) typer.hideOverlay();
    if (!settings.universalTyping) typer.detach();
    renderMode();
  });
  renderMode();

  try {
    const [classifier, landmarker] = await Promise.all([
      SignClassifier.create({
        modelUrl: "models/best_model_tl.onnx",
        classesUrl: "models/classes.json",
        wasmPaths: "lib/onnxruntime/",
      }),
      createHandLandmarker(),
    ]);
    state.classifier = classifier;
    state.landmarker = landmarker;
  } catch (err) {
    console.error("Model loading failed:", err);
    setCameraMessage(`Could not load the models: ${err.message}`);
    showStatus("Model loading failed. Reload the extension.", "error");
    return;
  }
  await startCamera();
}

async function createHandLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks("lib/mediapipe/wasm");
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: "models/hand_landmarker.task", delegate },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    return await HandLandmarker.createFromOptions(fileset, options("GPU"));
  } catch (err) {
    console.warn("GPU delegate unavailable, falling back to CPU:", err);
    return HandLandmarker.createFromOptions(fileset, options("CPU"));
  }
}

/** Lets the background worker toggle recognition from the keyboard shortcut. */
function connectToBackground() {
  const port = chrome.runtime.connect({ name: "sidepanel" });
  chrome.windows.getCurrent().then((win) => port.postMessage({ type: "hello", windowId: win.id }));
  port.onMessage.addListener((message) => {
    if (message.type === "toggle") togglePause();
  });
  // The service worker is stopped when idle, which drops the port. Reconnect so the
  // keyboard shortcut still reaches this panel.
  port.onDisconnect.addListener(() => setTimeout(connectToBackground, 500));
}

// ============================================================ camera

async function startCamera() {
  ui.cameraPermissionBtn.classList.add("hidden");
  setCameraMessage("Starting camera…");
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
  } catch (err) {
    console.error("Camera error:", err);
    if (err.name === "NotAllowedError" || err.name === "SecurityError") {
      setCameraMessage("Sign-O needs camera access.");
      showStatus("Click “Allow camera access”, allow it, then come back here.", "error");
      ui.cameraPermissionBtn.classList.remove("hidden");
    } else {
      setCameraMessage(`Camera unavailable: ${err.message}`);
      showStatus("Check that no other app is using the camera.", "error");
    }
    return;
  }
  ui.video.srcObject = state.stream;
  await ui.video.play();
  ui.overlay.width = ui.video.videoWidth;
  ui.overlay.height = ui.video.videoHeight;
  setCameraMessage(null);
  state.running = true;
  ui.pauseBtn.disabled = false;
  renderPause();
  showStatus(readyMessage());
  const loopId = ++state.loopId;
  scheduleFrame(loopId);
}

function stopCamera() {
  state.running = false;
  state.loopId++;
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  ui.video.srcObject = null;
  overlayCtx.clearRect(0, 0, ui.overlay.width, ui.overlay.height);
  engine.reset();
  state.snapshot = engine.snapshot();
  renderPrediction();
  typer.hideOverlay();
  typer.detach();
}

async function togglePause() {
  if (state.running) {
    stopCamera();
    setCameraMessage("Paused. Press Resume or Alt+Shift+S.");
    showStatus("Paused — camera is off.");
    renderPause();
  } else if (state.classifier && state.landmarker) {
    await startCamera();
  }
}

function scheduleFrame(loopId) {
  const next = () => loopId === state.loopId && processFrame(loopId);
  if (ui.video.requestVideoFrameCallback) ui.video.requestVideoFrameCallback(next);
  else requestAnimationFrame(next);
}

// ============================================================ per-frame loop

function processFrame(loopId) {
  const video = ui.video;
  const now = performance.now();
  if (video.readyState >= 2 && video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;
    let hands = [];
    try {
      hands = toHands(state.landmarker.detectForVideo(video, now));
    } catch (err) {
      console.warn("Hand tracking failed:", err);
    }
    drawHands(hands);
    state.signHand = pickSignHand(hands);

    if (state.signHand) {
      if (!state.predicting && now - state.lastPredictionAt >= PREDICTION_INTERVAL_MS) {
        state.lastPredictionAt = now;
        predict(state.signHand.landmarks);
      }
    } else {
      engine.handLost(now);
    }
    handlePinch(hands, now);
  }

  const snapshot = engine.tick(now, engineOptions());
  if (snapshot.commit >= 0) commitLetter(snapshot.commit);
  state.snapshot = snapshot;
  renderPrediction();
  pushOverlay(now);
  scheduleFrame(loopId);
}

async function predict(landmarks) {
  state.predicting = true;
  try {
    const probs = await state.classifier.classify(ui.video, landmarks);
    if (!state.running) return;
    const snapshot = engine.observe(probs, performance.now(), engineOptions());
    if (snapshot.commit >= 0) commitLetter(snapshot.commit);
    state.snapshot = snapshot;
  } catch (err) {
    console.error("Prediction failed:", err);
    showStatus(`Prediction failed: ${err.message}`, "error");
  } finally {
    state.predicting = false;
  }
}

function engineOptions() {
  return {
    minConfidence: settings.minConfidence,
    holdTimeMs: settings.holdTimeMs,
    autoCommit: settings.currentMode === "auto",
  };
}

function toHands(result) {
  if (!result?.landmarks) return [];
  return result.landmarks.map((landmarks, i) => {
    const category = result.handedness?.[i]?.[0]?.categoryName;
    // MediaPipe assumes a mirrored (selfie) image. Our frame is not mirrored, so swap
    // the label to get the user's real hand.
    const side = category === "Left" ? "right" : category === "Right" ? "left" : "unknown";
    const thumb = landmarks[4];
    const index = landmarks[8];
    const pinchDistance = Math.hypot(thumb.x - index.x, thumb.y - index.y);
    let minX = 1, minY = 1, maxX = 0, maxY = 0;
    for (const p of landmarks) {
      minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
    }
    return { side, landmarks, pinch: pinchDistance < settings.pinchThreshold, area: (maxX - minX) * (maxY - minY) };
  });
}

/** Pinch mode signs with the non-click hand; other modes use the largest (closest) hand. */
function pickSignHand(hands) {
  const candidates = settings.currentMode === "pinch" ? hands.filter((h) => h.side !== settings.clickHand) : hands;
  return candidates.reduce((best, h) => (!best || h.area > best.area ? h : best), null);
}

function handlePinch(hands, now) {
  if (settings.currentMode !== "pinch") return;
  const clickHand = hands.find((h) => h.side === settings.clickHand);
  const pinching = Boolean(clickHand?.pinch);
  // Fire on the moment the pinch closes, so holding a pinch types only once.
  if (pinching && !state.pinchWasDown && now - state.lastPinchAt >= PINCH_COOLDOWN_MS) {
    state.lastPinchAt = now;
    if (canType(state.snapshot)) commitLetter(state.snapshot.index);
    else showStatus("Pinch detected, but no confident letter to type yet.");
  }
  state.pinchWasDown = pinching;
}

function canType(snapshot) {
  return Boolean(
    state.running &&
    state.signHand &&
    snapshot.index >= 0 &&
    snapshot.confidence >= settings.minConfidence &&
    state.classifier.classes[snapshot.index]?.character,
  );
}

// ============================================================ typing

async function commitLetter(index) {
  const character = state.classifier.classes[index]?.character;
  if (!character) return;
  engine.lock(index); // never repeat a letter until the hand is released
  const ok = await runTyping(() => typer.insert(character), `Typed ${character}`);
  if (ok) {
    state.typedFlash = { character, until: performance.now() + 900 };
    ui.letter.classList.remove("typed");
    void ui.letter.offsetWidth; // restart the animation
    ui.letter.classList.add("typed");
    if (settings.soundEnabled) playTypedSound();
  }
}

async function runTyping(action, successMessage) {
  state.typing = true;
  try {
    const result = await action();
    if (result.ok) showStatus(successMessage, "ok", 1800);
    else showStatus(result.message, "error");
    return result.ok;
  } finally {
    state.typing = false;
    ui.undoBtn.disabled = !typer.canUndo();
  }
}

// ============================================================ rendering

function renderPrediction() {
  const snap = state.snapshot;
  const cls = snap.index >= 0 && state.classifier ? state.classifier.classes[snap.index] : null;
  const handVisible = Boolean(state.signHand) && state.running;

  ui.letter.textContent = handVisible && cls ? cls.character : "—";
  ui.label.textContent = handVisible && cls ? cls.label : state.running ? "No hand" : "Paused";
  const pct = handVisible ? Math.round(snap.confidence * 100) : 0;
  ui.confidenceText.textContent = `${pct}%`;
  ui.confidenceFill.style.width = `${pct}%`;
  ui.confidenceFill.classList.toggle("low", snap.confidence < settings.minConfidence);

  const progress = settings.currentMode === "auto" && handVisible ? snap.progress : 0;
  ui.ring.style.strokeDashoffset = String(RING_CIRCUMFERENCE * (1 - progress));
  ui.ring.classList.toggle("locked", snap.phase === "locked");

  ui.phase.textContent = phaseMessage(snap, handVisible);
  ui.typeBtn.disabled = !canType(snap) || state.typing;
  renderTop3(handVisible ? snap.probs : null);
}

function phaseMessage(snap, handVisible) {
  if (!state.running) return "Recognition is paused.";
  if (!handVisible) {
    return settings.currentMode === "pinch"
      ? `Sign with your ${otherHand()} hand; pinch with your ${settings.clickHand} hand to type.`
      : "Show one hand to the camera.";
  }
  if (snap.confidence < settings.minConfidence) return "Not sure yet — hold the sign clearly in view.";
  if (settings.currentMode === "auto") {
    if (snap.phase === "holding") return "Hold steady…";
    if (snap.phase === "locked") return "Typed ✓ Change the sign, or lower your hand to repeat it.";
    return "Keep the sign steady to type it.";
  }
  if (settings.currentMode === "pinch") return `Pinch with your ${settings.clickHand} hand to type.`;
  return "Click “Type letter” to type it.";
}

let lastTop3Key = "";
function renderTop3(probs) {
  if (!probs || !state.classifier) {
    if (lastTop3Key) ui.top3.replaceChildren();
    lastTop3Key = "";
    return;
  }
  const ranked = Array.from(probs, (p, i) => [i, p]).sort((a, b) => b[1] - a[1]).slice(0, 3);
  const key = ranked.map(([i, p]) => `${i}:${Math.round(p * 100)}`).join("|");
  if (key === lastTop3Key) return;
  lastTop3Key = key;
  ui.top3.replaceChildren(
    ...ranked.map(([i, p]) => {
      const li = document.createElement("li");
      const ch = document.createElement("span");
      ch.className = "ch";
      ch.textContent = state.classifier.classes[i].character;
      const pct = document.createElement("span");
      pct.textContent = `${Math.round(p * 100)}%`;
      li.title = state.classifier.classes[i].label;
      li.append(ch, pct);
      return li;
    }),
  );
}

function drawHands(hands) {
  const { width, height } = ui.overlay;
  overlayCtx.clearRect(0, 0, width, height);
  for (const hand of hands) {
    const isSign = hand === pickSignHand(hands);
    const pts = hand.landmarks.map((p) => [p.x * width, p.y * height]);
    overlayCtx.lineWidth = 3;
    overlayCtx.strokeStyle = hand.pinch && settings.currentMode === "pinch" ? "#d4af37" : isSign ? "#4caf50" : "#9e9e9e";
    overlayCtx.beginPath();
    for (const [a, b] of HAND_CONNECTIONS) {
      overlayCtx.moveTo(...pts[a]);
      overlayCtx.lineTo(...pts[b]);
    }
    overlayCtx.stroke();
    overlayCtx.fillStyle = isSign ? "#ff7a2f" : "#bdbdbd";
    for (const [x, y] of pts) {
      overlayCtx.beginPath();
      overlayCtx.arc(x, y, 3.5, 0, Math.PI * 2);
      overlayCtx.fill();
    }
  }
}

function pushOverlay(now) {
  if (!settings.showOverlay || now - state.lastOverlayAt < OVERLAY_INTERVAL_MS) return;
  state.lastOverlayAt = now;
  const snap = state.snapshot;
  const handVisible = Boolean(state.signHand);
  const cls = snap.index >= 0 ? state.classifier.classes[snap.index] : null;
  const flash = state.typedFlash && state.typedFlash.until > now ? state.typedFlash.character : null;
  typer.overlay({
    visible: handVisible || Boolean(flash),
    character: flash || (cls && snap.confidence >= settings.minConfidence ? cls.character : "…"),
    progress: settings.currentMode === "auto" ? snap.progress : 0,
    locked: snap.phase === "locked",
    typed: Boolean(flash),
  });
}

function renderMode() {
  ui.segments.forEach((btn) => btn.setAttribute("aria-checked", String(btn.dataset.mode === settings.currentMode)));
  ui.modeHint.textContent = {
    auto: `Hold a sign steady for ${(settings.holdTimeMs / 1000).toFixed(1)} s and it types automatically.`,
    pinch: `Sign with your ${otherHand()} hand, pinch your ${settings.clickHand} thumb and index finger to type.`,
    mouse: "Letters are typed only when you click “Type letter”.",
  }[settings.currentMode];
}

function renderPause() {
  ui.pauseBtn.textContent = state.running ? "Pause" : "Resume";
  ui.pauseBtn.classList.toggle("paused", !state.running);
}

function otherHand() {
  return settings.clickHand === "left" ? "right" : "left";
}

function readyMessage() {
  return settings.universalTyping
    ? "Ready. Click into any text box on the page, then sign."
    : "Ready. Click into a text box on the page, then sign.";
}

function setCameraMessage(text) {
  ui.cameraMessage.textContent = text || "";
  ui.cameraMessage.classList.toggle("hidden", !text);
}

let statusTimer = null;
function showStatus(message, kind = "", revertAfterMs = 0) {
  clearTimeout(statusTimer);
  ui.status.textContent = message;
  ui.status.className = `status ${kind}`;
  if (revertAfterMs) {
    statusTimer = setTimeout(() => showStatus(state.running ? readyMessage() : "Paused — camera is off."), revertAfterMs);
  }
}

// ============================================================ UI events

function bindUi() {
  ui.segments.forEach((btn) => btn.addEventListener("click", () => saveSettings({ currentMode: btn.dataset.mode })));
  ui.pauseBtn.addEventListener("click", togglePause);
  ui.typeBtn.addEventListener("click", () => canType(state.snapshot) && commitLetter(state.snapshot.index));
  ui.spaceBtn.addEventListener("click", () => runTyping(() => typer.insert(" "), "Space"));
  ui.backspaceBtn.addEventListener("click", () => runTyping(() => typer.backspace(), "Backspace"));
  ui.undoBtn.addEventListener("click", () => runTyping(() => typer.undo(), "Undone"));
  ui.clearBtn.addEventListener("click", () => runTyping(() => typer.clear(), "Field cleared"));
  ui.settingsBtn.addEventListener("click", () => chrome.runtime.openOptionsPage());
  ui.cameraPermissionBtn.addEventListener("click", () => chrome.tabs.create({ url: chrome.runtime.getURL("camera.html") }));

  // Retry the camera when the user comes back from granting permission.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && !state.running && !ui.cameraPermissionBtn.classList.contains("hidden")) startCamera();
  });
  window.addEventListener("focus", () => {
    if (!state.running && !ui.cameraPermissionBtn.classList.contains("hidden")) startCamera();
  });

  document.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLButtonElement && (e.key === " " || e.key === "Enter")) return;
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
      e.preventDefault();
      runTyping(() => typer.undo(), "Undone");
    } else if (e.key === " ") {
      e.preventDefault();
      runTyping(() => typer.insert(" "), "Space");
    } else if (e.key === "Backspace") {
      e.preventDefault();
      runTyping(() => typer.backspace(), "Backspace");
    }
  });

  window.addEventListener("pagehide", () => {
    typer.hideOverlay();
    typer.detach();
  });
}
