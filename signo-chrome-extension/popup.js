// Sign-O Chrome Extension - popup
// Camera -> MediaPipe HandLandmarker (hand presence, pinch) -> MobileNetV3 ONNX classifier
// (same model and preprocessing as the desktop app) -> confirm -> type into the page.
// All code and models are bundled locally: Manifest V3 forbids loading remote scripts.

import { FilesetResolver, HandLandmarker } from "./lib/mediapipe/vision_bundle.mjs";

const ort = globalThis.ort;

const MODEL_URL = "models/best_model_tl.onnx";
const CLASSES_URL = "models/classes.json";
const HAND_MODEL_URL = "models/hand_landmarker.task";
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];
const PREDICTION_INTERVAL_MS = 400; // matches the desktop app
const PINCH_COOLDOWN_MS = 1000;

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

const settings = {
  currentMode: "pinch", // pinch | dwell | mouse
  dwellTimeMs: 1500,
  pinchThreshold: 0.06,
  clickHand: "left",
  minConfidence: 0.5,
};

const state = {
  classes: [], // [{label, character}] in model output order
  imageSize: [224, 224],
  session: null,
  landmarker: null,
  lastPrediction: null,
  lastPredictionTime: 0,
  predicting: false,
  lastPinchTime: 0,
  dwellLabel: null,
  dwellStart: 0,
  lastVideoTime: -1,
};

const $ = (id) => document.getElementById(id);
const dom = {
  video: $("cameraFeed"),
  overlay: $("landmarksOverlay"),
  predictionDisplay: $("predictionDisplay"),
  predictionDesc: $("predictionDesc"),
  top3: $("top3"),
  typeBtn: $("typeBtn"),
  spaceBtn: $("spaceBtn"),
  backspaceBtn: $("backspaceBtn"),
  clearBtn: $("clearBtn"),
  cameraPermissionBtn: $("cameraPermissionBtn"),
  modeBtns: document.querySelectorAll(".mode-btn"),
  statusText: $("statusText"),
  modeHint: $("modeHint"),
};
const overlayCtx = dom.overlay.getContext("2d");
const inputCanvas = document.createElement("canvas");
const inputCtx = inputCanvas.getContext("2d", { willReadFrequently: true });

// ============================================
// Startup
// ============================================
document.addEventListener("DOMContentLoaded", init);
if (document.readyState !== "loading") init();

let started = false;
async function init() {
  if (started) return;
  started = true;
  setupEventListeners();
  await loadSettings();
  try {
    setStatus("Loading models…");
    await Promise.all([loadClassifier(), loadHandLandmarker()]);
  } catch (err) {
    console.error("Model loading failed:", err);
    setStatus(`Could not load models: ${err.message}`, true);
    dom.predictionDesc.textContent = "Models failed to load.";
    return;
  }
  dom.predictionDesc.textContent = "Show one hand to see a prediction.";
  await startCamera();
}

async function loadSettings() {
  const stored = await chrome.storage.sync.get(Object.keys(settings));
  applySettings(stored);
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "sync") return;
    const updated = {};
    for (const [key, { newValue }] of Object.entries(changes)) updated[key] = newValue;
    applySettings(updated);
  });
}

function applySettings(values) {
  for (const key of Object.keys(settings)) {
    if (values[key] === undefined) continue;
    settings[key] = typeof settings[key] === "number" ? Number(values[key]) : values[key];
  }
  updateModeUI();
}

async function loadClassifier() {
  const meta = await (await fetch(CLASSES_URL)).json();
  state.classes = meta.classes;
  state.imageSize = meta.image_size;
  [inputCanvas.width, inputCanvas.height] = state.imageSize;

  ort.env.wasm.wasmPaths = "lib/onnxruntime/";
  ort.env.wasm.numThreads = 1; // threaded WASM needs cross-origin isolation
  state.session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
}

async function loadHandLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks("lib/mediapipe/wasm");
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    state.landmarker = await HandLandmarker.createFromOptions(fileset, options("GPU"));
  } catch (err) {
    console.warn("GPU delegate unavailable, using CPU:", err);
    state.landmarker = await HandLandmarker.createFromOptions(fileset, options("CPU"));
  }
}

// ============================================
// Camera
// ============================================
async function startCamera() {
  dom.cameraPermissionBtn.classList.add("hidden");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    dom.video.srcObject = stream;
    await dom.video.play();
    dom.overlay.width = dom.video.videoWidth;
    dom.overlay.height = dom.video.videoHeight;
    setStatus(modeStatusText());
    scheduleFrame();
  } catch (err) {
    console.error("Camera error:", err);
    if (err.name === "NotAllowedError" || err.name === "SecurityError") {
      // Popups cannot display Chrome's permission prompt; ask in a tab instead.
      setStatus("Camera permission needed. Click the button above, allow it, then reopen Sign-O.", true);
      dom.cameraPermissionBtn.classList.remove("hidden");
    } else {
      setStatus(`Camera unavailable: ${err.message}`, true);
    }
  }
}

function scheduleFrame() {
  if (dom.video.requestVideoFrameCallback) dom.video.requestVideoFrameCallback(processFrame);
  else requestAnimationFrame(processFrame);
}

// ============================================
// Per-frame loop
// ============================================
function processFrame() {
  const video = dom.video;
  if (!video.srcObject) return;
  if (video.currentTime !== state.lastVideoTime && video.readyState >= 2) {
    state.lastVideoTime = video.currentTime;
    let result = null;
    try {
      result = state.landmarker.detectForVideo(video, performance.now());
    } catch (err) {
      console.warn("Hand tracking failed:", err);
    }
    const hands = toHands(result);
    drawHands(hands);

    if (hands.length) {
      const now = performance.now();
      if (!state.predicting && now - state.lastPredictionTime >= PREDICTION_INTERVAL_MS) {
        state.lastPredictionTime = now;
        predictFrame();
      }
      handleConfirmation(hands);
    } else if (state.lastPrediction) {
      clearPrediction("Show one hand to see a prediction.");
    }
  }
  scheduleFrame();
}

function toHands(result) {
  if (!result || !result.landmarks) return [];
  return result.landmarks.map((landmarks, i) => {
    const category = result.handedness?.[i]?.[0]?.categoryName || "Unknown";
    // MediaPipe labels handedness as if the image were mirrored. The camera frame is
    // not mirrored, so swap to get the user's real hand.
    const handedness = category === "Left" ? "right" : category === "Right" ? "left" : "unknown";
    const thumb = landmarks[4];
    const index = landmarks[8];
    const pinchDistance = Math.hypot(thumb.x - index.x, thumb.y - index.y);
    return { handedness, landmarks, pinch: pinchDistance < settings.pinchThreshold };
  });
}

function drawHands(hands) {
  const { width, height } = dom.overlay;
  overlayCtx.clearRect(0, 0, width, height);
  for (const hand of hands) {
    const pts = hand.landmarks.map((p) => [p.x * width, p.y * height]);
    overlayCtx.lineWidth = 3;
    overlayCtx.strokeStyle = hand.pinch ? "#d4af37" : "#2e7d32";
    overlayCtx.beginPath();
    for (const [a, b] of HAND_CONNECTIONS) {
      overlayCtx.moveTo(...pts[a]);
      overlayCtx.lineTo(...pts[b]);
    }
    overlayCtx.stroke();
    overlayCtx.fillStyle = "#ff7a2f";
    for (const [x, y] of pts) {
      overlayCtx.beginPath();
      overlayCtx.arc(x, y, 4, 0, Math.PI * 2);
      overlayCtx.fill();
    }
  }
}

// ============================================
// Classification (mirrors signo_app/predictor.py)
// ============================================
async function predictFrame() {
  state.predicting = true;
  try {
    const tensor = frameToTensor(dom.video);
    const output = await state.session.run({ input: tensor });
    const probs = softmax(output.logits.data);
    const ranked = Array.from(probs, (p, i) => [i, p]).sort((a, b) => b[1] - a[1]).slice(0, 3);
    const top3 = ranked.map(([i, p]) => ({ ...state.classes[i], confidence: p }));
    state.lastPrediction = top3[0];
    state.lastPrediction.top3 = top3;
    updatePredictionUI();
  } catch (err) {
    console.error("Prediction failed:", err);
    setStatus(`Prediction failed: ${err.message}`, true);
  } finally {
    state.predicting = false;
  }
}

// Centered white padding to 224x224, RGB, ImageNet normalization, CHW float32.
function frameToTensor(video) {
  const [w, h] = state.imageSize;
  const scale = Math.min(w / video.videoWidth, h / video.videoHeight);
  const dw = Math.round(video.videoWidth * scale);
  const dh = Math.round(video.videoHeight * scale);
  inputCtx.fillStyle = "#ffffff";
  inputCtx.fillRect(0, 0, w, h);
  inputCtx.imageSmoothingEnabled = true;
  inputCtx.imageSmoothingQuality = "high";
  inputCtx.drawImage(video, Math.floor((w - dw) / 2), Math.floor((h - dh) / 2), dw, dh);

  const { data } = inputCtx.getImageData(0, 0, w, h);
  const plane = w * h;
  const chw = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      chw[c * plane + i] = (data[i * 4 + c] / 255 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
    }
  }
  return new ort.Tensor("float32", chw, [1, 3, h, w]);
}

function softmax(logits) {
  let max = -Infinity;
  for (const v of logits) max = Math.max(max, v);
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function canType(prediction) {
  return Boolean(prediction && prediction.character && prediction.confidence >= settings.minConfidence);
}

function updatePredictionUI() {
  const p = state.lastPrediction;
  dom.predictionDisplay.textContent = p.character || "—";
  dom.predictionDesc.textContent = `${p.label} · ${(p.confidence * 100).toFixed(1)}%` +
    (p.confidence < settings.minConfidence ? ` (below ${Math.round(settings.minConfidence * 100)}% threshold)` : "");
  dom.top3.textContent = p.top3
    .map((t) => `${t.character} ${t.label}: ${(t.confidence * 100).toFixed(1)}%`)
    .join("\n");
  dom.typeBtn.disabled = !canType(p);
}

function clearPrediction(message) {
  state.lastPrediction = null;
  state.dwellLabel = null;
  dom.predictionDisplay.textContent = "— — —";
  dom.predictionDesc.textContent = message;
  dom.top3.textContent = "";
  dom.typeBtn.disabled = true;
}

// ============================================
// Confirmation (mirrors the desktop app)
// ============================================
function handleConfirmation(hands) {
  const prediction = state.lastPrediction;
  if (settings.currentMode === "mouse" || !canType(prediction)) return;
  const now = performance.now();
  let triggered = false;

  if (settings.currentMode === "pinch") {
    const clickHand = hands.find((h) => h.handedness === settings.clickHand);
    if (clickHand && clickHand.pinch && now - state.lastPinchTime >= PINCH_COOLDOWN_MS) {
      state.lastPinchTime = now;
      triggered = true;
    }
  } else if (settings.currentMode === "dwell") {
    if (state.dwellLabel === prediction.label) {
      triggered = now - state.dwellStart >= settings.dwellTimeMs;
    } else {
      state.dwellLabel = prediction.label;
      state.dwellStart = now;
    }
  }

  if (triggered) {
    state.dwellLabel = null;
    sendEdit("insert", prediction.character);
  }
}

// ============================================
// Typing into the active tab
// ============================================
async function sendEdit(action, text) {
  let tab;
  try {
    [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  } catch (err) {
    setStatus(`No active tab: ${err.message}`, true);
    return;
  }
  if (!tab || !tab.id || !/^(https?|file):/.test(tab.url || "")) {
    setStatus("Chrome does not allow typing on this page. Open a normal web page.", true);
    return;
  }
  const message = { type: "SIGNO_EDIT", action, text };
  let response;
  try {
    response = await chrome.tabs.sendMessage(tab.id, message);
  } catch {
    // Tab was open before the extension loaded: inject the content script, then retry.
    try {
      await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["content.js"] });
      response = await chrome.tabs.sendMessage(tab.id, message);
    } catch (err) {
      setStatus(`Could not reach the page: ${err.message}`, true);
      return;
    }
  }
  if (!response || !response.ok) {
    setStatus(
      response && response.reason === "no-field"
        ? "Click into a text box on the page first, then reopen Sign-O."
        : `Typing failed: ${response ? response.reason : "no response"}`,
      true,
    );
    return;
  }
  const labels = { insert: `Typed: ${text}`, backspace: "Backspace", clear: "Field cleared" };
  flashStatus(labels[action]);
}

// ============================================
// UI wiring
// ============================================
function setupEventListeners() {
  dom.modeBtns.forEach((btn) =>
    btn.addEventListener("click", () => chrome.storage.sync.set({ currentMode: btn.dataset.mode })),
  );
  dom.typeBtn.addEventListener("click", () => {
    if (canType(state.lastPrediction)) sendEdit("insert", state.lastPrediction.character);
  });
  dom.spaceBtn.addEventListener("click", () => sendEdit("insert", " "));
  dom.backspaceBtn.addEventListener("click", () => sendEdit("backspace"));
  dom.clearBtn.addEventListener("click", () => sendEdit("clear"));
  dom.cameraPermissionBtn.addEventListener("click", () =>
    chrome.tabs.create({ url: chrome.runtime.getURL("camera.html") }),
  );
  document.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLButtonElement && (e.key === " " || e.key === "Enter")) return;
    if (e.key === " ") { e.preventDefault(); sendEdit("insert", " "); }
    if (e.key === "Backspace") { e.preventDefault(); sendEdit("backspace"); }
  });
}

function modeStatusText() {
  return {
    pinch: `Pinch with your ${settings.clickHand} hand to type; sign with the other.`,
    dwell: `Hold a sign steady for ${(settings.dwellTimeMs / 1000).toFixed(1)} s to type it.`,
    mouse: "Click “Type letter” to type the shown letter.",
  }[settings.currentMode];
}

function updateModeUI() {
  dom.modeBtns.forEach((btn) => btn.classList.toggle("active", btn.dataset.mode === settings.currentMode));
  dom.modeHint.textContent = modeStatusText();
  state.dwellLabel = null;
}

let statusTimer = null;
function setStatus(message, isError = false) {
  clearTimeout(statusTimer);
  dom.statusText.textContent = message;
  dom.statusText.style.color = isError ? "#c62828" : "";
}

function flashStatus(message) {
  setStatus(message);
  statusTimer = setTimeout(() => setStatus(modeStatusText()), 2000);
}
