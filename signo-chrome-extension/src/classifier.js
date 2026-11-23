// MobileNetV3-Small sign classifier (ONNX Runtime Web) with hand-centred cropping.
//
// The model was trained on photos framed around the hand (hand + part of the forearm on a
// plain background). The webcam frame includes the person and the room, so we crop a square
// around the hand landmarks first, then apply the same policy as signo_app/predictor.py:
// centred white padding to 224x224, RGB, ImageNet normalisation, CHW float32.

const ort = globalThis.ort; // loaded by lib/onnxruntime/ort.wasm.min.js

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];
const CROP_SCALE = 1.5; // crop side = longest side of the landmark box x this factor
const WRIST_SHIFT = 0.12; // move the crop centre toward the wrist to include some forearm

export class SignClassifier {
  static async create({ modelUrl, classesUrl, wasmPaths }) {
    const meta = await (await fetch(classesUrl)).json();
    ort.env.wasm.wasmPaths = wasmPaths;
    ort.env.wasm.numThreads = 1; // threaded WASM needs cross-origin isolation
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return new SignClassifier(session, meta);
  }

  constructor(session, meta) {
    this.session = session;
    this.classes = meta.classes; // [{label, character}] in model output order
    [this.width, this.height] = meta.image_size;
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.width;
    this.canvas.height = this.height;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
  }

  /** Classify the hand in `video`. `landmarks` are MediaPipe normalized points (or null for full frame). */
  async classify(video, landmarks) {
    const rect = landmarks ? handCropRect(landmarks, video.videoWidth, video.videoHeight) : fullFrameRect(video);
    this.drawCrop(video, rect);
    const output = await this.session.run({ input: this.toTensor() });
    return softmax(output.logits.data);
  }

  drawCrop(video, rect) {
    const { ctx, width: w, height: h } = this;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    // Letterbox the crop rect into the canvas, then draw only the part inside the frame;
    // anything outside the camera image stays white (same as the training padding colour).
    const scale = Math.min(w / rect.w, h / rect.h);
    const offX = (w - rect.w * scale) / 2;
    const offY = (h - rect.h * scale) / 2;
    const sx = Math.max(0, rect.x);
    const sy = Math.max(0, rect.y);
    const ex = Math.min(video.videoWidth, rect.x + rect.w);
    const ey = Math.min(video.videoHeight, rect.y + rect.h);
    if (ex <= sx || ey <= sy) return;
    ctx.drawImage(
      video,
      sx, sy, ex - sx, ey - sy,
      offX + (sx - rect.x) * scale, offY + (sy - rect.y) * scale, (ex - sx) * scale, (ey - sy) * scale,
    );
  }

  toTensor() {
    const { width: w, height: h } = this;
    const { data } = this.ctx.getImageData(0, 0, w, h);
    const plane = w * h;
    const chw = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      for (let c = 0; c < 3; c++) {
        chw[c * plane + i] = (data[i * 4 + c] / 255 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
      }
    }
    return new ort.Tensor("float32", chw, [1, 3, h, w]);
  }
}

/** Square crop (in video pixels) around the hand, nudged toward the wrist. */
export function handCropRect(landmarks, frameW, frameH) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of landmarks) {
    minX = Math.min(minX, p.x * frameW);
    maxX = Math.max(maxX, p.x * frameW);
    minY = Math.min(minY, p.y * frameH);
    maxY = Math.max(maxY, p.y * frameH);
  }
  const side = Math.max(maxX - minX, maxY - minY) * CROP_SCALE;
  let cx = (minX + maxX) / 2;
  let cy = (minY + maxY) / 2;
  const wrist = landmarks[0];
  cx += (wrist.x * frameW - cx) * WRIST_SHIFT;
  cy += (wrist.y * frameH - cy) * WRIST_SHIFT;
  return { x: cx - side / 2, y: cy - side / 2, w: side, h: side };
}

function fullFrameRect(video) {
  return { x: 0, y: 0, w: video.videoWidth, h: video.videoHeight };
}

export function softmax(logits) {
  let max = -Infinity;
  for (const v of logits) max = Math.max(max, v);
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp(logits[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}
