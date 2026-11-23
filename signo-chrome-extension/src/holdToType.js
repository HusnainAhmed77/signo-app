// Hold-to-type state machine. Pure logic, no browser APIs, so it can be unit-tested.
//
//   idle ──hand──▶ searching ──stable letter──▶ holding ──held holdTimeMs──▶ locked (letter typed)
//     ▲                ▲   ◀──letter changes/drops──┘                          │
//     └──hand gone ≥ RELEASE_MS (from any phase)    └──── released: hand relaxes or changes letter
//
// While locked, the same letter is never typed twice. To repeat a letter, lower your hand
// or relax it briefly, then sign it again.

export const SMOOTHING_WINDOW = 4; // predictions averaged (~0.6 s at 150 ms per prediction)
export const MIN_MARGIN = 0.15; // top-1 must beat top-2 by this much in smoothed probability
export const RELEASE_MS = 250; // hand must be gone this long to count as "lowered"
export const RELEASE_OBSERVATIONS = 2; // consecutive predictions that must differ to unlock

export class HoldToType {
  constructor() {
    this.reset();
  }

  reset() {
    this.phase = "idle";
    this.history = [];
    this.smoothed = null;
    this.candidate = -1;
    this.holdStart = 0;
    this.locked = -1;
    this.releaseCount = 0;
    this.lostSince = null;
  }

  /** Feed one classifier probability vector (hand visible). Returns a snapshot. */
  observe(probs, now, { minConfidence, holdTimeMs, autoCommit }) {
    this.lostSince = null;
    this.history.push(Float32Array.from(probs));
    if (this.history.length > SMOOTHING_WINDOW) this.history.shift();
    this.smoothed = average(this.history);

    const [top, second] = topTwo(this.smoothed);
    const qualifies =
      this.smoothed[top] >= minConfidence && this.smoothed[top] - (second >= 0 ? this.smoothed[second] : 0) >= MIN_MARGIN;

    if (this.phase === "locked") {
      const stillSame = top === this.locked && this.smoothed[top] >= minConfidence * 0.6;
      this.releaseCount = stillSame ? 0 : this.releaseCount + 1;
      if (this.releaseCount >= RELEASE_OBSERVATIONS) {
        this.locked = -1;
        this.releaseCount = 0;
        this.phase = "searching";
      }
    }

    if (this.phase !== "locked") {
      if (!qualifies) {
        this.phase = "searching";
        this.candidate = -1;
      } else if (top !== this.candidate || this.phase !== "holding") {
        this.phase = "holding";
        this.candidate = top;
        this.holdStart = now;
      }
    }
    return this.tick(now, { holdTimeMs, autoCommit });
  }

  /** Hand not visible in this frame. */
  handLost(now) {
    if (this.lostSince === null) this.lostSince = now;
  }

  /** Advance timers (call every frame). Returns a snapshot; `commit` is set at most once per hold. */
  tick(now, { holdTimeMs, autoCommit }) {
    if (this.lostSince !== null && now - this.lostSince >= RELEASE_MS && this.phase !== "idle") {
      this.reset();
    }
    let commit = -1;
    let progress = 0;
    if (this.phase === "holding") {
      progress = Math.min(1, (now - this.holdStart) / holdTimeMs);
      if (autoCommit && progress >= 1) {
        commit = this.candidate;
        this.lock(commit);
      }
    } else if (this.phase === "locked") {
      progress = 1;
    }
    return this.snapshot(progress, commit);
  }

  /** Mark a letter as typed by another trigger (pinch / button) so it is not repeated. */
  lock(index) {
    this.phase = "locked";
    this.locked = index;
    this.candidate = -1;
    this.releaseCount = 0;
  }

  snapshot(progress = 0, commit = -1) {
    const index = this.smoothed ? topTwo(this.smoothed)[0] : -1;
    return {
      phase: this.phase,
      index,
      confidence: index >= 0 ? this.smoothed[index] : 0,
      probs: this.smoothed,
      progress,
      commit,
    };
  }
}

function average(vectors) {
  const out = new Float32Array(vectors[0].length);
  for (const v of vectors) for (let i = 0; i < out.length; i++) out[i] += v[i];
  for (let i = 0; i < out.length; i++) out[i] /= vectors.length;
  return out;
}

function topTwo(values) {
  let first = -1;
  let second = -1;
  for (let i = 0; i < values.length; i++) {
    if (first < 0 || values[i] > values[first]) {
      second = first;
      first = i;
    } else if (second < 0 || values[i] > values[second]) {
      second = i;
    }
  }
  return [first, second];
}
