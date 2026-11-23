// camera.js - one-time camera permission request in a normal tab.
const statusEl = document.getElementById("status");

async function requestCamera() {
  statusEl.className = "";
  statusEl.textContent = "Waiting for permission…";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    stream.getTracks().forEach((track) => track.stop());
    statusEl.className = "ok";
    statusEl.textContent =
      "Camera allowed. You can close this tab, focus a text box on any page, and open the Sign-O popup.";
  } catch (err) {
    statusEl.className = "err";
    statusEl.textContent =
      err.name === "NotAllowedError"
        ? "Camera was blocked. Click the camera icon in the address bar, allow it, then try again."
        : `Could not open the camera: ${err.message}`;
  }
}

document.getElementById("grantBtn").addEventListener("click", requestCamera);
requestCamera();
