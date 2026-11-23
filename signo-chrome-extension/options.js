// options.js - Sign-O Options UI
// Handles settings UI interactions and storage

document.addEventListener("DOMContentLoaded", () => {
  // DOM elements
  const modeGroup = document.getElementById("modeGroup");
  const handGroup = document.getElementById("handGroup");
  const dwellSettings = document.getElementById("dwellSettings");
  const saveBtn = document.getElementById("saveBtn");
  const dwellTimeInput = document.getElementById("dwellTimeInput");

  // Load current settings
  chrome.storage.sync.get(
    ["currentMode", "dwellTimeMs", "pinchThreshold", "clickHand"],
    (settings) => {
      // Set mode buttons
      modeGroup.querySelectorAll(".radio-option").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.mode === settings.currentMode);
      });

      // Show/hide dwell settings based on mode
      if (settings.currentMode === "dwell") {
        dwellSettings.style.display = "block";
        dwellTimeInput.value = settings.dwellTimeMs || 1500;
      } else {
        dwellSettings.style.display = "none";
      }

      // Set hand selection
      handGroup.querySelectorAll(".radio-option").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.hand === settings.clickHand);
      });
    }
  );

  // Mode button clicks
  modeGroup.addEventListener("click", (e) => {
    const target = e.target.closest(".radio-option");
    if (!target) return;

    // Update UI
    modeGroup.querySelectorAll(".radio-option").forEach((btn) => btn.classList.remove("active"));
    target.classList.add("active");

    const mode = target.dataset.mode;
    CONFIG.currentMode = mode;

    // Show/hide dwell settings
    if (mode === "dwell") {
      dwellSettings.style.display = "block";
    } else {
      dwellSettings.style.display = "none";
    }
  });

  // Hand button clicks
  handGroup.addEventListener("click", (e) => {
    const target = e.target.closest(".radio-option");
    if (!target) return;

    handGroup.querySelectorAll(".radio-option").forEach((btn) => btn.classList.remove("active"));
    target.classList.add("active");

    chrome.storage.sync.set({ clickHand: target.dataset.hand }, () => {
      console.log("Hand preference saved:", target.dataset.hand);
    });
  });

  // Dwell time input
  dwellTimeInput.addEventListener("change", () => {
    const value = parseInt(dwellTimeInput.value) || 1500;
    dwellTimeInput.value = Math.max(500, Math.min(3000, value));
    
    chrome.storage.sync.set({ dwellTimeMs: dwellTimeInput.value }, () => {
      console.log("Dwell time saved:", dwellTimeInput.value + "ms");
    });
  });

  // Save button
  saveBtn.addEventListener("click", () => {
    // Collect all settings
    const modeButtons = modeGroup.querySelectorAll(".radio-option");
    let selectedMode = "pinch";
    modeButtons.forEach((btn) => {
      if (btn.classList.contains("active")) {
        selectedMode = btn.dataset.mode;
      }
    });

    // Save all settings
    const settings = {
      currentMode: selectedMode,
    };

    // Add dwell time if in dwell mode
    if (selectedMode === "dwell") {
      const dwellValue = parseInt(dwellTimeInput.value) || 1500;
      settings.dwellTimeMs = Math.max(500, Math.min(3000, dwellValue));
    }

    // Save hand selection (already saved on click, but also here)
    const handButtons = handGroup.querySelectorAll(".radio-option");
    let clickHand = "left";
    handButtons.forEach((btn) => {
      if (btn.classList.contains("active")) {
        clickHand = btn.dataset.hand;
      }
    });
    settings.clickHand = clickHand;

    chrome.storage.sync.set(settings, () => {
      console.log("All settings saved:", settings);
      // Notify content/popup scripts
      chrome.runtime.sendMessage({
        type: "SETTINGS_UPDATED",
        settings: settings,
      });
      // Simple feedback
      saveBtn.textContent = "Saved!";
      setTimeout(() => saveBtn.textContent = "Save Settings", 1500);
    });
  });
});

// Global config object that popup.js also uses
window.CONFIG = window.CONFIG || {
  currentMode: "pinch",
  dwellTimeMs: 1500,
  pinchThreshold: 0.06,
};
