/**
 * OpenNeuro live stats loader.
 *
 * Fetches docs/assets/data/openneuro-stats.json on every page load
 * (cache: "no-store") and fills three DOM targets:
 *
 *   #on-public-datasets  — formatted integer
 *   #on-participants      — formatted integer
 *   #on-modalities        — <div> filled with .on-modality-pill <span>s
 *
 * The JSON is pre-built by scripts/update_openneuro_stats.py and kept
 * fresh by the openneuro-stats GitHub Action (every 6 h).
 *
 * Runs on initial load and after every MkDocs Material SPA page switch.
 */

(function () {
  "use strict";

  function hasPlaceholders() {
    return !!(
      document.getElementById("on-public-datasets") ||
      document.getElementById("on-participants") ||
      document.getElementById("on-modalities")
    );
  }

  function setText(id, value) {
    var el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  function fmt(n) {
    return new Intl.NumberFormat("en-US").format(n);
  }

  function renderModalities(list) {
    var container = document.getElementById("on-modalities");
    if (!container || !list || !list.length) return;
    container.innerHTML = "";
    list.forEach(function (name) {
      var pill = document.createElement("span");
      pill.className = "on-modality-pill";
      pill.textContent = name;
      container.appendChild(pill);
    });
  }

  function statsUrl() {
    /* document.baseURI is always absolute — set from the MkDocs Material
       <base> tag, or falls back to location.href when no <base> is present. */
    return new URL("assets/data/openneuro-stats.json", document.baseURI).href;
  }

  function applyFallback() {
    setText("on-public-datasets", "1,800+");
    setText("on-participants", "80,000+");
    renderModalities(["MRI", "PET", "MEG", "EEG", "iEEG"]);
  }

  function hydrate() {
    if (!hasPlaceholders()) return;

    fetch(statsUrl(), { cache: "no-store" })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (stats) {
        setText("on-public-datasets", fmt(stats.public_datasets));
        setText("on-participants", fmt(stats.participants));
        renderModalities(stats.modalities || []);
      })
      .catch(function (err) {
        console.warn("[qortex] OpenNeuro stats unavailable:", err);
        applyFallback();
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", hydrate);
  } else {
    hydrate();
  }

  /* MkDocs Material SPA navigation — re-run after every page switch */
  document.addEventListener("DOMContentSwitch", hydrate);
})();
