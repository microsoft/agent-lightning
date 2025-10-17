// Copyright (c) Microsoft. All rights reserved.

// ---- CSS helpers ---------------------------------------------------------
function matVar(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}
function toRGBA(color, a = 1) {
  if (!color) return `rgba(0,0,0,${a})`;
  const m = color.match(/^#?([\da-f]{3}|[\da-f]{6})$/i);
  if (m) {
    const hex =
      m[1].length === 3
        ? m[1]
            .split("")
            .map((x) => x + x)
            .join("")
        : m[1];
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${a})`;
  }
  const nums = color.match(/[\d.]+/g) || [0, 0, 0, 1];
  const [r, g, b] = nums.map(Number);
  return `rgba(${r | 0}, ${g | 0}, ${b | 0}, ${a})`;
}

// ---- Theme defaults (pulled from MkDocs Material CSS vars) ---------------
function applyThemeDefaults() {
  const font =
    matVar("--md-text-font").replace(/['"]/g, "") || "Roboto, sans-serif";
  const text = matVar("--md-default-fg-color") || "#1f2937";
  const border = matVar("--md-typeset-border-color") || "rgba(0,0,0,.12)";
  const bg = "#777777";

  Chart.defaults.font.family = font;
  Chart.defaults.font.size = 16;
  Chart.defaults.color = text;
  Chart.defaults.borderColor = border;
  Chart.defaults.backgroundColor = bg;

  Chart.defaults.scale.grid.color = border;
  Chart.defaults.scale.ticks.color = text;

  Chart.defaults.plugins.legend.labels.color = text;
  Chart.defaults.plugins.tooltip.titleColor = text;
  Chart.defaults.plugins.tooltip.bodyColor = text;
  Chart.defaults.plugins.tooltip.backgroundColor = toRGBA(bg, 0.3);
  Chart.defaults.plugins.tooltip.borderColor = border;
  Chart.defaults.plugins.tooltip.borderWidth = 1;

  // Sensible global behavior
  Chart.defaults.responsive = true;
  Chart.defaults.maintainAspectRatio = false;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    Chart.defaults.animation = false;
  }
}

// ---- Dataset color defaults (Material primary/accent) --------------------
const colorScheme = ["#c45259", "#5276c4", "#f69047", "#7cc452", "#c2b00a"];

function applyDatasetDefaults(config) {
  if (!config.data || !Array.isArray(config.data.datasets)) return;

  config.data.datasets = config.data.datasets.map((ds, index) => {
    const color = colorScheme[index % colorScheme.length];

    const result = {
      ...ds,
      borderColor: toRGBA(color, 0.8),
      backgroundColor: toRGBA(color, 0.3),
      pointBackgroundColor: color,
      pointBorderColor: color,
    };
    return result;
  });
}

// ---- Deep merge (config JSON + our defaults) ----------------------------
function deepMerge(target, src) {
  if (!src || typeof src !== "object") return target;
  for (const k of Object.keys(src)) {
    const v = src[k];
    if (v && typeof v === "object" && !Array.isArray(v)) {
      target[k] = deepMerge(target[k] || {}, v);
    } else {
      target[k] = v;
    }
  }
  return target;
}

// ---- Build final config for a canvas ------------------------------------
function buildConfig(baseCfg) {
  const type = (baseCfg.type || "").toLowerCase();

  // site-wide chart defaults
  const globalDefaults = {
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "top" },
        tooltip: { enabled: true },
      },
      layout: { padding: { top: 8, right: 8, bottom: 0, left: 0 } },
      normalized: true,
      alignToPixels: true,
      animations: {
        y: {
          from: (ctx) => 300,
          duration: 800,
          easing: "easeOutCubic",
        },
        // optional little point fade-in
        radius: {
          from: 0,
          to: 3,
          duration: 300,
          delay: (ctx) => ctx.dataIndex * 100,
        },
      },
      // subtle curve looks nicer with draw-in
      elements: { line: { tension: 0.3 } },
    },
  };

  const merged = deepMerge({}, globalDefaults);
  applyDatasetDefaults(baseCfg);
  deepMerge(merged, baseCfg); // user config wins
  console.log(merged);
  return merged;
}

(function () {
  const registry = new WeakMap(); // canvas -> chart

  // ---- Render all canvases with data-chart JSON ---------------------------
  function renderAll() {
    document.querySelectorAll("canvas[data-chart]").forEach((canvas) => {
      if (registry.get(canvas)) return; // already initialized

      let cfg;
      try {
        cfg = JSON.parse(canvas.getAttribute("data-chart"));
      } catch (e) {
        console.error("Invalid data-chart JSON:", e, canvas);
        return;
      }

      const ctx = canvas.getContext("2d");

      const finalCfg = buildConfig(cfg);
      const chart = new Chart(ctx, finalCfg);
      registry.set(canvas, chart);
    });
  }

  // ---- Retheme on scheme/primary/accent change ----------------------------
  function retheme() {
    applyThemeDefaults();
    registry.forEach?.((chart) => chart.update("none"));
    // Fallback for browsers without WeakMap iteration support:
    document.querySelectorAll("canvas[data-chart]").forEach((c) => {
      const ch = registry.get(c);
      if (ch) ch.update("none");
    });
  }

  // Initial theme + render (works on hard refresh)
  function boot() {
    applyThemeDefaults();
    renderAll();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }

  // Observe theme flips. Attributes might be on <html> or <body>.
  const attrs = [
    "data-md-color-scheme",
    "data-md-color-primary",
    "data-md-color-accent",
  ];
  const obs = new MutationObserver(retheme);
  obs.observe(document.documentElement, {
    attributes: true,
    attributeFilter: attrs,
    subtree: true,
  });

  // Re-render on SPA navigations (Material)
  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(() => {
      // new canvases may appear after navigation
      renderAll();
      retheme();
    });
  }
})();
