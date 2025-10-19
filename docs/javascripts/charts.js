// Copyright (c) Microsoft. All rights reserved.

// ---- CSS helpers ---------------------------------------------------------
function matVar(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}
function toRGBA(color, a = 1) {
  if (!color) return `rgba(0,0,0,${a})`;
  const m = color.match(/^#?([\da-f]{3}|[\da-f]{6})$/i);
  if (m) {
    const hex = m[1].length === 3 ? m[1].split("").map((x) => x + x).join("") : m[1];
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
  const font = matVar("--md-text-font").replace(/['"]/g, "") || "Roboto, sans-serif";
  const text = matVar("--md-default-fg-color") || "#1f2937";
  const border = "rgba(128, 128, 128, 0.1)"
  const bg = "#777777";

  Chart.defaults.font.family = font;
  Chart.defaults.font.size = 16;
  Chart.defaults.color = text;
  Chart.defaults.borderColor = border;
  Chart.defaults.backgroundColor = bg;

  Chart.defaults.scale.grid.color = border;
  Chart.defaults.scale.ticks.color = text;
  Chart.defaults.scale.title.color = text;

  Chart.defaults.plugins.legend.labels.color = text;
  Chart.defaults.plugins.tooltip.titleColor = text;
  Chart.defaults.plugins.tooltip.bodyColor = text;
  Chart.defaults.plugins.tooltip.backgroundColor = toRGBA(bg, 0.5);
  Chart.defaults.plugins.tooltip.borderColor = border;
  Chart.defaults.plugins.tooltip.borderWidth = 1;

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
    return {
      ...ds,
      borderColor: toRGBA(color, 0.8),
      backgroundColor: toRGBA(color, 0.3),
      pointBackgroundColor: color,
      pointBorderColor: color,
    };
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
          duration: 1500,
          easing: "easeOutCubic",
        },
        radius: {
          from: 0,
          to: 3,
          duration: 300,
          delay: (ctx) => ctx.dataIndex * 30,
        },
      },
      elements: { line: { tension: 0.3 } },
    },
  };
  const merged = deepMerge({}, globalDefaults);
  applyDatasetDefaults(baseCfg);
  deepMerge(merged, baseCfg); // user config wins
  return merged;
}

(function () {
  // registry stores per-canvas state: { chart, cfg }
  const registry = new WeakMap(); // canvas -> { chart, cfg }

  // IntersectionObserver to (re)animate when visible
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;

        const canvas = entry.target;
        const state = registry.get(canvas);
        if (!state || !state.cfg) return;

        // Respect reduced-motion: if disabled, just update without animation
        const prefersReduced =
          window.matchMedia("(prefers-reduced-motion: reduce)").matches;

        // Destroy & rebuild to guarantee a fresh animation
        if (state.chart) {
          try {
            state.chart.destroy();
          } catch (_) {}
        }
        const ctx = canvas.getContext("2d");
        const cfg = buildConfig(JSON.parse(JSON.stringify(state.cfg)));

        // If reduced motion, skip animations
        if (prefersReduced) {
          cfg.options = cfg.options || {};
          cfg.options.animation = false;
        }

        state.chart = new Chart(ctx, cfg);
        registry.set(canvas, state);
      });
    },
    { threshold: 0.3 } // animate when ~30% visible
  );

  // ---- Render all canvases with data-chart JSON ---------------------------
  function renderAll() {
    document.querySelectorAll("canvas[data-chart]").forEach((canvas) => {
      let parsedCfg;
      try {
        parsedCfg = JSON.parse(canvas.getAttribute("data-chart"));
      } catch (e) {
        console.error("Invalid data-chart JSON:", e, canvas);
        return;
      }

      // store config; chart will be created by IntersectionObserver when visible
      if (!registry.get(canvas)) {
        registry.set(canvas, { chart: null, cfg: parsedCfg });
        io.observe(canvas);
      }
    });
  }

  // ---- Retheme on scheme/primary/accent change ----------------------------
  function retheme() {
    applyThemeDefaults();
    // Update visible charts without forcing animation
    document.querySelectorAll("canvas[data-chart]").forEach((c) => {
      const state = registry.get(c);
      if (state?.chart) {
        // Fix the issue that scale options color are not updated when theme changes
        const scaleOptions = state.chart.options.scales;
        for (const key of Object.keys(scaleOptions)) {
          scaleOptions[key].ticks.color = Chart?.defaults?.scale?.ticks?.color;
          scaleOptions[key].title.color = Chart?.defaults?.scale?.title?.color;
        }
        state.chart.update("none");
      }
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
  const attrs = ["data-md-color-scheme", "data-md-color-primary", "data-md-color-accent"];
  const obs = new MutationObserver(retheme);
  obs.observe(document.documentElement, {
    attributes: true,
    attributeFilter: attrs,
    subtree: true,
  });

  // Re-scan on SPA navigations (Material)
  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(() => {
      renderAll(); // new canvases
      retheme();   // keep colors in sync
    });
  }
})();
