function initCharts(root = document) {
  const canvases = root.querySelectorAll("canvas[data-chart]");

  for (const el of canvases) {
    // Avoid double-instantiation on SPA-style page swaps
    if (el._chart) { el._chart.destroy(); }
    console.log("Creating chart for", el);

    const cfg = JSON.parse(el.getAttribute("data-chart"));
    // Make charts responsive by default
    cfg.options = Object.assign(
      { responsive: true, maintainAspectRatio: false },
      cfg.options || {}
    );
    const ctx = el.getContext("2d");
    el._chart = new Chart(ctx, cfg);
  }
}

function initChartThemes() {
  const charts = [];

  function matVar(name) {
    return getComputedStyle(document.body)
      .getPropertyValue(name)
      .trim();
  }

  function applyTheme() {
    Chart.defaults.font.family = matVar('--md-text-font');
    Chart.defaults.color = matVar('--md-default-fg-color');
    Chart.defaults.borderColor = matVar('--md-typeset-border-color');
    Chart.defaults.scale.grid.color = matVar('--md-typeset-border-color');
    Chart.defaults.scale.ticks.color = matVar('--md-default-fg-color');
    charts.forEach(c => c.update());
  }

  // Observe theme switch (Material sets data-md-color-scheme)
  new MutationObserver(applyTheme).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-md-color-scheme']
  });

  // initial theme
  applyTheme();
}

initChartThemes();

// Material for MkDocs: re-run on each virtual page change
if (window.document$) {
  document$.subscribe(() => initCharts(document));
} else {
  // Fallback if not using Material or instant loading is off
  window.addEventListener("DOMContentLoaded", () => initCharts(document));
}
