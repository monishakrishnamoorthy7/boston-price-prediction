(() => {
  const views = [...document.querySelectorAll("[data-view]")];
  const navItems = [...document.querySelectorAll("[data-nav]")];
  const validViews = new Set(views.map((view) => view.dataset.view));
  const initialView = document.body.dataset.initialView === "predict" ? "predict" : "home";

  const showView = (name, moveFocus = false) => {
    const viewName = validViews.has(name) ? name : initialView;
    views.forEach((view) => {
      const active = view.dataset.view === viewName;
      view.hidden = !active;
      view.setAttribute("aria-hidden", String(!active));
    });
    navItems.forEach((item) => {
      const active = item.dataset.nav === viewName;
      item.classList.toggle("active", active);
      item.setAttribute("aria-current", active ? "page" : "false");
    });
    if (moveFocus) document.querySelector(`[data-view="${viewName}"]`)?.focus();
    animateResult();
  };

  const navigate = (name, replace = false) => {
    const nextHash = `#${validViews.has(name) ? name : initialView}`;
    if (window.location.hash !== nextHash) {
      const method = replace ? "replaceState" : "pushState";
      window.history[method]({}, "", nextHash);
    }
    showView(nextHash.slice(1), true);
  };

  navItems.forEach((item) => {
    item.addEventListener("click", (event) => {
      event.preventDefault();
      navigate(item.dataset.nav);
    });
  });
  window.addEventListener("hashchange", () => showView(window.location.hash.slice(1), true));
  window.addEventListener("popstate", () => showView(window.location.hash.slice(1), true));

  function animateResult() {
    const result = document.querySelector("[data-view='predict'] [data-value]");
    if (!result || result.closest("[hidden]")) return;
    const target = Number(result.dataset.value);
    const started = performance.now();
    const tick = (now) => {
      const progress = Math.min((now - started) / 900, 1);
      result.textContent = (target * (1 - Math.pow(1 - progress, 3))).toFixed(2);
      if (progress < 1) window.requestAnimationFrame(tick);
    };
    window.requestAnimationFrame(tick);
  }

  const chart = document.querySelector(".chart");
  if (chart) {
    const rows = JSON.parse(chart.dataset.chart);
    const svg = chart.querySelector("svg");
    const points = (key) => rows.map((row, index) => {
      const x = (index / Math.max(rows.length - 1, 1)) * 100;
      const y = 100 - (Math.min(row[key], 52) / 52) * 100;
      return `${x},${y}`;
    }).join(" ");
    ["actual", "predicted"].forEach((key, index) => {
      const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      line.setAttribute("points", points(key));
      line.setAttribute("fill", "none");
      line.setAttribute("stroke", index ? "#bc7753" : "#183b32");
      line.setAttribute("stroke-width", "1.5");
      line.setAttribute("vector-effect", "non-scaling-stroke");
      svg.append(line);
    });
  }

  showView(window.location.hash.slice(1) || initialView);
})();
