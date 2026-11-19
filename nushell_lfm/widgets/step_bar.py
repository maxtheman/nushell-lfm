"""A step bar for long marimo apps: sits in the page, sticks to the top once scrolled past, follows the scroll."""

import anywidget
import traitlets


class StepBar(anywidget.AnyWidget):
    """The step bar. It sits below the header, sticks to the top of the window once the page scrolls past it,
    and highlights the step whose section is on screen."""

    _esm = """
    export default {
      render({ model, el }) {
        document.querySelectorAll(".nb-steps, .nb-steps-style").forEach((n) => n.remove());
        const style = document.createElement("style");
        style.className = "nb-steps-style";
        style.textContent = `
          .nb-steps { position: fixed; z-index: 9; box-sizing: border-box; display: flex; gap: 6px; align-items: center; justify-content: flex-start; flex-wrap: wrap;
                      padding: 8px 4px; font: 14px/1.3 system-ui, sans-serif; color: CanvasText;
                      border-top: 1px solid color-mix(in srgb, CanvasText 18%, transparent); border-bottom: 1px solid color-mix(in srgb, CanvasText 18%, transparent); }
          .nb-steps.stuck { padding: 8px 24px; backdrop-filter: blur(10px); background: color-mix(in srgb, Canvas 88%, transparent); border-top-color: transparent; }
          .nb-steps a { color: inherit; text-decoration: none; padding: 4px 12px; border-radius: 999px; white-space: nowrap; border: 1px solid transparent; transition: background 120ms, color 120ms; }
          .nb-steps a:hover { border-color: color-mix(in srgb, CanvasText 30%, transparent); }
          .nb-steps a.on { background: CanvasText; color: Canvas; }
          .nb-steps span { opacity: 0.45; }
          @media (max-width: 720px) { .nb-steps { font-size: 12px; gap: 2px; } .nb-steps a { padding: 3px 8px; } .nb-steps.stuck { padding: 6px 10px; } }
        `;
        document.head.appendChild(style);
        const hold = document.createElement("div");
        hold.style.height = "46px";
        el.appendChild(hold);
        const bar = document.createElement("nav");
        bar.className = "nb-steps";
        const links = model.get("steps").map(([id, label], i) => {
          if (i) { const sep = document.createElement("span"); sep.textContent = "›"; bar.appendChild(sep); }
          const a = document.createElement("a");
          a.textContent = label;
          a.href = "#" + id;
          a.onclick = (e) => {
            e.preventDefault();
            const target = document.getElementById(id);
            if (target) { target.style.scrollMarginTop = "64px"; target.scrollIntoView({ behavior: "smooth", block: "start" }); }
          };
          bar.appendChild(a);
          return [id, a];
        });
        document.body.appendChild(bar);
        let queued = false;
        const place = () => {
          if (queued) return;
          queued = true;
          requestAnimationFrame(() => {
            queued = false;
            hold.style.height = bar.offsetHeight + "px";
            const r = hold.getBoundingClientRect();
            const stuck = r.top <= 0;
            bar.style.top = Math.max(0, r.top) + "px";
            bar.style.left = (stuck ? 0 : r.left) + "px";
            bar.style.width = stuck ? "100vw" : r.width + "px";
            bar.classList.toggle("stuck", stuck);
            bar.style.paddingLeft = stuck ? Math.max(10, r.left + 4) + "px" : "";  // stay in line with the page column
            let active = null;
            for (const [id] of links) {
              const target = document.getElementById(id);
              if (target && target.getBoundingClientRect().top <= window.innerHeight * 0.25) active = id;
            }
            const last = document.getElementById(links[links.length - 1][0]);
            if (last && last.getBoundingClientRect().top <= window.innerHeight - 160) active = links[links.length - 1][0];
            for (const [id, a] of links) a.classList.toggle("on", id === active);
          });
        };
        document.addEventListener("scroll", place, { capture: true, passive: true });
        window.addEventListener("resize", place);
        const watch = new ResizeObserver(place);
        watch.observe(hold);
        place();
        return () => {
          watch.disconnect();
          document.removeEventListener("scroll", place, { capture: true });
          window.removeEventListener("resize", place);
          bar.remove();
          style.remove();
        };
      },
    };
    """
    steps = traitlets.List().tag(sync=True)
