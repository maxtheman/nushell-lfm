import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="LFM3 3D Activation Widget (anywidget)")


@app.cell
def _():
    import anywidget
    import marimo as mo
    import traitlets

    LAYER_TYPES = [
        "conv", "conv", "full_attention", "conv", "sliding_attention", "conv",
        "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
        "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv",
        "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
    ]

    DEFAULT_EXAMPLES = [
        {
            "id": "ex0",
            "promptTokens": ["<bos>", "find", "errors", "in", "logs"],
            "responseTokens": ["ls", "**/*", "|", "grep", "-rn", "'err'"],
            "keyCols": [1, 2, 4, 8, 9],
            "syntaxCols": [7],
            "faultSpan": [7, 9],
            "d1Verdict": "invalid_nushell",
            "seed": 202,
        },
        {
            "id": "ex1",
            "promptTokens": ["<bos>", "search", "repo", "for", "db", "pool"],
            "responseTokens": ["glob", "**/*.py", "|", "find", "-i", "'db'"],
            "keyCols": [1, 4, 5, 6, 9, 11],
            "syntaxCols": [8],
            "faultSpan": None,
            "d1Verdict": "valid",
            "seed": 101,
        },
        {
            "id": "ex2",
            "promptTokens": ["<bos>", "open", "cfg", "or", "list", "options"],
            "responseTokens": ["try", "{", "open", "$p", "}", "catch", "{", "glob", "*}"],
            "keyCols": [1, 2, 5, 8, 13],
            "syntaxCols": [6, 7, 10, 11, 12],
            "faultSpan": None,
            "d1Verdict": "valid",
            "seed": 303,
        },
    ]

    class Lfm3ActivationWidget(anywidget.AnyWidget):
        """Minimal LFM3 3D anywidget: 3D model + D1/LoRA toggles + scroll-to-anim + bottom [In/Out] carousel."""

        examples = traitlets.List(default_value=DEFAULT_EXAMPLES).tag(sync=True)
        layer_types = traitlets.List(default_value=LAYER_TYPES).tag(sync=True)
        active_example_idx = traitlets.Int(0).tag(sync=True)
        layer_idx = traitlets.Int(2).tag(sync=True)
        show_d1 = traitlets.Bool(True).tag(sync=True)
        show_lora = traitlets.Bool(True).tag(sync=True)
        active_token_idx = traitlets.Int(8).tag(sync=True)

        _css = """
        .lfm3-root { font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; user-select: none; }
        .lfm3-stage { position: relative; width: 100%; height: 380px; }
        .lfm3-stage canvas { width: 100%; height: 100%; display: block; cursor: grab; touch-action: none; }
        .lfm3-stage canvas:active { cursor: grabbing; }
        .lfm3-title { position: absolute; top: 8px; left: 10px; pointer-events: none; display: flex; align-items: baseline; gap: 8px; }
        .lfm3-title strong { font-size: 14px; letter-spacing: -0.02em; }
        .lfm3-title span { font-size: 11px; font-family: ui-monospace, SFMono-Regular, monospace; opacity: 0.65; }
        .lfm3-scroll-hint { position: absolute; right: 10px; top: 50%; transform: translateY(-50%); pointer-events: none; text-align: right; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 10px; opacity: 0.7; line-height: 1.25; }
        .lfm3-toggles { position: absolute; bottom: 8px; left: 10px; display: flex; gap: 14px; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 11px; }
        .lfm3-toggles label { display: flex; align-items: center; gap: 6px; cursor: pointer; }
        .lfm3-carousel-wrap { border-top: 1px solid rgba(148,163,184,0.3); padding: 8px 4px 2px; }
        .lfm3-carousel { display: flex; gap: 8px; overflow-x: auto; scrollbar-width: none; }
        .lfm3-carousel::-webkit-scrollbar { display: none; }
        .lfm3-card { flex-shrink: 0; border-radius: 6px; border: 1px solid rgba(148,163,184,0.35); padding: 6px 10px; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 11px; display: flex; flex-direction: column; gap: 4px; min-width: 240px; }
        .lfm3-card.active { border-color: #6366f1; }
        .lfm3-card.inactive { opacity: 0.6; cursor: pointer; }
        .lfm3-card.inactive:hover { opacity: 1; }
        .lfm3-row { display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }
        .lfm3-tag { font-size: 10px; opacity: 0.6; width: 26px; flex-shrink: 0; }
        .lfm3-tok { padding: 1px 5px; border-radius: 4px; border: 1px solid transparent; transition: background 80ms; }
        """

        _esm = """
        function mulberry32(a) {
          return function() {
            let t = a += 0x6D2B79F5;
            t = Math.imul(t ^ (t >>> 15), t | 1);
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
          };
        }

        function render({ model, el }) {
          el.innerHTML = `
            <div class="lfm3-root">
              <div class="lfm3-stage">
                <canvas></canvas>
                <div class="lfm3-title"><strong>LFM3</strong><span class="lfm3-layer-badge"></span></div>
                <div class="lfm3-scroll-hint">Scroll<br/>to anim<br/><strong class="lfm3-layer-frac"></strong></div>
                <div class="lfm3-toggles">
                  <label><input class="chk-d1" type="checkbox" /> <span>D1</span></label>
                  <label><input class="chk-lora" type="checkbox" /> <span>LoRA</span></label>
                </div>
              </div>
              <div class="lfm3-carousel-wrap"><div class="lfm3-carousel"></div></div>
            </div>
          `;

          const canvas = el.querySelector("canvas");
          const ctx = canvas.getContext("2d");
          const layerBadge = el.querySelector(".lfm3-layer-badge");
          const layerFrac = el.querySelector(".lfm3-layer-frac");
          const chkD1 = el.querySelector(".chk-d1");
          const chkLora = el.querySelector(".chk-lora");
          const carousel = el.querySelector(".lfm3-carousel");

          let yaw = -0.46, pitch = 0.34, zoom = 0.90;
          let layerFloat = model.get("layer_idx") || 2;
          let hoverTok = null;
          let attnData = null, scene = null;

          const getEx = () => (model.get("examples") || [])[model.get("active_example_idx") || 0];
          const allTokens = (ex) => [...ex.promptTokens, ...ex.responseTokens];
          const activeTok = () => (hoverTok !== null ? hoverTok : model.get("active_token_idx"));

          function computeAttention(ex, isSliding, rng) {
            const T = allTokens(ex).length;
            const win = isSliding ? 3 : T;
            const groups = [];
            for (let g = 0; g < 4; g++) {
              const mat = [];
              for (let q = 0; q < T; q++) {
                const row = new Array(T).fill(0);
                for (let k = 0; k <= q; k++) {
                  if (isSliding && (q - k) > win && k !== 0) { row[k] = 0; continue; }
                  let s = 0.05 + rng() * 0.08;
                  if (g === 0) { if (k === 0) s += 0.90; if (k === q - 1 || k === q) s += 0.70; }
                  else if (g === 1) { s += Math.exp(-(q - k) * 0.75) * 1.15; }
                  else if (g === 2) { if ((ex.keyCols || []).includes(k)) s += 1.15; if (k === 0) s += 0.20; }
                  else { if ((ex.syntaxCols || []).includes(k)) s += 1.05; if (k === q) s += 0.30; }
                  row[k] = s;
                }
                const maxV = Math.max(1e-6, ...row);
                for (let k = 0; k < T; k++) row[k] = k <= q ? Math.min(1, row[k] / maxV) : 0.0;
                mat.push(row);
              }
              groups.push(mat);
            }
            const avg = [];
            for (let q = 0; q < T; q++) {
              const row = [];
              for (let k = 0; k < T; k++) {
                row.push(k > q ? 0.0 : Math.min(1, (groups[0][q][k] + groups[1][q][k] + groups[2][q][k] + groups[3][q][k]) / 4 * 1.15));
              }
              avg.push(row);
            }
            return { groups, avg };
          }

          function buildScene() {
            const ex = getEx();
            if (!ex) return { boxes: [], wires: [], nodes: [] };
            const lts = model.get("layer_types") || [];
            const li = model.get("layer_idx") || 0;
            const lt = lts[li] || "full_attention";
            const isConv = lt === "conv", isFull = lt === "full_attention", isSliding = lt === "sliding_attention";
            const nT = allTokens(ex).length;
            const rng = mulberry32((ex.seed || 101) + li * 31);
            const dSeq = 0.75 + nT * 0.085;
            const showD1 = model.get("show_d1"), showLora = model.get("show_lora");

            attnData = computeAttention(ex, isSliding, rng);

            const makeGrid = (nx, ny, nz, loraMode = false) => {
              const g = { front: [], top: [], side: [] };
              for (let i = 0; i < ny; i++) g.front.push(Array.from({ length: nx }, () => 0.18 + rng() * 0.72));
              for (let t = 0; t < nz; t++) {
                const isFault = ex.faultSpan && t >= ex.faultSpan[0] && t <= ex.faultSpan[1];
                const isResp = t >= ex.promptTokens.length;
                const boost = loraMode ? (isFault ? 0.45 : isResp ? 0.25 : -0.1) : ((ex.keyCols || []).includes(t) ? 0.22 : 0);
                g.top.push(Array.from({ length: nx }, () => Math.min(1, Math.max(0.08, 0.22 + rng() * 0.65 + boost))));
              }
              for (let i = 0; i < ny; i++) g.side.push(Array.from({ length: nz }, () => 0.22 + rng() * 0.65));
              return g;
            };

            const boxes = [], wires = [], nodes = [];
            const addBox = (id, x, y, z, wx, hy, dz, color, nx, ny, nz, customGrid = null, isAttn = false, isConvCore = false, isLora = false) => {
              boxes.push({ id, x, y, z, wx, hy, dz, color, nx, ny, nz, grid: customGrid || makeGrid(nx, ny, nz, isLora), isAttn, isConvCore, isLora });
            };
            const addWire = (pts, color = "#94a3b8", dashed = false) => wires.push({ pts, color, dashed });
            const addNode = (x, y, z, label, color = "#cbd5e1") => nodes.push({ x, y, z, label, color });

            const sx = -2.45;
            addBox("input", sx, -3.2, 0, 0.85, 0.22, dSeq, "#64748b", 2, 1, nT);
            addBox("embed", sx, -2.5, 0, 1.20, 0.30, dSeq, "#6366f1", 4, 2, nT);
            addWire([[sx, -3.09, 0], [sx, -2.65, 0]]);
            addBox("norm1", sx, -1.75, 0, 1.20, 0.16, dSeq, "#94a3b8", 4, 1, nT);
            addWire([[sx, -2.35, 0], [sx, -1.83, 0]]);
            const seqColor = isConv ? "#0ea5e9" : isFull ? "#a855f7" : "#ec4899";
            addBox("seq_spine", sx, -0.95, 0, 1.20, 0.48, dSeq, seqColor, 4, 2, nT);
            addWire([[sx, -1.67, 0], [sx, -1.19, 0]]);
            addBox("norm1_post", sx, -0.32, 0, 1.20, 0.16, dSeq, "#94a3b8", 4, 1, nT);
            addWire([[sx, -0.71, 0], [sx, -0.40, 0]]);
            addNode(sx, 0.15, 0, "+");
            addWire([[sx, -0.24, 0], [sx, 0.03, 0]]);
            addWire([[sx, -2.12, 0], [sx - 0.98, -2.12, 0], [sx - 0.98, 0.15, 0], [sx - 0.14, 0.15, 0]], "#38bdf8");
            addBox("norm2", sx, 0.68, 0, 1.20, 0.16, dSeq, "#94a3b8", 4, 1, nT);
            addWire([[sx, 0.27, 0], [sx, 0.60, 0]]);
            addBox("ffn_spine", sx, 1.40, 0, 1.95, 0.50, dSeq, "#10b981", 8, 2, nT);
            addWire([[sx, 0.76, 0], [sx, 1.15, 0]]);
            addNode(sx, 2.08, 0, "+");
            addWire([[sx, 1.65, 0], [sx, 1.96, 0]]);
            addWire([[sx, 0.40, 0], [sx - 1.25, 0.40, 0], [sx - 1.25, 2.08, 0], [sx - 0.14, 2.08, 0]], "#34d399");
            addBox("norm_out", sx, 2.62, 0, 1.20, 0.18, dSeq, "#94a3b8", 4, 1, nT);
            addWire([[sx, 2.20, 0], [sx, 2.53, 0]]);

            if (showD1) {
              addBox("d1_broken", sx - 0.80, 3.25, 0, 0.38, 0.28, 0.38, "#ef4444", 2, 1, 2);
              addBox("d1_reason", sx + 0.0, 3.25, 0, 0.85, 0.28, 0.38, "#ef4444", 5, 1, 2);
              addBox("d1_span", sx + 0.85, 3.25, 0, 0.42, 0.28, dSeq, "#ef4444", 2, 1, nT);
              addWire([[sx, 2.71, 0], [sx, 3.11, 0]], "#ef4444");
            } else {
              addBox("lm_head", sx, 3.25, 0, 2.15, 0.30, dSeq, "#f43f5e", 8, 2, nT);
              addWire([[sx, 2.71, 0], [sx, 3.10, 0]], "#f43f5e");
            }
            addWire([[sx + 0.62, -0.95, 0], [0.15, -0.95, 0]], seqColor, true);

            const rx = 2.05;
            if (isConv) {
              addBox("conv_in", rx, -2.65, 0, 2.35, 0.34, dSeq, "#38bdf8", 9, 2, nT);
              addBox("conv_B", rx - 1.05, -1.70, 0, 0.66, 0.32, dSeq, "#0284c7", 3, 2, nT);
              addBox("conv_x", rx, -1.70, 0, 0.66, 0.32, dSeq, "#0ea5e9", 3, 2, nT);
              addBox("conv_C", rx + 1.05, -1.70, 0, 0.66, 0.32, dSeq, "#0284c7", 3, 2, nT);
              addWire([[rx - 0.8, -2.48, 0], [rx - 1.05, -1.86, 0]], "#38bdf8");
              addWire([[rx, -2.48, 0], [rx, -1.86, 0]], "#38bdf8");
              addWire([[rx + 0.8, -2.48, 0], [rx + 1.05, -1.86, 0]], "#38bdf8");
              addNode(rx - 0.50, -0.98, 0, "×");
              addWire([[rx - 1.05, -1.54, 0], [rx - 0.60, -1.04, 0]], "#0ea5e9");
              addWire([[rx, -1.54, 0], [rx - 0.40, -1.04, 0]], "#0ea5e9");
              addBox("conv1d", rx - 0.50, -0.15, 0, 1.10, 0.68, dSeq, "#2563eb", 4, 4, nT, null, false, true);
              addWire([[rx - 0.50, -0.85, 0], [rx - 0.50, -0.49, 0]], "#2563eb");
              addNode(rx + 0.35, 0.62, 0, "×");
              addWire([[rx - 0.50, 0.19, 0], [rx + 0.22, 0.58, 0]], "#2563eb");
              addWire([[rx + 1.05, -1.54, 0], [rx + 1.05, 0.62, 0], [rx + 0.48, 0.62, 0]], "#0284c7");
              addBox("conv_out", rx + 0.35, 1.38, 0, 1.20, 0.34, dSeq, "#38bdf8", 4, 2, nT);
              addWire([[rx + 0.35, 0.74, 0], [rx + 0.35, 1.21, 0]], "#38bdf8");
              if (showLora) {
                addBox("lora_conv_a", rx + 1.65, -2.45, 0.15, 0.22, 0.44, dSeq, "#f59e0b", 1, 2, nT, null, false, false, true);
                addBox("lora_conv_b", rx + 1.65, 1.38, 0.15, 0.22, 0.44, dSeq, "#f59e0b", 1, 2, nT, null, false, false, true);
                addWire([[rx + 1.18, -2.65, 0], [rx + 1.54, -2.45, 0.15]], "#f59e0b", true);
                addWire([[rx + 0.96, 1.38, 0], [rx + 1.54, 1.38, 0.15]], "#f59e0b", true);
              }
            } else {
              const gCol = isFull ? "#a855f7" : "#ec4899";
              addBox("gqa_q", rx - 1.15, -2.10, 0, 0.64, 0.92, dSeq, gCol, 3, 4, nT);
              addBox("gqa_k", rx - 0.28, -2.10, 0, 0.64, 0.28, dSeq, "#c084fc", 3, 2, nT);
              addBox("gqa_v", rx + 0.56, -2.10, 0, 0.64, 0.28, dSeq, "#c084fc", 3, 2, nT);
              addBox("gqa_gate", rx + 1.52, -0.95, 0, 0.58, 0.40, dSeq, "#64748b", 3, 2, nT);
              addBox("q_norm", rx - 1.15, -1.25, 0, 0.64, 0.15, dSeq, "#94a3b8", 3, 1, nT);
              addBox("k_norm", rx - 0.28, -1.25, 0, 0.64, 0.15, dSeq, "#94a3b8", 3, 1, nT);
              addWire([[rx - 1.15, -1.64, 0], [rx - 1.15, -1.33, 0]], gCol);
              addWire([[rx - 0.28, -1.96, 0], [rx - 0.28, -1.33, 0]], "#c084fc");
              const tNow = Math.min(nT - 1, Math.max(0, activeTok() ?? nT - 1));
              const frontGqa = attnData.groups.map((mat) => mat[tNow]);
              const sideGqa = attnData.groups.map((mat) => mat.map((row, q) => {
                const act = row.slice(0, q + 1);
                return Math.min(1, (act.reduce((a, b) => a + b, 0) / act.length) * 1.2);
              }));
              addBox("gqa_core", rx - 0.28, -0.15, 0, dSeq, 0.88, dSeq, gCol, nT, 4, nT, { top: attnData.avg, front: frontGqa, side: sideGqa }, true);
              addWire([[rx - 1.15, -1.17, 0], [rx - 0.70, -0.60, 0]], gCol);
              addWire([[rx - 0.28, -1.17, 0], [rx - 0.28, -0.60, 0]], "#c084fc");
              addWire([[rx + 0.56, -1.96, 0], [rx + 0.22, -0.60, 0]], "#c084fc");
              addNode(rx + 0.35, 0.72, 0, "×");
              addWire([[rx - 0.28, 0.29, 0], [rx + 0.22, 0.67, 0]], gCol);
              addWire([[rx + 1.52, -0.75, 0], [rx + 1.52, 0.72, 0], [rx + 0.48, 0.72, 0]], "#64748b");
              addBox("gqa_out", rx + 0.35, 1.45, 0, 1.20, 0.34, dSeq, gCol, 4, 2, nT);
              addWire([[rx + 0.35, 0.84, 0], [rx + 0.35, 1.28, 0]], gCol);
              if (showLora) {
                addBox("lora_gqa_q", rx - 1.72, -2.10, 0.2, 0.20, 0.68, dSeq, "#f59e0b", 1, 3, nT, null, false, false, true);
                addBox("lora_gqa_o", rx + 1.32, 1.45, 0.2, 0.20, 0.34, dSeq, "#f59e0b", 1, 2, nT, null, false, false, true);
                addWire([[rx - 1.47, -2.10, 0], [rx - 1.62, -2.10, 0.2]], "#f59e0b", true);
                addWire([[rx + 0.96, 1.45, 0], [rx + 1.22, 1.45, 0.2]], "#f59e0b", true);
              }
            }
            if (showLora) {
              addBox("lora_ffn", sx + 1.28, 1.40, 0.2, 0.20, 0.50, dSeq, "#f59e0b", 1, 2, nT, null, false, false, true);
              addWire([[sx + 0.98, 1.40, 0], [sx + 1.18, 1.40, 0.2]], "#f59e0b", true);
            }
            return { boxes, wires, nodes };
          }

          function rotatePt(x, y, z) {
            const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
            const x1 = x * cy - z * sy, z1 = x * sy + z * cy;
            return { x: x1, y: (-y) * cp - z1 * sp, z: (-y) * sp + z1 * cp };
          }

          function computeCamera(W, H) {
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            const padX = 40, padTop = 28, padBottom = 30, fov = 16.0, baseDist = 20.0;
            const pts = [];
            for (const b of scene.boxes) {
              const hx = b.wx / 2, hy = b.hy / 2, hz = b.dz / 2;
              for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) pts.push([b.x + sx * hx, b.y + sy * hy, b.z + sz * hz]);
            }
            for (const w of scene.wires) for (const pt of w.pts) pts.push(pt);
            for (const n of scene.nodes) pts.push([n.x, n.y, n.z]);
            for (const [x, y, z] of pts) {
              const r = rotatePt(x, y, z), persp = fov / Math.max(3.0, baseDist + r.z);
              const px = r.x * persp, py = -r.y * persp;
              if (px < minX) minX = px; if (px > maxX) maxX = px;
              if (py < minY) minY = py; if (py > maxY) maxY = py;
            }
            const availW = Math.max(80, W - padX * 2), availH = Math.max(80, H - padTop - padBottom);
            const fitScale = Math.min(availW / Math.max(0.1, maxX - minX), availH / Math.max(0.1, maxY - minY)) * zoom;
            const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
            const centerX = W / 2, centerY = padTop + availH / 2;
            return {
              project(x, y, z) {
                const r = rotatePt(x, y, z), persp = fov / Math.max(3.0, baseDist + r.z);
                return { sx: centerX + (r.x * persp - cx) * fitScale, sy: centerY + (-r.y * persp - cy) * fitScale, depth: r.z };
              }
            };
          }

          const hexToRgb = (h) => { const n = parseInt(h.slice(1), 16); return [(n>>16)&255, (n>>8)&255, n&255]; };

          function drawGriddedQuad(p00, p10, p11, p01, grid2d, rgb, shade, hiR, hiC) {
            const rows = grid2d.length, cols = grid2d[0].length;
            const lerp2 = (u, v) => [
              (p00.sx + (p10.sx - p00.sx) * u) * (1 - v) + (p01.sx + (p11.sx - p01.sx) * u) * v,
              (p00.sy + (p10.sy - p00.sy) * u) * (1 - v) + (p01.sy + (p11.sy - p01.sy) * u) * v
            ];
            const hasFocus = (hiR && hiR.size > 0) || (hiC && hiC.size > 0);
            for (let r = 0; r < rows; r++) {
              for (let c = 0; c < cols; c++) {
                const val = grid2d[r][c];
                const a = lerp2(c/cols, r/rows), b = lerp2((c+1)/cols, r/rows);
                const d = lerp2((c+1)/cols, (r+1)/rows), e = lerp2(c/cols, (r+1)/rows);
                ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.lineTo(d[0], d[1]); ctx.lineTo(e[0], e[1]); ctx.closePath();
                const inFocus = (!hiR || hiR.has(r)) && (!hiC || hiC.has(c));
                if (val <= 0.001) ctx.fillStyle = "rgba(15,23,42,0.72)";
                else {
                  const lum = shade * (0.35 + 0.78 * val) * (hasFocus ? (inFocus ? 1.25 : 0.35) : 1.0);
                  ctx.fillStyle = `rgba(${Math.min(255,Math.round(rgb[0]*lum))},${Math.min(255,Math.round(rgb[1]*lum))},${Math.min(255,Math.round(rgb[2]*lum))},${inFocus?0.96:0.72})`;
                }
                ctx.fill();
                ctx.strokeStyle = (hasFocus && inFocus && val > 0.001) ? "rgba(255,255,255,0.65)" : "rgba(15,23,42,0.28)";
                ctx.lineWidth = (hasFocus && inFocus && val > 0.001) ? 0.9 : 0.45;
                ctx.stroke();
              }
            }
          }

          function draw() {
            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();
            canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr);
            ctx.save(); ctx.scale(dpr, dpr);
            const W = rect.width, H = rect.height;
            ctx.clearRect(0, 0, W, H);
            const li = model.get("layer_idx") || 0;
            const lts = model.get("layer_types") || [];
            const lt = lts[li] || "full_attention";
            layerBadge.textContent = `L${li} · ${lt === "conv" ? "Gated Short Conv (k=7)" : lt === "full_attention" ? "Full GQA" : "Sliding GQA"}`;
            layerFrac.textContent = `${li + 1} / ${lts.length || 24}`;

            const cam = computeCamera(W, H);
            const tFocus = activeTok();
            if (tFocus !== null && attnData) {
              const gqaBox = scene.boxes.find((b) => b.isAttn);
              if (gqaBox && attnData.groups[0][tFocus]) gqaBox.grid.front = attnData.groups.map((mat) => mat[tFocus]);
            }
            for (const w of scene.wires) {
              ctx.save(); ctx.strokeStyle = w.color; ctx.lineWidth = w.dashed ? 1.2 : 1.5;
              if (w.dashed) ctx.setLineDash([4, 3]);
              ctx.beginPath();
              w.pts.forEach((pt, idx) => { const p = cam.project(pt[0], pt[1], pt[2]); if (idx===0) ctx.moveTo(p.sx,p.sy); else ctx.lineTo(p.sx,p.sy); });
              ctx.stroke(); ctx.restore();
            }
            const sorted = scene.boxes.map((b) => ({ b, p: cam.project(b.x, b.y, b.z) })).sort((a, b) => b.p.depth - a.p.depth);
            for (const { b } of sorted) {
              const hx = b.wx / 2, hy = b.hy / 2, hz = b.dz / 2;
              const v = [
                cam.project(b.x-hx,b.y-hy,b.z+hz), cam.project(b.x+hx,b.y-hy,b.z+hz), cam.project(b.x+hx,b.y+hy,b.z+hz), cam.project(b.x-hx,b.y+hy,b.z+hz),
                cam.project(b.x-hx,b.y-hy,b.z-hz), cam.project(b.x+hx,b.y-hy,b.z-hz), cam.project(b.x+hx,b.y+hy,b.z-hz), cam.project(b.x-hx,b.y+hy,b.z-hz),
              ];
              const rgb = hexToRgb(b.color);
              let activeSlices = null;
              if (tFocus !== null && b.nz > 2) {
                activeSlices = new Set([tFocus]);
                if (b.isConvCore) for (let k = Math.max(0, tFocus - 6); k <= tFocus; k++) activeSlices.add(k);
              }
              const faces = [
                { idx: [0,1,2,3], grid: b.grid.front, shade: 1.00, hiR: null, hiC: null },
                { idx: [5,4,7,6], grid: b.grid.front, shade: 0.72, hiR: null, hiC: null },
                { idx: [4,5,1,0], grid: b.grid.top,   shade: 1.18, hiR: activeSlices, hiC: null },
                { idx: [3,2,6,7], grid: b.grid.top,   shade: 0.65, hiR: activeSlices, hiC: null },
                { idx: [1,5,6,2], grid: b.grid.side,  shade: 0.84, hiR: null, hiC: activeSlices },
                { idx: [4,0,3,7], grid: b.grid.side,  shade: 0.78, hiR: null, hiC: activeSlices },
              ];
              const vis = faces.map((f) => {
                const pts = f.idx.map((i) => v[i]);
                const cross = (pts[1].sx-pts[0].sx)*(pts[2].sy-pts[0].sy) - (pts[1].sy-pts[0].sy)*(pts[2].sx-pts[0].sx);
                const depth = pts.reduce((a, p) => a + p.depth, 0) / 4;
                return { ...f, pts, cross, depth };
              }).filter((f) => f.cross > 0).sort((a, b) => b.depth - a.depth);
              for (const f of vis) {
                drawGriddedQuad(f.pts[0], f.pts[1], f.pts[2], f.pts[3], f.grid, rgb, f.shade, f.hiR, f.hiC);
                ctx.beginPath(); ctx.moveTo(f.pts[0].sx,f.pts[0].sy); ctx.lineTo(f.pts[1].sx,f.pts[1].sy);
                ctx.lineTo(f.pts[2].sx,f.pts[2].sy); ctx.lineTo(f.pts[3].sx,f.pts[3].sy); ctx.closePath();
                ctx.strokeStyle = "rgba(15,23,42,0.55)"; ctx.lineWidth = 0.8; ctx.stroke();
              }
            }
            for (const n of scene.nodes) {
              const p = cam.project(n.x, n.y, n.z);
              ctx.beginPath(); ctx.arc(p.sx, p.sy, 7.5, 0, Math.PI * 2);
              ctx.fillStyle = "#0f172a"; ctx.fill(); ctx.strokeStyle = n.color; ctx.lineWidth = 1.4; ctx.stroke();
              ctx.fillStyle = "#f8fafc"; ctx.font = "bold 10px ui-monospace, SFMono-Regular, monospace";
              ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(n.label, p.sx, p.sy + 0.5);
            }
            ctx.restore();
          }

          function renderCarousel() {
            const exs = model.get("examples") || [];
            const cur = model.get("active_example_idx") || 0;
            const li = model.get("layer_idx") || 0;
            const isConv = ((model.get("layer_types") || [])[li]) === "conv";
            const tFocus = activeTok();
            carousel.innerHTML = "";

            exs.forEach((ex, idx) => {
              const isSel = idx === cur;
              const card = document.createElement("div");
              card.className = `lfm3-card ${isSel ? "active" : "inactive"}`;
              if (!isSel) {
                card.onclick = () => {
                  hoverTok = null;
                  model.set("active_example_idx", idx);
                  model.set("active_token_idx", ex.promptTokens.length + 2);
                  model.save_changes();
                };
              }
              const makeRow = (label, toks, offset) => {
                const row = document.createElement("div"); row.className = "lfm3-row";
                const tag = document.createElement("span"); tag.className = "lfm3-tag"; tag.textContent = label;
                row.appendChild(tag);
                toks.forEach((tok, i) => {
                  const gi = offset + i;
                  const sp = document.createElement("span"); sp.className = "lfm3-tok"; sp.textContent = tok;
                  if (isSel) {
                    sp.style.cursor = "pointer";
                    const isAct = gi === tFocus;
                    const isFault = ex.faultSpan && gi >= ex.faultSpan[0] && gi <= ex.faultSpan[1];
                    let w = 0;
                    if (tFocus !== null && gi <= tFocus) {
                      w = isConv ? ((tFocus - gi) < 7 ? 1.0 - (tFocus - gi) * 0.12 : 0) : (attnData && attnData.avg[tFocus] ? attnData.avg[tFocus][gi] : 0);
                    }
                    if (isAct) { sp.style.background = "#6366f1"; sp.style.color = "#fff"; sp.style.borderColor = "#6366f1"; }
                    else if (w > 0.12) {
                      const a = (0.16 + w * 0.64).toFixed(2);
                      sp.style.background = isConv ? `rgba(14,165,233,${a})` : `rgba(168,85,247,${a})`;
                      sp.style.borderColor = isFault ? "#ef4444" : "rgba(148,163,184,0.35)";
                    } else {
                      sp.style.borderColor = isFault ? "#ef4444" : "rgba(148,163,184,0.3)";
                      if (tFocus !== null && gi > tFocus) sp.style.opacity = "0.4";
                    }
                    sp.onmouseenter = () => { hoverTok = gi; renderCarousel(); draw(); };
                    sp.onmouseleave = () => { hoverTok = null; renderCarousel(); draw(); };
                    sp.onclick = (e) => { e.stopPropagation(); model.set("active_token_idx", gi); model.save_changes(); };
                  }
                  row.appendChild(sp);
                });
                return row;
              };
              card.appendChild(makeRow("In", ex.promptTokens, 0));
              card.appendChild(makeRow("Out", ex.responseTokens, ex.promptTokens.length));
              carousel.appendChild(card);
            });
          }

          chkD1.checked = model.get("show_d1");
          chkLora.checked = model.get("show_lora");
          chkD1.onchange = () => { model.set("show_d1", chkD1.checked); model.save_changes(); };
          chkLora.onchange = () => { model.set("show_lora", chkLora.checked); model.save_changes(); };

          let drag = false, lx = 0, ly = 0;
          canvas.addEventListener("pointerdown", (e) => { drag = true; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId); });
          canvas.addEventListener("pointermove", (e) => {
            if (!drag) return;
            yaw += (e.clientX - lx) * 0.008; pitch = Math.max(-1.25, Math.min(1.25, pitch + (e.clientY - ly) * 0.008));
            lx = e.clientX; ly = e.clientY; draw();
          });
          canvas.addEventListener("pointerup", (e) => { drag = false; canvas.releasePointerCapture(e.pointerId); });
          canvas.addEventListener("wheel", (e) => {
            e.preventDefault();
            const maxL = Math.max(0, (model.get("layer_types") || []).length - 1);
            layerFloat = Math.max(0, Math.min(maxL, layerFloat + Math.sign(e.deltaY) * 0.5));
            const nextL = Math.round(layerFloat);
            yaw += Math.sign(e.deltaY) * 0.03;
            if (nextL !== model.get("layer_idx")) { model.set("layer_idx", nextL); model.save_changes(); }
            else draw();
          }, { passive: false });

          const sync = () => {
            chkD1.checked = model.get("show_d1");
            chkLora.checked = model.get("show_lora");
            scene = buildScene();
            renderCarousel();
            draw();
          };
          for (const k of ["examples", "active_example_idx", "layer_idx", "show_d1", "show_lora", "active_token_idx"]) {
            model.on(`change:${k}`, sync);
          }
          window.addEventListener("resize", draw);
          sync();
        }
        export default { render };
        """

    return DEFAULT_EXAMPLES, LAYER_TYPES, Lfm3ActivationWidget, mo


@app.cell
def _(Lfm3ActivationWidget, mo):
    lfm3_widget = mo.ui.anywidget(Lfm3ActivationWidget())
    lfm3_widget
    return (lfm3_widget,)


if __name__ == "__main__":
    app.run()
