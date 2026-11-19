"""Nushell in a notebook cell: Nushell compiled to WebAssembly, over an in-memory copy of a set of files."""

import anywidget
import traitlets


class NuRepl(anywidget.AnyWidget):
    """A Nushell session in the page: Nushell compiled to WebAssembly (nu-wasm/) over an in-memory copy of the
    environment. Variables, the current directory and written files persist until the environment changes."""

    _esm = """
    export default {
      async render({ model, el }) {
        el.innerHTML = `<div class="repl"><div class="scroll"></div>
          <div class="line"><span class="prompt">~ ❯</span><input spellcheck="false" autocomplete="off" placeholder="Loading Nushell…" disabled /></div></div>`;
        const scroll = el.querySelector(".scroll"), input = el.querySelector("input"), prompt = el.querySelector(".prompt");
        let nu;
        try {
          nu = await import(model.get("base") + "nu_wasm.js");
          await nu.default();
        } catch (err) {
          input.placeholder = "Nushell failed to load — build it with: mise run build-wasm";
          return;
        }
        const say = (text, cls) => { const d = document.createElement("div"); d.className = cls; d.textContent = text; scroll.append(d); scroll.scrollTop = scroll.scrollHeight; };
        const where = () => JSON.parse(nu.session_run("pwd")).stdout.replace(/^\\/workspace/, "~") || "~";
        const open = () => {
          nu.set_files(JSON.stringify(model.get("files")));
          nu.session_reset();
          scroll.textContent = "";
          say(`Nushell ${nu.nu_version()} · WebAssembly · ${Object.keys(model.get("files")).length} files · try: ${model.get("hint")}`, "dim");
          prompt.textContent = "~ ❯";
        };
        const history = []; let at = 0;
        input.disabled = false; input.placeholder = "type Nushell and press Enter";
        input.addEventListener("keydown", (e) => {
          if (e.key === "Enter" && input.value.trim()) {
            const code = input.value; input.value = ""; history.push(code); at = history.length;
            if (code.trim() === "clear") { scroll.textContent = ""; return; }
            say(`${prompt.textContent} ${code}`, "cmd");
            const result = JSON.parse(nu.session_run(code));
            const text = result.ok ? result.stdout : result.stderr;
            if (text) say(text, result.ok ? "out" : "err");
            if (result.changed.length) say(`wrote ${result.changed.join(", ")}`, "dim");
            prompt.textContent = `${where()} ❯`;
          } else if (e.key === "ArrowUp" && at > 0) { e.preventDefault(); input.value = history[--at]; }
          else if (e.key === "ArrowDown") { e.preventDefault(); at = Math.min(history.length, at + 1); input.value = history[at] ?? ""; }
        });
        model.on("change:files", open);
        open();
      },
    };
    """
    _css = """
    .repl { background: #14161a; color: #d7dae0; border-radius: 8px; border: 1px solid #2a2e35; font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
    .repl .scroll { max-height: 300px; min-height: 64px; overflow: auto; padding: 10px 14px 4px; }
    .repl .out { white-space: pre; line-height: 1.2; overflow-x: auto; }
    .repl .err { white-space: pre-wrap; color: #ef8a62; }
    .repl .cmd { color: #6cc08b; white-space: pre-wrap; margin-top: 4px; }
    .repl .dim { color: #8b929c; white-space: pre-wrap; }
    .repl .line { display: flex; gap: 8px; align-items: center; border-top: 1px solid #2a2e35; padding: 8px 14px; }
    .repl .prompt { color: #6cc08b; white-space: nowrap; }
    .repl input { flex: 1; min-width: 0; font: inherit; color: inherit; background: transparent; border: 0; outline: 0; }
    """
    files = traitlets.Dict().tag(sync=True)
    hint = traitlets.Unicode().tag(sync=True)
    base = traitlets.Unicode().tag(sync=True)  # where the WebAssembly build is served from
