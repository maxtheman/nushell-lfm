// Static server for the demo: this folder, plus the WebAssembly Nushell build from ../public/nu-wasm.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const roots = [["/nu-wasm/", path.join(here, "..", "public", "nu-wasm")], ["/", here]];
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm", ".jinja": "text/plain" };
const port = Number(process.env.PORT || 8377);

http
  .createServer((req, res) => {
    const url = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    const [prefix, root] = roots.find(([p]) => url.startsWith(p));
    const file = path.join(root, url.slice(prefix.length) || "index.html");
    if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404).end("not found");
      return;
    }
    res.writeHead(200, { "Content-Type": types[path.extname(file)] || "application/octet-stream", "Content-Length": fs.statSync(file).size, "Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*" });
    fs.createReadStream(file).pipe(res);
  })
  .listen(port, () => console.log(`Demo: http://localhost:${port}`));
