// Run every dataset sample in the WebAssembly build of Nushell and compare with what native nu returned.
// Usage: node scripts/check_nu_wasm.mjs <cases.json>   (scripts/check_nu_wasm.py writes the cases and calls this)
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const pkg = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "public", "nu-wasm");
const nu = await import(path.join(pkg, "nu_wasm.js"));
await nu.default({ module_or_path: fs.readFileSync(path.join(pkg, "nu_wasm_bg.wasm")) });

const { environments, cases } = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
// Native runs happen in a temporary directory and stamp real file times; neither is part of the answer.
const neutral = (text) =>
  text.replace(/\/(?:private\/)?(?:var|tmp)\/[^"\s]*?\/workspace/g, "/workspace").replace(/"modified":\s*"[^"]*"/g, '"modified":""');
const same = (a, b) => {
  try { return JSON.stringify(JSON.parse(neutral(a))) === JSON.stringify(JSON.parse(neutral(b))); }
  catch { return neutral(a).trim() === neutral(b).trim(); }
};

// One thing legitimately differs between a real disk and the in-memory workspace: the order `glob` returns
// entries in (directory order on disk, sorted in the browser). `loose` ignores order, and only for calls that
// use `glob`. (Modified times are already neutralised above: native prints them in the local time zone.)
const canon = (v) => Array.isArray(v) ? v.map(canon).map((x) => JSON.stringify(x)).sort() : v && typeof v === "object"
  ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, canon(x)])) : v;
const flat = (text) => text.replace(/[\u2500-\u257f]/g, " ").split("\n").map((l) => l.replace(/\s+/g, " ").trim().replace(/^\d+ /, "")).filter(Boolean).sort().join("\n");
const loose = (code, a, b) => {
  if (!/\bglob\b/.test(code)) return false;
  try { return JSON.stringify(canon(JSON.parse(neutral(a)))) === JSON.stringify(canon(JSON.parse(neutral(b)))); }
  catch { return flat(a) === flat(b); }
};

const table = {};
const differ = [];
for (const c of cases) {
  nu.set_files(JSON.stringify(environments[c.env]));
  const r = JSON.parse(nu.run(c.code));
  const row = (table[c.type] ??= { samples: 0, "same status": 0, "same output": 0 });
  row.samples++;
  const status = r.ok === c.ok && r.parse_error === c.parse_error;
  const output = status && (!c.ok || same(r.stdout, c.stdout));
  row["same status"] += status;
  row["same output"] += output;
  const platform = status && c.ok && !output && loose(c.code, r.stdout, c.stdout);
  if (c.type === "valid") row["differs only in glob order"] = (row["differs only in glob order"] ?? 0) + platform;
  if (!output && !platform) differ.push({ id: c.id, native: c.ok ? c.stdout.slice(0, 120) : "(fails)", wasm: (r.ok ? r.stdout : r.stderr).slice(0, 120) });
}
console.log(`Nushell ${nu.nu_version()} in WebAssembly vs native, ${cases.length} samples in ${Object.keys(environments).length} environments`);
console.table(table);
for (const d of differ) console.log("differs:", JSON.stringify(d));
const valid = table.valid ?? { samples: 0, "same output": 0 };
const explained = valid["differs only in glob order"] ?? 0;
if (valid["same output"] + explained !== valid.samples) {
  console.error("FAIL: a correct call gives a different answer in WebAssembly");
  process.exit(1);
}
console.log(`OK: of ${valid.samples} correct calls, ${valid["same output"]} give the same answer and ${explained} differ only in the order glob returns entries.`);
