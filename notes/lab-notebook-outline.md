# Lab notebook outline

Numbers are in `results-tables.md`; the longer notes with charts are in `lab-notes.md`.

## 1. What was built

- A marimo notebook that generates a Nushell tool-call dataset from seeds and fine-tunes LFM3 1.7B on it with MLX, offline, on one Mac.
- Scoring by execution: a call is correct only if running it returns the reference value.
- The same model in the browser: 4-bit ONNX on WebGPU next to Nushell compiled to WebAssembly.

## 2. Result

- 214 of 231 held-out requests correct (92.6%); base model 3 of 40 (8%). Table 1.
- 176 of 191 on the requests used for nothing else; per environment 88–98%.
- The same 231 requests through two hosted models: 22.1% and 13.0%. Table 2.
- Chart: `charts/1-executed-correct.png`.

## 3. How the data is made

- Environments: 20 small projects generated from seeds (Faker, polyfactory); 4 held out whole.
- Chains: walks on a graph of data kinds; kept only if they run, repeat, and return something. 401 kept.
- Sampling weights: command counts in the Nushell book and cookbook; public Unix usage data for category shares and at most 30% of chains.
- Requests: written by the base model in six styles, 3 per chain; 22% fell back to a composed description.
- Table 5.

## 4. What the data showed about itself

- JSON files: 5 training chains in the first run; 4 of that run's 11 wrong calls were `open --raw` on JSON.
- The first request check accepted "document" as keeping `doc`.
- 47 of 1,203 requests repeat part of the rewriting instruction.
- 6 of the 17 wrong calls are requests that lack what the reference needs. Table 6, chart `charts/5-wrong-calls.png`.
- Reference values depend on the machine in three ways: `glob` order, directory size from `ls`, time zone. Found by running every call in WebAssembly as well.
- Held out by project, not by pattern: 15 / 50 / 27 of 77 chains.

## 5. Training

- Sweep over rank and learning rate, then adapter placement, each chosen by executed-correct on 24 requests. Table 3, chart `charts/3-sweep-and-ablation.png`.
- With 24 requests, only the gap between 75% and 42% is outside noise.
- Data size: 72% with 162 chains, 92% with 401 chains (two runs, different held-out environments). Chart `charts/2-data-size.png`.
- Final adapter: 825 rows × 2 epochs, 414 updates, 3.5 GB peak. The run took 50 minutes including the guidance experiments.

## 6. Export and browser

- Adapter added to the 4-bit ONNX build as low-rank branches; 22 of 24 before and after export. Table 4.
- WebAssembly Nushell matches native on 382 of 401 reference calls; the other 19 differ in `glob` order only.
- Demo: full-window terminal, held-out requests to pick from, the call is run and compared in the page.

## 7. Experiments left out of the shipped pipeline

- Grader on D1 2B: 96% failure type, 97% broken or not.
- Guided decoding: no decoder beat the plain adapter. The adapter's first choice leads its runner-up by a median of 11.1 nats; a second token is within the floor at 19 of 1,290 tokens. Chart `charts/4-guidance-margins.png`.
- Low-bit base weights: 91% at 4 bits, 73% at 3 bits, 0% at 2 bits without retraining. With block retraining and distillation, 2 bits gets 82.3%.
- Table 7.

## 8. Open items

- Cross-fitted `P_bad` and combined guidance.
