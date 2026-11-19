# Training sketch transcription

Transcribed from the drawing titled **training** in the A2UI **training** Space on November 18, 2026. [Board image](training-sketch.png)

## 1. Set up the environment

- Run Nushell in WASM.
- Generate directories and files, including filenames, metadata, and SQL.
- Use WebGPU and Nushell. A separate `nu` is written under `webgpu`.

## 2. Set up the data and grader

Use the Nushell docs to build coverage for:

- Search
- Data processing
- File manipulation, with SQL or CSV noted beside it
- An open category marked `??`
- Compositions
- Nu TUI or functions

A handwritten `Not` appears next to the data-processing and file-manipulation entries. Its scope is unclear.

Start with function chains and create broken versions. Add prompts, then make **K variations** using LFM with thinking enabled. The sketch gives the approximate dataset size as **function chains × breaks per chain × K prompt variants**.

Write user prompts with inconsistent formatting, spelling mistakes or abbreviations, and less technical language. A request may name only some components while meaning the whole system. It may also describe a part by analogy, sometimes within the same domain and sometimes from an unrelated domain. The pasted prompt notes include an unexplained `vv` under the component example.

Add agentic data later. Leave audio and video for a later stage.

### Failure examples pasted into the sketch

| Failure | Example in the pasted reference |
|---|---|
| Request misunderstood | Misheard filename, missing negation, or wrong number |
| Wrong tool or malformed call | Another tool, prose, or unreadable call syntax |
| Invalid Nushell | Unclosed delimiter, invalid function signature, or syntax from another language |
| Wrong data source | Wrong file or column, or example data in place of the CSV |
| Wrong operation or argument | Count instead of sum, wrong sort column or direction, or wrong filter |
| Broken chain or function | Missing or reordered steps, uncalled function, or wrong intermediate value |
| Wrong output or side effect | Wrong output shape, extra text, or a write to the wrong path |
| Runtime or environment failure | Timeout, unsupported feature, or version difference |
| Grader false pass | Hard-coded result matches a single fixture, or the grader misses a required function or file change |
| Grader false fail | An equivalent answer gets rejected over formatting, order, type, or an unspecified tie-break |

## 3. Train the grader

- Feed a function chain to **D1 2B**. Have it identify the **break, reason, and span**.
- Sweep LoRA adapter placement, batch size, learning rate and multiplier, dropout, and gradient clipping.
- Compare **Autoguidance** with the original. The sketch also says **annealed** and **with a second KV cache**. Which arm uses annealing is unclear.

## 4. Train the target

- Train **LFM3** and sweep its hyperparameters.
- **4a.** `P_good`.
- **4b.** `→ D1 P_bad`, preserving the arrow placement from the sketch.
- The two branches are joined by a mark labeled **2× LoRA**.

## 5. Final in WebGPU

Run the final setup in WebGPU.

## Right-hand diagram

From top to bottom, the sketch shows:

1. **Target environments** with a directory tree and **Nu in WASM**.
2. **Table of commands**.
3. **Chains and source**, using ditto marks to carry forward the table label.
4. **Breaking procedure and examples / samples**.
5. **Prompt**, with ditto marks after it.
6. **D1 trainer** with **losses** and **confusion**.
7. **LFM trainer** with **losses**.
