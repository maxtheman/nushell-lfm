# Experiments

Work that is not part of the shipped pipeline. Everything here imports `nushell_lfm` and reads or writes under `runs/`; nothing in `nushell_lfm`, the notebook or `scripts/` imports from here.

| File | What it is |
|---|---|
| `breaks.py` | Ten failure types: broken variants of a chain, each kept only if running it agrees with its label. |
| `grader.py` | A LoRA on D1 2B that reads a request and a call and answers: broken or not, which failure type, where. |
| `data.py` | The dataset those two need: the target rows plus broken and equivalent variants (fingerprint `edd14fe576df`). |
| `guidance.py` | Guided decoding for the LFM3 adapter: steering away from an adapter trained on wrong calls, a weaker adapter, the base model, or the prompt without its request; cross-fitting to collect the adapter's own mistakes. |
| `run_guidance_followup.py` | Scores those decoders on every held-out request. |
| `lowbit.py`, `run_low_bit.py` | The model at 2 to 4 bits per weight: plain requantisation, and block-wise retraining followed by distillation. |
| `model_view_extras.py` | Capture code for D1 and for guidance token by token, taken out of the 3D view. |
| `defs.py` | Loads the pipeline for these scripts. |

Results so far (dataset `edd14fe576df`, same target rows as the shipped dataset):

- Grader: failure type 96%, broken or not 97%, span found 94% on 400 held-out rows (base model 27% and 61%).
- Guided decoding did not improve on the plain adapter: 89.6–92.6% across the decoders tried, against 92.6% unguided, on 231 requests.
- Requantised base weights with the existing adapter: 91% at 4 bits, 73% at 3 bits, 0% at 2 bits.
- 2 bits with block-wise retraining and distillation (`lowbit.py`): 82.3% (190 of 231), block weights in 412 MB.

`notes/lab-notes.md` has the numbers behind these. Run a script from the project root, for example:

```sh
mise exec -- .venv/bin/python experiments/run_guidance_followup.py --stage table
```
