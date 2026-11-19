# Sources and papers

## Models

- **LFM3 1.7B** (Liquid AI): the base model that is fine-tuned, in its 8-bit MLX build for training and its 4-bit ONNX build (`onnx-community/LFM3-1.7B-ONNX`) for the browser. It also writes the requests in the dataset.

## Data

- **Nushell** 0.117.0, <https://github.com/nushell/nushell>. Its built-in command signatures (`help commands`) are the command catalogue in `assets/nu_commands.json`; the same release is compiled to WebAssembly in `nu-wasm/`.
- **The Nushell book and cookbook**, <https://github.com/nushell/nushell.github.io>. Command counts in their example code are the main sampling prior (`data/nushell_examples_usage.json`), and the book's "Coming from Bash" and "Nu map" pages are the Unix-to-Nushell table (`data/unix_to_nushell.json`). The exact commit is recorded in both files.
- **NL2Bash**: Xi Victoria Lin, Chenglong Wang, Luke Zettlemoyer, Michael D. Ernst. *NL2Bash: A Corpus and Semantic Parser for Natural Language Interface to the Linux Operating System.* LREC 2018. <https://github.com/TellinaTool/nl2bash>. Counts of the utility at the head of each pipeline stage (`data/command_usage.json`).
- **spignelon/bash_history**, a Hugging Face dataset of shell history lines, <https://huggingface.co/datasets/spignelon/bash_history>. Same counts.
- **Shell Aliases on GitHub**: Michael Schröder, Jürgen Cito. *An Empirical Investigation of Command-Line Customization.* Empirical Software Engineering, 2022. arXiv:2012.10206. Data: Zenodo, doi:10.5281/zenodo.4007049. Same counts, taken from alias bodies.
- **UNIX User Data**, UCI Machine Learning Repository, <https://archive.ics.uci.edu/dataset/141/unix+user+data>. Read only with `--include-restricted`; not part of the committed counts.

`data/command_usage.json` lists every source with its URL, pinned revision, retrieval date and checksum.

## Methods

- **LoRA**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. arXiv:2106.09685.
- **Classifier-free guidance** (experiments): Jonathan Ho, Tim Salimans. *Classifier-Free Diffusion Guidance.* 2022. arXiv:2207.12598.
- **Autoguidance** (experiments): Tero Karras, Miika Aittala, Tuomas Kynkäänniemi, Jaakko Lehtinen, Timo Aila, Samuli Laine. *Guiding a Diffusion Model with a Bad Version of Itself.* NeurIPS 2024. arXiv:2406.02507.
- **Ternary weights** (experiments): Fengfu Li, Bin Liu, Xiaoxing Wang, Bo Zhang, Junchi Yan. *Ternary Weight Networks.* 2016. arXiv:1605.04711.

## Software

- **MLX**, **mlx-lm** and **mlx-vlm** (training and inference on Apple Silicon); **marimo** (the notebook); **anywidget** and **wigglystuff** (controls and live code views); **Faker** and **polyfactory** (generated project contents); **NetworkX** (the graph chains are sampled from); **Altair** (charts); **ONNX** and **ONNX Runtime** (the browser export and its check); **Transformers.js** and **onnxruntime-web** (the model on WebGPU); **wasm-pack** (the WebAssembly build of Nushell).
