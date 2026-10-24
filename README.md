# Agentic Financial RAG

Agentic retrieval-augmented generation for financial document QA, with reproducible benchmark runners, validation and refinement ablations, tests, and verified results.

**Raghavendranath Kandula — Independent Researcher, United States**  
Contact: raghavendranath.tr@gmail.com

Companion software for *Agentic RAG with Self-Validation for Conversational Financial Document Analysis*, NLP4KGC 2026. The camera-ready version reports new evaluations using the implementation, model settings, and experimental records provided in this release. These records supersede the earlier experimental summary; this release does not substantiate the earlier historical scores.

## Results

| Method | FinanceBench literal agreement | FinQA numerical agreement | ConvFinQA all-turn agreement |
| --- | ---: | ---: | ---: |
| DeepSeek single-shot | 6/50 (.120) | 28/48 (.583) | 132/178 (.742) |
| DeepSeek full agentic | 10/50 (.200) | 35/48 (.729) | 134/178 (.753) |
| Qwen3-32B single-shot | 8/50 (.160) | 28/48 (.583) | 114/178 (.640) |

On the same 50 FinanceBench questions, three-pass generic revision and validated fixed-evidence refinement both match the **same nine references**. Full refinement matches ten. The policies differ in total checking calls and early stopping; this is not an isolated, compute-matched estimate of validation's effect. See [complete results and uncertainty](results/RESULTS.txt).

**FinanceBench literal agreement is a limited diagnostic, not human factual accuracy.** Most selected references contain prose. A correct paraphrase may fail exact matching. Independent full-answer/grounding evaluation has not been performed and remains future work. Internal acceptance, executable arithmetic, and request replay do not certify financial correctness.

## Quick start: inspect and test

Use Python 3.12. The exact tested scientific environment is in `requirements.lock.txt`; it includes the model/index dependencies needed for fresh inference. For software tests and offline replay alone, the smaller `requirements-replay.txt` is sufficient.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-replay.txt
python research.py results
python research.py tests
```

CI runs the test suites and release checks. CI does not call a paid API, train T5, or independently assess financial answers.

## Reproduce the saved results offline

The large data, index and raw-trajectory files are distributed as `benchmark-artifacts.zip`, separately from the Git source tree. Download that release asset and compare its SHA-256 against `ARTIFACTS.json`. In the delivered local folder it is already under `artifacts/`, and its contents are already installed.

```bash
python tools/unpack_artifacts.py artifacts/benchmark-artifacts.zip
python research.py check
python research.py verify
```

`check` verifies all frozen scientific snapshot files. `verify` replays generation/validation requests using recorded responses with unrecorded model requests forbidden, checks histories, evidence and arithmetic, then recomputes scores from separated reference files. It requires **no API key or embedding download**. It verifies the recorded top-k evidence and vector/index file hashes; it does not re-embed all source documents or claim to independently remeasure nearest-neighbor quality.

Some offline scorers refresh derived report files. Run `check` before replay; the raw requests, saved predictions, original manifests and inputs remain preserved. `tools/release_check.py` validates tracked code, expected result counts and excluded paths. [Protocol details](docs/EXPERIMENTS.md) explain sampling, model versions, units, metrics and adapter differences.

## Run new experiments

Install the full environment if you will rebuild embeddings or use hosted models:

```bash
python -m pip install -r requirements.lock.txt
python tools/live_run.py --help
python tools/live_run.py --benchmark finqa --out work/finqa-new --device cpu --budget-usd 20
python tools/score_live.py --run work/finqa-new
```

`live_run.py` reads `OPENROUTER_API_KEY` from the local environment or a file supplied with `--key-file`. It never prints the key. Set the credential locally; do not put it in a notebook, committed config, shell command that embeds the actual value, or chat. The client requires a dedicated server-side credit limit no greater than the configured budget. Every run uses a new output directory, pins one model/provider without fallback, records raw requests/responses and reserves cost before calls. Ambiguous paid failures stop for inspection instead of automatic retry. Do not overwrite or tune the published benchmark runs after viewing their labels.

A live provider may retire a model or change its backend. A matching model ID, seed and configuration do not guarantee bitwise reproducibility of newly sampled responses. The saved-record replay is deterministic; a fresh hosted evaluation is a separate result. The live wrapper preserves the benchmark's documented generation/validation adapter; it does not use legacy account-supervisor scripts that required the original key's exact $20 limit.

To rebuild the selected FinQA BGE/FAISS index, use a separate working copy and the pinned revision from the existing index manifest. The generic index CLI supports CPU or available accelerator devices. Source preparation and model training scripts are included for inspection; **the published selection manifests are the authority for the reported sample**, not a new resampling command.

## Layout

- `research.py`: result display, checksums, test suites and offline replay/scoring.
- `tools/live_run.py`: fresh FinQA, FinanceBench and ConvFinQA inference with local credentials.
- `experiments/nlp4kgc-validation-v3`: preserved FinQA adapter and records.
- `experiments/nlp4kgc-three-benchmarks`: preserved ConvFinQA adapter and FinQA/ConvFinQA Qwen comparators.
- `experiments/nlp4kgc-validation-v4`: final FinanceBench adapter and Qwen comparator.
- `experiments/nlp4kgc-regeneration-ablation`: isolated no-validator-feedback revision control.
- `experiments/t5-small`: exploratory supervised program generator; different lexical reader, optional training.
- `results/`: compact verified result summaries.
- `SNAPSHOT-MANIFEST.json`: checksums for frozen scientific inputs, source snapshots and raw records.
- `artifacts/`: local release asset, excluded from Git; upload it as a release attachment.

Versioned engines retain their original file layout and scientific hashes to support exact replay. They are not presented as one uniform adapter: FinQA/ConvFinQA use program-executed final values and a candidate-aware checking call; FinanceBench preserves typed declared values and adds source-first audit plus comparison. A fixed-evidence control and generic revision control are available on FinanceBench only.

The optional T5-small experiment uses a pinned public base checkpoint, 4,988 training examples and three fixed epochs. Its original trained checkpoint is not committed. Saved predictions can be rescored offline; training a new checkpoint is a separate stochastic run. See `docs/T5.md`.

## Research scope

The contribution is an auditable online retrieval/generation/refinement workflow. The release contains no controlled prior-error injection, dependency-graph analyses or offline repair experiments. FinanceBench's evaluation sample had been inspected in preceding implementation audits, and its final evaluation/ablation must be described as post-hoc, not a fresh holdout. No independent human validator-reliability or faithfulness claim is made. There is no additional random EDGAR study, alternate-embedding comparison, or controlled A100 timing experiment.

OpenAI Codex assisted with implementation, software checks, saved-record analysis and preparation of this release. Experimental DeepSeek and Qwen use is specified in the protocols. Passing tests is evidence about software behavior, not an independent human review of the answers.

## License and citation

Original software is MIT-licensed. The paper, benchmark material, financial source reports, model weights and copied upstream evaluator have their own applicable terms; the software license does not relicense them. See [third-party notices](THIRD_PARTY_NOTICES.md). No source report PDFs, private manuscript notes, credentials, runtime caches or model caches are committed. Use `CITATION.cff` and cite the benchmark papers when reusing their data.

See [publishing instructions](docs/PUBLISHING.md) for uploading this prepared local repository and its release asset. The manuscript is delivered separately for author review; do not publish a manuscript draft as an approved proceedings version.
