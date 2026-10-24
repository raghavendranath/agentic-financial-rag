# Experimental specification

## Data, retrieval and exposure

| Dataset | Pinned source commit | Evaluation | Development | Retrieval corpus |
| --- | --- | --- | --- | --- |
| FinQA | `0f16e2867befa6840783e58be38c9efb9229d742` | 50 official test questions, 48 numerical + 2 Boolean | 20 official dev questions | 107 public processed report pages; known-filing filter |
| ConvFinQA | `cf3eed2d5984960bf06bb8145bcea5e80b0222a6` | 50 public dev conversations / 178 turns | 5 train conversations / 18 turns | 193 public processed report pages; known-filing filter |
| FinanceBench | `cc39aeb4afdf33909ee1412188bf89035950c2eb` | 50 questions from public 150-case release | 5 other questions | 44 complete PDFs / 6,261 pages / 3,555 chunks; known-report filter |

FinQA uses SHA-256 order, sampling seed 1820, one question per filing, excluding previously observed filings. ConvFinQA uses SHA-256 order with the `three-benchmarks-1821` seed string, one conversation per filing, excluding previously observed FinQA filings. All fixed IDs and exact source/input hashes are in each data manifest. This does not establish disjointness from every other study using public datasets.

FinanceBench's evaluation sample was inspected in prior implementation audits. Its final protocol and additional self-revision protocol were frozen before their respective model calls, but the sample is exposed and results are post-hoc. No unseen-holdout claim is made. The release preserves the selected IDs and every final attempt, including unknown/invalid answers. There is no answer-key-driven best-of-attempt selection.

BGE-M3 (`BAAI/bge-m3`, revision `5617a9f61b028005a4858fdac845db406aefb181`) generates normalized 1,024-dimensional dense embeddings. Token chunks have length 1,500 with overlap 200, exact FAISS `IndexFlatIP`, top ten, maximum embedding sequence 8,192. Inference does not use gold evidence-page selection. FinanceBench sorted-text extraction uses PyMuPDF 2.0. This is document QA with provided document identity, not unrestricted cross-report search.

## Models and adapters

- DeepSeek: `deepseek/deepseek-v4.1-flash-1017`, OpenRouter provider `inference-net/fp8`, returned provider InferenceNet.
- Qwen: `qwen/qwen3-32b`, provider `deepinfra/fp8`.
- Temperature 0, seed 18, maximum output 4,096 tokens, reasoning disabled, no routing fallback.
- Maximum three answer attempts; last attempt is retained, including failed/unknown outputs. Checking can stop a validated arm early.
- FinQA/ConvFinQA preserve program-executed final values and a candidate-aware same-model checking call. The raw declared value is also logged.
- Final FinanceBench preserves the typed declared display with unit checks, a candidate-blind source-first same-model audit, then a separate comparison. Strict operational acceptance is not independent factual verification.
- Single-shot/full share exact first drafts in FinQA and FinanceBench. ConvFinQA maintains each arm's own generated history, so later-turn questions can have different context and initial drafts. Its comparison is end-to-end.
- Qwen shares initial evidence in FinQA/FinanceBench, but maintains its own conversational history in ConvFinQA.

The FinanceBench generic three-pass control receives only a previous declared/raw draft, a constant reconsideration instruction and pass number. No verdict, calculator output, parser diagnosis, source audit or label is supplied. Evidence stays fixed; all three passes run; the third is always scored. Explicit-validation fixed evidence uses more total requests and adaptive stopping. The comparison does not isolate validation alone.

## Scorers

Numeric parsing/comparison normalizes whole scalar formats and rounds to five decimals. There is no sign or factor-100 forgiveness. On one predicted scalar and one reference scalar, the reported scalar numerical F1 is numeric-item accuracy. It is not a multi-number extraction F1 or symbolic-program-accuracy claim. FinQA excludes its two Boolean references from the 48-item numerical denominator and reports the 50-item total separately.

FinanceBench compares whole-scalar references numerically; otherwise NFKC Unicode normalization, case folding, collapsed whitespace and removal of trailing `.?!;` apply. No semantic paraphrase judgment occurs. Thirty-four of the selected references contain prose; literal mismatches are not labels of factual error. A correct headline value with erroneous explanation would not be established as fully correct by a scalar match. Human full-answer/source assessment is future work.

ConvFinQA's primary aggregation weights all 178 turns. Secondary aggregation considers the first up to three available turns (133). Completely correct conversation rate requires every turn to match. These aggregations must not be interchanged.

Bootstrap intervals use 2,000 filing-cluster draws, seed 18. They describe sample variation and do not correct sample exposure, multiple development choices, public-data contamination, metric limitations or unmatched compute.

## What is and is not verified

Offline replay verifies exact recorded requests, raw replies, history construction, selected retrieved evidence, arithmetic execution, final selection/stopping rules, score counts and hashes. It prevents unrecorded paid inference. It is not a new independent model run, human financial annotation, complete retrieval re-embedding, human ground-truth correction, or a claim that validator acceptance implies correctness. No earlier automated-judge scores are carried forward.
