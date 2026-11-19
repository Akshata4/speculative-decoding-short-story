# Paper notes

Structured reading notes for the four papers behind this short story: taxonomy, setup, key numbers, ablations and limitations. All numbers are as reported by the authors.

---

## 1. Speculative Decoding: Performance or Illusion? (anchor paper)

- **Authors:** Xiaoxuan Liu*, Jiaxiang Yu*, Jongseok Park, Ion Stoica, Alvin Cheung (UC Berkeley)
- **arXiv:** [2601.11580](https://arxiv.org/abs/2601.11580). v1 31 Dec 2025, v2 18 Mar 2026. cs.CL / cs.AI, CC BY 4.0
- **Code:** [SpecDecode-Bench/vllm](https://github.com/SpecDecode-Bench/vllm) (profiling suite) and [SpecDecode-Bench/simulator](https://github.com/SpecDecode-Bench/simulator) (CPU simulator, ~30–40 min run)
- **Type:** Systematic benchmark, the first in a production engine

**Taxonomy of methods evaluated**

| Family | Methods | Drafter |
|---|---|---|
| Draft-model based | Llama-3.2-1B → Llama3-70B; Qwen3-0.6B → Qwen3-8B | Separate small LM |
| Learned heads (no separate model) | EAGLE, EAGLE-3 (fine-tuned heads); MTP (co-trained heads) | Light layers on the target |
| Training-free | n-gram / prompt lookup | String match over the context |
| Verification shape | Chain vs tree (k = 6, 21), tested in SGLang | n/a |

**Setup**

| Setting | Details |
|---|---|
| Engine | vLLM 0.10.1.1 / 0.11.1rc1 with CUDA graphs and chunked prefill; SGLang 0.5.9 for trees |
| Hardware | H100 80 GB; TP=4 for the 70B and 106B models |
| Models | Llama3.1-8B, Llama3-70B, Qwen3-8B (non-thinking and thinking), GLM-4.5-Air (106B MoE) |
| Data | CNN/DailyMail, ShareGPT, InstructCoder, GSM8K, AIME22–24, GPQA-Main |
| Batch sizes | 1–128 (reasoning: 1–16; GLM: 1–4) |
| Draft length | k = 3 (and 5 for n-gram) |
| Decoding | Temperature 0; max 8K output tokens (32K for reasoning) |
| Metric | Token throughput |

**Speedup model:** E[speedup] = (1 − α^(k+1)) / ((1 − α)(k·c + 1))

**Key results**

- **Batch decay:** EAGLE on Llama3.1-8B / GSM8K falls from 1.73× to 1.21× (batch 1 → 128). On ShareGPT by batch 32, the 8B drops 4.3% (1.68 → 1.61) and the 70B drops 14.0% (1.96 → 1.72).
- **Time breakdown:**

  | Component | Share of runtime |
  |---|---|
  | Verification | 42–95% |
  | EAGLE drafting (batch 1) | 12–20% |
  | n-gram drafting | < 2% |
  | Sampling | < 1.7% |
  | vLLM overhead | 3–12% |

- **Memory:** n-gram +0; EAGLE-3 +4.9% static (Qwen3-8B); draft model raises KV per token from 144 to 256 KiB (1.77×).
- **Draft model cost:** ~12.5% of a target pass for the 70B (best method there) vs ~37.5% for Qwen3-8B (worst there).
- **Trees:** at batch 1, chain 1.65×, k=6 1.68×, k=21 1.85× (Qwen3-8B, GSM8K). At batch 64 the k=21 tree is below 1× everywhere while the chain stays above 1×. Accepted length rises 2.25 → 2.92 while the acceptance rate falls 0.415 → 0.095.
- **Reasoning:** Qwen3-8B thinking gets EAGLE-3 1.64–1.80× and n-gram 1.50–1.58×. GLM-4.5-Air MTP gets 1.3–1.8×, with per-position acceptance 0.92 / 0.68 / 0.38.
- **Acceptance variability:** n-gram is heavy-tailed (bursts above 15 tokens); EAGLE has a median of 2–4. On InstructCoder, n-gram's mean is 7.27 vs 4.24 for EAGLE. Above a BLEU-4 overlap of ~0.6, n-gram beats EAGLE at all batch sizes (up to +53% at k=3, +100% at k=5).
- **Upper bound:** oracle k gives ~2.75× vs 2.1× for the best fixed k (InstructCoder, batch 1). An oracle combining EAGLE and n-gram gives up to 2.2× more than the best single fixed strategy.
- **Nondeterminism:** greedy outputs differ across batch sizes and variants because of kernel floating-point effects.

**Ablations:** batch size · k (3 vs 5) · chain vs tree · model size · oracle k · oracle method choice · BLEU-n threshold.

**Future work (authors):** verify only likely-accepted tokens; a lightweight adaptive method selector; better later-position MTP heads; cheaper drafters for small targets; EAGLE KV maintenance on pause/resume; overlap measured against the evolving context.

---

## 2. Is Multimodal Speculative Decoding Ready for Diffusion-Based Parallel Drafting? A Survey and Empirical Diagnosis

- **Authors:** Yantao Li, Huanlin Gao, Fang Zhao, Chao Tan, Qiang Hui, Shuting Liu, Fuyuan Shi, Ting Lu, Shaoan Zhao, Xueqiang Guo, Xinpei Su, Jianbing Zhang, Xinyu Dai, Kai Wang, Shiguo Lian (Nanjing University; China Unicom Data Science & AI Research Institute)
- **arXiv:** [2608.20743](https://arxiv.org/abs/2608.20743). v1 21 Aug 2026, v2 29 Aug 2026. cs.AI
- **Type:** Survey + empirical diagnosis

**Taxonomy (drafter-side parallelism)**

| Level | Definition | Examples |
|---|---|---|
| L0 | Sequential: one drafter forward = one draft depth | Small AR LMs, EAGLE family |
| L1 | Multi-position: fixed future offsets predicted from a shared prefix | Medusa, MTP, SpecFLASH |
| L2 | Block-parallel generative: whole block jointly generated or refined | DFlash, DSpark |

Orthogonal axes (these don't change the L-level): candidate width and tree construction (SpecInfer, Sequoia, DDTree), and the verification rule.

**Domain survey**

- **VLM:** SPD-MLLM, MSD, MASSV, DREAM, HiViS, SpecVLM, ViSpec, TIGER, Spec-LLaVA, ViSkip; L1 SpecFLASH; L2 Fast-dVLM
- **Video:** Sparse-to-Dense, SpecVLM-video, Sparrow, HIPPO, ParallelVLM, LVSpec
- **VLA:** Spec-VLA, KERV, HeiSD, SV-VLA; L2 Realtime-VLA FLASH, Fast-dDrive
- **Speech:** SpecASR, ParaASR, CTC-SSD, VADUSA, Codec-MTP, PCG, UGSD
- **AR image generation:** LANTERN(++), Spatial SD, Continuous SD, SJD / SJD2 / SJD++, GSD, SDVG

**Benchmark setup**

| Setting | Details |
|---|---|
| Targets | Qwen3-VL-4B / 8B, Qwen3.6-27B (dense), Qwen3.6-35A3B (MoE; described but no results reported) |
| Drafters | EAGLE-3 (L0), MTP (L1), DFlash (L2), DSpark (L2) |
| Data | 600 samples = 100 each from GQA, Flickr30K, TextVQA, CharXiv, MMMU, ConvBench |
| Serving | SGLang, greedy decoding, max 2,048 new tokens |
| Metrics | Mean accepted tokens (MAT), wall-clock speedup, stage latency |

**Key results (Table 4)**

| Target | EAGLE-3 | MTP | DSpark | DFlash |
|---|---|---|---|---|
| Qwen3.6-27B: MAT / speedup | 2.31 / 1.69× | 3.22 / 1.64× | 3.52 / 2.04× | **4.38 / 2.60×** |
| Qwen3-VL-4B: speedup | 0.71× | n/a | n/a | **1.95×** (MAT 2.50) |
| Qwen3-VL-8B: speedup | 0.88× | n/a | n/a | **2.14×** (MAT 2.75) |

**Diagnostics**

- **Stage latency (27B):** vision conditioning ~26–27 ms vs drafting 1.14 ms (DFlash) and 1.78 ms (DSpark).
- **HR-Bench:** 4K gives 0.96–1.11×; 8K gives 0.83–0.99×, even though MAT stays at 2.5–3.2.
- **Context ablation:** masking visual KV for the drafter lowers MAT by ~4%; removing all prefill context lowers it by 11–14%.

**Verdict:** "partially ready", depending on model, task and system.

**Limitations:** model, drafter checkpoint, recipe and backend are confounded; multimodal framework support is immature.

**Future directions:** lightweight multimodal conditioning; dependency modelling inside L2 blocks; dynamic routing across L-levels with fallback to plain decoding; tree-aware parallel-drafter training; multimodal relaxed acceptance; standardized evaluation.

---

## 3. A Practical Investigation of Training-free Relaxed Speculative Decoding

- **Authors:** Guoxuan Xia (Imperial College London), Luka Ribar, Paul Balanca (Graphcore; work done at Graphcore)
- **arXiv:** [2607.08690](https://arxiv.org/abs/2607.08690). v1 9 Jul 2026. cs.LG / cs.AI, CC BY 4.0
- **Code:** [graphcore-research/relaxed-spec-dec-repro](https://github.com/graphcore-research/relaxed-spec-dec-repro)
- **Type:** Unified framework / taxonomy + benchmark (not a pure survey)

**Framework:** replace the target p with relaxed distributions π at three points: rejection (π_rej), residual resampling (π_res) and the bonus token (π_bonus). A parameter α controls how far the relaxation goes.

**Taxonomy**

| Family | Methods | Idea |
|---|---|---|
| KL-bounded | CACTUS (reverse KL), mentored decoding (forward KL) | Maximise acceptance under a divergence budget |
| Threshold / defer | Fuzzy and reducible-fuzzy SD (JS divergence), speculative cascades (spec-casc-opt) | Accept when close, else defer |
| Mix | Ensemble α·q + (1−α)·p | Blend the two distributions |
| Contrast | Speculative contrastive decoding | Aims to improve capability |

Out of scope: learned methods (Judge Decoding, AutoJudge, DIVERSED).

**Setup**

| Setting | Details |
|---|---|
| Drafter → target pairs | Qwen3 0.6B → 32B, Qwen3 8B → 32B, Qwen3.5 MTP → 27B, Llama 3.2 1B → 3.1 70B (GPQA) |
| Tasks | AIME 2024, GPQA Diamond, LiveCodeBench v6, in thinking mode |
| Sweeps | k ∈ {3, 5, 10, 20}; c_rel ∈ {0.05, 0.2, 0.5}; ~10³ queries per point |
| Hardware | GH200 cost table, vLLM 0.20.1. Speedup is modelled from accepted length and response length |

**Key results**

- **CACTUS mean accepted tokens (AIME24, k = 20), strict → max α:**

  | Pair | Strict | Relaxed |
  |---|---|---|
  | 8B → 32B | 8.15 | 14.95 |
  | 0.6B → 32B | 5.25 | 9.36 |
  | MTP → 27B | 5.36 | 10.68 |

- **vLLM overhead:** a 0.6B drafter's c_rel is 0.266 in plain decode but 0.847 inside the stock spec-dec loop.
- Weak and MTP drafters cause rambling or longer answers that can cancel the gains.
- The same α does not transfer across tasks (AIME vs GPQA).
- Changing k moves the speedup about as much as relaxation does, and it stays lossless.
- Only CACTUS and mentored decoding stay useful with MTP drafters.

**Takeaways:** tune k first; validate with real measurements; relaxed SD requires capability re-evaluation; account for changes in response length; prefer reducible fuzzy over fuzzy; many relaxed methods assume a strong language-model drafter.

---

## 4. Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimization

- **Authors:** Jiantong Jiang, Peiyu Yang (corresponding), Feng Liu (University of Melbourne); Rui Zhang (HUST)
- **arXiv:** [2607.08057](https://arxiv.org/abs/2607.08057). v1 9 Jul 2026. Findings of ACL 2026, pp. 38450–38476
- **Type:** Survey (no new experiments)

**Taxonomy (sKis: system-aware KV infrastructure for serving)**

- **Temporal: execution and scheduling**
  - KV-centric scheduling at request, token and kernel level (e.g. FlashInfer)
  - Pipelining and overlap (FastDecode, Neo, CachedAttention, KVPR)
  - Hardware-aware execution: prefill/decode disaggregation (Mooncake, Infinite-LLM, MuxServe) and compute offloading (CPU, CSD, PIM)
- **Spatial: placement and migration**
  - Memory hierarchy, cross-device (ArkVale, OmniKV, ClusterKV, PQCache, FlexGen, CXL) and intra-GPU (HBM to L2 prefetch)
  - Compute-device orchestration (DistServe, TraCT, CacheGen)
- **Structural: representation and retention**
  - Compression:
    - Quantization: KIVI, KVQuant, GEAR, CQ, VQ-LLM, CommVQ
    - Low-rank: xKV, Palu, ShadowKV, LoRC, EigenAttention
    - Structural merge/prune: KVMerger, MiniCache, KVSharer, CHAI, ThinK
  - Retention:
    - Allocation and reuse: PagedAttention, Prompt Cache, LazyLLM
    - Eviction, static: SnapKV, PyramidKV
    - Eviction, dynamic: H2O, Scissorhands, StreamingLLM, TOVA
    - Budgeting: Ada-KV, CAKE, DynamicKV

**Analyses**

- Behaviour × objective matrix (mean and tail latency, throughput, GPU memory, I/O, energy, quality)
- Co-design affinity network

**Observations**

| # | Observation |
|---|---|
| O1 | Structural work is the most studied and mostly targets memory |
| O2 | Temporal methods drive latency and throughput, but tail latency is under-reported |
| O3 | Spatial methods mainly target I/O |
| O4 | Energy is under-explored |
| O5 | Quality loss is universal and must be controllable |
| O6 | Hardware-aware execution and device orchestration form the strongest co-design pair |
| O7 | Compression is isolated from other behaviours |

**Open challenges**

| # | Challenge |
|---|---|
| C1 | SLO-driven tail control |
| C2 | Energy-aware design |
| C3 | Trustworthiness |
| C4 | Portability across interconnects |
| C5 | Joint eviction / offload / prefetch |
| C6 | Unified benchmarks |

**Relevance here:** it does not cover speculative decoding. Draft tokens, trees and draft models all add KV state that this layer must schedule, place and store, and the Berkeley paper flags EAGLE KV maintenance as future work.
