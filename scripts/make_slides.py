"""Build the slide deck (PNG per slide -> PDF + PPTX with speaker notes).

Slides that show a chart reuse figures/*.png; text slides are drawn here in the
same visual style. Speaker notes come from video/script.md ("## Slide N" blocks).

Usage:  python scripts/make_slides.py
Outputs: slides/png/slide_XX.png, slides/speculative-decoding-deck.pdf,
         slides/speculative-decoding-deck.pptx
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
OUT = ROOT / "slides"
PNG = OUT / "png"
PNG.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE, AQUA, VIOLET, NAVY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#0d366b"
FILLS = {BLUE: "#e6f0fc", ORANGE: "#fdebe2", AQUA: "#dcf3ea", VIOLET: "#e9e6fa"}
plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
W, H, DPI = 12, 6.75, 200


def new_slide(title, subtitle=None, kicker=None):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if kicker:
        ax.text(0.04, 0.925, kicker.upper(), fontsize=11, color=BLUE, weight="bold")
    ax.text(0.04, 0.855, title, fontsize=26, weight="bold", color=INK, va="center")
    if subtitle:
        ax.text(0.04, 0.785, subtitle, fontsize=14, color=INK2, va="center")
    return fig, ax


def card(ax, x, y, w, h, head, body, col, head_size=14, body_size=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                fc=FILLS.get(col, "#f4f3f0"), ec="none"))
    ax.add_patch(FancyBboxPatch((x, y), 0.006, h, boxstyle="square,pad=0", fc=col, ec="none"))
    ax.text(x + 0.022, y + h - 0.045, head, fontsize=head_size, weight="bold", color=INK, va="top")
    ax.text(x + 0.022, y + h - 0.105, body, fontsize=body_size, color=INK2, va="top", linespacing=1.45)


def bullets(ax, items, x=0.05, y=0.7, size=15, gap=0.085, color=INK):
    for i, it in enumerate(items):
        ax.text(x, y - i * gap, "•", fontsize=size, color=BLUE, va="top", weight="bold")
        ax.text(x + 0.022, y - i * gap, it, fontsize=size, color=color, va="top", linespacing=1.35)


# ---------------------------------------------------------------------------
def s_title():
    return FIG / "f00_cover.png"


def s_question():
    fig, ax = new_slide("The question", "Speculative decoding promises a 'free' 2–3× speedup. Does it hold up in production?",
                        "Why this story")
    card(ax, 0.04, 0.38, 0.29, 0.32, "The promise",
         "Small drafter guesses, big model\nchecks k tokens in one pass.\nLossless: identical output\ndistribution.", BLUE)
    card(ax, 0.355, 0.38, 0.29, 0.32, "The catch",
         "Most headline numbers come from\nresearch prototypes at batch 1.\nReal servers batch 64–128\nrequests with paged KV caches.", ORANGE)
    card(ax, 0.67, 0.38, 0.29, 0.32, "This talk",
         "4 papers from 2026: one map\nof the field, one production\nreality check, two frontiers,\nand what to deploy.", AQUA)
    ax.text(0.04, 0.22, "Agenda:  refresher → unified taxonomy → vLLM reality check → multimodal & diffusion drafters\n"
            "→ relaxed verification → KV-cache serving layer → cheat-sheet → my two cents",
            fontsize=13, color=INK2, linespacing=1.6)
    return fig


def s_papers():
    fig, ax = new_slide("Four papers, one story", "All released in 2026; two surveys and two systematic benchmarks", "Sources")
    rows = [
        ("Speculative Decoding: Performance or Illusion?", "Liu, Yu, Park, Stoica, Cheung · UC Berkeley",
         "arXiv 2601.11580", "Benchmark in vLLM — the anchor", BLUE),
        ("Is Multimodal SD Ready for Diffusion-Based Parallel Drafting?", "Li et al. · Nanjing Univ. & China Unicom",
         "arXiv 2608.20743", "Survey + benchmark — drafter taxonomy", ORANGE),
        ("A Practical Investigation of Training-free Relaxed SD", "Xia, Ribar, Balanca · Imperial College / Graphcore",
         "arXiv 2607.08690", "Taxonomy + benchmark — verification", AQUA),
        ("Towards Efficient LLM Serving: System-Aware KV Cache Survey", "Jiang, Yang, Zhang, Liu · Melbourne / HUST",
         "arXiv 2607.08057", "Survey (ACL Findings 2026) — serving layer", VIOLET),
    ]
    for i, (t, a, idn, role, col) in enumerate(rows):
        y = 0.6 - i * 0.145
        ax.add_patch(FancyBboxPatch((0.04, y), 0.92, 0.12, boxstyle="round,pad=0,rounding_size=0.01",
                                    fc=FILLS[col], ec="none"))
        ax.add_patch(FancyBboxPatch((0.04, y), 0.006, 0.12, boxstyle="square,pad=0", fc=col, ec="none"))
        ax.text(0.06, y + 0.08, t, fontsize=14, weight="bold", color=INK, va="center")
        ax.text(0.06, y + 0.035, a, fontsize=11.5, color=INK2, va="center")
        ax.text(0.94, y + 0.08, idn, fontsize=12, color=col, weight="bold", ha="right", va="center")
        ax.text(0.94, y + 0.035, role, fontsize=11.5, color=INK2, ha="right", va="center")
    return fig


def s_levels():
    fig, ax = new_slide("Who drafts? The L0 → L1 → L2 ladder",
                        "Taxonomy from Li et al. (2608.20743): how much parallelism lives on the drafter side",
                        "Taxonomy · column 1")
    card(ax, 0.04, 0.3, 0.29, 0.4, "L0 · Sequential",
         "One drafter step = one token.\n\nSmall draft LMs, EAGLE,\nEAGLE-3.\n\n+ accurate, sees its own guesses\n− k serial steps per draft", BLUE)
    card(ax, 0.355, 0.3, 0.29, 0.4, "L1 · Multi-position",
         "Several heads predict +1, +2, +3\nfrom one hidden state.\n\nMedusa, MTP heads, SpecFLASH.\n\n+ one shot, cheap\n− later positions guessed 'blind'", ORANGE)
    card(ax, 0.67, 0.3, 0.29, 0.4, "L2 · Block-parallel",
         "Whole block generated or refined\njointly (often diffusion).\n\nDFlash, DSpark.\n\n+ high acceptance, bounded depth\n− new, immature framework support", AQUA)
    ax.text(0.04, 0.2, "Outside the ladder: n-gram / prompt lookup — copy what followed the current suffix earlier in "
            "the context.\nNo training, no GPU, no memory. Orthogonal axes: chain vs tree shape, strict vs relaxed "
            "verification.", fontsize=12.5, color=INK2, linespacing=1.6)
    return fig


def s_setup():
    fig, ax = new_slide("The reality check: SD inside vLLM",
                        "Liu et al. (2601.11580) — first systematic study in a production engine", "Paper 1 · setup")
    card(ax, 0.04, 0.37, 0.44, 0.35, "What they ran",
         "Engine: vLLM 0.10–0.11, CUDA graphs + chunked prefill on\nHardware: H100 80 GB (TP=4 for 70B / 106B)\n"
         "Models: Llama3.1-8B, Llama3-70B, Qwen3-8B (± thinking),\n             GLM-4.5-Air (106B MoE)", BLUE, body_size=11.5)
    card(ax, 0.52, 0.37, 0.44, 0.35, "How they measured",
         "Methods: n-gram, EAGLE, EAGLE-3, draft model, MTP;\n               tree verification in SGLang\n"
         "Data: CNN/DM, ShareGPT, InstructCoder, GSM8K,\n           AIME, GPQA   ·   Batch 1 → 128\n"
         "Metric: token throughput (outputs vary run to run)", ORANGE, body_size=11.5)
    ax.text(0.04, 0.27, "Ablations:  batch size  ·  draft length (3 vs 5)  ·  chain vs tree (k = 3/6/21)  ·  model size  ·  "
            "oracle draft length  ·  oracle method choice", fontsize=12.5, color=INK2)
    ax.text(0.04, 0.18, "Contributions:  production-grade evaluation  ·  time & memory breakdown  ·  acceptance "
            "variability  ·  theoretical upper bound", fontsize=12.5, color=INK2)
    return fig


def s_more_findings():
    fig, ax = new_slide("More from the vLLM study", "Smaller findings that matter when you deploy", "Paper 1 · details")
    items = [
        ("Drafter cost ratio decides draft models", "70B target: drafter ≈ 12.5% of a pass → best method.\n"
         "8B target: ≈ 37.5% → worst method.", BLUE),
        ("Reasoning workloads are a sweet spot", "Qwen3-8B thinking on AIME/GPQA: EAGLE-3 1.64–1.80×,\nn-gram 1.50–1.58×.", AQUA),
        ("MTP heads fade with distance", "GLM-4.5-Air: acceptance 0.92 → 0.68 → 0.38 at positions\n1/2/3 → only 1.3–1.8×.", ORANGE),
        ("Memory is mostly fine — except draft models", "n-gram +0, EAGLE-3 +4.9% static; draft model raises\nKV per token 144 → 256 KiB (1.77×).", VIOLET),
    ]
    for i, (h, b, c) in enumerate(items):
        x = 0.04 + (i % 2) * 0.47
        y = 0.43 - (i // 2) * 0.26
        card(ax, x, y, 0.45, 0.23, h, b, c, body_size=12)
    ax.text(0.04, 0.08, "Also: greedy outputs differ across batch sizes (kernel floating-point nondeterminism) → "
            "benchmark throughput, not one exact answer.", fontsize=11.5, color=MUTED)
    return fig


def s_mm_landscape():
    fig, ax = new_slide("Speculative decoding beyond text",
                        "Survey half of Li et al. (2608.20743): every modality adapts draft & verify differently",
                        "Paper 2 · survey")
    items = [
        ("Vision-language", "Reuse / prune visual tokens for the drafter:\nMASSV, ViSpec, SpecVLM, DREAM, HiViS", BLUE),
        ("Video", "Sparse attention in drafting, input pruning,\npipeline overlap (HIPPO), relaxed acceptance", ORANGE),
        ("Vision-language-action", "Robot / driving actions: relaxed acceptance\n(Spec-VLA, KERV), L2 action chunks", AQUA),
        ("Speech & audio", "ASR & TTS with tolerance-based acceptance;\nL1 heads (VADUSA, Codec-MTP)", VIOLET),
        ("AR image generation", "Relaxed latent acceptance (LANTERN),\nspeculative Jacobi decoding", BLUE),
        ("Benchmark half", "Qwen3-VL-4B/8B, Qwen3.6-27B · 6 tasks:\nGQA, Flickr30K, TextVQA,\nCharXiv, MMMU, ConvBench", ORANGE),
    ]
    for i, (h, b, c) in enumerate(items):
        x = 0.04 + (i % 3) * 0.31
        y = 0.43 - (i // 3) * 0.27
        card(ax, x, y, 0.29, 0.24, h, b, c, head_size=13.5, body_size=11)
    ax.text(0.04, 0.08, "Key insight: for continuous outputs (actions, audio, latents) 'exactly equal' is the wrong "
            "acceptance test — 'close enough' is natural.", fontsize=11.5, color=MUTED)
    return fig


def s_relaxed_framework():
    fig, ax = new_slide("How strict should the check be?",
                        "Xia, Ribar & Balanca (2607.08690): one framework for training-free relaxed SD",
                        "Paper 3 · taxonomy")
    ax.text(0.04, 0.7, "Relaxation = replace the target distribution p with a looser π at three points:",
            fontsize=14, color=INK)
    for i, (h, c) in enumerate([("accept / reject test", BLUE), ("resampling after a rejection", ORANGE),
                                ("the bonus token", AQUA)]):
        ax.add_patch(FancyBboxPatch((0.04 + i * 0.31, 0.56), 0.29, 0.09,
                                    boxstyle="round,pad=0,rounding_size=0.01", fc=FILLS[c], ec="none"))
        ax.text(0.185 + i * 0.31, 0.605, h, fontsize=13, ha="center", va="center", color=INK, weight="bold")
    card(ax, 0.04, 0.17, 0.29, 0.33, "KL-bounded",
         "CACTUS, mentored decoding\n\nMax acceptance within a\ndivergence budget from p.\nBest family for MTP drafters.", BLUE, body_size=11.5)
    card(ax, 0.355, 0.17, 0.29, 0.33, "Threshold / defer",
         "Fuzzy SD (reducible variant),\nspeculative cascades\n\nAccept if the two models are\n'close enough'; else defer.", ORANGE, body_size=11.5)
    card(ax, 0.67, 0.17, 0.29, 0.33, "Mix / contrast",
         "Ensemble α·q + (1−α)·p,\nspeculative contrastive decoding\n\nThe latter aims to improve\nquality, not speed.", AQUA, body_size=11.5)
    ax.text(0.04, 0.08, "Testbed: Qwen3 0.6B/8B → 32B, Qwen3.5 MTP → 27B, Llama 1B → 70B · AIME24, GPQA-D, "
            "LiveCodeBench · k = 3–20", fontsize=11.5, color=MUTED)
    return fig


def s_two_cents():
    fig, ax = new_slide("My two cents", None, "Opinion")
    card(ax, 0.04, 0.13, 0.44, 0.65, "What they get right",
         "• Measuring inside the engine people deploy, at\n   batch sizes people run (Berkeley)\n\n"
         "• Verification dominates → research should move\n   from better guessers to smarter checking\n\n"
         "• Reporting end-to-end speedup incl. longer\n   answers (relaxed SD)\n\n"
         "• Honest 'partially ready' verdict for multimodal", AQUA, body_size=12.5)
    card(ax, 0.52, 0.13, 0.44, 0.65, "Where I'm skeptical",
         "• Oracle gains (2.75×, +2.2×) assume perfect\n   foresight; a real router costs time and errs\n\n"
         "• Multimodal: model, drafter & backend change\n   together — trust the direction, not the size\n\n"
         "• Relaxed SD speedups are mostly modelled,\n   not wall-clock\n\n"
         "• KV survey skips SD — SD-aware KV scheduling\n   is an obvious open combination", ORANGE, body_size=12.5)
    return fig


def s_future():
    fig, ax = new_slide("Where the field is heading",
                        "From 'one drafter, fixed k, always on' to a runtime policy", "Open problems")
    bullets(ax, [
        "Verify less: only check tokens likely to be accepted; cheaper verification kernels",
        "Adaptive per-step routing: choose n-gram / EAGLE / MTP / off and the draft length k online",
        "Better MTP heads: train for positions 2, 3, … not just the next token",
        "L2 diffusion drafters with dependency modelling inside the block; tree-aware training",
        "Multimodal: compress visual context for the drafter; fall back to plain decoding at high resolution",
        "Serving co-design: KV schedulers aware of draft tokens; energy & tail-latency reporting",
        "Relaxed SD as a product knob — only with task-specific capability evaluation",
    ], y=0.7, size=14.5, gap=0.08)
    return fig


def s_end():
    fig, ax = new_slide("Thank you", "Code, figures, article and slides are in the GitHub repository", "References")
    refs = [
        "[1] Liu, Yu, Park, Stoica, Cheung. Speculative Decoding: Performance or Illusion? arXiv:2601.11580 (2026)",
        "[2] Li et al. Is Multimodal Speculative Decoding Ready for Diffusion-Based Parallel Drafting? arXiv:2608.20743 (2026)",
        "[3] Xia, Ribar, Balanca. A Practical Investigation of Training-free Relaxed Speculative Decoding. arXiv:2607.08690 (2026)",
        "[4] Jiang, Yang, Zhang, Liu. Towards Efficient LLM Serving: A Survey on System-Aware KV Cache Optimization.\n"
        "      arXiv:2607.08057, ACL Findings 2026",
        "[5] Leviathan, Kalman, Matias. Fast Inference from Transformers via Speculative Decoding. ICML 2023",
    ]
    for i, r in enumerate(refs):
        ax.text(0.04, 0.68 - i * 0.095, r, fontsize=12, color=INK2, va="top", linespacing=1.4)
    ax.text(0.04, 0.12, "All charts drawn by the author from numbers reported in the papers.", fontsize=11.5, color=MUTED)
    return fig


SLIDES = [
    s_title, s_question, s_papers,
    FIG / "f01_draft_verify.png", FIG / "f02_speedup_formula.png", FIG / "f03_taxonomy.png", s_levels,
    s_setup, FIG / "f04_batch_decay.png", FIG / "f05_time_breakdown.png", FIG / "f06_tree_vs_chain.png",
    FIG / "f07_ngram_oracle.png", s_more_findings,
    s_mm_landscape, FIG / "f08_multimodal.png", FIG / "f09_mm_bottleneck.png",
    s_relaxed_framework, FIG / "f10_relaxed.png",
    FIG / "f11_kv_taxonomy.png", FIG / "f12_decision_guide.png",
    s_two_cents, s_future, s_end,
]


def render():
    paths = []
    n = len(SLIDES)
    for i, s in enumerate(SLIDES, 1):
        out = PNG / f"slide_{i:02d}.png"
        src = s() if callable(s) else s
        if isinstance(src, Path):
            img = Image.open(src).convert("RGB").resize((int(W * DPI), int(H * DPI)), Image.LANCZOS)
            fig = plt.figure(figsize=(W, H))
            fig.figimage(img, 0, 0)
        else:
            fig = src
        if i > 1:
            fig.text(0.975, 0.02, f"{i} / {n}", ha="right", fontsize=10, color=MUTED)
        fig.savefig(out, dpi=DPI, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths.append(out)
    return paths


def notes():
    script = (ROOT / "video" / "script.md").read_text()
    blocks = re.split(r"^## Slide (\d+).*$", script, flags=re.M)
    return {int(blocks[i]): blocks[i + 1].strip() for i in range(1, len(blocks), 2)}


def build_pdf(paths):
    with PdfPages(OUT / "speculative-decoding-deck.pdf") as pdf:
        for p in paths:
            fig = plt.figure(figsize=(W, H))
            fig.figimage(Image.open(p), 0, 0)
            pdf.savefig(fig, dpi=DPI)
            plt.close(fig)


def build_pptx(paths):
    from pptx import Presentation
    from pptx.util import Emu
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)
    nts = notes()
    for i, p in enumerate(paths, 1):
        sl = prs.slides.add_slide(prs.slide_layouts[6])
        sl.shapes.add_picture(str(p), 0, 0, prs.slide_width, prs.slide_height)
        if i in nts:
            sl.notes_slide.notes_text_frame.text = nts[i]
    prs.save(OUT / "speculative-decoding-deck.pptx")


if __name__ == "__main__":
    ps = render()
    build_pdf(ps)
    build_pptx(ps)
    print(f"{len(ps)} slides written")
