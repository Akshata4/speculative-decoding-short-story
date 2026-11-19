"""Generate every figure used in the article and slide deck.

All charts are drawn from numbers *reported* in the four papers (cited in each
figure footer) or computed from the Berkeley paper's expected-speedup formula.
Nothing is copied from the papers' own figures.

Usage:  python scripts/make_figures.py          (writes PNGs into figures/)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

# ---- design tokens (validated reference palette, light mode) ---------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
BLUE_LIGHT = "#cde2fb"
BLUE_MID = "#86b6ef"
GRAY_FILL = "#f0efec"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 13,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "savefig.dpi": 200,
})

W, H = 12, 6.75  # 16:9


def finish(fig, name, title, subtitle=None, source=None):
    fig.suptitle(title, x=0.04, y=0.975, ha="left", fontsize=20, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.04, 0.905, subtitle, ha="left", fontsize=13, color=INK2)
    if source:
        fig.text(0.04, 0.025, "Source: " + source, ha="left", fontsize=10, color=MUTED)
    fig.savefig(OUT / name, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, text, fc, ec=None, color=INK, size=12, weight="normal", r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec or fc, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            color=color, weight=weight, wrap=True)


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                                 color=color, lw=lw))


def blank_axes(fig, rect=(0.03, 0.08, 0.94, 0.78)):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


# ---------------------------------------------------------------------------
# 0. Cover
# ---------------------------------------------------------------------------
def cover():
    fig = plt.figure(figsize=(W, H))
    ax = blank_axes(fig, (0, 0, 1, 1))
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc="#0d366b"))
    rng = np.random.default_rng(7)
    # a row of "draft" tokens being checked
    xs = np.linspace(0.06, 0.94, 14)
    for i, x in enumerate(xs):
        acc = i < 9
        fc = "#3987e5" if acc else "#104281"
        ax.add_patch(FancyBboxPatch((x - 0.025, 0.18), 0.05, 0.09,
                                    boxstyle="round,pad=0,rounding_size=0.01", fc=fc, ec="none"))
        ax.text(x, 0.225, "✓" if acc else "·", ha="center", va="center", color="white", fontsize=16)
    ax.text(0.06, 0.68, "Speculative Decoding", color="white", fontsize=44, weight="bold")
    ax.text(0.06, 0.57, "Does the speedup survive production?", color="#b7d3f6", fontsize=26)
    ax.text(0.06, 0.42, "A short story from four 2026 papers: taxonomy, benchmarks, and what to deploy",
            color="#86b6ef", fontsize=15)
    fig.savefig(OUT / "f00_cover.png", facecolor="#0d366b")
    plt.close(fig)
    print("wrote f00_cover.png")


# ---------------------------------------------------------------------------
# 1. How draft-then-verify works
# ---------------------------------------------------------------------------
def draft_verify():
    fig = plt.figure(figsize=(W, H))
    ax = blank_axes(fig)
    # left: normal decoding
    ax.text(0.0, 0.95, "Normal decoding: 1 big-model pass per token", fontsize=14, weight="bold", color=INK)
    for i in range(5):
        box(ax, 0.0 + i * 0.09, 0.78, 0.075, 0.1, f"pass\n{i+1}", BLUE_LIGHT, color=INK, size=10)
    ax.text(0.0, 0.71, "5 tokens  →  5 sequential passes of the large model", fontsize=11, color=INK2)

    ax.text(0.0, 0.58, "Speculative decoding: guess cheaply, check in one pass", fontsize=14,
            weight="bold", color=INK)
    # drafter
    box(ax, 0.0, 0.32, 0.17, 0.17, "Drafter\n(small model, EAGLE\nhead, MTP head,\nor n-gram lookup)",
        "#fdebe2", ORANGE, size=10.5)
    toks = ["the", "cat", "sat", "on", "mat"]
    for i, t in enumerate(toks):
        box(ax, 0.23 + i * 0.075, 0.37, 0.065, 0.08, t, "#fdebe2", ORANGE, size=11)
    ax.text(0.23, 0.48, "k = 5 draft tokens", fontsize=11, color=INK2)
    arrow(ax, 0.17, 0.405, 0.225, 0.405)
    arrow(ax, 0.60, 0.405, 0.66, 0.405)
    box(ax, 0.66, 0.30, 0.16, 0.21, "Target LLM\nverifies all k\nin ONE forward\npass", BLUE_LIGHT, BLUE, size=11)
    arrow(ax, 0.82, 0.405, 0.86, 0.405)
    # accepted result
    res = [("the", True), ("cat", True), ("sat", True), ("on", False), ("a", None)]
    for i, (t, ok) in enumerate(res):
        fc = "#d2f1e4" if ok else ("#fbdada" if ok is False else BLUE_LIGHT)
        ec = AQUA if ok else (RED if ok is False else BLUE)
        y = 0.62 - i * 0.085
        box(ax, 0.87, y, 0.12, 0.07, (t + ("  ✓" if ok else "  ✗" if ok is False else "  (bonus)")),
            fc, ec, size=10.5)
    ax.text(0.87, 0.71, "accepted 3 + 1 bonus", fontsize=10, color=INK2, ha="left")
    ax.text(0.0, 0.17,
            "Same output distribution as normal decoding (lossless), but 4 tokens for the price of\n"
            "1 big-model pass + 5 cheap drafter passes.  Speedup depends on how many guesses survive.",
            fontsize=12, color=INK2)
    finish(fig, "f01_draft_verify.png", "How speculative decoding works",
           "A cheap drafter proposes tokens; the large model checks them all at once",
           "Author's illustration of the draft-then-verify loop (Leviathan et al. 2023; Chen et al. 2023).")


# ---------------------------------------------------------------------------
# 2. The speedup formula, plotted
# ---------------------------------------------------------------------------
def speedup_formula():
    def es(alpha, k, c):
        return (1 - alpha ** (k + 1)) / ((1 - alpha) * (k * c + 1))

    fig, axes = plt.subplots(1, 2, figsize=(W, H), sharey=True)
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.07, right=0.97, wspace=0.08)
    ks = np.arange(1, 13)
    alphas = [(0.9, BLUE), (0.75, ORANGE), (0.6, AQUA)]
    for ax, c, lab in [(axes[0], 0.05, "Cheap drafter (c = 0.05, e.g. n-gram / EAGLE head)"),
                       (axes[1], 0.375, "Expensive drafter (c = 0.375, 0.6B draft for 8B target)")]:
        for a, col in alphas:
            y = es(a, ks, c)
            ax.plot(ks, y, color=col, lw=2.2, marker="o", ms=5)
            best = ks[np.argmax(y)]
            ax.text(ks[-1] + 0.2, y[-1], f"α = {a}", color=INK2, va="center", fontsize=11)
            ax.plot(best, y.max(), "o", ms=10, mfc="none", mec=col, mew=1.8)
        ax.axhline(1, color=BASE, lw=1.2, ls="--")
        ax.set_title(lab, fontsize=13)
        ax.set_xlabel("Draft length k (tokens proposed per step)")
        ax.set_xlim(0.5, 14)
    axes[0].set_ylabel("Expected speedup (×)")
    axes[0].text(13.8, 1.05, "break-even (1×)", color=MUTED, fontsize=10, ha="right")
    finish(fig, "f02_speedup_formula.png", "Why speedups are fragile: the expected-speedup formula",
           "E[speedup] = (1 − α^(k+1)) / ((1 − α)(k·c + 1)).  α = acceptance rate, c = drafter cost relative to "
           "target.  Circles mark the best k.",
           "Formula from Liu et al., 'Speculative Decoding: Performance or Illusion?' (arXiv 2601.11580), §4.1; "
           "curves by the author (assumes verify ≈ one decode step).")


# ---------------------------------------------------------------------------
# 3. Unified taxonomy map
# ---------------------------------------------------------------------------
def taxonomy():
    fig = plt.figure(figsize=(W, H))
    ax = blank_axes(fig, (0.02, 0.07, 0.96, 0.8))
    cols = [
        ("1  WHO DRAFTS?", BLUE, "#e6f0fc", [
            ("Training-free lookup", "n-gram / prompt lookup"),
            ("L0  Sequential drafter", "small draft LM, EAGLE / EAGLE-3"),
            ("L1  Multi-position heads", "Medusa, MTP, SpecFLASH"),
            ("L2  Block-parallel (diffusion)", "DFlash, DSpark"),
        ]),
        ("2  WHAT SHAPE?", ORANGE, "#fdebe2", [
            ("Chain", "one guess per position"),
            ("Tree", "SpecInfer, Sequoia, k = 6 / 21 nodes"),
            ("Adaptive length", "oracle / predicted k"),
            ("Method routing", "pick n-gram vs EAGLE per step"),
        ]),
        ("3  HOW STRICT A CHECK?", AQUA, "#dcf3ea", [
            ("Lossless (strict)", "exact rejection sampling"),
            ("Relaxed, KL-bounded", "CACTUS, mentored decoding"),
            ("Relaxed, threshold/defer", "fuzzy SD, speculative cascades"),
            ("Relaxed, mix/contrast", "ensemble, spec. contrastive"),
        ]),
        ("4  WHAT SERVES IT?", VIOLET, "#e9e6fa", [
            ("Temporal: when", "KV-aware scheduling, P/D split"),
            ("Spatial: where", "GPU↔CPU↔SSD offload, migration"),
            ("Structural: how", "paging, quantization, eviction"),
            ("Engine", "vLLM, SGLang, batch size"),
        ]),
    ]
    cw, gap = 0.235, 0.02
    for ci, (head, col, fill, items) in enumerate(cols):
        x = ci * (cw + gap)
        box(ax, x, 0.86, cw, 0.11, head, col, color="white", size=13, weight="bold")
        for ri, (t, sub) in enumerate(items):
            y = 0.66 - ri * 0.2
            ax.add_patch(FancyBboxPatch((x, y), cw, 0.17, boxstyle="round,pad=0,rounding_size=0.015",
                                        fc=fill, ec="none"))
            ax.text(x + 0.015, y + 0.115, t, fontsize=12, weight="bold", color=INK, va="center")
            ax.text(x + 0.015, y + 0.05, sub, fontsize=10.5, color=INK2, va="center")
    fig.text(0.04, 0.035,
             "Columns 1 & 2: Li et al. (arXiv 2608.20743) L0–L2 taxonomy + Liu et al. (2601.11580).  Column 3: Xia et al. "
             "(2607.08690).  Column 4: Jiang et al. (2607.08057). Synthesis by the author.",
             fontsize=10, color=MUTED)
    fig.suptitle("One map of the field: four independent design choices", x=0.04, y=0.975, ha="left",
                 fontsize=20, fontweight="bold", color=INK)
    fig.text(0.04, 0.905, "Every speculative decoding system picks one option per column — and the columns interact",
             fontsize=13, color=INK2)
    fig.savefig(OUT / "f03_taxonomy.png", facecolor=SURFACE)
    plt.close(fig)
    print("wrote f03_taxonomy.png")


# ---------------------------------------------------------------------------
# 4. Speedup decays with batch size
# ---------------------------------------------------------------------------
def batch_decay():
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(top=0.8, bottom=0.17, left=0.08, right=0.8)
    series = [
        ("EAGLE · Llama3-70B · ShareGPT", [1, 32], [1.96, 1.72], BLUE),
        ("EAGLE · Llama3.1-8B · ShareGPT", [1, 32], [1.68, 1.61], ORANGE),
        ("EAGLE · Llama3.1-8B · GSM8K", [1, 128], [1.73, 1.21], AQUA),
    ]
    for lab, x, y, col in series:
        ax.plot(x, y, color=col, lw=2.4, marker="o", ms=9, mec=SURFACE, mew=2)
        drop = (y[0] - y[1]) / y[0] * 100
        ax.text(x[1] * 1.12, y[1], f"{lab}\n{y[0]:.2f}× → {y[1]:.2f}×  (−{drop:.0f}%)",
                va="center", fontsize=11, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_ylim(1.0, 2.1)
    ax.axhline(1, color=BASE, lw=1.2)
    ax.set_xlabel("Concurrent requests (batch size, log scale)")
    ax.set_ylabel("Speedup over no-SD decoding (×)")
    ax.grid(axis="x", visible=False)
    finish(fig, "f04_batch_decay.png", "The speedup shrinks as the server gets busy",
           "Only the reported end-points are plotted; bigger models and busier servers lose more",
           "Liu et al., arXiv 2601.11580 (vLLM, H100). Lines connect two reported points, not a measured curve.")


# ---------------------------------------------------------------------------
# 5. Where the time goes
# ---------------------------------------------------------------------------
def time_breakdown():
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(top=0.8, bottom=0.17, left=0.3, right=0.95)
    rows = [
        ("Verification (target model)", 42, 95, BLUE),
        ("Drafting — EAGLE (batch 1)", 12, 20, ORANGE),
        ("Engine overhead (vLLM)", 3, 12, MUTED),
        ("Drafting — n-gram", 0, 2, AQUA),
        ("Sampling", 0, 1.7, MUTED),
    ]
    for i, (lab, lo, hi, col) in enumerate(rows[::-1]):
        ax.barh(i, hi - lo, left=lo, height=0.5, color=col)
        ax.text(hi + 1.5, i, f"{lo:g}–{hi:g}%" if lo else f"< {hi:g}%", va="center", fontsize=12, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=13, color=INK)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Share of end-to-end execution time (range across models, datasets, batch sizes)")
    ax.grid(axis="y", visible=False)
    finish(fig, "f05_time_breakdown.png", "Checking, not guessing, is the bottleneck",
           "Verification grows with model size and batch size; cheap drafters are nearly free",
           "Liu et al., arXiv 2601.11580, §5.1 (reported ranges).")


# ---------------------------------------------------------------------------
# 6. Tree vs chain
# ---------------------------------------------------------------------------
def tree_vs_chain():
    fig, axes = plt.subplots(1, 2, figsize=(W, H), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.07, right=0.97, wspace=0.3)
    ax = axes[0]
    labs, vals = ["Chain", "Tree k = 6", "Tree k = 21"], [1.65, 1.68, 1.85]
    ax.bar(labs, vals, color=[BLUE, ORANGE, AQUA], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.2f}×", ha="center", fontsize=13, color=INK)
    ax.axhline(1, color=BASE, lw=1.2, ls="--")
    ax.set_ylim(0, 2.2)
    ax.set_title("Batch 1: trees win a little", fontsize=14)
    ax.set_ylabel("Speedup (×)  — Qwen3-8B, GSM8K")
    ax.grid(axis="x", visible=False)
    ax.text(1, 0.35, "At batch 64 the k = 21 tree drops below 1×\n(slower than no SD) on every workload;\n"
            "the chain stays above 1×", ha="center", fontsize=11, color=INK,
            bbox=dict(fc="#fbdada", ec=RED, boxstyle="round,pad=0.5"))

    ax = axes[1]
    ax.bar([0, 1], [2.25, 2.92], width=0.5, color=[BLUE, AQUA])
    for xi, v, r in zip([0, 1], [2.25, 2.92], ["41.5%", "9.5%"]):
        ax.text(xi, v + 0.06, f"{v:.2f} tokens", ha="center", fontsize=13, color=INK)
        ax.text(xi, 0.5, f"{r}\nof drafted\ntokens\naccepted", ha="center", fontsize=11, color="white")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Narrow (chain)", "Widest tree (k = 21)"])
    ax.set_ylim(0, 3.5)
    ax.set_ylabel("Accepted tokens per verification step")
    ax.set_title("More tokens kept, far more work wasted", fontsize=14)
    ax.grid(axis="x", visible=False)
    finish(fig, "f06_tree_vs_chain.png", "Tree verification: a batch-1 trick",
           "Wider trees keep ~30% more tokens per step, but only 1 in 10 drafted tokens survives verification",
           "Liu et al., arXiv 2601.11580, §3.2 (SGLang v0.5.9).")


# ---------------------------------------------------------------------------
# 7. n-gram vs EAGLE and the oracle gap
# ---------------------------------------------------------------------------
def ngram_oracle():
    fig, axes = plt.subplots(1, 2, figsize=(W, H))
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.07, right=0.97, wspace=0.3)
    ax = axes[0]
    ax.bar(["EAGLE", "n-gram"], [4.24, 7.27], color=[BLUE, AQUA], width=0.5)
    for i, v in enumerate([4.24, 7.27]):
        ax.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=14, color=INK)
    ax.set_ylim(0, 8.5)
    ax.set_title("Code editing: the 'dumb' method wins", fontsize=14)
    ax.set_ylabel("Mean accepted tokens per step (InstructCoder)")
    ax.grid(axis="x", visible=False)
    ax.text(1, 1.0, "Output repeats the prompt\n→ copying beats predicting", ha="center", fontsize=11,
            color="white")

    ax = axes[1]
    ax.bar(["No SD", "Best fixed k", "Oracle k"], [1.0, 2.1, 2.75], color=[GRAY_FILL, BLUE_MID, BLUE], width=0.55,
           edgecolor=[BASE, BLUE_MID, BLUE])
    for i, v in enumerate([1.0, 2.1, 2.75]):
        ax.text(i, v + 0.06, f"{v:.2f}×" if i else "1×", ha="center", fontsize=14, color=INK)
    ax.set_ylim(0, 3.2)
    ax.set_title("Headroom: knowing k in advance", fontsize=14)
    ax.set_ylabel("Speedup (×) — InstructCoder, batch 1")
    ax.grid(axis="x", visible=False)
    finish(fig, "f07_ngram_oracle.png", "No single method wins — and the ceiling is higher than we reach",
           "Per-step choice of method and draft length is the open opportunity",
           "Liu et al., arXiv 2601.11580, §6–8 (oracle = perfect predictor, not a real system).")


# ---------------------------------------------------------------------------
# 8. Multimodal: which drafter level wins
# ---------------------------------------------------------------------------
def multimodal():
    fig, axes = plt.subplots(1, 2, figsize=(W, H), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.07, right=0.97, wspace=0.28)
    ax = axes[0]
    m = ["EAGLE-3\n(L0)", "MTP\n(L1)", "DSpark\n(L2)", "DFlash\n(L2)"]
    sp = [1.69, 1.64, 2.04, 2.60]
    mat = [2.31, 3.22, 3.52, 4.38]
    ax.bar(m, sp, color=[BLUE, ORANGE, AQUA, AQUA], width=0.55)
    for i, (s, a) in enumerate(zip(sp, mat)):
        ax.text(i, s + 0.05, f"{s:.2f}×", ha="center", fontsize=13, color=INK, weight="bold")
        ax.text(i, 0.12, f"{a:.2f}\ntok/step", ha="center", fontsize=10, color="white")
    ax.axhline(1, color=BASE, lw=1.2, ls="--")
    ax.set_ylim(0, 3)
    ax.set_title("Qwen3.6-27B, six vision-language tasks", fontsize=14)
    ax.set_ylabel("Wall-clock speedup (×)")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    xs = np.arange(2)
    ax.bar(xs - 0.18, [0.71, 0.88], width=0.34, color=BLUE, label="EAGLE-3 (L0)")
    ax.bar(xs + 0.18, [1.95, 2.14], width=0.34, color=AQUA, label="DFlash (L2)")
    for xi, v in zip(list(xs - 0.18) + list(xs + 0.18), [0.71, 0.88, 1.95, 2.14]):
        ax.text(xi, v + 0.05, f"{v:.2f}×", ha="center", fontsize=12, color=INK)
    ax.axhline(1, color=RED, lw=1.4, ls="--")
    ax.text(1.45, 1.1, "below this line = slower than no SD", color=RED, fontsize=10, ha="right", va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels(["Qwen3-VL-4B", "Qwen3-VL-8B"])
    ax.set_ylim(0, 3)
    ax.set_title("Small native VLMs", fontsize=14)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(axis="x", visible=False)
    finish(fig, "f08_multimodal.png", "Multimodal: block-parallel (diffusion) drafters lead",
           "L2 drafters propose a whole block at once; old-style EAGLE-3 can even slow small VLMs down",
           "Li et al., arXiv 2608.20743, Table 4 (SGLang, greedy, 600 samples).")


# ---------------------------------------------------------------------------
# 9. Multimodal bottleneck: vision prefill + high resolution
# ---------------------------------------------------------------------------
def mm_bottleneck():
    fig, axes = plt.subplots(1, 2, figsize=(W, H))
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.16, right=0.97, wspace=0.45)
    ax = axes[0]
    rows = [("Vision conditioning", 26.5, VIOLET), ("DSpark drafting", 1.78, AQUA), ("DFlash drafting", 1.14, AQUA)]
    for i, (lab, v, col) in enumerate(rows[::-1]):
        ax.barh(i, v, color=col, height=0.5)
        ax.text(v + 0.6, i, f"{'~' if i == 2 else ''}{v:g} ms", va="center", fontsize=12, color=INK)
    ax.set_yticks(range(3))
    ax.set_yticklabels([r[0] for r in rows[::-1]], color=INK)
    ax.set_xlim(0, 34)
    ax.set_xlabel("Latency per stage (ms), Qwen3.6-27B")
    ax.set_title("Images cost more than drafting", fontsize=14)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    for i, (lab, lo, hi) in enumerate([("HR-Bench 4K", 0.96, 1.11), ("HR-Bench 8K", 0.83, 0.99)]):
        ax.plot([lo, hi], [i, i], color=BLUE if i == 0 else ORANGE, lw=10, solid_capstyle="round")
        ax.text(hi + 0.02, i, f"{lo:.2f}–{hi:.2f}×", va="center", fontsize=12, color=INK)
    ax.axvline(1, color=RED, lw=1.4, ls="--")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["4K images", "8K images"], color=INK)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(0.75, 1.3)
    ax.set_xlabel("Speedup range across drafters (×)")
    ax.set_title("High resolution erases the gain", fontsize=14)
    ax.grid(axis="y", visible=False)
    finish(fig, "f09_mm_bottleneck.png", "Why multimodal is only 'partially ready'",
           "Even with ~2.5–3.2 accepted tokens per step, big images make prefill dominate",
           "Li et al., arXiv 2608.20743, Fig. 3 and Table 5.")


# ---------------------------------------------------------------------------
# 10. Relaxed SD: more tokens accepted, but at what cost
# ---------------------------------------------------------------------------
def relaxed():
    fig, axes = plt.subplots(1, 2, figsize=(W, H), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.2, right=0.97, wspace=0.35)
    ax = axes[0]
    rows = [("Qwen3 8B → 32B\n(strong LM drafter)", 8.15, 14.95),
            ("Qwen3.5 MTP → 27B\n(MTP head drafter)", 5.36, 10.68),
            ("Qwen3 0.6B → 32B\n(weak LM drafter)", 5.25, 9.36)]
    for i, (lab, a, b) in enumerate(rows[::-1]):
        ax.plot([a, b], [i, i], color=BASE, lw=3, zorder=1)
        ax.scatter([a], [i], s=140, color=BLUE, zorder=2, label="Strict (lossless)" if i == 0 else None)
        ax.scatter([b], [i], s=140, color=ORANGE, zorder=2, label="Relaxed (CACTUS, max α)" if i == 0 else None)
        ax.text(a, i + 0.22, f"{a:.2f}", ha="center", fontsize=11, color=INK2)
        ax.text(b, i + 0.22, f"{b:.2f}", ha="center", fontsize=11, color=INK2)
    ax.set_yticks(range(3))
    ax.set_yticklabels([r[0] for r in rows[::-1]], color=INK, fontsize=11.5)
    ax.set_xlim(3, 17)
    ax.set_ylim(-0.6, 3.2)
    ax.set_xlabel("Mean accepted tokens per step (AIME24, draft length 20)")
    ax.set_title("Relaxing the check ~doubles acceptance", fontsize=14)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    ax.axis("off")
    ax.set_title("…but the bill arrives elsewhere", fontsize=14)
    notes = [
        ("Accuracy must be re-tested", "per task: the same α behaves\ndifferently on AIME vs GPQA"),
        ("Weak / MTP drafters", "cause rambling, longer answers\n→ can be slower end-to-end"),
        ("Tune draft length first", "it's lossless and moves speed\nas much as relaxation does"),
    ]
    for i, (h, t) in enumerate(notes):
        y = 0.83 - i * 0.32
        ax.add_patch(FancyBboxPatch((0, y - 0.2), 1, 0.27, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc="#fdebe2", ec="none", transform=ax.transAxes))
        ax.text(0.04, y + 0.02, h, fontsize=12.5, weight="bold", color=INK, transform=ax.transAxes)
        ax.text(0.04, y - 0.1, t, fontsize=11, color=INK2, transform=ax.transAxes)
    finish(fig, "f10_relaxed.png", "Relaxed verification: faster, but no longer free",
           "Accepting 'close-enough' tokens trades the lossless guarantee for speed",
           "Xia, Ribar & Balanca, arXiv 2607.08690, Table 3 and §5.1 findings.")


# ---------------------------------------------------------------------------
# 11. KV-cache serving taxonomy
# ---------------------------------------------------------------------------
def kv_taxonomy():
    fig = plt.figure(figsize=(W, H))
    ax = blank_axes(fig, (0.02, 0.08, 0.96, 0.78))
    box(ax, 0.36, 0.82, 0.28, 0.13, "System-aware KV cache\noptimization", "#0d366b", color="white",
        size=14, weight="bold")
    branches = [
        (0.02, "TEMPORAL\nwhen is KV computed / used?", BLUE, "#e6f0fc",
         ["KV-aware request scheduling", "Compute / I/O overlap", "Prefill–decode disaggregation", "Offload to CPU / PIM"]),
        (0.35, "SPATIAL\nwhere does KV live?", ORANGE, "#fdebe2",
         ["GPU ↔ CPU ↔ SSD tiers", "Keep 'important' KV on GPU", "KV transfer between nodes", "CXL memory pools"]),
        (0.68, "STRUCTURAL\nhow is KV stored?", VIOLET, "#e9e6fa",
         ["Paged allocation, prefix reuse", "Quantization (KIVI, KVQuant)", "Low-rank (Palu, ShadowKV)", "Eviction (H2O, SnapKV, StreamingLLM)"]),
    ]
    for x, head, col, fill, items in branches:
        arrow(ax, 0.5, 0.82, x + 0.15, 0.71, color=BASE)
        box(ax, x, 0.58, 0.3, 0.13, head, col, color="white", size=12.5, weight="bold")
        for i, it in enumerate(items):
            box(ax, x, 0.44 - i * 0.12, 0.3, 0.1, it, fill, size=11.5)
    fig.text(0.04, 0.035, "Source: Jiang, Yang, Zhang & Liu, 'Towards Efficient LLM Serving: A Survey on System-Aware "
             "KV Cache Optimization' (arXiv 2607.08057, ACL Findings 2026). Redrawn by the author.",
             fontsize=10, color=MUTED)
    fig.suptitle("The serving layer under every speedup", x=0.04, y=0.975, ha="left", fontsize=20,
                 fontweight="bold", color=INK)
    fig.text(0.04, 0.905, "Speculative decoding adds KV for drafts, trees and draft models — this is the system that "
             "has to absorb it", fontsize=13, color=INK2)
    fig.savefig(OUT / "f11_kv_taxonomy.png", facecolor=SURFACE)
    plt.close(fig)
    print("wrote f11_kv_taxonomy.png")


# ---------------------------------------------------------------------------
# 12. Deployment decision guide (author's synthesis)
# ---------------------------------------------------------------------------
def decision_guide():
    fig = plt.figure(figsize=(W, H))
    ax = blank_axes(fig, (0.02, 0.06, 0.96, 0.8))
    rows = [
        ("Prompt and output overlap a lot\n(code edits, RAG quoting, rewriting)", "n-gram / prompt lookup",
         "free to try, no training, no memory"),
        ("Low concurrency, latency-critical\n(chat, single-user, on-device)", "EAGLE-3 or MTP, chain;\ntry small trees",
         "biggest gains at batch 1–8"),
        ("Large target (70B+) with a\nmatching tiny model available", "Draft model",
         "drafter cost ratio stays small"),
        ("High-throughput batch serving\n(batch 64+)", "Short chain (k ≈ 2–3) or SD off",
         "verification dominates; trees hurt"),
        ("Vision-language serving", "L2 block drafter (DFlash);\nwatch image resolution",
         "prefill can erase the gain"),
        ("Need more speed and can\nre-run your evals", "Relaxed acceptance (CACTUS)\nwith a strong drafter",
         "not lossless any more"),
    ]
    ax.text(0.0, 0.97, "If your workload is…", fontsize=13, weight="bold", color=INK2)
    ax.text(0.40, 0.97, "start with…", fontsize=13, weight="bold", color=INK2)
    ax.text(0.71, 0.97, "because…", fontsize=13, weight="bold", color=INK2)
    for i, (w, m, why) in enumerate(rows):
        y = 0.84 - i * 0.145
        ax.add_patch(FancyBboxPatch((0, y - 0.06), 1, 0.13, boxstyle="round,pad=0,rounding_size=0.015",
                                    fc="#f4f3f0" if i % 2 == 0 else SURFACE, ec="none"))
        ax.text(0.01, y, w, fontsize=11.5, color=INK, va="center")
        ax.text(0.40, y, m, fontsize=12, color=BLUE, va="center", weight="bold")
        ax.text(0.71, y, why, fontsize=11.5, color=INK2, va="center")
    fig.suptitle("My cheat-sheet: which speculative decoding to deploy", x=0.04, y=0.975, ha="left", fontsize=20,
                 fontweight="bold", color=INK)
    fig.text(0.04, 0.905, "Synthesised from the four papers — always confirm with a benchmark on your own traffic",
             fontsize=13, color=INK2)
    fig.text(0.04, 0.03, "Author's opinion, based on arXiv 2601.11580, 2608.20743, 2607.08690, 2607.08057.",
             fontsize=10, color=MUTED)
    fig.savefig(OUT / "f12_decision_guide.png", facecolor=SURFACE)
    plt.close(fig)
    print("wrote f12_decision_guide.png")


if __name__ == "__main__":
    cover()
    draft_verify()
    speedup_formula()
    taxonomy()
    batch_decay()
    time_breakdown()
    tree_vs_chain()
    ngram_oracle()
    multimodal()
    mm_bottleneck()
    relaxed()
    kv_taxonomy()
    decision_guide()
