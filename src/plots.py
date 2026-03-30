
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

plt.rcParams.update({
    "font.family"     : "serif",
    "font.size"       : 11,
    "axes.titlesize"  : 12,
    "axes.labelsize"  : 11,
    "xtick.labelsize" : 9,
    "ytick.labelsize" : 9,
    "figure.dpi"      : 150,
    "savefig.bbox"    : "tight",
    "savefig.dpi"     : 150,
})

CAT_SHORT = [
    "Adult\nContent", "Child\nAbuse", "Economic\nHarm", "Fraud/\nDeception",
    "Hate/Harass/\nViolence", "Illegal\nActivity", "Malware\nViruses",
    "Physical\nHarm", "Political\nCampaigning", "Privacy\nViolation",
    "Tailored\nFinancial",
]


def pearson_r(x, y):
    x, y = np.array(x), np.array(y)
    return np.corrcoef(x, y)[0, 1]


def scatter_with_labels(ax, x, y, labels, xlabel, ylabel, title, color, annotate=True):
    ax.scatter(x, y, c=color, s=80, zorder=3, edgecolors="white", linewidths=0.5)
    if annotate:
        for xi, yi, lab in zip(x, y, labels):
            ax.annotate(lab, (xi, yi), textcoords="offset points",
                        xytext=(4, 4), fontsize=7.5, alpha=0.85)
    r = pearson_r(x, y)
    m, b = np.polyfit(x, y, 1)
    xline = np.linspace(min(x), max(x), 100)
    ax.plot(xline, m * xline + b, "--", color=color, alpha=0.5, linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n(Pearson r = {r:.3f}, n=11)")
    ax.grid(True, alpha=0.3)
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="../results/")
    parser.add_argument("--output_dir",  default="../figures/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = torch.load(os.path.join(args.results_dir, "combined_analysis.pt"),
                      map_location="cpu")

    categories       = data["categories"]
    harm_intercatsim = data["harm_intercategory"].numpy()
    ref_intercatsim  = data["ref_intercategory"].numpy()
    harm_ref_align   = data["harm_ref_alignment"].numpy()
    harm_norms       = data["harm_norms"].numpy()
    ref_norms        = data["ref_norms"].numpy()
    acceptance       = data["acceptance_rates"].numpy() * 100  # percent
    dir_data = torch.load(os.path.join(args.results_dir, "directions_analysis.pt"),
                        map_location="cpu")
    mean_sim_matrix = dir_data["mean_sim_matrix"].numpy()
    ls, le           = data["layer_start"].item(), data["layer_end"].item()

    short = CAT_SHORT

    fig, ax = plt.subplots(figsize=(7, 5.5))
    mask = np.eye(len(categories), dtype=bool)
    im = ax.imshow(mean_sim_matrix, vmin=0.5, vmax=1.0, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Cosine similarity")
    ax.set_xticks(range(len(categories)))
    ax.set_yticks(range(len(categories)))
    ax.set_xticklabels([c.replace("/", "/\n") for c in categories],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(categories, fontsize=8)
    for i in range(len(categories)):
        for j in range(len(categories)):
            val = mean_sim_matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6.5, color="black" if val < 0.85 else "white")
    ax.set_title(f"Inter-category harmfulness direction cosine similarity\n"
                 f"(Qwen2-7B, layers {ls}-{le}, replicating Zhao et al. Fig. 9a)")
    plt.tight_layout()
    path = os.path.join(args.output_dir, "fig1_similarity_matrix.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")

    fig, ax = plt.subplots(figsize=(9, 4))
    x     = np.arange(len(categories))
    width = 0.38
    b1 = ax.bar(x - width/2, harm_intercatsim, width, label="Harmfulness (t_inst)",
                color="#E07B54", alpha=0.85)
    b2 = ax.bar(x + width/2, ref_intercatsim,  width, label="Refusal (t_post-inst)",
                color="#5B8DB8", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel("Mean off-diagonal cosine similarity")
    ax.set_ylim(0.5, 1.05)
    ax.axhline(0.65, color="#E07B54", linestyle="--", linewidth=0.9, alpha=0.6,
               label="Zhao et al. HF reference (~0.65)")
    ax.axhline(0.92, color="#5B8DB8", linestyle="--", linewidth=0.9, alpha=0.6,
               label="Zhao et al. Ref reference (~0.92)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Harmfulness directions are more category-specific than refusal directions\n"
                 "(lower similarity = more distinct)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(args.output_dir, "fig2_harm_vs_ref_sim.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    r = scatter_with_labels(ax, ref_norms, acceptance, categories,
                            xlabel="Refusal direction norm (layers 9-20 avg)",
                            ylabel="Acceptance rate (%)",
                            title="Refusal direction norm predicts baseline vulnerability",
                            color="#5B8DB8")
    path = os.path.join(args.output_dir, "fig3_ref_norm_vs_acceptance.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}  [r={r:.3f}]")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    r = scatter_with_labels(ax, harm_norms, acceptance, categories,
                            xlabel="Harmfulness direction norm (layers 9-20 avg)",
                            ylabel="Acceptance rate (%)",
                            title="Harmfulness direction norm vs baseline vulnerability",
                            color="#E07B54")
    path = os.path.join(args.output_dir, "fig4_hf_norm_vs_acceptance.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}  [r={r:.3f}]")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    r = scatter_with_labels(ax, harm_ref_align, acceptance, categories,
                            xlabel="Harmfulness-refusal direction alignment (cosine sim)",
                            ylabel="Acceptance rate (%)",
                            title="Harmfulness-refusal alignment vs baseline vulnerability",
                            color="#6BAE75")
    path = os.path.join(args.output_dir, "fig5_alignment_vs_acceptance.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}  [r={r:.3f}]")

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#c0392b" if a > 20 else "#e67e22" if a > 10 else "#27ae60"
              for a in acceptance]
    bars = ax.bar(range(len(categories)), acceptance, color=colors, alpha=0.85)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel("Acceptance rate (%)")
    ax.set_title("Per-category baseline acceptance rate on Qwen2-7B\n"
                 "(higher = model more often fails to refuse)")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, acceptance):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
    legend_elements = [
        mpatches.Patch(color="#27ae60", alpha=0.85, label="Low vulnerability (≤10%)"),
        mpatches.Patch(color="#e67e22", alpha=0.85, label="Medium (10-20%)"),
        mpatches.Patch(color="#c0392b", alpha=0.85, label="High vulnerability (>20%)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")
    plt.tight_layout()
    path = os.path.join(args.output_dir, "fig6_acceptance_rates.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")

    print("\n── Summary of correlations with acceptance rate ─────────────────────")
    metrics = {
        "HF inter-category sim"  : harm_intercatsim,
        "Ref inter-category sim" : ref_intercatsim,
        "HF-Refusal alignment"   : harm_ref_align,
        "HF direction norm"      : harm_norms,
        "Ref direction norm"     : ref_norms,
    }
    for name, vals in metrics.items():
        r = pearson_r(vals, acceptance)
        print(f"  {name:<30}  r = {r:>7.4f}")


if __name__ == "__main__":
    main()