import argparse
import torch
import torch.nn.functional as F
CATEGORIES = [
    ("Adult Content",             "Adult Content"),
    ("Child Abuse",               "Child Abuse"),
    ("Economic Harm",             "Economic Harm"),
    ("Fraud_Deception",           "Fraud/Deception"),
    ("Hate_Harass_Violence",      "Hate/Harass/Violence"),
    ("Illegal Activity",          "Illegal Activity"),
    ("Malware Viruses",           "Malware Viruses"),
    ("Physical Harm",             "Physical Harm"),
    ("Political Campaigning",     "Political Campaigning"),
    ("Privacy Violation Activity","Privacy Violation"),
    ("Tailored Financial Advice", "Tailored Financial Advice"),
]
FILTER_FILENAMES = [
    "Adult_Content",
    "Child_Abuse",
    "Economic_Harm",
    "Fraud_Deception",
    "Hate_Harass_Violence",
    "Illegal_Activity",
    "Malware_Viruses",
    "Physical_Harm",
    "Political_Campaigning",
    "Privacy_Violation_Activity",
    "Tailored_Financial_Advice",
]


def compute_cosine_similarity_matrix(
    directions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_categories, n_layers, _ = directions.shape
    layer_sim = torch.zeros(n_layers, n_categories, n_categories)
    for l in range(n_layers):
        d = directions[:, l, :]
        layer_sim[l] = d @ d.T
    mean_sim = layer_sim.mean(dim=0)
    return layer_sim, mean_sim


def compute_off_diagonal_means(sim_matrix: torch.Tensor) -> torch.Tensor:
    n = sim_matrix.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    off_diag_means = torch.zeros(n)
    for i in range(n):
        off_diag_means[i] = sim_matrix[i][mask[i]].mean()
    return off_diag_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",         default="../run/",
                        help="Directory containing qwen2-dir-{category}.pt files")
    parser.add_argument("--activations_dir", default="../activations/",
                        help="Directory containing refusal_labels.pt from filter_refused.py")
    parser.add_argument("--output_dir",      default="../results/")
    parser.add_argument("--layer_start",     default=9,  type=int,
                        help="Start layer for similarity computation (inclusive)")
    parser.add_argument("--layer_end",       default=20, type=int,
                        help="End layer for similarity computation (inclusive)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    n_categories = len(CATEGORIES)

    print("Loading direction vectors from Zhao et al. extraction pipeline...")
    all_dirs = []
    for fname, label in CATEGORIES:
        path = os.path.join(args.run_dir, f"qwen2-dir-{fname}.pt")
        d = torch.load(path, map_location="cpu").float()  # [n_layers, hidden_dim]
        all_dirs.append(d)
        print(f"  {label:<35}  shape={d.shape}  norm_l14={d[14].norm():.2f}")

    dirs_raw    = torch.stack(all_dirs, dim=0)
    dirs_normed = F.normalize(dirs_raw, dim=-1)

    n_layers   = dirs_raw.shape[1]
    hidden_dim = dirs_raw.shape[2]
    print(f"\n  n_categories={n_categories}  n_layers={n_layers}  hidden_dim={hidden_dim}")

    print(f"\nComputing cosine similarity matrix (layers {args.layer_start}-{args.layer_end})...")
    dirs_subset           = dirs_normed[:, args.layer_start:args.layer_end+1, :]
    layer_sim, mean_sim   = compute_cosine_similarity_matrix(dirs_subset)
    off_diag_means        = compute_off_diagonal_means(mean_sim)

    print("\n── Inter-category cosine similarity (off-diagonal means) ───────────")
    print(f"  {'Category':<35}  {'Mean sim to others':>20}")
    print(f"  {'-'*35}  {'-'*20}")
    for i, (_, label) in enumerate(CATEGORIES):
        print(f"  {label:<35}  {off_diag_means[i].item():>20.4f}")
    overall = off_diag_means.mean().item()
    print(f"\n  Overall mean (layers {args.layer_start}-{args.layer_end}): {overall:.4f}")
    print(f"  Zhao et al. Qwen2 reference:                ~0.65 (range 0.60-0.73)")

    acceptance_rates = torch.zeros(n_categories)
    refusal_labels_path = os.path.join(args.activations_dir, "refusal_labels.pt")

    if os.path.exists(refusal_labels_path):
        print(f"\nLoading refusal labels from {refusal_labels_path}...")
        labels = torch.load(refusal_labels_path, map_location="cpu")

        print(f"\n── Per-category acceptance rates (vulnerability signal) ─────────────")
        print(f"  {'Category':<35}  {'Refused':>8}  {'Total':>7}  {'Accept Rate':>12}")
        print(f"  {'-'*35}  {'-'*8}  {'-'*7}  {'-'*12}")

        for i, (fname, label) in enumerate(zip(FILTER_FILENAMES, [c[1] for c in CATEGORIES])):
            if fname in labels:
                mask      = labels[fname]
                n_refused = mask.sum().item()
                n_total   = len(mask)
                accept    = 1.0 - (n_refused / n_total)
                acceptance_rates[i] = accept
                print(f"  {label:<35}  {n_refused:>8}  {n_total:>7}  {accept:>11.1%}")
            else:
                print(f"  {label:<35}  NOT FOUND in refusal_labels.pt")
    else:
        print(f"\nNo refusal_labels.pt found at {refusal_labels_path}.")
        print("Run filter_refused.py first to get acceptance rates.")

    if acceptance_rates.any():
        print("\n── Geometric properties vs acceptance rate (preview) ───────────────")
        print(f"  {'Category':<35}  {'Off-diag sim':>13}  {'Accept Rate':>12}")
        print(f"  {'-'*35}  {'-'*13}  {'-'*12}")
        for i, (_, label) in enumerate(CATEGORIES):
            print(f"  {label:<35}  {off_diag_means[i].item():>13.4f}  {acceptance_rates[i].item():>11.1%}")

        x = off_diag_means
        y = acceptance_rates
        x_mean = x.mean()
        y_mean = y.mean()
        r = ((x - x_mean) * (y - y_mean)).sum() / (
            ((x - x_mean)**2).sum().sqrt() * ((y - y_mean)**2).sum().sqrt()
        )
        print(f"\n  Pearson r (sim vs acceptance): {r.item():.4f}")
        print(f"  Interpretation: {'positive' if r > 0 else 'negative'} correlation — "
              f"{'more similar directions -> more vulnerable' if r > 0 else 'more distinct directions -> more vulnerable'}")

    out_path = os.path.join(args.output_dir, "directions_analysis.pt")
    torch.save({
        "directions"       : dirs_normed,
        "directions_raw"   : dirs_raw,
        "categories"       : [c[1] for c in CATEGORIES],
        "layer_sim_matrix" : layer_sim,
        "mean_sim_matrix"  : mean_sim,
        "off_diag_means"   : off_diag_means,
        "acceptance_rates" : acceptance_rates,
        "layer_start"      : args.layer_start,
        "layer_end"        : args.layer_end,
    }, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()