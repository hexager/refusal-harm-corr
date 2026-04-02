import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
CATEGORIES = [
    ("Adult Content",              "Adult Content"),
    ("Child Abuse",                "Child Abuse"),
    ("Economic Harm",              "Economic Harm"),
    ("Fraud_Deception",            "Fraud/Deception"),
    ("Hate_Harass_Violence",       "Hate/Harass/Violence"),
    ("Illegal Activity",           "Illegal Activity"),
    ("Malware Viruses",            "Malware Viruses"),
    ("Physical Harm",              "Physical Harm"),
    ("Political Campaigning",      "Political Campaigning"),
    ("Privacy Violation Activity", "Privacy Violation"),
    ("Tailored Financial Advice",  "Tailored Financial Advice"),
]

FILTER_FILENAMES = [
    "Adult_Content", "Child_Abuse", "Economic_Harm", "Fraud_Deception",
    "Hate_Harass_Violence", "Illegal_Activity", "Malware_Viruses",
    "Physical_Harm", "Political_Campaigning", "Privacy_Violation_Activity",
    "Tailored_Financial_Advice",
]


def pearson_r(x, y):
    x, y = np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)
    return np.corrcoef(x, y)[0, 1]


def off_diagonal_means(dirs_normed: torch.Tensor, layer_start: int, layer_end: int) -> torch.Tensor:
    n_cat = dirs_normed.shape[0]
    subset = dirs_normed[:, layer_start:layer_end+1, :]   # [n_cat, n_layers, hidden]
    sim = torch.einsum('ilh,jlh->ijl', subset, subset).mean(dim=-1)
    mask = ~torch.eye(n_cat, dtype=torch.bool)
    return torch.stack([sim[i][mask[i]].mean() for i in range(n_cat)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",         default="../run/")
    parser.add_argument("--activations_dir", default="../activations/")
    parser.add_argument("--output_dir",      default="../results/")
    parser.add_argument("--layer_start",     default=9,  type=int)
    parser.add_argument("--layer_end",       default=20, type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    n_cat = len(CATEGORIES)
    ls, le = args.layer_start, args.layer_end

    print("Loading harmfulness directions (t_inst)...")
    harm_raw = []
    for fname, label in CATEGORIES:
        path = os.path.join(args.run_dir, f"qwen2-dir-{fname}.pt")
        harm_raw.append(torch.load(path, map_location="cpu").float())
    harm_raw    = torch.stack(harm_raw, dim=0)   # [n_cat, n_layers, hidden]
    harm_normed = F.normalize(harm_raw, dim=-1)

    print("Loading refusal directions (t_post-inst)...")
    ref_raw = []
    for fname, label in CATEGORIES:
        path = os.path.join(args.run_dir, f"qwen2-refuse-dir-{fname}.pt")
        ref_raw.append(torch.load(path, map_location="cpu").float())
    ref_raw    = torch.stack(ref_raw, dim=0)     # [n_cat, n_layers, hidden]
    ref_normed = F.normalize(ref_raw, dim=-1)

    acceptance_rates = torch.zeros(n_cat)
    labels_path = os.path.join(args.activations_dir, "refusal_labels.pt")
    if os.path.exists(labels_path):
        labels = torch.load(labels_path, map_location="cpu")
        for i, fname in enumerate(FILTER_FILENAMES):
            if fname in labels:
                mask = labels[fname]
                acceptance_rates[i] = 1.0 - mask.float().mean()


    harm_intercategory = off_diagonal_means(harm_normed, ls, le)

    ref_intercategory  = off_diagonal_means(ref_normed, ls, le)

    harm_ref_alignment = torch.zeros(n_cat)
    for i in range(n_cat):
        h = harm_normed[i, ls:le+1, :]   # [n_layers, hidden]
        r = ref_normed[i,  ls:le+1, :]   # [n_layers, hidden]
        harm_ref_alignment[i] = (h * r).sum(dim=-1).mean()

    harm_norms = harm_raw[:, ls:le+1, :].norm(dim=-1).mean(dim=-1)  # [n_cat]
    ref_norms  = ref_raw[:,  ls:le+1, :].norm(dim=-1).mean(dim=-1)  # [n_cat]

    labels_str = [c[1] for c in CATEGORIES]

    print(f"\n── Full geometric analysis (layers {ls}-{le}) ──────────────────────────────────")
    print(f"  {'Category':<35}  {'HF inter-sim':>12}  {'Ref inter-sim':>13}  "
          f"{'HF-Ref align':>12}  {'HF norm':>8}  {'Accept%':>8}")
    print(f"  {'-'*35}  {'-'*12}  {'-'*13}  {'-'*12}  {'-'*8}  {'-'*8}")

    for i, label in enumerate(labels_str):
        print(f"  {label:<35}  {harm_intercategory[i]:>12.4f}  {ref_intercategory[i]:>13.4f}  "
              f"{harm_ref_alignment[i]:>12.4f}  {harm_norms[i]:>8.2f}  "
              f"{acceptance_rates[i].item()*100:>7.1f}%")

    print(f"\n── Pearson r with acceptance rate ──────────────────────────────────────────────")
    metrics = {
        "HF inter-category sim"   : harm_intercategory,
        "Ref inter-category sim"  : ref_intercategory,
        "HF-Refusal alignment"    : harm_ref_alignment,
        "HF direction norm"       : harm_norms,
        "Ref direction norm"      : ref_norms,
    }
    for name, vals in metrics.items():
        r = pearson_r(vals, acceptance_rates)
        bar_len = int(abs(r) * 20)
        bar = ("+" if r > 0 else "-") * bar_len
        print(f"  {name:<30}  r={r:>7.4f}  {bar}")

    print(f"\n  n=11 categories. |r| > 0.55 ~ p<0.1, |r| > 0.66 ~ p<0.05 (two-tailed)")
    print(f"  Zhao et al. reference: HF inter-sim ~0.65, Ref inter-sim ~0.89-0.95")
    print(f"  Our Qwen2-7B:          HF inter-sim {harm_intercategory.mean():.4f}, "
          f"Ref inter-sim {ref_intercategory.mean():.4f}")

    out_path = os.path.join(args.output_dir, "combined_analysis.pt")
    torch.save({
        "categories"          : labels_str,
        "harm_directions"     : harm_normed,
        "ref_directions"      : ref_normed,
        "harm_intercategory"  : harm_intercategory,
        "ref_intercategory"   : ref_intercategory,
        "harm_ref_alignment"  : harm_ref_alignment,
        "harm_norms"          : harm_norms,
        "ref_norms"           : ref_norms,
        "acceptance_rates"    : acceptance_rates,
        "layer_start"         : ls,
        "layer_end"           : le,
    }, out_path)
    print(f"\nSaved -> {out_path}")
    print("\nRef norms vs acceptance rates:")
    for i, label in enumerate(labels_str):
        print(f"  {label:<35}  ref_norm={ref_norms[i]:.4f}  accept={acceptance_rates[i]*100:.1f}%")

    print(f"\n── Robustness check: r without near-zero acceptance categories ─────────")
    thresholds = [5, 10]
    for thresh in thresholds:
        mask = acceptance_rates.numpy() > (thresh / 100)
        n_kept = mask.sum()
        if n_kept < 4:
            print(f"  threshold>{thresh}%: only {n_kept} categories, skipping")
            continue
        kept_cats = [labels_str[i] for i, m in enumerate(mask) if m]
        print(f"\n  threshold>{thresh}% (n={n_kept}): {kept_cats}")
        for name, vals in metrics.items():
            vals_np = vals.numpy() if hasattr(vals, 'numpy') else vals
            r_full    = pearson_r(vals_np, acceptance_rates.numpy())
            r_trimmed = pearson_r(vals_np[mask], acceptance_rates.numpy()[mask])
            change = r_trimmed - r_full
            print(f"    {name:<30}  r_full={r_full:>7.4f}  r_trimmed={r_trimmed:>7.4f}  "
                f"delta={change:>+7.4f}  {'HOLDS' if abs(r_trimmed) > 0.5 else 'WEAKENS'}")
if __name__ == "__main__":
    main()