import torch
import torch.nn.functional as F

CATEGORIES = [
    ("Adult Content",            "Adult Content"),
    ("Child Abuse",              "Child Abuse"),
    ("Economic Harm",            "Economic Harm"),
    ("Fraud_Deception",          "Fraud/Deception"),
    ("Hate_Harass_Violence",     "Hate/Harass/Violence"),
    ("Illegal Activity",         "Illegal Activity"),
    ("Malware Viruses",          "Malware Viruses"),
    ("Physical Harm",            "Physical Harm"),
    ("Political Campaigning",    "Political Campaigning"),
    ("Privacy Violation Activity", "Privacy Violation"),
    ("Tailored Financial Advice", "Tailored Financial Advice"),
]

# Load all directions
dirs = []
for fname, label in CATEGORIES:
    d = torch.load(f"run/qwen2-dir-{fname}.pt", map_location="cpu").float()
    dirs.append(d)

# Stack [n_categories, n_layers, hidden_dim]
dirs = torch.stack(dirs, dim=0)
print(f"Directions shape: {dirs.shape}")

# Normalize
dirs_normed = F.normalize(dirs, dim=-1)

# Compute similarity averaged over middle layers 9-20
dirs_mid = dirs_normed[:, 9:21, :]  # [n_categories, 12, hidden_dim]
sim = torch.einsum('ilh,jlh->ijl', dirs_mid, dirs_mid).mean(dim=-1)

# Print off-diagonal means
n = len(CATEGORIES)
mask = ~torch.eye(n, dtype=torch.bool)
print(f"\n── Inter-category cosine similarity (layers 9-20) ──────────────────")
for i, (_, label) in enumerate(CATEGORIES):
    off_diag_mean = sim[i][mask[i]].mean().item()
    print(f"  {label:<35}  {off_diag_mean:.4f}")

print(f"\n  Overall mean: {sim[mask].mean().item():.4f}")
print(f"  Zhao et al. Qwen2 reference: ~0.65")