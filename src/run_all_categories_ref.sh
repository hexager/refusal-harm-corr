#!/bin/bash
mkdir -p ../run
mkdir -p output
CATEGORIES=(
    "Adult Content"
    "Child Abuse"
    "Economic Harm"
    "Fraud_Deception"
    "Hate_Harass_Violence"
    "Illegal Activity"
    "Malware Viruses"
    "Physical Harm"
    "Political Campaigning"
    "Privacy Violation Activity"
    "Tailored Financial Advice"
)

FILENAMES=(
    "Adult Content"
    "Child Abuse"
    "Economic Harm"
    "Fraud_Deception"
    "Hate_Harass_Violence"
    "Illegal Activity"
    "Malware Viruses"
    "Physical Harm"
    "Political Campaigning"
    "Privacy Violation Activity"
    "Tailored Financial Advice"
)

for cat in "${FILENAMES[@]}"; do
    echo "Processing: $cat"
    bash get_diff_mean.sh \
        "../data/catqa_${cat}.json" \
        "../data/alpaca_data_instruction.json" \
        0 \
        "../run/qwen2-refuse-dir-${cat}.pt" \
        "../run/qwen2-refuse-${cat}.pt" \
        "../run/qwen2-refuse-harmless.pt" \
        0 200 0 "qwen" "refuse"
    echo "Done: $cat"
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

done
EOF