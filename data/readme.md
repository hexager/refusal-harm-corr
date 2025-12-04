
## Harmful instructions

- [advbench](https://github.com/arobey1/advbench)
- [Sorry](https://sorry-bench.github.io/) // we use this dataset to obtain accepted harmful instructions.
- [CATQA](https://huggingface.co/datasets/declare-lab/CategoricalHarmfulQA) // This dataset provide categorical harmfulness instructions.

## Harmless instuctions

- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [Xstest](https://github.com/paul-rottger/xstest) // we use this dataset to obtain refused harmless instructions. 

## Jailbreak 

- [Persuasion](https://huggingface.co/datasets/CHATS-Lab/Persuasive-Jailbreaker-Data)
- [Template-based attack](https://github.com/sherdencooper/GPTFuzz/tree/master/datasets)
- [GCG](https://github.com/GraySwanAI/nanoGCG)   ///we use nanoGCG to learn adversarial suffix.


For extracting directions (or other latent-space analysis) for each model, you need to first infer that model to get the accepted/ refused instructions accordingly (using [src/run_inference.sh](https://github.com/CHATS-lab/LLMs_Encode_Harmfulness_Refusal_Separately/blob/main/src/run_inference.sh)). The results may differ depending on the model.
