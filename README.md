# LLM prompt perturbation

This repository contains our code that implements the prompt perturbation method proposed in [InferDPT](https://arxiv.org/pdf/2310.12214). We basically follow the idea proposed in InferDPT except that we use cosine similarity to compute the similarity/distance between embeddings except Euclidean distance.

## Entry points

1. First run `create_perturbation_files.py` to create files necessary to perturb LLM prompt.
2. Then, run `perturb_prompt.py` to perturb prompts.
