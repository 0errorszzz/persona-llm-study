# persona-llm-study
learning and testing
# Persona Interpretation in Llama-2-7B

This project investigates the mechanistic interpretability of "Personas" within Large Language Models, specifically focusing on identifying specific neurons that encode social roles.

## Project Status: Real-Weight Probing
We have successfully transitioned from mock architectures to real-weight activation extraction using `meta-llama/Llama-2-7b-hf`.

## Key Findings (Teacher vs. Student)
Using a contrastive activation analysis at **Layer 15 (MLP gate_proj)**, we identified the top neurons that distinguish between the "Teacher" and "Student" personas.

### Discriminative Neurons (Top 5 Indices)
- **Target Layer:** Layer 15 (Intermediate size: 11008)
- **Identified Indices:** `[8372, 725, 6718, 3575, 4536]`
- **Activation Delta:** ~0.0018 (Float16 precision)

## Environment & Setup
- **Cluster:** SDSC Expanse
- **Model:** Llama-2-7B-hf (Accessed via Hugging Face gated repo)
- **Hardware:** NVIDIA GPU (min. 16GB VRAM for fp16 inference)

## Next Steps
- **Cross-Validation:** Test if these indices remain stable across similar roles (e.g., Professor vs. Pupil).
- **Ablation Studies:** Zero-out these neurons to observe the causal effect on model output.