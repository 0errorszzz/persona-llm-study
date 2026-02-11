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
## Neuron Interpretation Results (Layer 15)
-Through cross-validation across multiple persona pairs, we have identified two distinct "Identity Switches":

### 1. The "Academic Authority" Neuron: Index #8372
- **Characteristics**: Extremely sensitive to educational roles.
- **Performance**: Rank 0 (Top 1) for *Teacher vs. Student*, Rank 5 for *Doctor vs. Nurse*.
- **Interpretation**: Encodes seniority based on knowledge transmission and mentorship.

### 2. The "Professional Hierarchy" Neuron: Index #8758
- **Characteristics**: Dominates corporate and institutional structures.
- **Performance**: Rank 0 (Top 1) for both *Doctor vs. Nurse* and *Manager vs. Intern*.
- **Interpretation**: Encodes power dynamics and administrative seniority within a professional organization.

## Methodology: How we found them
1. **Extraction**: Used `gate_proj` hooks at Layer 15 of Llama-2-7B to capture 11,008-dimensional activations.
2. **Contrastive Analysis**: Calculated the absolute difference ($\Delta$) between high-status and low-status persona prompts.
3. **Cross-Validation**: Filtered neurons by testing across 3 different domains (Education, Medicine, Corporate) to eliminate word-specific noise.
### Gender Neuron Experiment Results
| Gender Pair | Top 5 Neurons (Indices) |
| :--- | :--- |
| a man vs a woman | [6953, 8595, 8140, 10437, 8195] |
| a male doctor vs a female doctor | [6953, 10878, 2826, 1095, 9542] |
| a son vs a daughter | [6953, 10283, 6063, 8595, 2400] |
## The stability of Neuron #6953 across varied prompts confirms that the model has a specialized, modularized circuit at Layer 15 for processing gender identity.
