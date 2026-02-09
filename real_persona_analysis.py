import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

def main():
    # 1. Use the real Llama-2-7b ID
    model_id = "meta-llama/Llama-2-7b-hf"
    print(f"Connecting to Hugging Face to load {model_id}...", flush=True)
    
    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    
    # 2. Load the model weights
    # We use torch.float16 to fit the ~13GB model into GPU memory
    # device_map="auto" will automatically handle the GPU placement on Expanse
    print("Loading weights (this might take a while if downloading)...", flush=True)
    model = LlamaForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )

    # 3. Setup the sensor (Hook) at Layer 15 MLP
    activations = []
    def hook_fn(module, input, output):
        # Capturing the expansion layer output (11008 dimensions)
        activations.append(output.detach())
    
    model.model.layers[15].mlp.gate_proj.register_forward_hook(hook_fn)

    # 4. Define our persona-specific prompts
    # We want to see how the model differentiates these two roles
    prompt_teacher = "The identity of this person is a professional teacher."
    prompt_student = "The identity of this person is a young student."

    # 5. Run inference
    for p in [prompt_teacher, prompt_student]:
        inputs = tokenizer(p, return_tensors="pt").to("cuda")
        print(f"Analyzing prompt: {p}")
        with torch.no_grad():
            model(**inputs)

    # 6. Extract and compare the 'Identity Neurons'
    # Slicing: [Batch 0, Last Token -1, All 11008 neurons]
    act_t = activations[0][0, -1, :]
    act_s = activations[1][0, -1, :]

    # Calculate absolute difference to find the most sensitive neurons
    diff = torch.abs(act_t - act_s)
    values, indices = torch.topk(diff, 5)

    print("\n" + "="*60)
    print("REAL NEURON DISCOVERY RESULTS")
    print("="*60)
    print(f"Top 5 Discriminative Neuron Indices: \n{indices.tolist()}")
    print(f"Difference Magnitudes: \n{[f'{v:.4f}' for v in values.tolist()]}")
    print("="*60)
    print("These indices are the specific 'switches' Llama uses to distinguish roles.")

if __name__ == "__main__":
    main()