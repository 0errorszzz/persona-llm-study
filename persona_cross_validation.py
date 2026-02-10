import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

def main():
    # 1. Configuration
    model_id = "meta-llama/Llama-2-7b-hf"
    # These are the "suspects" we found in the Teacher/Student experiment
    candidate_indices = [8372, 725, 6718, 3575, 4536]
    
    # 2. Loading Model & Tokenizer
    print(f"Loading model: {model_id}...")
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )

    # 3. Define Contrastive Pairs (High status/Senior vs Low status/Junior)
    test_pairs = [
        ("professional teacher", "young student"),
        ("senior doctor", "junior nurse"),
        ("experienced manager", "new intern")
    ]

    # 4. Hook Setup
    activations = {}
    def hook_fn(module, input, output):
        # We capture the last token activation at Layer 15 MLP
        activations['current'] = output.detach()

    # Target: Layer 15 gate_proj
    model.model.layers[15].mlp.gate_proj.register_forward_hook(hook_fn)

    print("\n" + "="*80)
    print(f"{'Persona Comparison Group':<45} | {'Top Neuron':<12} | {'#8372 Rank'}")
    print("-" * 80)

    # 5. Cross-Validation Loop
    for senior, junior in test_pairs:
        pair_activations = []
        for role in [senior, junior]:
            prompt = f"The identity of this person is a {role}."
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                model(**inputs)
            
            # Extract last token activation (index -1)
            pair_activations.append(activations['current'][0, -1, :])
        
        # Calculate Absolute Difference (Delta)
        diff = torch.abs(pair_activations[0] - pair_activations[1])
        
        # Finding the strongest neuron for this specific pair
        top_neuron_this_pair = torch.argmax(diff).item()
        
        # Calculate the rank of our primary suspect (#8372)
        # Rank 0 means it's the most different, Rank 11007 means it's the least
        rank_8372 = (diff > diff[8372]).sum().item()
        
        print(f"{senior + ' vs ' + junior:<45} | {top_neuron_this_pair:<12} | Rank {rank_8372}")

    print("="*80)
    print("Note: If Rank is low (e.g., < 50), the neuron is likely a general 'Persona Switch'.")

if __name__ == "__main__":
    main()