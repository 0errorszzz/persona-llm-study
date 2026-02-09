import torch
from transformers import LlamaConfig, LlamaForCausalLM

def main():
    # 1. Initialize configuration for Llama-2-7b structure
    # This does not require internet access or gated permissions
    print("Initializing architecture for Llama-2-7b...", flush=True)
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008, # This is the SwiGLU expansion dimension
        num_hidden_layers=32,
        num_attention_heads=32
    )
    
    # 2. Instantiate the model with random weights to save time and skip auth
    # We use float16 to mimic the real experimental environment
    model = LlamaForCausalLM(config).to(torch.float16).to("cuda")

    # 3. Setup a dictionary to store our captured tensors
    captured_data = {}

    # 4. Define the Hook function to grab the MLP intermediate output
    def hook_fn(module, input, output):
        # We store the output of the 'gate_proj' or 'up_proj'
        captured_data['mlp_intermediate'] = output.detach()
    
    # Register the hook on Layer 15's MLP gate_proj
    # In Llama, the MLP expansion happens here
    model.model.layers[15].mlp.gate_proj.register_forward_hook(hook_fn)

    # 5. Create a dummy input (Batch size=1, Sequence length=8)
    # This simulates a short sentence like "The teacher is in the classroom"
    dummy_input = torch.randint(0, 32000, (1, 8)).to("cuda")

    print("Executing forward pass (Inference)...", flush=True)
    with torch.no_grad():
        model(dummy_input)

    # 6. Analysis of the captured high-dimensional data
    if 'mlp_intermediate' in captured_data:
        tensor = captured_data['mlp_intermediate'] # Shape: [1, 8, 11008]
        
        # Focus on the last token's activation
        last_token_act = tensor[0, -1, :] 
        
        # Get Top 5 strongest neurons in this 11008-dim space
        values, indices = torch.topk(last_token_act, 5)

        print("\n" + "="*50)
        print(f"Data captured from Layer 15 MLP")
        print(f"Expansion Dimension: {tensor.shape[-1]} (from 4096)")
        print("-" * 50)
        print("Top 5 Active Neurons (Indices):", indices.tolist())
        print("Top 5 Active Values (Mocked):", [f"{v:.4f}" for v in values.tolist()])
        print("="*50)
        print("\nNote: These values are random now. Once access is granted,")
        print("these indices will represent specific semantic concepts.")
    else:
        print("Error: Hook failed to capture data.")

if __name__ == "__main__":
    main()