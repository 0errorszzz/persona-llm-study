import torch
from transformers import LlamaConfig, LlamaForCausalLM

def main():
    # 1. Load the default configuration for Llama-2
    # This does not download weights; it only defines the architecture parameters
    print("Loading Llama model configuration...")
    config = LlamaConfig() 
    
    # 2. Initialize the model with the configuration
    # This creates a model with random weights to save time and memory
    model = LlamaForCausalLM(config)

    # 3. Print the overall model structure
    print("\n=== Model Architecture Overview ===")
    print(model)

    # 4. Inspect the specific components of the first layer (Layer 0)
    # Most layers in Llama are identical in structure
    if hasattr(model.model, 'layers'):
        first_layer = model.model.layers[0]
        
        print("\n=== Detailed Structure of Layer 0 ===")
        print(f"Self-Attention Component: \n{first_layer.self_attn}")
        print(f"\nMLP (Feed-Forward) Component: \n{first_layer.mlp}")

if __name__ == "__main__":
    main()