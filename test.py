from transformers import pipeline
import torch

# 1. check GPU
print(f"CUDA: {torch.cuda.is_available()}")
print(f"current GPU count: {torch.cuda.device_count()}")

# 2. samll model test
# device_map="auto" 
pipe = pipeline(
    "text-generation", 
    model="gpt2", 
    device_map="auto"
)

# 3. test run
result = pipe("Hello, I am a language model, and today I will", max_length=30)
print("\n--- model generating result ---")
print(result[0]['generated_text'])