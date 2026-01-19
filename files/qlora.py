# QLoRA (Quantized LoRA) Implementation
# QLoRA enables fine-tuning of larger models with significantly reduced VRAM usage through:
# 1. 4-bit quantization of the base model
# 2. Storing gradients in 16-bit for stability
# This allows for larger batch sizes and sequence lengths during training,
# enabling better context understanding and more efficient training on consumer GPUs.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from trl import SFTTrainer

# Load base model and tokenizer
model_id = "facebook/opt-350m"  # Example base model
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

## QLoRA specific: Configure 4-bit quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",      # Normal Float 4 for better precision
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True  # Nested quantization for additional memory savings
)

## QLoRA specific: Load model in 4-bit precision instead of 16-bit
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quant_config
)

## QLoRA specific: Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Define LoRA configuration
lora_config = LoraConfig(
    r=16,  # Rank dimension -- lower would be faster but underfit, higher would be slower and more expensive
    lora_alpha=32,  # Alpha parameter for LoRA scaling
    lora_dropout=0.05,  # Dropout probability for LoRA layers
    bias="none", # bias params not adjusted 
    task_type="CAUSAL_LM",  # Task type for causal language modeling
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "down_proj", "gate_proj"],  # Target attention modules for adaptation
    # q_proj, v_proj, k_proj, o_proj: Attention mechanism modules
    # down_proj, gate_proj: Feed-forward network modules for more comprehensive model adaptation
)

# Apply LoRA adapters to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Print trainable vs total parameters

# Load a sample dataset (replace with your actual dataset)
dataset = load_dataset("imdb", split="train[:1000]")  # Mock dataset
# Split dataset into train and validation
train_val_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_val_split["train"]
val_dataset = train_val_split["test"]

## QLoRA specific: Adjusted training arguments for larger batches while keeping informative comments
training_args = TrainingArguments(
    output_dir="./qlora_finetuned_model",  # Directory where model checkpoints will be saved
    per_device_train_batch_size=8,  # Larger batch size possible with QLoRA's memory efficiency
    gradient_accumulation_steps=4,  # Effectively creates a batch size of 32 (8*4) to stabilize training
    warmup_steps=100,  # Gradually increases learning rate for first 100 steps to prevent early divergence
    max_steps=1000,  # Total training steps; alternative to num_epochs when dataset size is unknown
    learning_rate=2e-4,  # Relatively high LR works well with LoRA since we're updating fewer parameters
    fp16=True,  # Mixed precision training to reduce memory usage and speed up training 
    #Gradients are computed in 32-bit for stability, while forward passes use 16-bit for efficiency.
    logging_steps=10,  # Log metrics every 10 steps to monitor training progress
    save_steps=200,  # Save checkpoint every 200 steps to enable resuming if interrupted
    save_total_limit=3,  # Keep only the 3 most recent checkpoints to save disk space
    load_best_model_at_end=True,     # Load the best checkpoint at the end of training
    metric_for_best_model="loss",     # Monitor loss to determine the best model
    evaluation_strategy="steps",      # Evaluate during training at specified intervals
    eval_steps=100,                   # Evaluate every 100 steps
)

# Set up the trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=4096,  # Choose based on: 1) Model's max context (e.g. 128k for Llama-2), 2) GPU memory constraints,
                        # 3) Typical length of your training examples. For large models, often start with 2-4k tokens
                        # and increase if needed and if memory allows. Monitor attention patterns to optimize.
                        # QLoRA allows for longer sequences due to memory efficiency
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model("./qlora_finetuned_model/final")

# Merge LoRA weights with base model for faster inference
merged_model = model.merge_and_unload()

## QLoRA specific: Quantize to 8-bit for efficient deployment while maintaining quality
quantized_model = AutoModelForCausalLM.from_pretrained(
    "qlora_finetuned_model/final",
    device_map="auto",
    load_in_8bit=True,  # Load with 8-bit quantization
)

# Save the quantized model
quantized_model.save_pretrained("./qlora_finetuned_model/quantized")
tokenizer.save_pretrained("./qlora_finetuned_model/quantized")

# Example inference with the merged model
input_text = "This movie was really"
inputs = tokenizer(input_text, return_tensors="pt").to(merged_model.device)
with torch.no_grad():
    outputs = merged_model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Load and test the trained model
print("\nTesting trained model...")

# Load the quantized model and tokenizer
test_model = AutoModelForCausalLM.from_pretrained(
    "./qlora_finetuned_model/quantized",
    device_map="auto",
    load_in_8bit=True
)
test_tokenizer = AutoTokenizer.from_pretrained("./qlora_finetuned_model/quantized")

# Test examples
test_examples = [
    "This movie was really",
    "The customer service was",
    "The food at this restaurant"
]

print("\nGenerating responses for test examples:")
for example in test_examples:
    inputs = test_tokenizer(example, return_tensors="pt").to(test_model.device)
    with torch.no_grad():
        outputs = test_model.generate(
            **inputs,
            max_length=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    response = test_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: {example}")
    print(f"Response: {response}")

