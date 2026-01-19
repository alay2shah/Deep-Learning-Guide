import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from trl import SFTTrainer

# Load base model and tokenizer
model_id = "facebook/opt-350m"  # Example base model
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load model in 16-bit precision for efficient training
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Define LoRA configuration
lora_config = LoraConfig(
    r=16,  # Rank dimension -- lower would be faster but underfit, higher would be slower and more expensive
    lora_alpha=32,  # Alpha parameter for LoRA scaling
    lora_dropout=0.05,  # Dropout probability for LoRA layers
    bias="none", # bias params not adjusted 
    use_rslora=True,  # Use rank-stabilized LoRA for better training stability
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

# Define training arguments
training_args = TrainingArguments(
    output_dir="./lora_finetuned_model",  # Directory where model checkpoints will be saved
    per_device_train_batch_size=4,  # Small batch size to fit in GPU memory; increase for larger GPUs
    gradient_accumulation_steps=4,  # Effectively creates a batch size of 16 (4*4) to stabilize training
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
    max_seq_length=2048,  # Choose based on: 1) Model's max context (e.g. 128k for Llama-2), 2) GPU memory constraints,
                        # 3) Typical length of your training examples. For large models, often start with 2-4k tokens
                        # and increase if needed and if memory allows. Monitor attention patterns to optimize.
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model("./lora_finetuned_model/final")

# Merge LoRA weights with base model for faster inference
merged_model = model.merge_and_unload()

# Quantize the merged model to 8-bit for deployment
quantized_model = AutoModelForCausalLM.from_pretrained(
    "lora_finetuned_model/final",
    device_map="auto",
    load_in_8bit=True,  # Load with 8-bit quantization
)

# Save the quantized model
quantized_model.save_pretrained("./lora_finetuned_model/quantized")
tokenizer.save_pretrained("./lora_finetuned_model/quantized")

# Example inference with the merged model
input_text = "This movie was really"
inputs = tokenizer(input_text, return_tensors="pt").to(merged_model.device)
with torch.no_grad():
    outputs = merged_model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# Prune outlier weights to reduce model size while maintaining performance
def prune_outliers(model, threshold=1e-4):
    """Prune weights below threshold to reduce model size"""
    pruned_count = 0
    total_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Calculate mean and std of weights
            mean = param.data.mean()
            std = param.data.std()
            
            # Create mask for weights within threshold standard deviations
            mask = (param.data > mean - threshold * std) & (param.data < mean + threshold * std)
            
            # Zero out outlier weights
            param.data = param.data * mask
            
            pruned_count += (~mask).sum().item()
            total_count += param.numel()

    print(f"Pruned {pruned_count/total_count*100:.2f}% of weights")
    return model

# Prune outliers from quantized model
pruned_model = prune_outliers(quantized_model)

# Save the pruned and quantized model
pruned_model.save_pretrained("./lora_finetuned_model/pruned_quantized")
