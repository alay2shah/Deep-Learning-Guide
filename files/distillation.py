"""
Knowledge Distillation for Large Language Models
----------------------------------------------
This script implements knowledge distillation from a 7B parameter teacher model 
to a 1B parameter student model. The teacher model has been fine-tuned using QLoRA.

Key concepts:
- Temperature scaling is used to soften probability distributions
- Student learns from both hard labels and teacher's soft predictions
- Loss combines cross-entropy on hard targets and KL divergence with soft targets

Potential drawbacks:
- Student model may still struggle with complex reasoning despite distillation
- Quality degradation is expected with 7x parameter reduction
- Training requires significant compute despite smaller student model
- Finding optimal temperature requires tuning
- Student may learn teacher's biases and mistakes
"""

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import wandb

# Configuration
class DistillationConfig:
    def __init__(self):
        self.teacher_model_path = "./qlora_finetuned_model/final"
        self.temperature = 2.0  # Softens probability distributions
        self.alpha = 0.5  # Weight for soft targets loss
        self.batch_size = 8
        self.grad_accum_steps = 4
        self.learning_rate = 1e-4
        self.num_epochs = 3
        self.warmup_steps = 100
        
config = DistillationConfig()

# Initialize models and tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_path)

# 7B teacher model (from QLoRA fine-tuning)
teacher_model = AutoModelForCausalLM.from_pretrained(
    config.teacher_model_path,
    device_map="auto",
    torch_dtype=torch.float16
)
teacher_model.eval()  # Teacher always in eval mode

# 1B student model (initialize from smaller pretrained model)
student_model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-1.3b",  # Example 1B-class model
    device_map="auto",
    torch_dtype=torch.float16
)

# Prepare dataset (using same data as original fine-tuning)
dataset = load_dataset("your_dataset")  # Replace with actual dataset
train_dataloader = DataLoader(
    dataset["train"],
    batch_size=config.batch_size,
    shuffle=True
)

# Initialize optimizer and scheduler
optimizer = torch.optim.AdamW(student_model.parameters(), lr=config.learning_rate)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=config.warmup_steps,
    num_training_steps=len(train_dataloader) * config.num_epochs
)

# Initialize wandb for tracking
wandb.init(project="llm_distillation", config=vars(config))

def compute_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """
    Compute the distillation loss combining soft and hard targets
    """
    # Soft targets loss (KL divergence)
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
    soft_targets_loss = F.kl_div(
        soft_prob,
        soft_targets,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Hard targets loss (cross-entropy)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Combined loss
    return (alpha * soft_targets_loss) + ((1 - alpha) * hard_loss)

# Training loop
for epoch in range(config.num_epochs):
    student_model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to(student_model.device)
        attention_mask = batch["attention_mask"].to(student_model.device)
        labels = batch["labels"].to(student_model.device)
        
        # Get teacher predictions
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            teacher_logits = teacher_outputs.logits
            
        # Get student predictions
        student_outputs = student_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        student_logits = student_outputs.logits
        
        # Calculate loss
        loss = compute_loss(
            student_logits,
            teacher_logits,
            labels,
            config.temperature,
            config.alpha
        )
        
        # Scale loss for gradient accumulation
        loss = loss / config.grad_accum_steps
        loss.backward()
        
        # Update weights after accumulation
        if (batch_idx + 1) % config.grad_accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        total_loss += loss.item()
        
        # Log metrics
        if batch_idx % 100 == 0:
            wandb.log({
                "loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0]
            })
            
    # End of epoch logging
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{config.num_epochs}, Average Loss: {avg_loss:.4f}")
    wandb.log({"epoch": epoch, "average_loss": avg_loss})

# Save the distilled student model
student_model.save_pretrained("./distilled_model/final")
tokenizer.save_pretrained("./distilled_model/final")
wandb.finish()
