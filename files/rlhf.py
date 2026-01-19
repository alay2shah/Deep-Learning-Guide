import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel
from trl import PPOTrainer, PPOConfig, DPOTrainer, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import numpy as np

# Load the SFT model (from qlora.py)
base_model_path = "./qlora_finetuned_model/final"
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(base_model_path)

# Mock dataset for demonstration
def create_mock_feedback_dataset():
    prompts = [
        "Write a helpful email to a colleague about a project deadline.",
        "Explain quantum computing to a high school student.",
        "Write a creative story about space exploration."
    ]
    
    chosen_responses = [
        "Thank you for your work on the project. I noticed we're approaching our deadline...",
        "Quantum computing is like having many parallel universes helping you solve a problem...",
        "The starship's engines hummed softly as Captain Chen reviewed the mission parameters..."
    ]
    
    rejected_responses = [
        "You're late with the project again. This is unacceptable...",
        "Quantum computing is extremely complex and you probably won't understand...",
        "Space is boring and empty, just like this story..."
    ]
    
    return {
        "prompt": prompts * 10,  # Repeat for more samples
        "chosen": chosen_responses * 10,
        "rejected": rejected_responses * 10
    }

# Create mock dataset
dataset = create_mock_feedback_dataset()

"""
PPO (Proximal Policy Optimization) Approach
-----------------------------------------
Use Case: When you have a reward model or can compute rewards directly.
Advantages:
- More flexible reward engineering
- Can optimize for multiple objectives
- Better for complex, multi-step tasks
Disadvantages:
- More complex to implement and tune
- Requires careful reward design
- Can be unstable during training
"""

def ppo_training_loop():
    # Initialize PPO config
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=8,
        gradient_accumulation_steps=4,
        optimize_cuda_cache=True,
        early_stopping=True,
        target_kl=0.1,  # KL divergence target for early stopping
        ppo_epochs=4,   # Number of PPO epochs per batch
        seed=42
    )

    # Create model with value head for PPO
    model_ppo = AutoModelForCausalLMWithValueHead.from_pretrained(base_model_path)
    
    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model_ppo,
        tokenizer=tokenizer,
        dataset=dataset["prompt"]  # Using only prompts for PPO
    )

    # Text generation settings
    generation_kwargs = {
        "min_length": 32,
        "max_length": 128,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95
    }
    
    # Mock reward function (replace with actual reward model in production)
    def mock_reward_model(texts):
        # Simplified reward based on text length and keyword presence
        rewards = []
        positive_keywords = ["thank", "helpful", "great", "interesting"]
        negative_keywords = ["bad", "terrible", "boring", "hate"]
        
        for text in texts:
            score = 0.5  # Base score
            text_lower = text.lower()
            
            # Adjust score based on keywords
            for keyword in positive_keywords:
                if keyword in text_lower:
                    score += 0.1
            for keyword in negative_keywords:
                if keyword in text_lower:
                    score -= 0.1
                    
            rewards.append(min(max(score, 0), 1))  # Clamp between 0 and 1
        return torch.tensor(rewards)

    # Training loop
    for epoch in range(5):  # Number of training epochs
        for batch_idx in range(0, len(dataset["prompt"]), ppo_config.batch_size):
            # Get batch of prompts
            prompt_batch = dataset["prompt"][batch_idx:batch_idx + ppo_config.batch_size]
            
            # Generate responses
            responses = []
            for prompt in prompt_batch:
                response = ppo_trainer.generate(
                    prompt,
                    **generation_kwargs
                )
                responses.extend(tokenizer.batch_decode(response, skip_special_tokens=True))
            
            # Calculate rewards
            rewards = mock_reward_model(responses)
            
            # Run PPO step
            stats = ppo_trainer.step(prompt_batch, responses, rewards)
            
            print(f"Epoch {epoch}, Batch {batch_idx}, Mean reward: {torch.mean(rewards):.3f}")

"""
DPO (Direct Preference Optimization) Approach
-------------------------------------------
Use Case: When you have pairs of preferred/non-preferred responses.
Advantages:
- Simpler to implement
- More stable training
- Doesn't require explicit reward modeling
Disadvantages:
- Less flexible than PPO
- Cannot optimize for complex/multi-objective rewards
- Requires high-quality preference pairs
"""

def dpo_training_loop():
    from trl import DPOTrainer
    
    # Initialize DPO trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Using implicit reference model (same as original model)
        tokenizer=tokenizer,
        beta=0.1,        # Temperature parameter for DPO loss
        max_length=512,
        max_prompt_length=128
    )

    # Prepare data for DPO (needs prompt, chosen, and rejected responses)
    train_dataset = {
        "prompt": dataset["prompt"],
        "chosen": dataset["chosen"],
        "rejected": dataset["rejected"]
    }

    # Training arguments
    training_args = {
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-5,
        "max_steps": 1000,
        "save_steps": 100,
        "output_dir": "./dpo_trained_model"
    }

    # Train the model with DPO
    dpo_trainer.train(
        train_dataset=train_dataset,
        **training_args
    )

    # Save the DPO-trained model
    dpo_trainer.save_model("./dpo_trained_model/final")

if __name__ == "__main__":
    # Choose which training loop to run
    use_ppo = True  # Set to False to use DPO instead
    
    if use_ppo:
        print("Starting PPO training loop...")
        ppo_training_loop()
    else:
        print("Starting DPO training loop...")
        dpo_training_loop()
