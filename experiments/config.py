#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for RLOO training"""
    
    # Model settings
    model_name: str = "/home/ms/trl-rloo/models/tiny-Qwen3ForCausalLM"
    
    # Training hyperparameters  
    learning_rate: float = 3e-6
    max_steps: int = -1  # Use epoch-based training instead of step-based
    eval_steps: int = 20
    save_steps: int = 50
    logging_steps: int = 1
    
    # Training settings
    gradient_accumulation_steps: int = 1
    num_ppo_epochs: int = 3
    num_mini_batches: int = 1
    response_length: int = 100  # Short summaries for CNN/DailyMail
    
    # Dataset settings
    dataset_split_ratio: float = 0.8
    max_seq_length: int = 3100  # Input (<5000) + Generated summary (100) tokens
    max_input_tokens: int = 3000  # Filter criterion: only use samples with <5K input tokens
    
    # Reproducibility
    seed: int = 42
    
    # Directories
    base_output_dir: str = "./results"
    
    # Evaluation
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    
    def get_output_dir(self, rloo_k: int) -> str:
        """Generate output directory based on model and rloo_k"""
        model_short = self.model_name.split("/")[-1]
        return f"{self.base_output_dir}/{model_short}_rloo{rloo_k}"
    
    def get_log_dir(self, rloo_k: int) -> str:
        """Generate log directory"""
        return f"{self.get_output_dir(rloo_k)}/logs"
        
    def to_dict(self):
        """Convert config to dictionary for logging"""
        return {
            "model_name": self.model_name,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_ppo_epochs": self.num_ppo_epochs,
            "num_mini_batches": self.num_mini_batches,
            "response_length": self.response_length,
            "dataset_split_ratio": self.dataset_split_ratio,
            "max_seq_length": self.max_seq_length,
            "max_input_tokens": self.max_input_tokens,
            "seed": self.seed,
        }