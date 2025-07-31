#!/usr/bin/env python3
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import random
import shutil
import time
from datetime import datetime

import numpy as np
import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)

# Import from local trl submodule
import sys
sys.path.insert(0, '/home/ms/trl-rloo/trl')
from trl import RLOOConfig, RLOOTrainer
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from config import TrainingConfig


def set_reproducibility_seeds(seed: int):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # HuggingFace transformers seed
    
    # Additional reproducibility settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_filtered_govreport_dataset(tokenizer, config: TrainingConfig):
    """Prepare filtered GovReport dataset with <10K input tokens for reproducible training"""
    print("Loading GovReport dataset...")
    
    # Load dataset
    dataset = load_dataset("tau/scrolls", "gov_report", trust_remote_code=True)
    
    # Use train split and filter sequentially for efficiency
    train_data = dataset["train"]
    print(f"Original train dataset size: {len(train_data)}")
    
    # Efficiently load only needed samples for memory testing
    target_samples = 50  # Small number for memory usage testing
    print(f"Loading only {target_samples} samples, truncating to max {config.max_input_tokens} input tokens if needed...")
    
    # Take only the first target_samples from the dataset
    limited_data = train_data.select(range(target_samples))
    processed_samples = []
    
    for i, sample in enumerate(limited_data):
        if i % 10 == 0:
            print(f"Processing sample {i}/{target_samples}...")
            
        # Get input text and truncate if needed
        input_text = sample["input"]
        input_tokens = tokenizer.tokenize(input_text)
        
        # Truncate if exceeds max tokens
        if len(input_tokens) > config.max_input_tokens:
            truncated_tokens = input_tokens[:config.max_input_tokens]
            input_text = tokenizer.convert_tokens_to_string(truncated_tokens)
            print(f"Sample {i}: Truncated from {len(input_tokens)} to {len(truncated_tokens)} tokens")
        
        # Create processed sample
        processed_sample = {
            "input": input_text,
            "output": sample["output"]
        }
        processed_samples.append(processed_sample)
            
    print(f"Processed {len(processed_samples)} samples with max {config.max_input_tokens} tokens")
    
    # Convert back to dataset format
    from datasets import Dataset
    filtered_dataset = Dataset.from_list(processed_samples)
    
    # Split filtered dataset deterministically
    dataset_size = len(filtered_dataset)
    train_size = int(dataset_size * config.dataset_split_ratio)
    
    # Set seed for deterministic splitting
    filtered_dataset = filtered_dataset.shuffle(seed=config.seed)
    
    train_dataset = filtered_dataset.select(range(train_size))
    eval_dataset = filtered_dataset.select(range(train_size, min(dataset_size, train_size + 50)))  # Limit eval samples
    
    print(f"Final split: {len(train_dataset)} train, {len(eval_dataset)} eval samples")
    
    def prepare_dataset(dataset, tokenizer):
        """Tokenize dataset for RLOO training with document summarization task"""
        
        def tokenize(element):
            # Use processed input text
            input_text = element["input"]
            
            # Create chat format for government report summarization task
            messages = [{"role": "user", "content": f"Summarize the following government report:\n\n{input_text}"}]
            
            input_ids = tokenizer.apply_chat_template(
                messages,
                padding=False,
                add_generation_prompt=True,
            )
            
            return {"input_ids": input_ids, "lengths": len(input_ids)}
        
        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=4,
        )
    
    # Tokenize datasets with deterministic processing
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)
        
        # Additional length filtering just in case
        train_dataset = train_dataset.filter(
            lambda x: x["lengths"] <= config.max_seq_length, 
            num_proc=4
        )
        eval_dataset = eval_dataset.filter(
            lambda x: x["lengths"] <= config.max_seq_length, 
            num_proc=4
        )
        
    print(f"After tokenization filtering: {len(train_dataset)} train, {len(eval_dataset)} eval samples")
    
    return train_dataset, eval_dataset


def main():
    parser = argparse.ArgumentParser(description="RLOO training with reproducible dataset filtering")
    parser.add_argument("--rloo_k", type=int, required=True, help="RLOO k parameter")
    parser.add_argument("--batch_size", type=int, required=True, help="Per device batch size")
    
    args = parser.parse_args()
    
    # Load configuration
    config = TrainingConfig()
    
    # Set reproducibility seeds first
    set_reproducibility_seeds(config.seed)
    
    # Setup directories based on parameters
    output_dir = config.get_output_dir(args.rloo_k)
    log_dir = config.get_log_dir(args.rloo_k)
    
    # Clean output directory but preserve logs directory for memory monitor
    if os.path.exists(output_dir):
        # Only remove model files, not logs
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path) and item != "logs":
                shutil.rmtree(item_path)
            elif os.path.isfile(item_path):
                os.remove(item_path)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 50)
    print("RLOO Training with GovReport Dataset")
    print("=" * 50)
    print(f"Model: {config.model_name}")
    print(f"RLOO k: {args.rloo_k}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max steps: {config.max_steps}")
    print(f"Max input tokens: {config.max_input_tokens}")
    print(f"Response length: {config.response_length}")
    print(f"Seed: {config.seed}")
    print(f"Output directory: {output_dir}")
    print(f"Log directory: {log_dir}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 50)
    
    # Setup tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="left")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    
    # Setup models
    print("Loading models...")
    policy_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    ref_policy_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    reward_model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=1)
    
    # Prepare filtered dataset
    print("Preparing filtered dataset...")
    train_dataset, eval_dataset = prepare_filtered_govreport_dataset(tokenizer, config)
    
    # Setup training configuration
    training_config = RLOOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        report_to="none",
        remove_unused_columns=False,
        rloo_k=args.rloo_k,
        num_ppo_epochs=config.num_ppo_epochs,
        num_mini_batches=config.num_mini_batches,
        response_length=config.response_length,
        seed=config.seed,
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = RLOOTrainer(
        config=training_config,
        processing_class=tokenizer,
        policy=policy_model,
        ref_policy=ref_policy_model, 
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Start training
    print("Starting training...")
    start_time = time.time()
    
    try:
        trainer.train()
        training_successful = True
    except Exception as e:
        print(f"Training failed with error: {e}")
        training_successful = False
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Save model if training was successful
    if training_successful:
        print("Saving model...")
        trainer.save_model(output_dir)
        print(f"Model saved to: {output_dir}")
    
    # Save training summary
    training_summary = {
        "config": config.to_dict(),
        "parameters": {
            "rloo_k": args.rloo_k,
            "batch_size": args.batch_size,
        },
        "training_results": {
            "successful": training_successful,
            "duration_seconds": training_time,
            "final_step": trainer.state.global_step if training_successful else 0,
        },
        "system_info": {
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
        },
        "directories": {
            "output_dir": output_dir,
            "log_dir": log_dir
        }
    }
    
    os.makedirs(log_dir, exist_ok=True)
    summary_file = os.path.join(log_dir, "training_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(training_summary, f, indent=2)
        
    print(f"Training summary saved to: {summary_file}")
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Success: {training_successful}")


if __name__ == "__main__":
    main()