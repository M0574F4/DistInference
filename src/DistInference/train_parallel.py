# train_parallel.py

import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
import evaluate
from datetime import datetime
import random
import numpy as np
import wandb  # Import wandb
import os  # For environment variables and rank detection

from get_fold_dataloaders import get_dataloaders_from_config
from models import My_Model
from general_utils import load_config

def set_seed(seed):
    """
    Set the seed for all relevant libraries to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Inference Training Script")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file.')
    # Add more arguments here to override specific config parameters if needed
    args = parser.parse_args()
    return args

def compute_metrics(eval_pred, num_classes):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()

    # Load the accuracy metric using evaluate
    accuracy = evaluate.load("accuracy")
    acc = accuracy.compute(predictions=predictions, references=labels)

    # Optionally, load and compute additional metrics
    # f1 = evaluate.load("f1")
    # precision = evaluate.load("precision")
    # recall = evaluate.load("recall")

    # Example of computing additional metrics
    # f1_score = f1.compute(predictions=predictions, references=labels, average='weighted')
    # precision_score = precision.compute(predictions=predictions, references=labels, average='weighted')
    # recall_score = recall.compute(predictions=predictions, references=labels, average='weighted')

    # Return all desired metrics
    return {
        "accuracy": acc["accuracy"],
        # "f1": f1_score["f1"],
        # "precision": precision_score["precision"],
        # "recall": recall_score["recall"],
    }

def main():
    # Set these flags before model initialization
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args = parse_args()
    config = load_config(args.config)

    # **Set the seed at the very beginning**
    seed = config.get('training', {}).get('seed', 42)  # Default to 42 if not specified
    set_seed(seed)

    # Determine the process rank
    rank = int(os.getenv('RANK', '0'))  # Default to 0 if RANK is not set

    # Initialize wandb only on the main process (rank 0)
    if rank == 0:
        # Option 1: Read the API key from 'key.txt'
        try:
            with open('key.txt', 'r') as file:
                wandb_key = file.read().strip()
            wandb.login(key=wandb_key)
        except FileNotFoundError:
            print("key.txt not found. Ensure that 'key.txt' contains your wandb API key.")
            raise

        # Initialize the wandb run
        wandb.init(
            project='activity_recognition',  # Replace with your project name
            config=config,
            name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            reinit=True  # Allows multiple runs in the same script
        )

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")

    # Instantiate the combined model
    model = My_Model(config)

    # DataLoaders
    config_path = args.config  # Use the config path provided via arguments
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, fold_index = get_dataloaders_from_config(config_path)

    # Define the total number of training steps
    total_train_steps = int(config['training']['max_steps'])

    # Retrieve batch size and gradient accumulation steps
    batch_size = config['batch_sizes']['train']
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)

    print(f"Setting max_steps to {total_train_steps} to keep training steps consistent across runs.")

    # Define training arguments with fixed max_steps
    training_args = TrainingArguments(
        output_dir='./results',
        # Removed num_train_epochs to allow max_steps to control training
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=config['batch_sizes']['validation'],
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_strategy=config['training'].get('evaluation_strategy', 'steps'),  # Updated key
        eval_steps=config['training']['eval_steps'],
        save_total_limit=config['training']['save_total_limit'],
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        fp16=True,  # Enable mixed precision if using GPU
        report_to=['wandb'] if rank == 0 else [],  # Enable wandb reporting only on rank 0
        run_name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        seed=seed,  # Use the same seed
        max_steps=total_train_steps,  # Set the total training steps
        gradient_accumulation_steps=gradient_accumulation_steps,
        ddp_find_unused_parameters=False,  # Optimize DDP training
        lr_scheduler_type='cosine',  # Set scheduler to cosine annealing
        warmup_steps=config['training'].get('warmup_steps', 0),  # Optional: Set warmup steps
        # Optionally, set other parameters as needed
    )

    # Define a data collator to handle noise addition during training
    def data_collator(batch):
        inputs, labels = zip(*batch)
        inputs = torch.stack(inputs)  # Shape: (batch, 1, freq_bins, time_bins)
        labels = torch.tensor(labels, dtype=torch.long)
        return {'input_ids': inputs, 'labels': labels}  # Changed 'inputs' to 'input_ids'

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, config['model']['num_classes']),
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Evaluate on validation set
    eval_results = trainer.evaluate()
    print(f"Validation Results: {eval_results}")

    # Log metrics to WandB
    if rank == 0:
        wandb.log({"validation_" + k: v for k, v in eval_results.items()})

    # Save the models
    trainer.save_model("./trained_model")

    # Test the model
    test_results = trainer.predict(test_dataset)
    print(f"Test Results: {test_results.metrics}")

    # Log test metrics to WandB
    if rank == 0:
        wandb.log({"test_" + k: v for k, v in test_results.metrics.items()})
        # Finish the WandB run
        wandb.finish()

if __name__ == "__main__":
    main()
