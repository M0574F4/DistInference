# train.py

import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
import evaluate
import wandb  # Import wandb
from datetime import datetime  # For timestamp
import random
import numpy as np
import shutil  # For removing folders if needed

from get_fold_dataloaders import get_dataloaders_from_config
from models import My_Model
from general_utils import load_config







def set_seed(seed):
    """
    Set the seed for all relevant libraries to ensure reproducibility.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Distributed Inference Training Script")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file.')
    # Add more arguments here to override specific config parameters if needed
    args = parser.parse_args()
    return args


def compute_metrics(eval_pred, num_classes):
    """
    Compute evaluation metrics.

    Args:
        eval_pred (tuple): Tuple containing logits and labels.
        num_classes (int): Number of classes.

    Returns:
        dict: Dictionary of computed metrics.
    """
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()

    # Load the accuracy metric using evaluate
    accuracy = evaluate.load("accuracy")
    acc = accuracy.compute(predictions=predictions, references=labels)

    # Return all desired metrics
    return {
        "accuracy": acc["accuracy"],
    }


def main():
    # Set deterministic flags before model initialization
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Parse arguments and load configuration
    args = parse_args()
    config = load_config(args.config)

    # Set the seed for reproducibility
    seed = config.get('training', {}).get('seed', 42)  # Default to 42 if not specified
    set_seed(seed)

    # Initialize wandb
    # Generate a timestamp for the run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}"

    # Log in to wandb using the API key
    with open('key.txt', 'r') as file:
        wandb_key = file.read().strip()
    wandb.login(key=wandb_key)

    # Initialize the wandb run
    wandb.init(
        project='activity_recognition',
        name=run_name,
        config=config,
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
    config_path = 'config.yaml'
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, fold_index = get_dataloaders_from_config(config_path)

    # Calculate the number of training steps
    batch_size = config['batch_sizes']['train']
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    total_train_steps = int(config['training']['max_steps'])//torch.cuda.device_count()
    
    # Calculate the number of epochs based on batch size
    dataset_size = len(train_dataset)
    steps_per_epoch = dataset_size // (batch_size * gradient_accumulation_steps)
    num_epochs = total_train_steps // steps_per_epoch + 1  # Add 1 to ensure enough steps
    
    print(f"Calculated number of epochs: {num_epochs} to reach {total_train_steps} steps with batch size {batch_size}")
    
    # Define training arguments with checkpoint saving disabled by default
    training_args = TrainingArguments(
        output_dir='./trained_model',  # Directory for checkpoints
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=config['batch_sizes']['validation'],
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
        logging_steps=config['training']['logging_steps'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        eval_steps=config['training']['eval_steps'],
        save_strategy='no',  # Disable automatic checkpoint saving; we'll use our callback
        load_best_model_at_end=False,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        fp16=True,  # Enable mixed precision if using GPU
        report_to=['wandb'],
        run_name=run_name,
        seed=seed,
        max_steps=total_train_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        
        # **Cosine Annealing Scheduler Parameters**
        lr_scheduler_type='cosine',
        warmup_steps=config['training'].get('warmup_steps', 0),
        # num_cycles=0.5  # (If needed)
    )

    # Define paths for checkpoints and final model
    checkpoint_path = Path('./trained_model')  # Directory for checkpoints
    checkpoint_path.mkdir(parents=True, exist_ok=True)  # Ensure it exists

    final_save_path = Path('./trained_model') / run_name  # Directory for the final model
    final_save_path.mkdir(parents=True, exist_ok=True)  # Ensure it exists

    # Define a data collator to handle noise addition during training
    def data_collator(batch):
        inputs, labels = zip(*batch)
        inputs = torch.stack(inputs)  # Shape: (batch, 1, W, H)
        labels = torch.tensor(labels, dtype=torch.long)        
        return {'input_ids': inputs, 'labels': labels}

    # Initialize the Trainer with the custom checkpoint callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, config['model']['num_classes']),
        data_collator=data_collator,
    )

    # -----------------------
    #  Training and Evaluation
    # -----------------------
    trainer.train()

    # Evaluate on validation set
    eval_results = trainer.evaluate()
    print(f"Validation Results: {eval_results}")

    # Log validation metrics to wandb
    wandb.log({"validation_" + k: v for k, v in eval_results.items()})

    # -----------------------
    #  Final Model Saving
    # -----------------------
    trainer.save_model(final_save_path)
    print(f"Model saved to {final_save_path}")

    # Test the model
    test_results = trainer.predict(test_dataset)
    print(f"Test Results: {test_results.metrics}")

    # Log test metrics to wandb
    wandb.log({"test_" + k: v for k, v in test_results.metrics.items()})

    # Optionally remove the 'results' folder if it exists (if it's no longer needed)
    results_path = Path('./results')
    if results_path.exists() and results_path.is_dir():
        shutil.rmtree(results_path)
        print(f"Removed the 'results' directory at {results_path}")

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
