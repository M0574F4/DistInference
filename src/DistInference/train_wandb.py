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
    # labels = labels.numpy()

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
    total_train_steps = int(config['training']['max_steps'])
    
    # Calculate the number of epochs based on batch size
    dataset_size = len(train_dataset)
    steps_per_epoch = dataset_size // (batch_size * gradient_accumulation_steps)
    num_epochs = total_train_steps // steps_per_epoch + 1  # Add 1 to ensure enough steps
    
    print(f"Calculated number of epochs: {num_epochs} to reach {total_train_steps} steps with batch size {batch_size}")
    
    # Define training arguments with checkpoint saving disabled
    training_args = TrainingArguments(
        output_dir='./trained_model',  # Changed from './results'
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=config['batch_sizes']['validation'],
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
        logging_steps=config['training']['logging_steps'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        eval_steps=config['training']['eval_steps'],
        # Removed checkpoint saving parameters
        save_strategy='no',  # Disable checkpoint saving
        load_best_model_at_end=False,  # Disabled because no checkpoints
        metric_for_best_model='accuracy',  # Optional: can be removed
        greater_is_better=True,
        fp16=True,  # Enable mixed precision if using GPU
        report_to=['wandb'],  # Enable wandb reporting
        run_name=run_name,  # Set the run name in TrainingArguments
        seed=seed,  # Use the same seed
        max_steps=total_train_steps,  # Set the total training steps
        gradient_accumulation_steps=gradient_accumulation_steps,
        
        # **Cosine Annealing Scheduler Parameters**
        lr_scheduler_type='cosine',  # Set scheduler to cosine annealing
        warmup_steps=config['training'].get('warmup_steps', 0),  # Optional: Set warmup steps
        # You can also set num_cycles if you want multiple cosine cycles
        # num_cycles=0.5,  # Default is 0.5
    )

    # Define a data collator to handle noise addition during training
    def data_collator(batch):
        inputs, labels = zip(*batch)
        inputs = torch.stack(inputs)  # Shape: (batch, 1, W, H)
        labels = torch.tensor(labels, dtype=torch.long)        
        return {'input_ids': inputs, 'labels': labels}

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

    # Log validation metrics to wandb
    wandb.log({"validation_" + k: v for k, v in eval_results.items()})

    # Save the models with timestamp
    save_path = Path('./trained_model') / run_name  # Use timestamp in the folder name
    save_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(save_path)
    print(f"Model saved to {save_path}")

    # Test the model
    test_results = trainer.predict(test_dataset)
    print(f"Test Results: {test_results.metrics}")

    # Log test metrics to wandb
    wandb.log({"test_" + k: v for k, v in test_results.metrics.items()})

    # Optionally remove the 'results' folder if it exists
    results_path = Path('./results')
    if results_path.exists() and results_path.is_dir():
        shutil.rmtree(results_path)
        print(f"Removed the 'results' directory at {results_path}")

    # Finish the wandb run
    wandb.finish()


    # Define a data collator to handle noise addition during training
    def data_collator(batch):
        inputs, labels = zip(*batch)
        inputs = torch.stack(inputs)  # Shape: (batch, 1, W, H)
        labels = torch.tensor(labels, dtype=torch.long)        
        return {'input_ids': inputs, 'labels': labels}

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

    # Log validation metrics to wandb
    wandb.log({"validation_" + k: v for k, v in eval_results.items()})

    # Save the models
    # Save the entire model
    trainer.save_model("./trained_model")

    # Test the model
    test_results = trainer.predict(test_dataset)
    print(f"Test Results: {test_results.metrics}")

    # Log test metrics to wandb
    wandb.log({"test_" + k: v for k, v in test_results.metrics.items()})

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
