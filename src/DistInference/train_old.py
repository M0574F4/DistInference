import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import evaluate

from get_fold_dataloaders import get_dataloaders_from_config
from models import My_Model
from general_utils import load_config



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
    args = parse_args()
    config = load_config(args.config)
    
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
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=config['training']['epochs'],
        per_device_train_batch_size=config['batch_sizes']['train'],
        per_device_eval_batch_size=config['batch_sizes']['validation'],
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        eval_steps=config['training']['eval_steps'],
        save_total_limit=config['training']['save_total_limit'],
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        fp16=True,  # Enable mixed precision if using GPU
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
        seed=42
    )

    # Train the model
    trainer.train()
    
    # Evaluate on validation set
    eval_results = trainer.evaluate()
    print(f"Validation Results: {eval_results}")
    
    # Save the models
    # Save the entire model
    trainer.save_model("./trained_model")
    
    # Test the model
    test_results = trainer.predict(test_dataset)
    print(f"Test Results: {test_results.metrics}")

if __name__ == "__main__":
    main()
