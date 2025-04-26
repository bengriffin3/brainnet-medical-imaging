import os
import torch
import numpy as np
import time
import pickle
import argparse
import logging
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm

# Import your own modules
from src.config.data import data_setup, data_loader, add_topo_features
from src.models.cnn import BrainTumorCNN, BrainTumorCNN_RN, BrainTumorCNN_Topo
from src.utils.prediction import train_model
from src.models.vision_transformer import run_training  # assuming you have a run_training function

def save_results(history, model_name, num_epochs, results_dir='results'):
    os.makedirs(results_dir, exist_ok=True)

    # Save entire history as a pickle file
    history_save_path = os.path.join(results_dir, f'{model_name}_history_{num_epochs}_epochs.pkl')
    with open(history_save_path, 'wb') as f:
        pickle.dump(history, f)

    logging.info(f"Training history saved to {history_save_path}")

def main(args):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Setup results directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Load data based on model type
    if args.model in ['cnn', 'cnn_res']:
        train_set, test_set, _ = data_setup()
        train_loader, test_loader = data_loader(train_set, test_set)
    elif args.model == 'cnn_topo':
        train_set, test_set, _ = data_setup()
        train_set, test_set = add_topo_features(train_set, test_set)
        train_loader, test_loader = data_loader(train_set, test_set)
    elif args.model in ['vit', 'vit_ft']:
        train_set, test_set, _ = data_setup(vision_transformer=True)
        train_loader, test_loader = data_loader(train_set, test_set)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    # Initialize model
    if args.model == 'cnn':
        model = BrainTumorCNN().to(device)
    elif args.model == 'cnn_res':
        model = BrainTumorCNN_RN().to(device)
    elif args.model == 'cnn_topo':
        model = BrainTumorCNN_Topo().to(device)
    elif args.model == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=4)
        model.to(device)
        model.train()
    elif args.model == 'vit_ft':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=4)
        for name, param in model.named_parameters():
            if 'blocks.11' in name or 'norm' in name or 'head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        model.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.head.in_features, 4)
        )
        model.to(device)
        model.train()

    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    logging.info(f'Model {args.model} initialized')

    # Train model
    start_time = time.time()
    logging.info(f"Starting training for {args.epochs} epochs...")

    if args.model.startswith('vit'):
        criterion = nn.CrossEntropyLoss()

        if args.model == 'vit_ft':
            head_params = list(model.head.parameters())
            other_params = [p for p in model.parameters() if p not in set(head_params)]
            optimizer = Adam([
                {'params': head_params, 'lr': 0.001},
                {'params': other_params, 'lr': 0.0001}
            ])
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            history = run_training(
                model, train_loader, test_loader,
                optimizer, scheduler, criterion,
                num_epochs=args.epochs, device=device,
                fine_tuning=True
            )
        else:
            optimizer = Adam(model.parameters(), lr=0.001)
            history = run_training(
                model, train_loader, test_loader,
                optimizer, None, criterion,
                num_epochs=args.epochs, device=device,
                fine_tuning=False
            )

    else:
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            num_epochs=args.epochs
        )

    end_time = time.time()
    training_duration = end_time - start_time
    logging.info(f"Training completed in {training_duration / 60:.2f} minutes")

    # Save full training history
    save_results(history, model_name=args.model, num_epochs=args.epochs, results_dir=results_dir)

    model_save_path = os.path.join(results_dir, f'{args.model}_model_{args.epochs}_epochs.pth')
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model weights saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train different models for brain tumor classification.")
    parser.add_argument('--model', type=str, required=True,
                        choices=['cnn', 'cnn_res', 'cnn_topo', 'vit', 'vit_ft'],
                        help='Choose which model to train: cnn, cnn_res, cnn_topo, vit, vit_ft')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')

    args = parser.parse_args()
    main(args)
