import os
import copy
import torch
import argparse
import dataloader
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.optim as optim
import matplotlib.pyplot as plt
from models.EEGNet import EEGNet, DeepConvNet
from matplotlib.ticker import MaxNLocator
from torch.utils.data import Dataset, DataLoader


class BCIDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index, ...], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.int64)
        return data, label

    def __len__(self):
        return self.data.shape[0]


def plot_train_acc(train_acc_list, test_acc_list, epochs, save_path='./results/accuracy_curve.png'):
    """Plot training and testing accuracy curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_acc_list, 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(range(1, epochs + 1), test_acc_list, 'r-', label='Test Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training and Testing Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy curve saved to {save_path}")
    plt.close()


def plot_train_loss(train_loss_list, epochs, save_path='./results/loss_curve.png'):
    """Plot training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_loss_list, 'g-', label='Train Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss curve saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path='./results/confusion_matrix.png'):
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def train(model, train_loader, test_loader, criterion, optimizer, args, device):
    """Training loop with validation."""
    best_acc = 0.0
    best_wts = None
    avg_acc_list = []
    test_acc_list = []
    avg_loss_list = []
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch [{epoch}/{args.num_epochs}]")
        print("-" * 80)
        
        # Training phase
        model.train()
        avg_acc = 0.0
        avg_loss = 0.0
        
        pbar = tqdm(train_loader, desc='Training')
        for i, data in enumerate(pbar):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()
            _, pred = torch.max(outputs.data, 1)
            avg_acc += pred.eq(labels).cpu().sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * pred.eq(labels).cpu().sum().item() / len(labels):.2f}%'
            })

        avg_loss /= len(train_loader.dataset)
        avg_loss_list.append(avg_loss)
        avg_acc = (avg_acc / len(train_loader.dataset)) * 100
        avg_acc_list.append(avg_acc)
        
        print(f'\nEpoch {epoch} Summary:')
        print(f'  Train Loss: {avg_loss:.4f}')
        print(f'  Train Acc.: {avg_acc:.2f}%')

        # Testing phase
        test_acc = test(model, test_loader, device)
        test_acc_list.append(test_acc)
        print(f'  Test Acc.:  {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_wts = copy.deepcopy(model.state_dict())
            print(f'  ✓ New best model! (Test Acc: {test_acc:.2f}%)')

    # Save best model
    os.makedirs('./weights', exist_ok=True)
    torch.save(best_wts, './weights/best.pt')
    print(f"\n{'=' * 80}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    
    return avg_acc_list, avg_loss_list, test_acc_list


def test(model, loader, device):
    """Test the model."""
    avg_acc = 0.0
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            avg_acc += pred.eq(labels).cpu().sum().item()

    avg_acc = (avg_acc / len(loader.dataset)) * 100
    return avg_acc


def evaluate_and_plot_confusion(model, loader, device):
    """Evaluate model and generate confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    plot_confusion_matrix(all_labels, all_preds)
    
    # Calculate metrics
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=['Class 0', 'Class 1']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EEG Classification Training')
    parser.add_argument("-num_epochs", type=int, default=300, help='Number of training epochs')
    parser.add_argument("-batch_size", type=int, default=64, help='Batch size')
    parser.add_argument("-lr", type=float, default=0.001, help='Learning rate')
    parser.add_argument("-model", type=str, default='eegnet', 
                       choices=['eegnet', 'deepconvnet'], help='Model to train')
    parser.add_argument("-activation", type=str, default='elu',
                       choices=['elu', 'relu', 'leakyrelu'], help='Activation function')
    parser.add_argument("-dropout", type=float, default=0.25, help='Dropout rate')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\nLoading BCI Competition Dataset...")
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    
    # Create datasets
    train_dataset = BCIDataset(train_data, train_label)
    test_dataset = BCIDataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    print(f"\nCreating {args.model.upper()} model...")
    if args.model == 'eegnet':
        model = EEGNet(activation=args.activation, dropout_rate=args.dropout)
    elif args.model == 'deepconvnet':
        model = DeepConvNet(activation=args.activation, dropout_rate=args.dropout)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

    model.to(device)
    criterion.to(device)

    # Train model
    train_acc_list, train_loss_list, test_acc_list = train(
        model, train_loader, test_loader, criterion, optimizer, args, device
    )

    # Plot results
    print("\nGenerating plots...")
    plot_train_acc(train_acc_list, test_acc_list, args.num_epochs)
    plot_train_loss(train_loss_list, args.num_epochs)
    
    # Load best model and evaluate
    print("\nEvaluating best model...")
    model.load_state_dict(torch.load('./weights/best.pt'))
    evaluate_and_plot_confusion(model, test_loader, device)
    
    print("\nAll done! Check './results/' for plots and './weights/' for model.")python inspect_data.py