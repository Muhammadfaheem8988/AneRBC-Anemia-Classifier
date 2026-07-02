# Module: Training and Validation Engine

import os
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

def train_and_validate(model, train_loader, val_loader, epochs=10, lr=0.001):
    """Trains the model and validates it epoch-by-epoch with real-time logging."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Initialization Complete. Training on device: {device} ---")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    total_batches = len(train_loader)
    
    for epoch in range(epochs):
        print(f"\n>>> Starting Epoch [{epoch+1}/{epochs}] <<<")
        start_time = time.time()
        
        # Training Phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 50 == 0 or (i + 1) == total_batches:
                print(f"    [Epoch {epoch+1}] Processing Batch {i+1}/{total_batches}...")
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * correct / total
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        
        # Validation Phase
        print(f"    [Epoch {epoch+1}] Training done. Running validation...")
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = 100 * correct / total
        
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        elapsed_time = time.time() - start_time
        
        print(f"+++ Epoch [{epoch+1}/{epochs}] Completed in {elapsed_time:.0f}s +++")
        print(f"    Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"    Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.2f}%")
        
    return history

def save_model_and_history(model, history, model_name, save_dir):
    """Saves the model's state_dict and training history to persistent storage."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    model_path = os.path.join(save_dir, f"{model_name}_weights.pth")
    history_path = os.path.join(save_dir, f"{model_name}_history.pkl")
    
    # Save model weights
    torch.save(model.state_dict(), model_path)
    
    # Save history dictionary
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
        
    print(f"[{model_name}] Saved successfully!")
    print(f"  - Weights: {model_path}")
    print(f"  - History: {history_path}")