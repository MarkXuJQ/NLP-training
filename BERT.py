import os
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['NO_PROXY'] = '*'

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd  
from sklearn.model_selection import train_test_split

class ReviewDataset(Dataset):
    def __init__(self, reviews, ratings, tokenizer, max_length=512):
        self.reviews = reviews
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.encodings = self.tokenizer(
            list(map(str, self.reviews)),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.ratings[idx] - 1, dtype=torch.long)
        }

def train_model(model, train_loader, val_loader, device, epochs=3, learning_rate=2e-5):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
            
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_accuracy = 0
    best_model_state = None
    
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Training Phase
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader, 1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels)
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                avg_loss = total_loss / batch_idx
                print(f"Batch {batch_idx}/{len(train_loader)} - Avg Loss: {avg_loss:.4f}", end='\r')

        # Validation Phase
        model.eval()
        predictions, true_labels = [], []
        
        print("\nValidating...")
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids,
                              attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict().copy()
            print(f"New best accuracy: {accuracy:.4f}")

        print(f'Epoch {epoch + 1} completed - Validation accuracy: {accuracy:.4f}')
        print('\nClassification Report:')
        print(classification_report(true_labels, predictions))

    print("\nTraining completed!")
    return best_model_state

def load_data():
    df = pd.read_csv('amazon_reviews.csv')  
    assert 'reviewText' in df.columns, "reviewText column not found in dataset"
    assert 'overall' in df.columns, "overall column not found in dataset"
    return df

def main():
    # Set device and display GPU info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.cuda.empty_cache()

    # Load and prepare data
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} reviews")

    batch_size = 16
    
    # Initialize model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=5 
    ).to(device)

    print(f"Model device: {next(model.parameters()).device}")

    # Split data into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['reviewText'].values,
        df['overall'].values,
        test_size=0.2,
        random_state=42
    )

    # Create datasets and dataloaders
    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer)
    val_dataset = ReviewDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        pin_memory=True if device.type == 'cuda' else False
    )

    print("Starting training...")
    best_model_state = train_model(model, train_loader, val_loader, device)
    torch.save(best_model_state, 'best_model.pth')

    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Rating')
    plt.xlabel('Predicted Rating')
    plt.show()
    
    print('\nFinal Classification Report:')
    print(classification_report(all_labels, all_predictions))

if __name__ == "__main__":
    main()

import torch
print(f"CUDA is available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name()}")
