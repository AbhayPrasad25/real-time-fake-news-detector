import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import os
from tqdm import tqdm

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data Preparation ------------------------------------------------------------
# Load data with error handling
try:
    fake_df = pd.read_csv('data/fake.csv')
    true_df = pd.read_csv('data/true.csv')
except FileNotFoundError:
    raise SystemExit("Error: Dataset files not found in 'data/' directory")

# Add labels and combine
fake_df['label'] = 0
true_df['label'] = 1
df = pd.concat([fake_df, true_df])

# Clean data
df = df.drop_duplicates(subset=['text'])  # Remove duplicates
df['text'] = df['text'].astype(str)  # Ensure text is string type

# Stratified split to maintain class balance
train_df, test_df = train_test_split(df, test_size=0.2, 
                                    stratify=df['label'], 
                                    random_state=42)

# Dataset Class ---------------------------------------------------------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Model Setup -----------------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    hidden_dropout_prob=0.3,  # Increased dropout for regularization
    attention_probs_dropout_prob=0.3
).to(device)

# Training Parameters ---------------------------------------------------------
BATCH_SIZE = 32 if torch.cuda.is_available() else 16
EPOCHS = 3  # Reduced epochs to prevent overfitting
LR = 2e-5
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-5)  # Weight decay

# Create DataLoaders ----------------------------------------------------------
train_dataset = NewsDataset(train_df['text'].values, 
                           train_df['label'].values, 
                           tokenizer)
test_dataset = NewsDataset(test_df['text'].values, 
                          test_df['label'].values, 
                          tokenizer)

train_loader = DataLoader(train_dataset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=True,
                         pin_memory=True if device.type == 'cuda' else False)

test_loader = DataLoader(test_dataset, 
                        batch_size=BATCH_SIZE*2,
                        pin_memory=True if device.type == 'cuda' else False)

# Training Loop ---------------------------------------------------------------
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    # Training Phase
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids, 
                       attention_mask=attention_mask, 
                       labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_train_loss = epoch_loss / len(train_loader)
    
    # Validation Phase
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, 
                           attention_mask=attention_mask, 
                           labels=labels)
            val_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_val_loss = val_loss / len(test_loader)
    val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"Val Accuracy: {val_acc:.2%}")
    print(classification_report(all_labels, all_preds, target_names=['Fake', 'Real']))
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        output_dir = 'model/fakenews_model'
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Saved best model!")

# Final Evaluation ------------------------------------------------------------
print("\nFinal Evaluation on Test Set:")
model = BertForSequenceClassification.from_pretrained(output_dir).to(device)
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=['Fake', 'Real']))