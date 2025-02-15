import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import os
from tqdm import tqdm

# Making a Dataframe 
fake_df = pd.read_csv('data/fake.csv')
true_df = pd.read_csv('data/true.csv')

# Adding label columns to the dataframes
fake_df['label'] = 0
true_df['label'] = 1

df = pd.concat([fake_df, true_df]).sample(frac = 1).reset_index(drop = True)

# Splitting the data into training and testing
train_df, test_df = train_test_split(df, test_size = 0.2, random_state=42)

# Dataset Class
class newsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)
    
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
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Initializing the tokenizer and the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Creating the training parameters
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define max_len
max_len = 256

# Creating the dataloaders
train_dataset = newsDataset(train_df['text'].values, train_df['label'].values, tokenizer, max_len)
test_dataset = newsDataset(test_df['text'].values, test_df['label'].values, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model.to(device)
optimizer = AdamW(model.parameters() , lr = LR)

# Triaining the model
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device) 
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    print(f"Average training loss: {avg_loss}")

# Saving the model
output_dir = 'model\fakenews_model'
os.makedirs(output_dir,exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Evaluating the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim = 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f'Test Accuracy: {correct / total:.2f}')