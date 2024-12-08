import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import AdamW

# Step 1: Load and preprocess data
# Load data
df_train = pd.read_csv("processed_training_dataset.csv")
df_test = pd.read_csv("processed_testing_dataset.csv")

# Drop rows with missing or invalid labels
df_train = df_train.dropna(subset=['label'])  # Drop rows with NaN values in the label column
df_train['label'] = df_train['label'].astype(int)  # Convert label to integer

df_test = df_test.dropna(subset=['label'])  # Do the same for the test dataset
df_test['label'] = df_test['label'].astype(int)

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Define a dataset class
class AbusiveCommentsDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_length=64):
        self.comments = comments
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, index):
        comment = str(self.comments[index])
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(int(label), dtype=torch.long)
        }

# Create datasets and dataloaders
train_dataset = AbusiveCommentsDataset(
    comments=df_train['processed_comment'].values,
    labels=df_train['label'].values,
    tokenizer=tokenizer
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = AbusiveCommentsDataset(
    comments=df_test['processed_comment'].values,
    labels=df_test['label'].values,
    tokenizer=tokenizer
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Step 2: Load BERT model with a classification layer
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
#model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model = model.to(device)

# Training setup
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 3

# Step 3: Training loop
model.train()
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} completed")

# Step 4: Evaluation on the test set
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Step 5: Save predictions to CSV
df_test['predicted_label'] = all_preds  # Add predictions to the DataFrame
df_test.to_csv("predicted_BERT_dataset.csv", index=False)
print("Predictions have been saved")
