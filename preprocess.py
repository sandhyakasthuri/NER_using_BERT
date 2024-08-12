import pandas as pd
from transformers import BertTokenizerFast, BertForTokenClassification, pipeline
import torch
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import AdamW

# Load your dataset
dataset_path = "data/ner_datasetreference.csv"
df = pd.read_csv(dataset_path, encoding='ISO-8859-1')

# Fill NaN values in the "Word" column, if any, with a placeholder or remove such rows
df['Word'] = df['Word'].fillna('')
df['Word'] = df['Word'].astype(str)

# Initialize the fast tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Group the words and labels by sentence
grouped = df.groupby('Sentence #')
sentences = grouped['Word'].apply(list).tolist()
labels = grouped['Tag'].apply(list).tolist()

# List unique labels from the entire dataset
unique_labels = df['Tag'].unique()
print("Unique labels in the dataset:", unique_labels)


# Define a label map including all labels found in the dataset
label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
reverse_label_map = {v: k for k, v in label_map.items()}

# Tokenize the sentences
tokenized_inputs = tokenizer(
    sentences,
    is_split_into_words=True,
    return_offsets_mapping=True,
    padding=True,
    truncation=True
)

# Extract necessary fields
input_ids = tokenized_inputs['input_ids']
attention_mask = tokenized_inputs['attention_mask']
offset_mapping = tokenized_inputs['offset_mapping']

# Align labels with tokens
aligned_labels = []

for i, doc_labels in enumerate(labels):
    doc_offset_labels = []
    last_label = 'O'
    for j, offset in enumerate(offset_mapping[i]):
        if offset[0] == offset[1]:
            doc_offset_labels.append(-100)
        else:
            token = tokenizer.convert_ids_to_tokens(input_ids[i][j])
            if token.startswith("##"):
                doc_offset_labels.append(label_map[last_label])
            else:
                current_label = doc_labels.pop(0) if len(doc_labels) > 0 else 'O'
                doc_offset_labels.append(label_map.get(current_label, label_map['O']))
                last_label = current_label
    aligned_labels.append(doc_offset_labels)

# Convert to tensors
input_ids = torch.tensor(input_ids)
attention_mask = torch.tensor(attention_mask)
labels = torch.tensor(aligned_labels)

# Save processed data
torch.save({'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}, 'processed_ner_data.pt')
print("Data preprocessing complete.")

# Create dataset
dataset = TensorDataset(input_ids, attention_mask, labels)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=16)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=16)

# Initialize model
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map))

# Check if GPU is available and move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=5e-5,no_deprecation_warning=True)

# Training loop
epochs = 2
for epoch in range(epochs):
    print(f"Starting epoch {epoch + 1}...")
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        if step % 10 == 0 and not step == 0:
            print(f"Batch {step}/{len(train_dataloader)}")
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}, Average Training Loss: {avg_train_loss}")
    
    # Validation loop
    print("Starting validation...")
    model.eval()
    val_loss = 0
    predictions, true_labels = [], []
    for step, batch in enumerate(val_dataloader):
        if step % 10 == 0 and not step == 0:
            print(f"Validation Batch {step}/{len(val_dataloader)}")
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        logits = outputs.logits
        val_loss += outputs.loss.item()

        # Convert predictions to label IDs
        logits = np.argmax(logits.detach().cpu().numpy(), axis=2)
        label_ids = b_labels.to('cpu').numpy()

        # Flatten the arrays and remove ignored index (-100)
        for i in range(len(label_ids)):
            for j in range(len(label_ids[i])):
                if label_ids[i][j] != -100:
                    predictions.append(logits[i][j])
                    true_labels.append(label_ids[i][j])

    unique_true_labels = set(true_labels)
    unique_predictions = set(predictions)
    print(f"Unique classes in true labels: {unique_true_labels}")
    print(f"Unique classes in predictions: {unique_predictions}")

    present_labels = sorted(unique_true_labels | unique_predictions)
    print(f"Present labels: {present_labels}")

    adjusted_target_names = [reverse_label_map[label] for label in present_labels]

    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_val_loss}")
    print("Validation Classification Report:")
    print(classification_report(true_labels, predictions, target_names=adjusted_target_names, labels=present_labels))

# Save the model and tokenizer after training
model.save_pretrained('ner_model')
tokenizer.save_pretrained('ner_model')

