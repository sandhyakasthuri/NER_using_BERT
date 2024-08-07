import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import numpy as np
from sklearn.metrics import classification_report
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="A parameter name that contains `beta` will be renamed internally to `bias`")
warnings.filterwarnings("ignore", message="A parameter name that contains `gamma` will be renamed internally to `weight`")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Define label map (should match the labels used in your dataset)
label_map = {
    'O': 0,
    'B-tim': 1,
    'B-per': 2,
    'B-art': 3,
    'B-org': 4,
    'B-nat': 5,
    'B-geo': 6,
    'B-eve': 7,
    'B-gpe': 8,
    'I-tim': 9,
    'I-per': 10,
    'I-art': 11,
    'I-org': 12,
    'I-nat': 13,
    'I-geo': 14,
    'I-eve': 15,
    'I-gpe': 16
}
reverse_label_map = {v: k for k, v in label_map.items()}

# Load preprocessed data
data = torch.load('processed_ner_data.pt', map_location=torch.device('cpu'))
input_ids = data['input_ids']
attention_mask = data['attention_mask']
labels = data['labels']

# Check unique labels
unique_labels = set(labels.flatten().tolist())
print("Unique labels in the dataset:", unique_labels)

# Ensure all labels are within the expected range
for label in unique_labels:
    if label not in label_map.values() and label != -100:
        print(f"Unexpected label found: {label}")

# Initialize the tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map))

# Check if GPU is available and move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create DataLoader
batch_size = 16
dataset = TensorDataset(input_ids, attention_mask, labels)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 3
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

    # Check unique classes in predictions and true_labels
    unique_true_labels = set(true_labels)
    unique_predictions = set(predictions)
    print(f"Unique classes in true labels: {unique_true_labels}")
    print(f"Unique classes in predictions: {unique_predictions}")

    # Filter label_map to include only the labels that are actually present in the predictions
    present_labels = sorted(unique_true_labels | unique_predictions)
    print(f"Present labels: {present_labels}")

    # Adjust target names to match present labels
    adjusted_target_names = [reverse_label_map[label] for label in present_labels]

    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_val_loss}")
    print("Validation Classification Report:")
    print(classification_report(true_labels, predictions, target_names=adjusted_target_names, labels=present_labels))

# Save the model and tokenizer after training
model.save_pretrained('ner_model')
tokenizer.save_pretrained('ner_model')
