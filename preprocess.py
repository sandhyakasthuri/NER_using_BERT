import pandas as pd
from transformers import BertTokenizerFast
import torch

# Load your dataset
dataset_path = "data/ner_datasetreference.csv"
df = pd.read_csv(dataset_path, encoding='ISO-8859-1')

# Fill NaN values in the "Word" column, if any, with a placeholder or remove such rows
df['Word'] = df['Word'].fillna('')
df['Word'] = df['Word'].astype(str)

# Initialize the fast tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Group the words by sentence
sentences = df.groupby("Sentence #")["Word"].apply(list).tolist()
labels = df.groupby("Sentence #")["Tag"].apply(list).tolist()

# Identify all unique labels in the dataset
unique_labels = set(label for sublist in labels for label in sublist)
print("Unique labels in the dataset:", unique_labels)

# Define a label map including all labels found in the dataset
label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}

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
            doc_offset_labels.append(-100)  # Special token or padding
        else:
            token = tokenizer.convert_ids_to_tokens(input_ids[i][j])
            if token.startswith("##"):
                doc_offset_labels.append(label_map[last_label])
            else:
                current_label = doc_labels.pop(0) if len(doc_labels) > 0 else 'O'
                doc_offset_labels.append(label_map.get(current_label, label_map['O']))  # Handle missing labels
                last_label = current_label
    aligned_labels.append(doc_offset_labels)

# Convert to tensors
input_ids = torch.tensor(input_ids)
attention_mask = torch.tensor(attention_mask)
labels = torch.tensor(aligned_labels)

# Save processed data
torch.save({'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}, 'processed_ner_data.pt')

print("Data preprocessing complete.")
