import os
import torch
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizerFast, BertForTokenClassification
import PyPDF2
import docx
import re

app = Flask(__name__)

# Load the model and tokenizer
model = BertForTokenClassification.from_pretrained('ner_model')
tokenizer = BertTokenizerFast.from_pretrained('ner_model')

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

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
    return text

def extract_text_from_word(file):
    doc = docx.Document(file)
    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

def preprocess_text(text):
    text = re.sub(r'\n+', ' ', text)  # Replace new lines with spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space
    return text

def predict_entities(text):
    inputs = tokenizer(text.split(), is_split_into_words=True, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)
    predicted_labels = [reverse_label_map[label_id.item()] for label_id in predictions[0]]
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    entities = [(token, label) for token, label in zip(tokens, predicted_labels) if label != 'O']
    return entities

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(file)
        elif file.filename.endswith('.docx'):
            text = extract_text_from_word(file)
        elif file.filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
    else:
        text = request.form.get('text')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    text = preprocess_text(text)
    entities = predict_entities(text)
    return jsonify({'entities': entities})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0')
