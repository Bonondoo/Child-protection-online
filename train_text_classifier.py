import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import BertTokenizer, TFBertModel

# Load pre-trained BERT model and tokenizer
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define LSTM model structure
class TextClassifier(keras.Model):
    def __init__(self, bert_model):
        super(TextClassifier, self).__init__()
        self.bert = bert_model
        self.lstm = layers.LSTM(128)
        self.dense = layers.Dense(1, activation='sigmoid')

    def call(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        lstm_output = self.lstm(bert_output.last_hidden_state)  # Use the output from BERT
        return self.dense(lstm_output)

# Example usage
# Define input data
texts = ["This is a positive example.", "This is a negative example."]
labels = [1, 0]  # Binary labels

# Tokenize inputs
encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='tf')

# Create and compile model
model = TextClassifier(bert_model)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model on your data
# model.fit(encodings['input_ids'], labels, epochs=5) # Uncomment to train