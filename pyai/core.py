import os
import json
import random
import re
import nltk
import pyjokes
import yfinance as yf
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import subprocess
import webbrowser

# ----- Sample Training Data -----
train_data = [
    (["My", "name", "is", "Aryan"],         ["O", "O", "O", "NAME"]),
    (["My", "name", "is", "Divyanshu"],         ["O", "O", "O", "NAME"]),
    (["My", "name", "is", "Vaishnavi"],         ["O", "O", "O", "NAME"]),
    (["My", "name", "is", "Bhumi"],         ["O", "O", "O", "NAME"]),
    (["My", "name", "is", "Yesh"],         ["O", "O", "O", "NAME"]),
    (["I", "live", "in", "Jaipur"],         ["O", "O", "O", "LOCATION"]),
    (["I", "am", "19", "years", "old"],     ["O", "O", "AGE", "O", "O"]),
    (["She", "is", "Meera"],                ["O", "O", "NAME"]),
    (["She", "stays", "in", "Goa"],         ["O", "O", "O", "LOCATION"]),
    (["Age", "is", "24"],                   ["O", "O", "AGE"]),
    (["I", "am", "Dev", "from", "Lucknow"], ["O", "O", "NAME", "O", "LOCATION"]),
    (["I", "am", "Dev", "from", "Jamshedpur"], ["O", "O", "NAME", "O", "LOCATION"]),
    (["I", "am", "Dev", "from", "Munger"], ["O", "O", "NAME", "O", "LOCATION"]),
    (["I", "am", "Dev", "from", "Patna"], ["O", "O", "NAME", "O", "LOCATION"]),
    (["I", "am", "Dev", "from", "Rachi"], ["O", "O", "NAME", "O", "LOCATION"]),
    (["I", "am", "Dev", "from", "India"], ["O", "O", "NAME", "O", "LOCATION"]),
    (["I", "am", "Dev", "from", "Jamalpur"], ["O", "O", "NAME", "O", "LOCATION"]),
    (["born", "in", "2003"],                ["O", "O", "AGE"]),
    (["I", "am", "Dev", "from", "Lucknow", ".", "My", "age", "is", "19"], ["O", "O", "NAME", "O", "LOCATION", "O", "O", "O", "O", "AGE"]),
]

# ----- Build Vocabulary -----
word_counter = Counter()
tag_set = set()

for words, tags in train_data:
    word_counter.update(words)
    tag_set.update(tags)

word2idx = {w: i+1 for i, w in enumerate(word_counter)}
word2idx["<PAD>"] = 0

tag2idx = {t: i for i, t in enumerate(tag_set)}
idx2tag = {i: t for t, i in tag2idx.items()}

# ----- Encode Function -----
def encode(words, tags, max_len):
    x = [word2idx.get(w, 0) for w in words]
    y = [tag2idx[t] for t in tags]

    if len(x) < max_len:
        pad_len = max_len - len(x)
        x += [0] * pad_len
        y += [tag2idx["O"]] * pad_len
    return x[:max_len], y[:max_len]

MAX_LEN = 100

X = []
Y = []

for words, tags in train_data:
    x, y = encode(words, tags, MAX_LEN)
    X.append(x)
    Y.append(y)

X = torch.tensor(X)
Y = torch.tensor(Y)

# ----- Dataset & DataLoader -----
class NERDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

loader = DataLoader(NERDataset(X, Y), batch_size=2, shuffle=True)

# ----- NER Model -----
class NERModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, tagset_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        tag_scores = self.fc(output)
        return tag_scores

# ----- Training -----
model = NERModel(vocab_size=len(word2idx), embed_dim=64, hidden_dim=64, tagset_size=len(tag2idx))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print('NER Training!')
for epoch in range(1, 51):
    model.train()
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs.view(-1, len(tag2idx)), batch_y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

print('NER Training Complete!')

# ----- Prediction -----
def predict_entities(sentence):
    model.eval()
    tokens = word_tokenize(sentence)
    encoded, _ = encode(tokens, ["O"] * len(tokens), MAX_LEN)
    input_tensor = torch.tensor([encoded])
    with torch.no_grad():
        outputs = model(input_tensor)
        predictions = torch.argmax(outputs, dim=2).squeeze().tolist()

    entities = {"NAME": None, "AGE": None, "LOCATION": None}
    for token, pred in zip(tokens, predictions):
        label = idx2tag.get(pred, "O")
        if label in entities and entities[label] is None:
            entities[label] = token
    return entities

# Step 1: Sample training data
train_sentences = [
    "How are you?",
    "Open the door",
    "The sun rises in the east",
    "What time is it?",
    "Close the window",
    "She is reading a book",
    "Is this your pen?",
    "Start the engine",
    "He likes football",
    "Where do you live?",
    "1+1 is 2",
    "I am Divyanshu",
    "Myself Divyanshu",
    "Do you know",
    "Shutdown /s /t 0",
    "Mkdir",
    "Start"
]

# Labels: 1=Question, 2=Command, 0=Statement, 3=Answer, 4=Name, 5=Know
train_labels = [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 3, 4, 4, 5, 6, 7, 8]

# Step 2: Vectorize the text
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_sentences).toarray()
y_train = np.array(train_labels)

# Step 3: Convert to PyTorch tensors
X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.long)  # Ensure correct dtype for labels

# Step 4: Define the model
class FrameClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FrameClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)  # Output dimension should be 6 for 6 classes
        )

    def forward(self, x):
        return self.net(x)

# Initialize model with correct output dimension (9 classes)
model = FrameClassifier(X_train.shape[1], 9)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()  # CrossEntropyLoss expects the target tensor to be of type torch.long
optimizer = optim.Adam(model.parameters(), lr=0.01)

print('Predict Frame Traning')
# Step 5: Train the model
epochs = 150
for epoch in range(epochs):
    model.train()
    outputs = model(X_tensor)
    loss = loss_fn(outputs, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('Predict Frame Traning Complete!')

# Step 6: Frame prediction function
frame_map = {0: "Statement", 1: "Question", 2: "Command", 3: "Answer", 4: "Name", 5: "Know", 6: "Shutdown", 7: "Make Dir", 8: "Start"}

def predict_frame(sentence):
    model.eval()
    input_vec = vectorizer.transform([sentence]).toarray()
    input_tensor = torch.tensor(input_vec, dtype=torch.float32)  # Ensure input is tensor type float32
    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output, dim=1).item()
    return frame_map.get(predicted)

class FrameClassifierAdvance(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class FramePredictorAdvance:
    def __init__(self, sentences, labels, class_map):
        self.vectorizer = CountVectorizer()
        self.X_train = self.vectorizer.fit_transform(sentences).toarray()
        self.y_train = np.array(labels)
        self.class_map = class_map
        self.model = FrameClassifierAdvance(self.X_train.shape[1], len(class_map))

    def train(self, epochs=150):
        X_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_train, dtype=torch.long)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        for epoch in range(epochs):
            self.model.train()
            outputs = self.model(X_tensor)
            loss = loss_fn(outputs, y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        print("Frame Classifier Training Complete!")

    def predict(self, sentence):
        input_vec = self.vectorizer.transform([sentence]).toarray()
        input_tensor = torch.tensor(input_vec, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted = torch.argmax(output, dim=1).item()
        return self.class_map.get(predicted)

class NERDatasetAdvance(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class NERModelAdvance(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, tagset_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.fc(output)

class NERTrainerAdvance:
    def __init__(self, train_data, max_len=100):
        self.max_len = max_len
        self.train_data = train_data
        self.word2idx = {}
        self.tag2idx = {}
        self.idx2tag = {}

        self.build_vocab()
        self.model = NERModelAdvance(len(self.word2idx), 64, 64, len(self.tag2idx))

    def build_vocab(self):
        word_counter = Counter()
        tag_set = set()
        for words, tags in self.train_data:
            word_counter.update(words)
            tag_set.update(tags)
        self.word2idx = {w: i + 1 for i, w in enumerate(word_counter)}
        self.word2idx["<PAD>"] = 0
        self.tag2idx = {t: i for i, t in enumerate(tag_set)}
        self.idx2tag = {i: t for t, i in self.tag2idx.items()}

    def encode(self, words, tags):
        x = [self.word2idx.get(w, 0) for w in words]
        y = [self.tag2idx[t] for t in tags]
        if len(x) < self.max_len:
            pad_len = self.max_len - len(x)
            x += [0] * pad_len
            y += [self.tag2idx["O"]] * pad_len
        return x[:self.max_len], y[:self.max_len]

    def train(self, epochs=50, batch_size=2):
        X, Y = [], []
        for words, tags in self.train_data:
            x, y = self.encode(words, tags)
            X.append(x)
            Y.append(y)

        X = torch.tensor(X)
        Y = torch.tensor(Y)
        dataset = NERDatasetAdvance(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs.view(-1, len(self.tag2idx)), batch_y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 10 == 0:
                print(f"Epoch {epoch} Loss: {total_loss:.4f}")
        print("NER Training Complete!")

    def predict(self, sentence):
        self.model.eval()
        tokens = word_tokenize(sentence)
        encoded, _ = self.encode(tokens, ["O"] * len(tokens))
        input_tensor = torch.tensor([encoded])
        with torch.no_grad():
            outputs = self.model(input_tensor)
            predictions = torch.argmax(outputs, dim=2).squeeze().tolist()
        entities = {"NAME": None, "AGE": None, "LOCATION": None}
        for token, pred in zip(tokens, predictions):
            label = self.idx2tag.get(pred, "O")
            if label in entities and entities[label] is None:
                entities[label] = token
        return entities

class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

class ChatbotAssistant:
    def __init__(self, intents_path, function_mapping=None):
        self.model = None
        self.intents_path = intents_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.function_mapping = function_mapping

        self.X = None
        self.Y = None

    @staticmethod
    def tokenize_and_lemmatizer(text):
        lemmatizer = nltk.WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]

        return words

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)

            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatizer(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append([pattern_words, intent['tag']])

            self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)

            intent_index = self.intents.index(document[1])

            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.Y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        Y_tensor = torch.tensor(self.Y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, Y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)  # lr stands for learning rate

        print('Model Traning!')
        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_Y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()
                running_loss += loss

        print("Training complete.")

    def save_model(self, model_path, dimension_path):
        torch.save(self.model.state_dict(), model_path)

        with open(dimension_path, 'w') as f:
            json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents)}, f)

    def load_model(self, model_path, dimension_path):
        with open(dimension_path, 'r') as f:
            dimensions = json.load(f)

        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    #@lru_cache(maxsize = None)
    def process_message(self, input_message):
        words = self.tokenize_and_lemmatizer(input_message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()

        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        if predicted_intent == "python_code_execution":  # If the intent is to execute Python code
            return self.execute_code(input_message)

        if predicted_intent == "joke":  # If the intent is to execute joke
            return pyjokes.get_joke()

        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])

        else:
            return "I didn't understand that."

    def execute_code(self, query):
        try:
            code = query.split('run python')[-1].strip()  # Extracting the Python code from the message
            # Run the code
            local_scope = {}
            exec(code, {"__builtins__": None}, local_scope)  # Restricting the built-ins to None
            result = local_scope.get('result', 'No output')  # Get the result of execution
            return f"The result of the Python code is: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

def predictFrameAdvance(sentence: str | None = None):
    train_sentences = [
    "How are you?",
    "Open the door",
    "The sun rises in the east",
    "What time is it?",
    "Close the window",
    "She is reading a book",
    "Is this your pen?",
    "Start the engine",
    "He likes football",
    "Where do you live?",
    "1+1 is 2",
    "I am Divyanshu",
    "Myself Divyanshu",
    "Do you know",
    "Shutdown /s /t 0",
    "Mkdir",
    "Start"
    ]

    frame_map = {0: "Statement", 1: "Question", 2: "Command", 3: "Answer", 4: "Name", 5: "Know", 6: "Shutdown", 7: "Make Dir", 8: "Start"}
    train_labels = [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 3, 4, 4, 5, 6, 7, 8]

    predict_frames = FramePredictorAdvance(train_sentences, train_labels, frame_map)
    return predict_frames.predict(sentence)

def predictNER(sentence: str | None = ''):
    train_data = [
    (["My", "name", "is", "Aryan"],         ["O", "O", "O", "NAME"]),
    (["My", "name", "is", "Divyanshu"],         ["O", "O", "O", "NAME"]),
    (["My", "name", "is", "Vaishnavi"],         ["O", "O", "O", "NAME"]),
    (["My", "name", "is", "Bhumi"],         ["O", "O", "O", "NAME"]),
    (["My", "name", "is", "Yesh"],         ["O", "O", "O", "NAME"]),
    (["I", "live", "in", "Jaipur"],         ["O", "O", "O", "LOCATION"]),
    (["I", "am", "19", "years", "old"],     ["O", "O", "AGE", "O", "O"]),
    (["She", "is", "Meera"],                ["O", "O", "NAME"]),
    (["She", "stays", "in", "Goa"],         ["O", "O", "O", "LOCATION"]),
    (["Age", "is", "24"],                   ["O", "O", "AGE"]),
    (["I", "am", "Dev", "from", "Lucknow"], ["O", "O", "NAME", "O", "LOCATION"]),
    (["I", "am", "Dev", "from", "Jamshedpur"], ["O", "O", "NAME", "O", "LOCATION"]),
    (["I", "am", "Dev", "from", "Munger"], ["O", "O", "NAME", "O", "LOCATION"]),
    (["I", "am", "Dev", "from", "Patna"], ["O", "O", "NAME", "O", "LOCATION"]),
    (["I", "am", "Dev", "from", "Rachi"], ["O", "O", "NAME", "O", "LOCATION"]),
    (["I", "am", "Dev", "from", "India"], ["O", "O", "NAME", "O", "LOCATION"]),
    (["I", "am", "Dev", "from", "Jamalpur"], ["O", "O", "NAME", "O", "LOCATION"]),
    (["born", "in", "2003"],                ["O", "O", "AGE"]),
    (["I", "am", "Dev", "from", "Lucknow", ".", "My", "age", "is", "19"], ["O", "O", "NAME", "O", "LOCATION", "O", "O", "O", "O", "AGE"]),
    ]

    return NERTrainerAdvance(train_data).predict(sentence)

class Brain:
    def __init__(self, intents_path: str = 'intents.json', **function_mapping):
        self.assistant = ChatbotAssistant(intents_path, function_mapping=function_mapping)

        self.assistant.parse_intents()
        self.assistant.prepare_data()
        self.assistant.train_model(8, 0.001, 100)

    def predict_message_type(self, message: str | None = None):
        '''It Returns:
            1. Statement.
            2. Question.
            3. Answer.
            4. Command.
            5. Shutdown.
            6. Name.
            7. Know.
            8. Make Dir.
            9. Start.'''
        return predict_frame(message)

    def pyai_say(self, *message, **options):
        print('PYAI :',*message, **options)

    def predict_entitie(self, message: str | None = None):
        '''It Returns:
            1. NAME.
            2. AGE.
            3. LOCATION.'''
        return predict_entities(message)

    def process_messages(self, message: str | None = None):
        return self.assistant.process_message(message)

class AdvanceBrain:
    def __init__(self, intents_path: str | None = 'intents.json', **function_mapping):
        self.assistant = ChatbotAssistant(intents_path, **function_mapping)
        self.assistant.parse_intents()
        self.assistant.prepare_data()
        self.assistant.train_model(8, 0.001, 100)

    def predict_message_type(self, message: str | None = None):
        '''It Returns:
            1. Statement.
            2. Question.
            3. Answer.
            4. Command.
            5. Shutdown.
            6. Name.
            7. Know.
            8. Make Dir.
            9. Start.'''
        return predictFrameAdvance(message)

    def predict_entitie(self, message: str | None = None):
        '''It Returns:
            1. NAME.
            2. AGE.
            3. LOCATION.'''
        return predictNER(message)

    def pyai_say(self, *message, **options):
        print('PYAI :',*message, **options)
        return ''

    def process_messages(self, message: str | None = None):
        return self.assistant.process_message(message)
