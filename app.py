# app.py
import os
import json
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib
import uuid
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from gensim.models import Word2Vec, KeyedVectors
import pickle
# Fix imports for transformers
from transformers import BertModel, BertTokenizer, BertConfig
# Use PyTorch's AdamW instead of transformers version
from torch.optim import AdamW
import time
import matplotlib.patches as mpatches
# Remove unused imports (get_linear_schedule_with_warmup)
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.jinja_env.globals['now'] = datetime.now  # Add this line to make now() available
app.secret_key = 'spam_detector_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['DATA_FOLDER'] = 'data'
app.config['WORD_EMBEDDINGS_FOLDER'] = 'embeddings'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'json', 'eml'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)
os.makedirs(app.config['WORD_EMBEDDINGS_FOLDER'], exist_ok=True)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PyTorch Email Dataset Class
class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        if self.tokenizer:
            # For BERT model
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        else:
            # For other models (will be handled in the specific model creation)
            return text, label

# Define PyTorch Models
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx, embedding_matrix=None):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # If pretrained embeddings are provided
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = False  # Freeze embeddings
            
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           batch_first=True,
                           dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        # text: [batch size, seq len]
        embedded = self.embedding(text)
        # embedded: [batch size, seq len, embedding dim]
        
        # Pack sequence
        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        
        # Run through LSTM
        #packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, (hidden, cell) = self.lstm(embedded)
        
        # Unpack sequence
        #output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Get last hidden state
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
            
        return self.fc(hidden)

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx, embedding_matrix=None):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # If pretrained embeddings are provided
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = False  # Freeze embeddings
            
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text: [batch size, seq len]
        embedded = self.embedding(text)
        # embedded: [batch size, seq len, embedding dim]
        
        # Need to transpose as conv1d expects [batch, channels, seq]
        embedded = embedded.permute(0, 2, 1)
        
        # Apply convolutions
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        
        # Apply max pooling
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        # Concatenate pooled outputs
        cat = self.dropout(torch.cat(pooled, dim=1))
        
        return self.fc(cat)

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', dropout=0.1):
        super().__init__()
        
        # Load pre-trained BERT model
        try:
            self.bert = BertModel.from_pretrained(bert_model_name)
        except:
            # If can't download, try to load from a local path
            config = BertConfig.from_pretrained(
                os.path.join(app.config['WORD_EMBEDDINGS_FOLDER'], 'bert-config')
            )
            self.bert = BertModel.from_pretrained(
                os.path.join(app.config['WORD_EMBEDDINGS_FOLDER'], 'bert-model'),
                config=config
            )
            
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get [CLS] token embedding (first token)
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classifier
        pooled_output = self.dropout(pooled_output)
        return self.fc(pooled_output)

# Spam Detector Class
class SpamDetector:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.model = None
        self.vectorizer = None
        self.trained = False
        self.df = None
        self.training_stats = {}
        self.word2vec_model = None
        self.tokenizer = None
        self.bert_tokenizer = None
        self.max_sequence_length = 200  # For deep learning models
        self.embedding_dim = 100  # Default dimension for word embeddings
        self.vocab = None  # For PyTorch text processing
        self.pad_idx = 1  # For padding in PyTorch models
        
    def _safe_extract_metrics(self, report, accuracy):
    # Default values in case metrics are missing
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        
        # Check if report contains class '1' (spam)
        if isinstance(report, dict):
            if '1' in report:
                precision = report['1'].get('precision', 0.0)
                recall = report['1'].get('recall', 0.0)
                f1 = report['1'].get('f1-score', 0.0)
            elif 1 in report:  # Some versions use int keys
                precision = report[1].get('precision', 0.0)
                recall = report[1].get('recall', 0.0)
                f1 = report[1].get('f1-score', 0.0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
    def preprocess_email(self, email):
        """Preprocess a single email"""
        # Convert to lowercase
        email = email.lower()
        
        # Remove URLs
        email = re.sub(r'http\S+|www\S+|https\S+', ' url ', email, flags=re.MULTILINE)
        
        # Remove numbers
        email = re.sub(r'\d+', ' num ', email)
        
        # Remove special characters and extra spaces
        email = re.sub(r'[^\w\s]', ' ', email)
        email = re.sub(r'\s+', ' ', email).strip()
        
        # Remove very short words
        email = ' '.join([word for word in email.split() if len(word) >= 2])
        
        return email
    
    def load_data(self):
        """Load email data from JSON files"""
        if not self.data_dir:
            return None
            
        emails = []
        groups = ['easy-ham-1', 'easy-ham-2', 'hard-ham-1', 'spam-1', 'spam-2']
        
        for group in groups:
            group_dir = os.path.join(self.data_dir, group)
            if not os.path.exists(group_dir):
                continue
                
            # Process all JSON files in the directory
            json_files = [f for f in os.listdir(group_dir) if f.endswith('.json')]
            for filename in json_files:
                file_path = os.path.join(group_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        email_data = json.load(f)
                        # Add spam label
                        is_spam = 1 if 'spam' in group else 0
                        emails.append({
                            'id': email_data['id'],
                            'group': email_data['group'],
                            'text': email_data['text'],
                            'is_spam': is_spam
                        })
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        if not emails:
            return None
            
        self.df = pd.DataFrame(emails)
        
        # Extract basic statistics
        stats = {
            'total_emails': len(self.df),
            'spam_emails': self.df['is_spam'].sum(),
            'ham_emails': len(self.df) - self.df['is_spam'].sum(),
            'spam_ratio': self.df['is_spam'].mean()
        }
        
        return stats
    
    def extract_features(self):
        """Extract features from emails"""
        if self.df is None:
            return None
        
        # Preprocess emails
        self.df['processed_text'] = self.df['text'].apply(self.preprocess_email)
        
        # Extract basic statistics
        self.df['email_length'] = self.df['text'].apply(len)
        self.df['word_count'] = self.df['text'].apply(lambda x: len(x.split()))
        
        # Extract common spam indicators
        spam_indicators = ['free', 'offer', 'money', 'credit', 'cash', 'win', 
                          'congratulations', 'prize', 'urgent', 'limited', 'guarantee']
        
        for indicator in spam_indicators:
            self.df[f'contains_{indicator}'] = self.df['text'].apply(
                lambda x: 1 if re.search(r'\b' + indicator + r'\b', x, re.IGNORECASE) else 0
            )
        
        # Check for HTML content
        self.df['has_html'] = self.df['text'].apply(
            lambda x: 1 if re.search(r'<html|<body|<div|<table', x, re.IGNORECASE) else 0
        )
        
        # Count URLs
        self.df['url_count'] = self.df['text'].apply(
            lambda x: len(re.findall(r'http\S+|www\S+|https\S+', x))
        )
        
        return True
    
    def create_word2vec_embeddings(self, save_model=True):
        """Create Word2Vec embeddings from the emails"""
        if self.df is None:
            return {"error": "No data loaded"}
            
        # Tokenize emails to prepare for Word2Vec
        tokenized_emails = [email.split() for email in self.df['processed_text']]
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec(
            sentences=tokenized_emails,
            vector_size=self.embedding_dim,
            window=5,
            min_count=2,
            workers=4,
            sg=1  # Skip-gram model (use CBOW if sg=0)
        )
        
        # Save the model if requested
        if save_model:
            model_path = os.path.join(app.config['WORD_EMBEDDINGS_FOLDER'], 'word2vec_model.bin')
            self.word2vec_model.save(model_path)
            
        return {"success": True, "vocabulary_size": len(self.word2vec_model.wv.key_to_index)}
    
    def load_word2vec_embeddings(self, model_path=None):
        """Load pre-trained Word2Vec embeddings"""
        if model_path is None:
            model_path = os.path.join(app.config['WORD_EMBEDDINGS_FOLDER'], 'word2vec_model.bin')
            
        if not os.path.exists(model_path):
            return {"error": f"Word2Vec model not found at {model_path}"}
            
        try:
            self.word2vec_model = Word2Vec.load(model_path)
            return {"success": True, "vocabulary_size": len(self.word2vec_model.wv.key_to_index)}
        except Exception as e:
            return {"error": f"Error loading Word2Vec model: {str(e)}"}
    
    def build_vocab(self, texts):
        """Build vocabulary for PyTorch models"""
        # Create a set of all words
        all_words = set()
        for text in texts:
            words = text.split()
            all_words.update(words)
        
        # Create word to index mapping
        word_to_idx = {'<pad>': self.pad_idx, '<unk>': 0}
        for i, word in enumerate(all_words, start=2):  # Start from 2 because 0,1 are special tokens
            word_to_idx[word] = i
            
        self.vocab = word_to_idx
        return word_to_idx
    
    def text_to_indices(self, text, max_length=None):
        """Convert text to indices using the vocabulary"""
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocab first.")
            
        if max_length is None:
            max_length = self.max_sequence_length
            
        words = text.split()
        indices = [self.vocab.get(word, 0) for word in words]  # 0 is <unk>
        
        # Pad or truncate
        if len(indices) < max_length:
            indices = indices + [self.pad_idx] * (max_length - len(indices))
        else:
            indices = indices[:max_length]
            
        return indices
    
    def create_embedding_matrix(self, word_to_idx, word_vector_model):
        """Create an embedding matrix for PyTorch embedding layer"""
        vocab_size = len(word_to_idx)
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))
        
        for word, i in word_to_idx.items():
            try:
                embedding_vector = word_vector_model.wv[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                # Word not in embedding vocabulary
                pass
                
        return embedding_matrix
    
    def prepare_pytorch_data(self, X_train, X_test, y_train, y_test, model_type='lstm'):
        """Prepare data for PyTorch models"""
        # Build vocabulary
        word_to_idx = self.build_vocab(X_train)
        
        if model_type == 'bert':
            # Load BERT tokenizer
            try:
                self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            except:
                # If can't download, try to load from a local path
                self.bert_tokenizer = BertTokenizer.from_pretrained(
                    os.path.join(app.config['WORD_EMBEDDINGS_FOLDER'], 'bert-tokenizer')
                )
                
            # Create PyTorch datasets
            train_dataset = EmailDataset(X_train, y_train, self.bert_tokenizer, self.max_sequence_length)
            test_dataset = EmailDataset(X_test, y_test, self.bert_tokenizer, self.max_sequence_length)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
            
            return train_loader, test_loader, None
            
        else: # LSTM or CNN
            # Convert text to indices
            X_train_indices = [self.text_to_indices(text) for text in X_train]
            X_test_indices = [self.text_to_indices(text) for text in X_test]
            
            # Get sequence lengths (for LSTM padding)
            train_lengths = [len(text.split()) if len(text.split()) < self.max_sequence_length 
                             else self.max_sequence_length for text in X_train]
            test_lengths = [len(text.split()) if len(text.split()) < self.max_sequence_length 
                           else self.max_sequence_length for text in X_test]
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.LongTensor(X_train_indices)
            y_train_tensor = torch.FloatTensor(y_train.values)
            X_test_tensor = torch.LongTensor(X_test_indices)
            y_test_tensor = torch.FloatTensor(y_test.values)
            train_lengths_tensor = torch.LongTensor(train_lengths)
            test_lengths_tensor = torch.LongTensor(test_lengths)
            
            # Create TensorDatasets
            train_dataset = TensorDataset(X_train_tensor, train_lengths_tensor, y_train_tensor)
            test_dataset = TensorDataset(X_test_tensor, test_lengths_tensor, y_test_tensor)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Create embedding matrix if we have a word2vec model
            embedding_matrix = None
            if self.word2vec_model:
                embedding_matrix = self.create_embedding_matrix(word_to_idx, self.word2vec_model)
                
            return train_loader, test_loader, embedding_matrix
    
    def train_pytorch_model(self, model, train_loader, test_loader, epochs=5, model_type='lstm'):
        """Train a PyTorch model"""
        # Define optimizer - using PyTorch's AdamW instead of transformers version
        optimizer = AdamW(model.parameters())
        
        # Define loss function
        criterion = nn.BCEWithLogitsLoss()
        
        # Move model to device
        model = model.to(device)
        
        # Training loop
        best_loss = float('inf')
        best_model = None
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            
            for batch in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                if model_type == 'bert':
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                elif model_type == 'lstm':
                    text, text_lengths, labels = batch
                    text, text_lengths, labels = text.to(device), text_lengths.to(device), labels.to(device)
                    
                    outputs = model(text, text_lengths)
                else:  # CNN
                    text, _, labels = batch
                    text, labels = text.to(device), labels.to(device)
                    
                    outputs = model(text)
                
                # Calculate loss
                loss = criterion(outputs.squeeze(), labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * labels.size(0)
                
            epoch_loss = running_loss / len(train_loader.dataset)
            
            # Evaluation phase
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    if model_type == 'bert':
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        
                        outputs = model(input_ids, attention_mask)
                    elif model_type == 'lstm':
                        text, text_lengths, labels = batch
                        text, text_lengths, labels = text.to(device), text_lengths.to(device), labels.to(device)
                        
                        outputs = model(text, text_lengths)
                    else:  # CNN
                        text, _, labels = batch
                        text, labels = text.to(device), labels.to(device)
                        
                        outputs = model(text)
                        
                    # Calculate loss
                    loss = criterion(outputs.squeeze(), labels)
                    val_loss += loss.item() * labels.size(0)
                    
                    # Get predictions
                    preds = torch.sigmoid(outputs.squeeze()) > 0.5
                    
                    # Save predictions and labels for metrics
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
            val_loss = val_loss / len(test_loader.dataset)
            
            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_preds)
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
            
            # Save the best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model.state_dict().copy()
        
        # Load the best model
        model.load_state_dict(best_model)
        
        # Final evaluation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                if model_type == 'bert':
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                elif model_type == 'lstm':
                    text, text_lengths, labels = batch
                    text, text_lengths, labels = text.to(device), text_lengths.to(device), labels.to(device)
                    
                    outputs = model(text, text_lengths)
                else:  # CNN
                    text, _, labels = batch
                    text, labels = text.to(device), labels.to(device)
                    
                    outputs = model(text)
                    
                # Get predictions
                preds = torch.sigmoid(outputs.squeeze()) > 0.5
                
                # Save predictions and labels for metrics
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, output_dict=True)
        cm = confusion_matrix(all_labels, all_preds)
        
        return model, accuracy, report, cm
    
    def train_model(self, model_type='logistic', hp_tuning=False, tuning_method='grid'):
        """Train a spam detection model"""
        if self.df is None:
            return {"error": "No data loaded"}
        
        # Split data
        X = self.df['processed_text']
        y = self.df['is_spam']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize vectorizer
        vectorizer = TfidfVectorizer(
            max_features=5000, 
            min_df=5, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Select model based on type
        if model_type == 'naive_bayes':
            # Naive Bayes model
            classifier = MultinomialNB()
            model_name = "Naive Bayes"
            
            if hp_tuning:
                # Define hyperparameters to tune
                param_grid = {
                    'classifier__alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
                    'classifier__fit_prior': [True, False]
                }
        
        elif model_type == 'random_forest':
            # Random Forest model
            classifier = RandomForestClassifier(random_state=42)
            model_name = "Random Forest"
            
            if hp_tuning:
                # Define hyperparameters to tune
                param_grid = {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [None, 10, 20, 30],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4]
                }
                
        elif model_type == 'xgboost':
            # XGBoost model
            classifier = xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            model_name = "XGBoost"
            
            if hp_tuning:
                # Define hyperparameters to tune
                param_grid = {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [3, 5, 7, 9],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
                    'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0]
                }
                
        elif model_type == 'lightgbm':
            # LightGBM model
            classifier = lgb.LGBMClassifier(random_state=42)
            model_name = "LightGBM"
            
            if hp_tuning:
                # Define hyperparameters to tune
                param_grid = {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [3, 5, 7, 9],
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__num_leaves': [31, 50, 70, 90],
                    'classifier__subsample': [0.7, 0.8, 0.9, 1.0]
                }
                
        elif model_type == 'voting':
            # Voting Ensemble model
            estimators = [
                ('lr', LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
                ('nb', MultinomialNB()),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('xgb', xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss'))
            ]
            classifier = VotingClassifier(estimators=estimators, voting='soft')
            model_name = "Voting Ensemble"
            
            # Not implementing hyperparameter tuning for voting ensemble due to complexity
            hp_tuning = False
            
        elif model_type == 'stacking':
            # Stacking Ensemble model
            estimators = [
                ('lr', LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
                ('nb', MultinomialNB()),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
            ]
            final_estimator = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')
            classifier = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
            model_name = "Stacking Ensemble"
            
            # Not implementing hyperparameter tuning for stacking ensemble due to complexity
            hp_tuning = False
            
        elif model_type == 'lstm':
            # Create Word2Vec embeddings if not already created
            if self.word2vec_model is None:
                self.create_word2vec_embeddings()
            
            # Prepare data for PyTorch
            train_loader, test_loader, embedding_matrix = self.prepare_pytorch_data(
                X_train, X_test, y_train, y_test, model_type='lstm'
            )
            
            # Initialize model
            vocab_size = len(self.vocab)
            hidden_dim = 128
            output_dim = 1
            n_layers = 2
            bidirectional = True
            dropout = 0.5
            
            model = LSTMClassifier(
                vocab_size, 
                self.embedding_dim, 
                hidden_dim, 
                output_dim, 
                n_layers, 
                bidirectional, 
                dropout, 
                self.pad_idx,
                embedding_matrix
            )
            
            # Train model
            model, accuracy, report, cm = self.train_pytorch_model(
                model, train_loader, test_loader, epochs=5, model_type='lstm'
            )
            
            # Store the model
            self.model = {
                'type': 'pytorch_lstm',
                'model': model,
                'vocab': self.vocab
            }
            model_name = "LSTM with Word Embeddings"
            
            self.trained = True
            
            # Save confusion matrix as image
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Ham', 'Spam'], 
                        yticklabels=['Ham', 'Spam'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            
            # Save plot to memory buffer
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            cm_image = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Extract metrics safely
            metrics = self._safe_extract_metrics(report, accuracy)

            # Compile training stats
            self.training_stats = {
                'model_type': model_name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'confusion_matrix': cm_image,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            
            return self.training_stats
            
        elif model_type == 'cnn':
            # Create Word2Vec embeddings if not already created
            if self.word2vec_model is None:
                self.create_word2vec_embeddings()
            
            # Prepare data for PyTorch
            train_loader, test_loader, embedding_matrix = self.prepare_pytorch_data(
                X_train, X_test, y_train, y_test, model_type='cnn'
            )
            
            # Initialize model
            vocab_size = len(self.vocab)
            n_filters = 100
            filter_sizes = [3, 4, 5]
            output_dim = 1
            dropout = 0.5
            
            model = CNNClassifier(
                vocab_size,
                self.embedding_dim,
                n_filters,
                filter_sizes,
                output_dim,
                dropout,
                self.pad_idx,
                embedding_matrix
            )
            
            # Train model
            model, accuracy, report, cm = self.train_pytorch_model(
                model, train_loader, test_loader, epochs=5, model_type='cnn'
            )
            
            # Store the model
            self.model = {
                'type': 'pytorch_cnn',
                'model': model,
                'vocab': self.vocab
            }
            model_name = "CNN with Word Embeddings"
            
            self.trained = True
            
            # Save confusion matrix as image
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Ham', 'Spam'], 
                        yticklabels=['Ham', 'Spam'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            
            # Save plot to memory buffer
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            cm_image = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Extract metrics safely
            metrics = self._safe_extract_metrics(report, accuracy)

            # Compile training stats
            self.training_stats = {
                'model_type': model_name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'confusion_matrix': cm_image,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            return self.training_stats
            
        elif model_type == 'bert':
            # Prepare data for BERT
            train_loader, test_loader, _ = self.prepare_pytorch_data(
                X_train, X_test, y_train, y_test, model_type='bert'
            )
            
            # Initialize model
            model = BERTClassifier(bert_model_name='bert-base-uncased', dropout=0.1)
            
            # Train model
            model, accuracy, report, cm = self.train_pytorch_model(
                model, train_loader, test_loader, epochs=3, model_type='bert'
            )
            
            # Store the model
            self.model = {
                'type': 'pytorch_bert',
                'model': model,
                'tokenizer': self.bert_tokenizer
            }
            model_name = "BERT Transformer"
            
            self.trained = True
            
            # Save confusion matrix as image
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Ham', 'Spam'], 
                        yticklabels=['Ham', 'Spam'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            
            # Save plot to memory buffer
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            cm_image = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Extract metrics safely
            metrics = self._safe_extract_metrics(report, accuracy)

            # Compile training stats
            self.training_stats = {
                'model_type': model_name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'confusion_matrix': cm_image,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            
            return self.training_stats
            
        else:  # Default to logistic regression
            classifier = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            model_name = "Logistic Regression"
            
            if hp_tuning:
                # Define hyperparameters to tune
                param_grid = {
                    'classifier__C': [0.1, 0.5, 1.0, 5.0, 10.0],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['liblinear', 'saga']
                }
        
        # Create the pipeline with TF-IDF vectorizer and classifier
        self.model = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        
        # If hyperparameter tuning is enabled, use grid or random search
        if hp_tuning and model_type not in ['voting', 'stacking', 'lstm', 'cnn', 'bert']:
            if tuning_method == 'random':
                search = RandomizedSearchCV(
                    self.model,
                    param_distributions=param_grid,
                    n_iter=10,
                    cv=5,
                    scoring='f1',
                    n_jobs=-1,
                    random_state=42
                )
                # Train with hyperparameter search
                search.fit(X_train, y_train)
                self.model = search.best_estimator_
                
                # Store best parameters for reporting
                best_params = search.best_params_
                
            else:  # grid search
                search = GridSearchCV(
                    self.model,
                    param_grid=param_grid,
                    cv=5,
                    scoring='f1',
                    n_jobs=-1
                )
                # Train with hyperparameter search
                search.fit(X_train, y_train)
                self.model = search.best_estimator_
                
                # Store best parameters for reporting
                best_params = search.best_params_
        else:
            # Regular training without hyperparameter tuning
            self.model.fit(X_train, y_train)
            best_params = None
        
        self.vectorizer = vectorizer
        self.trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Save confusion matrix as image
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Ham', 'Spam'], 
                    yticklabels=['Ham', 'Spam'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # Save plot to memory buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        cm_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Extract metrics safely
        metrics = self._safe_extract_metrics(report, accuracy)

        # Compile training stats
        self.training_stats = {
            'model_type': model_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'confusion_matrix': cm_image,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        return self.training_stats
    
    def save_model(self, filename=None):
        """Save the trained model to disk"""
        if not self.trained:
            return {"error": "No trained model to save"}
        
        if filename is None:
            filename = f"spam_model_{uuid.uuid4().hex[:8]}.pkl"
        
        model_path = os.path.join(app.config['MODEL_FOLDER'], filename)
        
        # Check if model is a PyTorch model
        if isinstance(self.model, dict) and 'type' in self.model and self.model['type'].startswith('pytorch'):
            pytorch_model_path = os.path.join(app.config['MODEL_FOLDER'], filename.replace('.pkl', '.pt'))
            
            # For PyTorch models, save using torch.save
            if self.model['type'] == 'pytorch_lstm' or self.model['type'] == 'pytorch_cnn':
                # Save the model
                torch.save({
                    'type': self.model['type'],
                    'state_dict': self.model['model'].state_dict(),
                    'vocab': self.vocab
                }, pytorch_model_path)
                
            elif self.model['type'] == 'pytorch_bert':
                # Save the BERT model
                torch.save({
                    'type': self.model['type'],
                    'state_dict': self.model['model'].state_dict()
                }, pytorch_model_path)
                
                # Save the tokenizer
                tokenizer_path = os.path.join(app.config['MODEL_FOLDER'], filename.replace('.pkl', '_tokenizer'))
                self.bert_tokenizer.save_pretrained(tokenizer_path)
                
            return {"success": True, "filename": filename.replace('.pkl', '.pt')}
        else:
            # Standard scikit-learn model
            joblib.dump(self.model, model_path)
            return {"success": True, "filename": filename}
    
    def load_model(self, filename):
        """Load a previously trained model"""
        model_path = os.path.join(app.config['MODEL_FOLDER'], filename)
        
        if not os.path.exists(model_path):
            return {"error": f"Model file not found: {filename}"}
        
        try:
            # Check if it's a PyTorch model (.pt extension)
            if filename.endswith('.pt'):
                # Load PyTorch model
                checkpoint = torch.load(model_path, map_location=device)
                
                if checkpoint['type'] == 'pytorch_lstm':
                    # Recreate LSTM model
                    self.vocab = checkpoint['vocab']
                    vocab_size = len(self.vocab)
                    hidden_dim = 128
                    output_dim = 1
                    n_layers = 2
                    bidirectional = True
                    dropout = 0.5
                    
                    model = LSTMClassifier(
                        vocab_size, 
                        self.embedding_dim, 
                        hidden_dim, 
                        output_dim, 
                        n_layers, 
                        bidirectional, 
                        dropout, 
                        self.pad_idx
                    )
                    
                    # Load state dict
                    model.load_state_dict(checkpoint['state_dict'])
                    
                    # Store model
                    self.model = {
                        'type': 'pytorch_lstm',
                        'model': model,
                        'vocab': self.vocab
                    }
                    
                elif checkpoint['type'] == 'pytorch_cnn':
                    # Recreate CNN model
                    self.vocab = checkpoint['vocab']
                    vocab_size = len(self.vocab)
                    n_filters = 100
                    filter_sizes = [3, 4, 5]
                    output_dim = 1
                    dropout = 0.5
                    
                    model = CNNClassifier(
                        vocab_size,
                        self.embedding_dim,
                        n_filters,
                        filter_sizes,
                        output_dim,
                        dropout,
                        self.pad_idx
                    )
                    
                    # Load state dict
                    model.load_state_dict(checkpoint['state_dict'])
                    
                    # Store model
                    self.model = {
                        'type': 'pytorch_cnn',
                        'model': model,
                        'vocab': self.vocab
                    }
                    
                elif checkpoint['type'] == 'pytorch_bert':
                    # Load BERT tokenizer
                    tokenizer_path = model_path.replace('.pt', '_tokenizer')
                    if os.path.exists(tokenizer_path):
                        self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
                    else:
                        # Try to load from Hugging Face
                        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                    
                    # Recreate BERT model
                    model = BERTClassifier(bert_model_name='bert-base-uncased', dropout=0.1)
                    
                    # Load state dict
                    model.load_state_dict(checkpoint['state_dict'])
                    
                    # Store model
                    self.model = {
                        'type': 'pytorch_bert',
                        'model': model,
                        'tokenizer': self.bert_tokenizer
                    }
            else:
                # Standard scikit-learn model
                self.model = joblib.load(model_path)
            
            self.trained = True
            return {"success": True}
        except Exception as e:
            return {"error": f"Error loading model: {str(e)}"}
    
    def predict(self, email_text):
        """Predict if an email is spam or ham"""
        if not self.trained:
            return {"error": "No trained model available"}
        
        # Preprocess the email
        processed_email = self.preprocess_email(email_text)
        
        # Check if it's a PyTorch model
        if isinstance(self.model, dict) and 'type' in self.model and self.model['type'].startswith('pytorch'):
            # Set the model to evaluation mode
            self.model['model'].eval()
            
            if self.model['type'] == 'pytorch_lstm':
                # Convert text to indices
                indices = self.text_to_indices(processed_email)
                text_tensor = torch.LongTensor([indices]).to(device)
                
                # Get text length
                text_length = min(len(processed_email.split()), self.max_sequence_length)
                text_length_tensor = torch.LongTensor([text_length]).to(device)
                
                # Get prediction
                with torch.no_grad():
                    output = self.model['model'](text_tensor, text_length_tensor)
                    prediction_proba = torch.sigmoid(output).item()
                    prediction = int(prediction_proba > 0.5)
                
            elif self.model['type'] == 'pytorch_cnn':
                # Convert text to indices
                indices = self.text_to_indices(processed_email)
                text_tensor = torch.LongTensor([indices]).to(device)
                
                # Get prediction
                with torch.no_grad():
                    output = self.model['model'](text_tensor)
                    prediction_proba = torch.sigmoid(output).item()
                    prediction = int(prediction_proba > 0.5)
                
            elif self.model['type'] == 'pytorch_bert':
                # Tokenize the text
                encoding = self.bert_tokenizer.encode_plus(
                    processed_email,
                    add_special_tokens=True,
                    max_length=self.max_sequence_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                # Get prediction
                with torch.no_grad():
                    input_ids = encoding['input_ids'].to(device)
                    attention_mask = encoding['attention_mask'].to(device)
                    
                    output = self.model['model'](input_ids, attention_mask)
                    prediction_proba = torch.sigmoid(output).item()
                    prediction = int(prediction_proba > 0.5)
            
            # Prepare result
            result = {
                'is_spam': bool(prediction),
                'spam_probability': float(prediction_proba),
                'classification': 'SPAM' if prediction else 'HAM',
                'model_type': self.model['type']
            }
            
        else:
            # Standard scikit-learn model
            # Make prediction
            prediction = self.model.predict([processed_email])[0]
            probability = self.model.predict_proba([processed_email])[0][1]
            
            result = {
                'is_spam': bool(prediction),
                'spam_probability': float(probability),
                'classification': 'SPAM' if prediction else 'HAM'
            }
            
            # Extract feature importance for logistic regression
            if isinstance(self.model.named_steps['classifier'], LogisticRegression):
                # Get the vectorizer from the pipeline
                vectorizer = self.model.named_steps['vectorizer']
                classifier = self.model.named_steps['classifier']
                
                # Transform the email text
                X_transformed = vectorizer.transform([processed_email])
                
                # Get feature names
                feature_names = vectorizer.get_feature_names_out()
                
                # Get coefficients
                coefficients = classifier.coef_[0]
                
                # Get the most important features in this email
                non_zero = X_transformed.nonzero()[1]
                
                # Create a list of (feature, coefficient, presence) tuples
                feature_importance = []
                for feature_idx in non_zero:
                    feature_importance.append({
                        'feature': feature_names[feature_idx],
                        'coefficient': float(coefficients[feature_idx]),
                        'presence': float(X_transformed[0, feature_idx])
                    })
                
                # Sort by absolute coefficient value
                feature_importance = sorted(
                    feature_importance, 
                    key=lambda x: abs(x['coefficient']), 
                    reverse=True
                )
                
                # Add top features to the result
                result['important_features'] = feature_importance[:10]
            
            # Extract feature importance for tree-based models
            elif any(isinstance(self.model.named_steps['classifier'], model_type) 
                    for model_type in [RandomForestClassifier, xgb.XGBClassifier, lgb.LGBMClassifier]):
                # Get the vectorizer from the pipeline
                vectorizer = self.model.named_steps['vectorizer']
                classifier = self.model.named_steps['classifier']
                
                # Transform the email text
                X_transformed = vectorizer.transform([processed_email])
                
                # Get feature names
                feature_names = vectorizer.get_feature_names_out()
                
                # Get feature importances
                if isinstance(classifier, RandomForestClassifier):
                    importances = classifier.feature_importances_
                elif isinstance(classifier, xgb.XGBClassifier):
                    importances = classifier.feature_importances_
                elif isinstance(classifier, lgb.LGBMClassifier):
                    importances = classifier.feature_importances_
                
                # Get the features present in this email
                non_zero = X_transformed.nonzero()[1]
                
                # Create a list of (feature, importance, presence) tuples
                feature_importance = []
                for feature_idx in non_zero:
                    feature_importance.append({
                        'feature': feature_names[feature_idx],
                        'importance': float(importances[feature_idx]),
                        'presence': float(X_transformed[0, feature_idx])
                    })
                
                # Sort by importance value
                feature_importance = sorted(
                    feature_importance, 
                    key=lambda x: x['importance'], 
                    reverse=True
                )
                
                # Add top features to the result
                result['important_features'] = feature_importance[:10]
        
        # Email statistics
        result['email_length'] = len(email_text)
        result['word_count'] = len(email_text.split())
        result['has_html'] = 1 if re.search(r'<html|<body|<div|<table', email_text, re.IGNORECASE) else 0
        result['url_count'] = len(re.findall(r'http\S+|www\S+|https\S+', email_text))
        
        return result
    
    # Add this method to the SpamDetector class in app.py

    def compare_models(self):
        """Train and compare multiple models on the same dataset"""
        if self.df is None:
            return {"error": "No data loaded"}
        
        # Models to compare
        models_to_compare = [
            'logistic',         # Traditional ML
            'naive_bayes',      # Traditional ML
            'random_forest',    # Traditional ML
            'xgboost',          # Gradient Boosting
            'lightgbm',         # Gradient Boosting
            'voting',           # Ensemble 
            'stacking',         # Ensemble
            'lstm',             # Deep Learning
            'cnn'               # Deep Learning
            # Note: BERT is excluded due to being resource intensive
        ]
        
        # Storage for results
        comparison_results = {
            'models': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'training_time': [],
            'model_type': []
        }
        
        # Split data
        X = self.df['processed_text']
        y = self.df['is_spam']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create Word2Vec embeddings if not already created (for deep learning models)
        if self.word2vec_model is None:
            self.create_word2vec_embeddings()
        
        # Train and evaluate each model
        for model_type in models_to_compare:
            print(f"\nTraining model: {model_type}")
            
            # Record training time
            start_time = time.time()
            
            # Train the model
            stats = self.train_model(model_type=model_type, hp_tuning=False)
            
            # Record results if successful
            if 'error' not in stats:
                training_time = time.time() - start_time
                
                # Categorize model types
                category = 'Traditional ML'
                if model_type in ['xgboost', 'lightgbm']:
                    category = 'Gradient Boosting'
                elif model_type in ['voting', 'stacking']:
                    category = 'Ensemble Methods'
                elif model_type in ['lstm', 'cnn', 'bert']:
                    category = 'Deep Learning'
                
                # Store results
                comparison_results['models'].append(stats.get('model_type', model_type))
                comparison_results['accuracy'].append(stats.get('accuracy', 0.0))
                comparison_results['precision'].append(stats.get('precision', 0.0))
                comparison_results['recall'].append(stats.get('recall', 0.0))
                comparison_results['f1'].append(stats.get('f1', 0.0))
                comparison_results['training_time'].append(training_time)
                comparison_results['model_type'].append(category)

                print(f"  Completed: {stats.get('model_type', model_type)}")
                print(f"  Accuracy: {stats.get('accuracy', 0.0):.4f}")
                print(f"  F1 Score: {stats.get('f1', 0.0):.4f}")
                print(f"  Training Time: {training_time:.2f}s")
            else:
                print(f"  Error: {stats['error']}")
        
        # Create visualization - Metrics comparison
        plt.figure(figsize=(12, 8))
        bar_width = 0.2
        index = np.arange(len(comparison_results['models']))
        
        plt.bar(index, comparison_results['accuracy'], bar_width, label='Accuracy', color='#3498db')
        plt.bar(index + bar_width, comparison_results['precision'], bar_width, label='Precision', color='#2ecc71')
        plt.bar(index + bar_width * 2, comparison_results['recall'], bar_width, label='Recall', color='#e74c3c')
        plt.bar(index + bar_width * 3, comparison_results['f1'], bar_width, label='F1 Score', color='#f39c12')
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Performance Metrics Comparison')
        plt.xticks(index + bar_width * 1.5, comparison_results['models'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save plot to memory buffer
        metrics_buf = BytesIO()
        plt.savefig(metrics_buf, format='png')
        plt.close()
        metrics_buf.seek(0)
        metrics_image = base64.b64encode(metrics_buf.getvalue()).decode('utf-8')
        
        # Create visualization - Training Time comparison
        plt.figure(figsize=(10, 6))
        colors = {
            'Traditional ML': '#3498db',
            'Gradient Boosting': '#2ecc71',
            'Ensemble Methods': '#e74c3c',
            'Deep Learning': '#f39c12'
        }
        
        bar_colors = [colors[category] for category in comparison_results['model_type']]
        
        bars = plt.bar(comparison_results['models'], comparison_results['training_time'], color=bar_colors)
        
        plt.xlabel('Models')
        plt.ylabel('Training Time (seconds)')
        plt.title('Training Time Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add a legend for model categories
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[cat], label=cat) for cat in set(comparison_results['model_type'])]
        plt.legend(handles=legend_elements)
        
        # Save plot to memory buffer
        time_buf = BytesIO()
        plt.savefig(time_buf, format='png')
        plt.close()
        time_buf.seek(0)
        time_image = base64.b64encode(time_buf.getvalue()).decode('utf-8')
        
        # Create visualization - F1 Score vs. Training Time
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            comparison_results['training_time'], 
            comparison_results['f1'], 
            c=[colors[category] for category in comparison_results['model_type']],
            s=100,
            alpha=0.7
        )
        
        # Add model names as annotations
        for i, model in enumerate(comparison_results['models']):
            plt.annotate(
                model.split()[0],  # Just take the first word for clarity
                (comparison_results['training_time'][i], comparison_results['f1'][i]),
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('F1 Score')
        plt.title('Model Efficiency: F1 Score vs. Training Time')
        plt.legend(handles=legend_elements)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot to memory buffer
        efficiency_buf = BytesIO()
        plt.savefig(efficiency_buf, format='png')
        plt.close()
        efficiency_buf.seek(0)
        efficiency_image = base64.b64encode(efficiency_buf.getvalue()).decode('utf-8')
        
        # Return the complete comparison results
        return {
            'raw_data': comparison_results,
            'metrics_image': metrics_image,
            'time_image': time_image,
            'efficiency_image': efficiency_image
        }

# Initialize the spam detector
spam_detector = SpamDetector(data_dir=app.config['DATA_FOLDER'])

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_available_models():
    models = []
    for filename in os.listdir(app.config['MODEL_FOLDER']):
        if filename.endswith('.pkl') or filename.endswith('.pt'):
            models.append(filename)
    return models

# Routes
@app.route('/')
def index():
    # Check if a model is loaded
    model_loaded = spam_detector.trained
    available_models = get_available_models()
    
    return render_template('index.html', 
                           model_loaded=model_loaded,
                           available_models=available_models)

@app.route('/load-data', methods=['GET', 'POST'])
def load_data():
    if request.method == 'POST':
        # Check if the data directory exists
        if not os.path.exists(app.config['DATA_FOLDER']):
            flash('Data directory not found', 'error')
            return redirect(url_for('index'))
        
        # Load data
        stats = spam_detector.load_data()
        
        if stats is None:
            flash('No data found in the data directory', 'error')
            return redirect(url_for('index'))
        
        # Extract features
        spam_detector.extract_features()
        
        flash('Data loaded successfully', 'success')
        return render_template('data_stats.html', stats=stats)
    
    return render_template('load_data.html')

@app.route('/train-model', methods=['GET', 'POST'])
def train_model():
    if request.method == 'POST':
        if spam_detector.df is None:
            flash('Please load data first', 'error')
            return redirect(url_for('load_data'))
        
        model_type = request.form.get('model_type', 'logistic')
        hp_tuning = 'hp_tuning' in request.form
        tuning_method = request.form.get('tuning_method', 'grid')
        
        # Train the model
        stats = spam_detector.train_model(model_type, hp_tuning, tuning_method)
        
        if 'error' in stats:
            flash(stats['error'], 'error')
            return redirect(url_for('index'))
        
        # Save the model
        save_result = spam_detector.save_model()
        
        if 'error' in save_result:
            flash(save_result['error'], 'warning')
        else:
            flash(f'Model trained and saved as {save_result["filename"]}', 'success')
        
        return render_template('model_trained.html', stats=stats)
    
    return render_template('train_model.html')

@app.route('/load-model', methods=['POST'])
def load_model():
    model_file = request.form.get('model_file')
    
    if not model_file:
        flash('No model selected', 'error')
        return redirect(url_for('index'))
    
    result = spam_detector.load_model(model_file)
    
    if 'error' in result:
        flash(result['error'], 'error')
    else:
        flash(f'Model loaded: {model_file}', 'success')
    
    return redirect(url_for('index'))

@app.route('/detect-spam', methods=['GET', 'POST'])
def detect_spam():
    if request.method == 'POST':
        if not spam_detector.trained:
            flash('Please train or load a model first', 'error')
            return redirect(url_for('index'))
        
        email_text = request.form.get('email_text', '')
        
        if not email_text:
            flash('Please enter email text', 'error')
            return render_template('detect_spam.html')
        
        # Analyze the email
        result = spam_detector.predict(email_text)
        
        if 'error' in result:
            flash(result['error'], 'error')
            return render_template('detect_spam.html')
        
        return render_template('spam_result.html', result=result, email_text=email_text)
    
    return render_template('detect_spam.html')

@app.route('/upload-email', methods=['POST'])
def upload_email():
    if not spam_detector.trained:
        return jsonify({'error': 'Please train or load a model first'})
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                email_text = f.read()
            
            # Analyze the email
            result = spam_detector.predict(email_text)
            
            if 'error' in result:
                return jsonify({'error': result['error']})
            
            # Add file info to result
            result['filename'] = filename
            
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': f'Error analyzing file: {str(e)}'})
    
    return jsonify({'error': 'File type not allowed'})

@app.route('/create-word-embeddings', methods=['POST'])
def create_word_embeddings():
    if spam_detector.df is None:
        flash('Please load data first', 'error')
        return redirect(url_for('load_data'))
    
    result = spam_detector.create_word2vec_embeddings()
    
    if 'error' in result:
        flash(result['error'], 'error')
    else:
        flash(f'Word embeddings created with vocabulary size: {result["vocabulary_size"]}', 'success')
    
    return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not spam_detector.trained:
        return jsonify({'error': 'No trained model available'})
    
    data = request.get_json()
    
    if not data or 'email' not in data:
        return jsonify({'error': 'No email text provided'})
    
    email_text = data['email']
    
    # Analyze the email
    result = spam_detector.predict(email_text)
    
    if 'error' in result:
        return jsonify({'error': result['error']})
    
    return jsonify(result)

@app.route('/compare-models', methods=['GET', 'POST'])
def compare_models():
    if request.method == 'POST':
        if spam_detector.df is None:
            flash('Please load data first', 'error')
            return redirect(url_for('load_data'))
        
        # Run model comparison
        results = spam_detector.compare_models()
        
        if 'error' in results:
            flash(results['error'], 'error')
            return redirect(url_for('index'))
        
        flash('Model comparison completed successfully', 'success')
        return render_template('model_comparison.html', results=results)
    
    return render_template('compare_models.html')

if __name__ == '__main__':
    app.run(debug=True)