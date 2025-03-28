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
from datetime import datetime  # Add this import
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import joblib
import uuid
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.jinja_env.globals['now'] = datetime.now  # Add this line to make now() available
app.secret_key = 'spam_detector_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['DATA_FOLDER'] = 'data'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'json', 'eml'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)

# Spam Detector Class
class SpamDetector:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.model = None
        self.vectorizer = None
        self.trained = False
        self.df = None
        self.training_stats = {}
    
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
    
    def train_model(self, model_type='logistic'):
        """Train a spam detection model"""
        if self.df is None:
            return {"error": "No data loaded"}
        
        # Split data
        X = self.df['processed_text']
        y = self.df['is_spam']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Select model type
        if model_type == 'naive_bayes':
            classifier = MultinomialNB()
            model_name = "Naive Bayes"
        elif model_type == 'random_forest':
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            model_name = "Random Forest"
        else:
            classifier = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            model_name = "Logistic Regression"
        
        # Create pipeline
        vectorizer = TfidfVectorizer(
            max_features=5000, 
            min_df=5, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.model = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        
        # Train the model
        self.model.fit(X_train, y_train)
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
        
        # Compile training stats
        self.training_stats = {
            'model_type': model_name,
            'accuracy': accuracy,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1': report['1']['f1-score'],
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
        joblib.dump(self.model, model_path)
        
        return {"success": True, "filename": filename}
    
    def load_model(self, filename):
        """Load a previously trained model"""
        model_path = os.path.join(app.config['MODEL_FOLDER'], filename)
        
        if not os.path.exists(model_path):
            return {"error": f"Model file not found: {filename}"}
        
        try:
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
        
        # Email statistics
        result['email_length'] = len(email_text)
        result['word_count'] = len(email_text.split())
        result['has_html'] = 1 if re.search(r'<html|<body|<div|<table', email_text, re.IGNORECASE) else 0
        result['url_count'] = len(re.findall(r'http\S+|www\S+|https\S+', email_text))
        
        return result

# Initialize the spam detector
spam_detector = SpamDetector(data_dir=app.config['DATA_FOLDER'])

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_available_models():
    models = []
    for filename in os.listdir(app.config['MODEL_FOLDER']):
        if filename.endswith('.pkl'):
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
        
        # Train the model
        stats = spam_detector.train_model(model_type)
        
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

if __name__ == '__main__':
    app.run(debug=True)