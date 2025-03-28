# Spam Detective

![Spam Detective Logo](https://img.shields.io/badge/Spam-Detective-blue?style=for-the-badge&logo=flask)

A sophisticated email spam detection system with a modern web interface built using Flask and machine learning. This application helps classify emails as spam or legitimate (ham) with detailed analysis and explanation.

## Features

- **Multiple Classification Algorithms**:
  - Logistic Regression (with feature importance explanations)
  - Naive Bayes (efficient for text classification)
  - Random Forest (robust against overfitting)

- **Interactive User Interface**:
  - Clean, responsive design using Bootstrap
  - Drag & drop email file uploads
  - Text input for pasting email content
  - Detailed result visualization

- **Advanced Analysis**:
  - Email content processing and feature extraction
  - Key spam indicator identification
  - Detailed classification reports with confidence scores
  - Visual explanation of classification decisions

- **Model Management**:
  - Train custom models on your own data
  - Save and load models for future use
  - Performance visualization with metrics dashboard
  - Confusion matrix visualization

- **Data Management**:
  - Support for loading custom email datasets
  - Statistics and visualization of dataset composition
  - Preprocessing and feature extraction

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/spam-detective.git
   cd spam-detective
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create necessary directories**:
   ```bash
   mkdir -p uploads models data
   ```

5. **Prepare the data directory structure**:
   The application expects data to be organized as follows:
   ```
   data/
   ├── easy-ham-1/
   ├── easy-ham-2/
   ├── hard-ham-1/
   ├── spam-1/
   └── spam-2/
   ```

   Each directory should contain email files in JSON format with the following structure:
   ```json
   {
     "id": "00001",
     "group": "easy-ham-1",
     "checksum": {
       "type": "MD5",
       "value": "7c53336b37003a9286aba55d2945844c"
     },
     "text": "Full email content including headers and body..."
   }
   ```

## Usage

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Access the web interface**:
   Open a browser and navigate to `http://127.0.0.1:5000/`

3. **Workflow**:
   - Load email dataset
   - Train a spam detection model
   - Detect spam in individual emails

### API Usage

The application provides a REST API for programmatic spam detection:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"email": "Your email content here"}'
```

Response format:
```json
{
  "is_spam": true,
  "spam_probability": 0.95,
  "classification": "SPAM",
  "important_features": [
    {"feature": "free", "coefficient": 1.23, "presence": 1.0},
    ...
  ],
  "email_length": 1250,
  "word_count": 213,
  "has_html": 1,
  "url_count": 3
}
```

## Project Structure

```
spam-detective/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── models/                 # Saved trained models
├── uploads/                # Temporary uploaded files
├── data/                   # Email datasets
└── templates/              # HTML templates
    ├── base.html           # Base template with navigation
    ├── index.html          # Home page
    ├── load_data.html      # Data loading interface
    ├── data_stats.html     # Dataset statistics
    ├── train_model.html    # Model training
    ├── model_trained.html  # Training results
    ├── detect_spam.html    # Spam detection interface
    └── spam_result.html    # Analysis results
```

## Technologies Used

- **Backend**:
  - Flask (Web framework)
  - scikit-learn (Machine learning)
  - pandas (Data processing)
  - NumPy (Numerical operations)
  - Matplotlib & Seaborn (Visualization)

- **Frontend**:
  - Bootstrap 5 (UI framework)
  - Chart.js (Interactive charts)
  - Font Awesome (Icons)
  - jQuery (DOM manipulation)

## Screenshots

<details>
<summary>Click to expand</summary>

### Home Page
[Home page screenshot placeholder]

### Data Statistics
[Data statistics screenshot placeholder]

### Model Training
[Model training screenshot placeholder]

### Spam Detection
[Spam detection screenshot placeholder]

### Analysis Results
[Analysis results screenshot placeholder]
</details>

## Development

### Adding Custom Models

To add a new classification algorithm:

1. Modify the `train_model` method in the `SpamDetector` class in `app.py`
2. Add the new algorithm option to the `train_model.html` template
3. Update the model comparison logic if needed

### Future Enhancements

- Email attachment analysis
- Integration with email clients
- Real-time email monitoring
- Advanced feature engineering options
- User accounts and saved analysis history

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The spam email classification dataset
- scikit-learn for machine learning tools
- Flask for the web framework
- The open-source community for various libraries

---

Made with ❤️ by Cluade 3.7 (Extend Thinking), Arun Prasad T D, Ganesh Sundhar S, Hari Krishnan N, Shruthikaa V