# AI Recipe Recommender 🍲🤖

A full-stack application that recommends recipes based on user mood and available ingredients using machine learning and FastAPI.

---

## 🚀 Quick Start

### Step 1: Open Project Folder
```bash
cd /d "E:\Ai Recipe Recommender"
```
### Step 2: Set Up Environment
```bash
# Create virtual environment
py -m venv venv

# Activate it (CMD)
venv\Scripts\activate

# Or in PowerShell
.\venv\Scripts\Activate.ps1

# Install dependencies
py -m pip install -r requirements.txt
```
### Step 3: Run Project
```bash
# Exploratory Data Analysis
py eda.py

# Run unit tests
py -m pytest test_preprocessing.py -v

# Train ML models
py train_and_export.py --data ../data/realistic_recipe_mood_dataset.csv --out models

# Start FastAPI backend
py app_fastapi.py
```

### Step 4: Open Frontend
Open frontend/index.html in your browser while the API is running.

## ⚠️ Python Version Notice
Use Python 3.11 or 3.12 only.
Do NOT use Python 3.14+, it causes module errors (e.g., distutils.msvccompiler, NumPy issues).

### Check version:

```bash
python --version
# Should be 3.11.x or 3.12.x
```

## 📁 Project Structure
```powershell
ai_recipe_recommender/
├── backend/
│   ├── train_and_export.py       # Trains 4 ML models
│   ├── app_fastapi.py            # REST API
│   ├── eda.py                    # Data analysis & visualizations
│   ├── test_preprocessing.py     # 9 unit tests
│   ├── requirements.txt          # Dependencies
│   └── models/                   # Auto-created after training
├── data/
│   └── realistic_recipe_mood_dataset.csv  # 400+ recipes
├── frontend/
│   ├── index.html                # Main UI
│   ├── style.css                 # Styling
│   └── script.js                 # API interaction
├── eda_output/                   # Auto-generated charts & reports
└── README.md                     # This file
```
## 🔧 Features

### Backend
- 4 ML models: Logistic Regression, Random Forest, Naive Bayes, SVM Linear
- Best model auto-selected based on F1-score
- Text processing: Title + Ingredients + Instructions → TF-IDF (2000 dims)

### Recommendation Algorithm
Score = α × Mood_Match + (1 - α) × Ingredient_Match
α = user slider (0–1)

Mood_Match = model confidence
Ingredient_Match = Jaccard similarity


**Performance:**
- Accuracy: 88.3%
- Macro F1-Score: 0.876
- Processing speed: <1 second per request

### Frontend
- Mood input (text or quick buttons)
- Ingredient input (comma-separated)
- Mood/ingredient balance slider (α)
- Recipe cards with matched ingredients highlighted

---

## 🌐 API Endpoints
- **POST `/predict`** – Predict mood from text  
- **POST `/recommend`** – Get recipe recommendations  
- **GET `/health`** – API status  
- **GET `/moods`** – List of available moods  

---

## 🧪 Testing & Evaluation
- 9/9 unit tests passed  
- Visual outputs: mood distribution, top ingredients, title word cloud, confusion matrix, model comparison  

---

## 🎯 Project Highlights
- Complete ML pipeline: loading → preprocessing → feature extraction → training → evaluation → deployment  
- Multiple algorithms, automatic best model selection, and visual performance comparison  
- Full-stack app: FastAPI backend + HTML/JS frontend  
- Professional quality: unit tests, charts, API documentation, error handling  

<div align="center">
🚀 **Ready to Use!** Follow exact steps in Quick Start and open `frontend/index.html` in your browser.
</div>
