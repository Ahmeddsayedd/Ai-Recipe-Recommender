# 🌙 AI Mood-Based Recipe Recommender
Emotion-aware intelligent recipe recommendation system powered by NLP mood detection and ingredient similarity scoring.

---

## 📌 Overview

This system suggests recipes based on your **emotional state + available ingredients** using TF-IDF text modeling, classical ML classification, and a dual-score ranking formula:

score = α × P(mood|recipe) + (1 - α) × ingredient_similarity

yaml
Copy code

---

## ✨ Features
J
### 🧠 Mood Classification
- Text-based mood detection (10 emotion classes)
- Real-time inference with probability confidence
- TF-IDF vectorization + ML classifiers

### 🍽 Smart Recipe Recommendation
- Ingredient-to-recipe similarity scoring
- Adjustable preference weight **α** (mood vs ingredients)
- Ranked recipe list in UI

### ⚙ Machine Learning Models
| Model | Purpose | Notes |
|-------|---------|-------|
| Logistic Regression | Baseline | Stable, interpretable |
| Random Forest | Nonlinear reasoning | 100 estimators |
| Naive Bayes | Probabilistic | Fast inference |
| SVM (Linear) | Margin maximization | Strong text performance |

### 🖥 Stack
- **Backend:** FastAPI
- **Frontend:** HTML, CSS, JavaScript
- **Offline inference** (no cloud required)

---

## 📂 Project Structure

ai_recipe_recommender/
├── backend/
│ ├── train_and_export_simple.py
│ ├── app_fastapi_simple.py
│ ├── eda.py
│ ├── test_preprocessing_fixed.py
│ ├── requirements.txt
│ └── models/
│ ├── best_model.pkl
│ ├── tfidf_vectorizer.pkl
│ ├── label_map.json
│ └── *.png
├── data/
│ └── realistic_recipe_mood_dataset.csv
├── frontend/
│ ├── index.html
│ ├── style.css
│ └── script.js
├── setup_and_run.bat
└── README.md



---

## ⚠ Python Version Support

| Version | Status |
|---------|--------|
| **3.11 / 3.12** | ✔ Fully supported |
| 3.13 | ⚠ Untested |
| **3.14+** | ❌ Breaks NumPy/pandas builds due to distutils removal |

---

## 🚀 Quick Launch (Recommended)

### 🛠 Manual Setup Step-by-Step
1️⃣ Create Virtual Environment
powershell
Copy code
cd "E:\Ai Recipe Recommender"
python -m venv venv
.\venv\Scripts\Activate.ps1
2️⃣ Install Dependencies
bash
Copy code
pip install -r backend/requirements.txt
📊 Data Processing & Model Training
1. Run EDA (Optional)
bash
Copy code
py backend/eda.py
2. Run Tests
bash
Copy code
py -m pytest backend/test_preprocessing_fixed.py -v
3. Train Models
bash
Copy code
py backend/train_and_export_simple.py --data data/realistic_recipe_mood_dataset.csv --out backend/models
Model outputs include:

best_model.pkl

tfidf_vectorizer.pkl

label_map.json

🌐 Run the API Server
bash
Copy code
py backend/app_fastapi_simple.py
API Base URL:

cpp
Copy code
http://127.0.0.1:8000
🖥 Frontend Usage
Open directly in browser:

bash
Copy code
frontend/index.html
UI includes:

Text mood input

Ingredient list input

α slider (mood/ingredient weight)

Ranked recipe cards

📡 API Endpoints
POST /predict
Predict mood from text.

json
Copy code
{
  "text": "I feel drained and stressed after work."
}
POST /recommend
Recipe suggestions using mood + ingredients.

json
Copy code
{
  "mood_text": "Feeling happy and energetic",
  "ingredients": "chicken, pasta, basil",
  "top_n": 5,
  "alpha": 0.7
}
GET /health
json
Copy code
{
  "status": "healthy"
}
📈 Evaluation Summary
Metric	Score
Accuracy	0.883
Macro F1	0.876
Weighted F1	0.883

Outputs include confusion matrix and comparison visuals.

❗ Known Limitations
No dietary filters (vegan, halal, gluten-free)

Dataset contains synthetic mood labeling

TF-IDF capped at 2000 features

Python 3.14 incompatible

🚧 Future Improvements
Dietary & allergen filtering

Personalized taste tracking

Transformer-based mood modeling (BERT)

Mobile deployment (Flutter / React Native)

❓ FAQ
Q: Can I retrain with more recipes?
A: Yes — add to dataset CSV and rerun training script.

Q: Do I need GPU?
A: No, CPU inference is fast.

Q: Supported OS?
A: Windows, macOS, Linux.