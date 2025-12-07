"""
FastAPI server for mood prediction and recipe recommendation.
Run with: python app_fastapi.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import joblib
import json
import numpy as np
import pandas as pd
import uvicorn
import os
from datetime import datetime
import hashlib

# Try to import TensorFlow for NN support
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Configuration
ARTIFACT_DIR = "models"  # relative to backend folder
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

app = FastAPI(
    title="AI Mood Recipe Recommender API",
    description="ML-powered recipe recommendations based on mood and ingredients",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS setup
origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "*"  # For development only
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded artifacts
vectorizer = None
best_model = None
nn_model = None
label_maps = None
recipes_df = None
INDEX_TO_LABEL = {}
LABEL_TO_INDEX = {}
CLASSES = []

class TextRequest(BaseModel):
    text: str

class RecommendRequest(BaseModel):
    mood: Optional[str] = None
    mood_text: Optional[str] = None
    ingredients: str = ""
    top_n: int = 5
    alpha: float = 0.7  # Mood weight vs ingredient weight

class FeedbackRequest(BaseModel):
    recipe_title: str
    predicted_mood: str
    actual_mood: Optional[str] = None
    rating: Optional[int] = None  # 1-5 stars
    feedback: Optional[str] = ""

def load_artifacts():
    """Load all ML artifacts"""
    global vectorizer, best_model, nn_model, label_maps, recipes_df
    global INDEX_TO_LABEL, LABEL_TO_INDEX, CLASSES
    
    print("🔧 Loading ML artifacts...")
    
    try:
        # Load vectorizer
        vectorizer_path = f"{ARTIFACT_DIR}/tfidf_vectorizer.pkl"
        if os.path.exists(vectorizer_path):
            vectorizer = joblib.load(vectorizer_path)
            print(f"✅ Vectorizer loaded (vocab size: {len(vectorizer.vocabulary_)})")
        else:
            raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
        
        # Load label maps
        label_map_path = f"{ARTIFACT_DIR}/label_map.json"
        if os.path.exists(label_map_path):
            with open(label_map_path, "r") as f:
                label_maps = json.load(f)
            
            INDEX_TO_LABEL = {int(k): v for k, v in label_maps['index_to_label'].items()}
            LABEL_TO_INDEX = label_maps['label_to_index']
            CLASSES = label_maps.get('classes', list(LABEL_TO_INDEX.keys()))
            print(f"✅ Label maps loaded ({len(CLASSES)} classes)")
        
        # Load best model
        best_model_path = f"{ARTIFACT_DIR}/best_model.pkl"
        if os.path.exists(best_model_path):
            best_model_info = joblib.load(best_model_path)
            
            # Check if it's a neural network info dict or a sklearn model
            if isinstance(best_model_info, dict) and 'model_type' in best_model_info:
                model_type = best_model_info.get('model_type')
                if model_type == 'neural_network' and TF_AVAILABLE:
                    nn_model_path = f"{ARTIFACT_DIR}/{best_model_info['model_file']}"
                    if os.path.exists(nn_model_path):
                        nn_model = load_model(nn_model_path)
                        print(f"✅ Neural Network model loaded from {nn_model_path}")
                    else:
                        raise FileNotFoundError(f"NN model file not found: {nn_model_path}")
                else:
                    raise ValueError(f"Unknown model type or TF not available: {model_type}")
            else:
                # It's a sklearn model
                best_model = best_model_info
                print(f"✅ Sklearn model loaded: {type(best_model).__name__}")
        else:
            raise FileNotFoundError(f"Best model not found at {best_model_path}")
        
        # Load recipes dataset
        recipes_path = f"{ARTIFACT_DIR}/recipes_for_recommend.csv"
        if os.path.exists(recipes_path):
            recipes_df = pd.read_csv(recipes_path)
            print(f"✅ Recipes dataset loaded ({len(recipes_df)} recipes)")
        else:
            raise FileNotFoundError(f"Recipes dataset not found at {recipes_path}")
        
        print("🎉 All artifacts loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        raise

def predict_mood_nn(text: str) -> tuple:
    """Predict mood using neural network"""
    if nn_model is None or vectorizer is None:
        raise RuntimeError("Neural Network model not loaded")
    
    # Transform text
    text_tfidf = vectorizer.transform([text]).toarray()
    
    # Predict
    probabilities = nn_model.predict(text_tfidf, verbose=0)[0]
    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx])
    mood = INDEX_TO_LABEL.get(predicted_idx, "unknown")
    
    return mood, confidence, probabilities.tolist()

def predict_mood_sklearn(text: str) -> tuple:
    """Predict mood using sklearn model"""
    if best_model is None:
        raise RuntimeError("Sklearn model not loaded")
    
    # Predict
    probabilities = best_model.predict_proba([text])[0]
    predicted_idx = int(np.argmax(probabilities))
    
    # Get class names from pipeline
    try:
        classes = best_model.classes_
        mood = classes[predicted_idx]
    except:
        # Fallback to label maps
        mood = INDEX_TO_LABEL.get(predicted_idx, "unknown")
    
    confidence = float(probabilities[predicted_idx])
    
    return mood, confidence, probabilities.tolist()

def predict_mood(text: str) -> tuple:
    """Predict mood from text using the appropriate model"""
    if nn_model is not None:
        return predict_mood_nn(text)
    elif best_model is not None:
        return predict_mood_sklearn(text)
    else:
        raise RuntimeError("No model available")

def ingredient_similarity(user_ings: str, recipe_ings: str) -> float:
    """Calculate similarity between user ingredients and recipe ingredients"""
    if not user_ings.strip():
        return 0.0
    
    # Parse ingredients
    user_set = set([
        ing.strip().lower() 
        for ing in user_ings.split(',') 
        if ing.strip()
    ])
    
    recipe_set = set([
        ing.strip().lower() 
        for ing in recipe_ings.split(',') 
        if ing.strip()
    ])
    
    if not user_set:
        return 0.0
    
    # Jaccard similarity
    intersection = user_set.intersection(recipe_set)
    union = user_set.union(recipe_set)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def calculate_recipe_scores(mood: str, user_ingredients: str, alpha: float = 0.7) -> pd.Series:
    """Calculate scores for all recipes"""
    if recipes_df is None or vectorizer is None:
        raise RuntimeError("Data not loaded")
    
    # Create cache key
    cache_key = hashlib.md5(
        f"{mood}:{user_ingredients}:{alpha}".encode()
    ).hexdigest()
    
    cache_file = f"{CACHE_DIR}/{cache_key}.pkl"
    
    # Check cache
    if os.path.exists(cache_file):
        try:
            cached_scores = joblib.load(cache_file)
            print(f"📦 Using cached scores for {mood}")
            return cached_scores
        except:
            pass
    
    print(f"🧮 Calculating scores for mood: {mood}")
    
    # Prepare recipe texts
    recipe_texts = (
        recipes_df['recipe_title'].fillna('') + ' ' +
        recipes_df['instructions'].fillna('') + ' ' +
        recipes_df['ingredients'].fillna('')
    ).tolist()
    
    # Get mood probabilities for all recipes
    if nn_model is not None:
        # NN prediction
        recipe_features = vectorizer.transform(recipe_texts).toarray()
        all_probs = nn_model.predict(recipe_features, verbose=0)
        mood_idx = LABEL_TO_INDEX.get(mood, 0)
        mood_scores = all_probs[:, mood_idx]
    else:
        # Sklearn prediction
        all_probs = best_model.predict_proba(recipe_texts)
        try:
            mood_idx = list(best_model.classes_).index(mood)
            mood_scores = all_probs[:, mood_idx]
        except ValueError:
            # Mood not in model classes
            mood_scores = np.zeros(len(recipes_df))
    
    # Calculate ingredient scores
    ingredient_scores = recipes_df['ingredients'].fillna('').apply(
        lambda x: ingredient_similarity(user_ingredients, x)
    ).values
    
    # Combine scores
    final_scores = alpha * mood_scores + (1.0 - alpha) * ingredient_scores
    
    # Cache results
    joblib.dump(pd.Series(final_scores, index=recipes_df.index), cache_file)
    
    return pd.Series(final_scores, index=recipes_df.index)

# Load artifacts on startup
@app.on_event("startup")
async def startup_event():
    """Load ML artifacts when server starts"""
    try:
        load_artifacts()
    except Exception as e:
        print(f"⚠️ Warning: Could not load artifacts: {e}")
        print("⚠️ API will run in limited mode")

# Health check endpoint
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "AI Mood Recipe Recommender API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "vectorizer": vectorizer is not None,
            "best_model": best_model is not None or nn_model is not None,
            "recipes": recipes_df is not None,
            "num_recipes": len(recipes_df) if recipes_df is not None else 0
        },
        "available_moods": CLASSES
    }
    return status

@app.get("/moods")
async def list_moods():
    """List all available moods"""
    return {
        "moods": CLASSES,
        "count": len(CLASSES),
        "descriptions": {
            mood: recipes_df[recipes_df['mood'] == mood]['mood_description'].iloc[0]
            if recipes_df is not None and mood in recipes_df['mood'].values
            else "No description available"
            for mood in CLASSES
        }
    }

@app.post("/predict_mood")
async def predict_mood_endpoint(request: TextRequest):
    """Predict mood from text description"""
    try:
        if vectorizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        mood, confidence, probabilities = predict_mood(request.text)
        
        # Get probability for each class
        class_probabilities = {}
        if nn_model is not None:
            for idx, prob in enumerate(probabilities):
                class_name = INDEX_TO_LABEL.get(idx, f"class_{idx}")
                class_probabilities[class_name] = float(prob)
        elif best_model is not None:
            try:
                classes = best_model.classes_
                for idx, class_name in enumerate(classes):
                    class_probabilities[class_name] = float(probabilities[idx])
            except:
                # Fallback
                for idx, class_name in enumerate(CLASSES):
                    if idx < len(probabilities):
                        class_probabilities[class_name] = float(probabilities[idx])
        
        return {
            "text": request.text,
            "predicted_mood": mood,
            "confidence": confidence,
            "probabilities": class_probabilities,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/recommend")
async def recommend_recipes(request: RecommendRequest):
    """Get recipe recommendations based on mood and ingredients"""
    try:
        # Determine mood
        if request.mood:
            mood = request.mood
            mood_source = "provided"
        elif request.mood_text:
            mood, confidence, _ = predict_mood(request.mood_text)
            mood_source = "predicted"
        else:
            raise HTTPException(
                status_code=400, 
                detail="Provide either 'mood' or 'mood_text'"
            )
        
        # Validate mood
        if mood not in LABEL_TO_INDEX and mood not in CLASSES:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown mood: {mood}. Available moods: {CLASSES}"
            )
        
        # Calculate scores
        scores = calculate_recipe_scores(
            mood, 
            request.ingredients, 
            request.alpha
        )
        
        # Get top N recipes
        recipes_df['score'] = scores
        top_recipes = recipes_df.sort_values('score', ascending=False).head(request.top_n)
        
        # Prepare response
        recommendations = []
        for _, recipe in top_recipes.iterrows():
            # Parse matched ingredients
            user_ingredients = set([
                ing.strip().lower() 
                for ing in request.ingredients.split(',') 
                if ing.strip()
            ])
            
            recipe_ingredients = set([
                ing.strip().lower() 
                for ing in str(recipe['ingredients']).split(',') 
                if ing.strip()
            ])
            
            matched = list(user_ingredients.intersection(recipe_ingredients))
            
            recommendations.append({
                "title": str(recipe['recipe_title']),
                "mood": str(recipe['mood']),
                "ingredients": str(recipe['ingredients']),
                "instructions": str(recipe['instructions']),
                "why_matches": str(recipe.get('why_it_matches_mood', '')),
                "score": float(recipe['score']),
                "matched_ingredients": matched,
                "match_count": len(matched),
                "total_ingredients": len(recipe_ingredients)
            })
        
        return {
            "request": {
                "mood_provided": request.mood,
                "mood_text": request.mood_text,
                "mood_used": mood,
                "mood_source": mood_source,
                "ingredients_count": len([
                    ing for ing in request.ingredients.split(',') if ing.strip()
                ]),
                "alpha": request.alpha,
                "top_n": request.top_n
            },
            "recommendations": recommendations,
            "statistics": {
                "total_recipes_considered": len(recipes_df),
                "average_score": float(scores.mean()),
                "max_score": float(scores.max())
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for model improvement"""
    try:
        # In a real system, you would save this to a database
        # For now, we'll save to a JSON file
        feedback_data = {
            "recipe_title": request.recipe_title,
            "predicted_mood": request.predicted_mood,
            "actual_mood": request.actual_mood,
            "rating": request.rating,
            "feedback": request.feedback,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to file (append)
        feedback_file = f"{CACHE_DIR}/feedback.json"
        
        if os.path.exists(feedback_file):
            with open(feedback_file, "r") as f:
                all_feedback = json.load(f)
        else:
            all_feedback = []
        
        all_feedback.append(feedback_data)
        
        with open(feedback_file, "w") as f:
            json.dump(all_feedback, f, indent=2)
        
        return {
            "status": "success",
            "message": "Feedback received",
            "feedback_id": len(all_feedback)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback error: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return {
        "error": str(exc),
        "status_code": 500,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("🚀 Starting AI Mood Recipe Recommender API...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🏥 Health Check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server")
    
    uvicorn.run(
        "app_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )