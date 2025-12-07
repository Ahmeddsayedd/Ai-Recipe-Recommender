"""
Train ML models for mood prediction without TensorFlow.
Usage: python train_and_export_simple.py --data ../data/realistic_recipe_mood_dataset.csv --out models
"""

import os
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare(path):
    """Load and prepare dataset"""
    print("📊 Loading dataset...")
    df = pd.read_csv(path)
    
    # Create combined text field
    df['text'] = (df['recipe_title'].fillna('') + ' ' +
                  df['instructions'].fillna('') + ' ' +
                  df['ingredients'].fillna('')).str.lower()
    
    # Clean text
    df['text'] = df['text'].str.replace(r'[^\w\s,]', ' ', regex=True)
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)
    
    # Labels
    df['mood'] = df['mood'].astype(str).str.strip()
    
    print(f"✅ Dataset loaded: {len(df)} samples")
    print(f"📊 Unique moods: {df['mood'].nunique()}")
    print(df['mood'].value_counts())
    
    return df

def tfidf_vectorize(X_train, max_features=2000):
    """Create TF-IDF vectorizer"""
    print(f"🔧 Creating TF-IDF vectorizer (max_features={max_features})...")
    vec = TfidfVectorizer(
        max_features=max_features, 
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )
    vec.fit(X_train)
    print(f"🔧 Vocabulary size: {len(vec.vocabulary_)}")
    return vec

def train_models_with_cv(X_train, y_train, X_val, y_val, vectorizer):
    """Train and evaluate multiple models with cross-validation"""
    results = {}
    
    # Define models to train
    models = {
        'logistic_regression': {
            'model': LogisticRegression(
                max_iter=1000,
                random_state=42,
                C=1.0,
                class_weight='balanced'
            ),
            'color': 'blue'
        },
        'random_forest': {
            'model': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=20,
                min_samples_split=5,
                class_weight='balanced',
                n_jobs=-1
            ),
            'color': 'green'
        },
        'naive_bayes': {
            'model': MultinomialNB(alpha=0.1),
            'color': 'orange'
        },
        'svm_linear': {
            'model': SVC(
                kernel='linear',
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            'color': 'red'
        }
    }
    
    print("\n" + "="*50)
    print("🤖 TRAINING MODELS")
    print("="*50)
    
    for name, config in models.items():
        print(f"\n📈 Training {name.replace('_', ' ').title()}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('clf', config['model'])
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Validate
        y_pred = pipeline.predict(X_val)
        val_f1 = f1_score(y_val, y_pred, average='macro')
        val_accuracy = np.mean(y_pred == y_val)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, 
            cv=cv, 
            scoring='f1_macro',
            n_jobs=-1
        )
        
        results[name] = {
            'model': pipeline,
            'val_f1': val_f1,
            'val_accuracy': val_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'color': config['color']
        }
        
        print(f"   ✅ Validation F1: {val_f1:.4f}")
        print(f"   ✅ Validation Accuracy: {val_accuracy:.4f}")
        print(f"   📊 CV F1: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    return results

def plot_model_comparison(results, out_folder):
    """Create visualization comparing models"""
    print("\n📊 Creating model comparison plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract data for plotting
    names = list(results.keys())
    val_f1_scores = [results[name]['val_f1'] for name in names]
    cv_scores = [results[name]['cv_mean'] for name in names]
    colors = [results[name]['color'] for name in names]
    
    # Plot 1: Validation F1 Scores
    bars1 = ax1.bar(names, val_f1_scores, color=colors, alpha=0.7)
    ax1.set_title('Model Performance (Validation F1 Score)')
    ax1.set_ylabel('F1 Score')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Plot 2: CV Scores with error bars
    x_pos = range(len(names))
    bars2 = ax2.bar(x_pos, cv_scores, color=colors, alpha=0.7,
                   yerr=[results[name]['cv_std'] for name in names],
                   capsize=5)
    ax2.set_title('Cross-Validation Performance')
    ax2.set_ylabel('F1 Score (Mean ± Std)')
    ax2.set_ylim(0, 1)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=45)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, 'model_comparison.png'), dpi=100)
    plt.close()
    
    print("✅ Model comparison plot saved")

def evaluate_best_model(best_model, X_test, y_test, classes, out_folder):
    """Comprehensive evaluation of the best model"""
    print("\n" + "="*50)
    print("🧪 FINAL EVALUATION ON TEST SET")
    print("="*50)
    
    # Predictions
    y_pred = best_model.predict(X_test)
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, 'confusion_matrix.png'), dpi=100)
    plt.close()
    
    # Calculate overall metrics
    accuracy = np.mean(y_pred == y_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n📈 Overall Metrics:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Macro F1: {macro_f1:.4f}")
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'per_class_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    with open(os.path.join(out_folder, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def main(args):
    """Main training pipeline"""
    print("="*50)
    print("🤖 MOOD RECIPE CLASSIFIER TRAINING")
    print("="*50)
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Load data
    df = load_and_prepare(args.data)
    texts = df['text'].values
    labels = df['mood'].values
    
    # Encode labels
    classes = sorted(list(set(labels)))
    label_to_index = {c: i for i, c in enumerate(classes)}
    index_to_label = {i: c for c, i in label_to_index.items()}
    
    print(f"\n📊 Found {len(classes)} unique moods:")
    for i, cls in enumerate(classes):
        count = sum(labels == cls)
        percentage = (count / len(labels)) * 100
        print(f"  {i+1:2d}. {cls:20s} - {count:4d} samples ({percentage:.1f}%)")
    
    # Split data
    print(f"\n🔀 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, 
        test_size=0.20, 
        random_state=42, 
        stratify=labels
    )
    
    print(f"📊 Training set: {len(X_train)} samples ({len(X_train)/len(texts)*100:.1f}%)")
    print(f"📊 Test set: {len(X_test)} samples ({len(X_test)/len(texts)*100:.1f}%)")
    
    # Create vectorizer
    vectorizer = tfidf_vectorize(X_train, max_features=2000)
    
    # Further split training data for validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.25, 
        random_state=42, 
        stratify=y_train
    )
    
    # Train models
    results = train_models_with_cv(X_train_final, y_train_final, X_val, y_val, vectorizer)
    
    # Find best model
    best_name = None
    best_f1 = -1
    
    for name, info in results.items():
        if info['val_f1'] > best_f1:
            best_f1 = info['val_f1']
            best_name = name
    
    print(f"\n🏆 Best model: {best_name} (F1: {best_f1:.4f})")
    
    # Create visualization
    plot_model_comparison(results, args.out)
    
    # Retrain best model on full training data
    print(f"\n🔄 Retraining best model ({best_name}) on full training data...")
    best_pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('clf', results[best_name]['model'].named_steps['clf'])
    ])
    
    best_pipeline.fit(X_train, y_train)
    
    # Save the best model
    joblib.dump(best_pipeline, os.path.join(args.out, "best_model.pkl"))
    print(f"✅ Best model saved as: {args.out}/best_model.pkl")
    
    # Save all models for reference
    for name, info in results.items():
        joblib.dump(info['model'], os.path.join(args.out, f"{name}_model.pkl"))
    
    # Save vectorizer and label maps
    joblib.dump(vectorizer, os.path.join(args.out, "tfidf_vectorizer.pkl"))
    
    label_maps = {
        'label_to_index': label_to_index,
        'index_to_label': index_to_label,
        'classes': classes
    }
    
    with open(os.path.join(args.out, "label_map.json"), "w") as f:
        json.dump(label_maps, f, indent=2)
    
    # Save dataset for recommendations
    df.to_csv(os.path.join(args.out, "recipes_for_recommend.csv"), index=False)
    
    # Evaluate on test set
    metrics = evaluate_best_model(best_pipeline, X_test, y_test, classes, args.out)
    
    # Save training summary
    summary = {
        'dataset_size': len(df),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'num_classes': len(classes),
        'best_model': best_name,
        'best_validation_f1': float(best_f1),
        'test_accuracy': metrics['accuracy'],
        'test_macro_f1': metrics['macro_f1'],
        'feature_count': len(vectorizer.vocabulary_),
        'models_trained': list(results.keys())
    }
    
    with open(os.path.join(args.out, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*50)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"\n📁 Artifacts saved to: {args.out}/")
    print(f"📊 Best model: {summary['best_model']}")
    print(f"📊 Test accuracy: {summary['test_accuracy']:.4f}")
    print(f"📊 Test F1: {summary['test_macro_f1']:.4f}")
    
    # Show artifact files
    print("\n📄 Generated files:")
    for file in sorted(os.listdir(args.out)):
        size = os.path.getsize(os.path.join(args.out, file))
        print(f"  - {file:30s} ({size/1024:.1f} KB)")
    
    print("\n🎉 Ready to use! Start API with: python app_fastapi.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mood recipe classifier (No TensorFlow)")
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument("--out", default="models", help="Output folder for models")
    parser.add_argument("--max_features", type=int, default=2000, help="Max TF-IDF features")
    args = parser.parse_args()
    
    main(args)