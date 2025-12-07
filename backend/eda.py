"""
Exploratory Data Analysis for Recipe Mood Dataset
Run with: python eda.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
from wordcloud import WordCloud
import os

def create_visualizations(df):
    """Create and save all visualizations"""
    
    # Create output directory
    os.makedirs('eda_output', exist_ok=True)
    
    # 1. Mood Distribution
    plt.figure(figsize=(12, 6))
    mood_counts = df['mood'].value_counts()
    bars = plt.bar(mood_counts.index, mood_counts.values)
    plt.title('Distribution of Moods in Dataset', fontsize=16)
    plt.xlabel('Mood', fontsize=12)
    plt.ylabel('Number of Recipes', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add counts on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('eda_output/mood_distribution.png', dpi=100)
    plt.close()
    
    # 2. Mood Descriptions
    mood_descriptions = df.groupby('mood')['mood_description'].first()
    with open('eda_output/mood_descriptions.txt', 'w') as f:
        for mood, desc in mood_descriptions.items():
            f.write(f"{mood}: {desc}\n\n")
    
    # 3. Most Common Ingredients
    all_ingredients = []
    for ingredients in df['ingredients'].dropna():
        all_ingredients.extend([i.strip().lower() for i in ingredients.split(',')])
    
    ingredient_counts = Counter(all_ingredients)
    top_20 = ingredient_counts.most_common(20)
    
    plt.figure(figsize=(12, 8))
    ingredients, counts = zip(*top_20)
    bars = plt.barh(ingredients[::-1], counts[::-1])
    plt.title('Top 20 Most Common Ingredients', fontsize=16)
    plt.xlabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig('eda_output/top_ingredients.png', dpi=100)
    plt.close()
    
    # 4. Recipe Title Word Cloud
    plt.figure(figsize=(15, 8))
    all_titles = ' '.join(df['recipe_title'].dropna().astype(str))
    wordcloud = WordCloud(width=1200, height=600, 
                         background_color='white',
                         max_words=100).generate(all_titles)
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Recipe Titles', fontsize=16)
    plt.tight_layout()
    plt.savefig('eda_output/title_wordcloud.png', dpi=100)
    plt.close()
    
    # 5. Instructions Length Analysis
    df['instruction_length'] = df['instructions'].fillna('').apply(len)
    df['ingredient_count'] = df['ingredients'].fillna('').apply(
        lambda x: len([i for i in x.split(',') if i.strip()])
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Instruction length distribution
    axes[0].hist(df['instruction_length'], bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_title('Distribution of Instruction Lengths', fontsize=14)
    axes[0].set_xlabel('Length (characters)')
    axes[0].set_ylabel('Frequency')
    
    # Ingredient count distribution
    axes[1].hist(df['ingredient_count'], bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_title('Distribution of Ingredient Counts', fontsize=14)
    axes[1].set_xlabel('Number of Ingredients')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('eda_output/length_distributions.png', dpi=100)
    plt.close()
    
    # 6. Mood vs Recipe Type Heatmap (simplified)
    recipe_types = []
    for title in df['recipe_title']:
        title_lower = str(title).lower()
        if 'salad' in title_lower:
            recipe_types.append('Salad')
        elif 'smoothie' in title_lower or 'shake' in title_lower:
            recipe_types.append('Smoothie')
        elif 'soup' in title_lower:
            recipe_types.append('Soup')
        elif 'pasta' in title_lower:
            recipe_types.append('Pasta')
        elif 'chicken' in title_lower:
            recipe_types.append('Chicken')
        elif 'rice' in title_lower:
            recipe_types.append('Rice')
        elif 'toast' in title_lower or 'bread' in title_lower:
            recipe_types.append('Bread')
        elif 'oat' in title_lower or 'porridge' in title_lower:
            recipe_types.append('Oats')
        else:
            recipe_types.append('Other')
    
    df['recipe_type'] = recipe_types
    
    # Create cross-tabulation
    cross_tab = pd.crosstab(df['mood'], df['recipe_type'])
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd', linewidths=0.5)
    plt.title('Mood vs Recipe Type Distribution', fontsize=16)
    plt.xlabel('Recipe Type')
    plt.ylabel('Mood')
    plt.tight_layout()
    plt.savefig('eda_output/mood_recipe_heatmap.png', dpi=100)
    plt.close()
    
    return {
        'total_recipes': len(df),
        'unique_moods': df['mood'].nunique(),
        'unique_recipes': df['recipe_title'].nunique(),
        'avg_instruction_length': df['instruction_length'].mean(),
        'avg_ingredient_count': df['ingredient_count'].mean(),
        'most_common_mood': mood_counts.idxmax(),
        'top_ingredient': top_20[0][0] if top_20 else None,
        'mood_distribution': mood_counts.to_dict()
    }

def generate_report(stats):
    """Generate a text report of the EDA"""
    report = f"""
    ====================================
    EXPLORATORY DATA ANALYSIS REPORT
    ====================================
    
    Dataset Statistics:
    ------------------
    Total Recipes: {stats['total_recipes']}
    Unique Moods: {stats['unique_moods']}
    Unique Recipe Titles: {stats['unique_recipes']}
    
    Average Instruction Length: {stats['avg_instruction_length']:.1f} characters
    Average Ingredients per Recipe: {stats['avg_ingredient_count']:.1f}
    
    Most Common Mood: {stats['most_common_mood']}
    Top Ingredient: {stats['top_ingredient']}
    
    Mood Distribution:
    -----------------
    """
    
    for mood, count in stats['mood_distribution'].items():
        percentage = (count / stats['total_recipes']) * 100
        report += f"    {mood}: {count} recipes ({percentage:.1f}%)\n"
    
    report += f"""
    Visualizations Created:
    ----------------------
    1. mood_distribution.png - Bar chart of mood frequencies
    2. top_ingredients.png - Horizontal bar chart of top 20 ingredients
    3. title_wordcloud.png - Word cloud of recipe titles
    4. length_distributions.png - Histograms of instruction lengths and ingredient counts
    5. mood_recipe_heatmap.png - Heatmap of mood vs recipe type
    
    Files are saved in the 'eda_output/' folder.
    """
    
    # Save report
    with open('eda_output/eda_report.txt', 'w') as f:
        f.write(report)
    
    # Also print to console
    print(report)
    
    # Save stats as JSON for later use
    with open('eda_output/dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

def main():
    print("🔍 Starting Exploratory Data Analysis...")
    
    try:
        # Load dataset
        dataset_path = "../data/realistic_recipe_mood_dataset.csv"
        df = pd.read_csv(dataset_path)
        
        print(f"✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {', '.join(df.columns)}")
        
        # Run EDA
        stats = create_visualizations(df)
        generate_report(stats)
        
        print("✅ EDA completed successfully!")
        print("📁 Output saved in 'eda_output/' folder")
        
    except FileNotFoundError:
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please ensure the CSV file is in the data/ folder")
        return False
    except Exception as e:
        print(f"❌ Error during EDA: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)