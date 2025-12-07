/**
 * AI Mood Recipe Recommender - Frontend JavaScript
 */

// Configuration
const API_BASE_URL = 'http://localhost:8000';
let currentMood = null;
let currentRecommendations = [];
let selectedRecipeForFeedback = null;
let currentRating = 0;

// DOM Elements
const elements = {
    moodText: document.getElementById('mood_text'),
    charCount: document.getElementById('charCount'),
    ingredients: document.getElementById('ingredients'),
    alpha: document.getElementById('alpha'),
    alphaValue: document.getElementById('alphaValue'),
    topN: document.getElementById('top_n'),
    advSettings: document.getElementById('advSettings'),
    loading: document.getElementById('loading'),
    results: document.getElementById('results'),
    recommendations: document.getElementById('recommendations'),
    stats: document.getElementById('stats'),
    error: document.getElementById('error'),
    apiStatus: document.getElementById('apiStatus'),
    feedbackModal: document.getElementById('feedbackModal'),
    modalRecipe: document.getElementById('modalRecipe'),
    ratingText: document.getElementById('ratingText'),
    feedbackText: document.getElementById('feedbackText')
};

// Character counter for mood text
elements.moodText.addEventListener('input', function() {
    elements.charCount.textContent = this.value.length;
    currentMood = null; // Reset quick mood when typing
});

// Alpha slider value display
function updateAlphaValue(value) {
    elements.alphaValue.textContent = `${value}%`;
}

// Set quick mood
function setQuickMood(mood) {
    currentMood = mood;
    const moodDescriptions = {
        'happy': "Feeling joyful and upbeat! 😊",
        'sad': "Feeling a bit down... 😔",
        'stressed': "Feeling overwhelmed and tense... 😫",
        'angry': "Feeling frustrated and tense... 😠",
        'tired': "Feeling exhausted and low-energy... 😴",
        'energetic': "Feeling pumped and full of energy! 💪",
        'lonely': "Feeling isolated and wanting comfort... 🏠",
        'relaxed': "Feeling calm and peaceful... 😌",
        'anxious': "Feeling nervous and on edge... 😰",
        'comfort-seeking': "Craving something warm and comforting... 🍲"
    };
    elements.moodText.value = moodDescriptions[mood] || `I'm feeling ${mood}`;
    elements.charCount.textContent = elements.moodText.value.length;
    
    // Highlight the selected button
    document.querySelectorAll('.mood-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
}

// Toggle advanced settings
function toggleAdvanced() {
    elements.advSettings.classList.toggle('hidden');
}

// Clear form
function clearForm() {
    elements.moodText.value = '';
    elements.charCount.textContent = '0';
    elements.ingredients.value = '';
    elements.alpha.value = 70;
    elements.alphaValue.textContent = '70%';
    elements.topN.value = '5';
    currentMood = null;
    currentRecommendations = [];
    
    // Reset UI
    elements.results.classList.add('hidden');
    elements.error.classList.add('hidden');
    
    // Remove active class from mood buttons
    document.querySelectorAll('.mood-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show success message
    showMessage('Form cleared! Ready for new input.', 'success');
}

// Show message
function showMessage(text, type = 'info') {
    elements.error.className = type;
    elements.error.textContent = text;
    elements.error.classList.remove('hidden');
    
    // Auto-hide success messages
    if (type === 'success') {
        setTimeout(() => {
            elements.error.classList.add('hidden');
        }, 3000);
    }
}

// Check API status
async function checkApiStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            updateApiStatus(true, `Connected - ${data.models_loaded.num_recipes} recipes loaded`);
        } else {
            updateApiStatus(false, 'API error');
        }
    } catch (error) {
        updateApiStatus(false, 'Cannot connect to API');
    }
}

function updateApiStatus(online, message) {
    const statusDot = elements.apiStatus.querySelector('.status-dot');
    const statusText = elements.apiStatus.querySelector('.status-text');
    
    statusDot.className = 'status-dot ' + (online ? 'online' : 'offline');
    statusText.textContent = message;
}

// Get recommendations
async function getRecommendations() {
    const moodText = elements.moodText.value.trim();
    const ingredients = elements.ingredients.value.trim();
    
    if (!moodText && !currentMood) {
        showMessage('Please describe your mood or select one from the quick buttons.', 'error');
        return;
    }
    
    // Show loading
    elements.loading.classList.remove('hidden');
    elements.results.classList.add('hidden');
    elements.error.classList.add('hidden');
    
    // Prepare request
    const requestData = {
        mood_text: moodText,
        ingredients: ingredients,
        top_n: parseInt(elements.topN.value),
        alpha: parseInt(elements.alpha.value) / 100
    };
    
    // Add mood if selected via quick button
    if (currentMood && !moodText) {
        requestData.mood = currentMood;
        requestData.mood_text = null;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'API request failed');
        }
        
        const data = await response.json();
        currentRecommendations = data.recommendations;
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        showMessage(`Error: ${error.message}`, 'error');
    } finally {
        elements.loading.classList.add('hidden');
    }
}

// Display results
function displayResults(data) {
    // Update stats
    elements.stats.innerHTML = `
        <i class="fas fa-chart-bar"></i>
        Found ${data.statistics.total_recipes_considered} recipes |
        Best match: ${(data.statistics.max_score * 100).toFixed(1)}%
    `;
    
    // Display recommendations
    let recommendationsHTML = '<div class="recipe-grid">';
    
    data.recommendations.forEach((recipe, index) => {
        const matchPercentage = (recipe.score * 100).toFixed(1);
        const moodEmoji = getMoodEmoji(recipe.mood);
        
        recommendationsHTML += `
            <div class="recipe-card">
                <div class="recipe-header">
                    <h3 class="recipe-title">${recipe.title}</h3>
                    <span class="recipe-score">${matchPercentage}% match</span>
                </div>
                
                <div class="recipe-meta">
                    <span><i class="fas fa-smile"></i> ${moodEmoji} ${recipe.mood}</span>
                    <span><i class="fas fa-carrot"></i> ${recipe.total_ingredients} ingredients</span>
                    <span><i class="fas fa-check-circle"></i> ${recipe.match_count} matches</span>
                </div>
                
                <div class="ingredients">
                    <h4><i class="fas fa-shopping-basket"></i> Ingredients:</h4>
                    <ul>
                        ${recipe.ingredients.split(',').map(ing => {
                            const trimmed = ing.trim();
                            const isMatched = recipe.matched_ingredients.some(m => 
                                m.toLowerCase() === trimmed.toLowerCase()
                            );
                            return `<li class="${isMatched ? 'matched' : ''}">
                                ${isMatched ? '<i class="fas fa-check"></i>' : '<i class="fas fa-circle"></i>'}
                                ${trimmed}
                            </li>`;
                        }).join('')}
                    </ul>
                </div>
                
                <div class="instructions">
                    <h4><i class="fas fa-list-ol"></i> Instructions:</h4>
                    <p>${recipe.instructions}</p>
                </div>
                
                ${recipe.why_matches ? `
                <div class="why-matches">
                    <h4><i class="fas fa-heart"></i> Why this matches your mood:</h4>
                    <p>${recipe.why_matches}</p>
                </div>
                ` : ''}
                
                <div class="recipe-actions">
                    <button class="btn-outline" onclick="openFeedback(${index})">
                        <i class="fas fa-star"></i> Rate This
                    </button>
                    <button class="btn-outline" onclick="saveRecipe(${index})">
                        <i class="fas fa-bookmark"></i> Save
                    </button>
                    <button class="btn-outline" onclick="shareRecipe(${index})">
                        <i class="fas fa-share-alt"></i> Share
                    </button>
                </div>
            </div>
        `;
    });
    
    recommendationsHTML += '</div>';
    elements.recommendations.innerHTML = recommendationsHTML;
    
    // Show results
    elements.results.classList.remove('hidden');
    
    // Scroll to results
    elements.results.scrollIntoView({ behavior: 'smooth' });
}

// Get emoji for mood
function getMoodEmoji(mood) {
    const emojiMap = {
        'happy': '😊',
        'sad': '😔',
        'stressed': '😫',
        'angry': '😠',
        'tired': '😴',
        'energetic': '💪',
        'lonely': '🏠',
        'relaxed': '😌',
        'anxious': '😰',
        'comfort-seeking': '🍲'
    };
    return emojiMap[mood] || '🍽️';
}

// Open feedback modal
function openFeedback(index) {
    if (!currentRecommendations[index]) return;
    
    selectedRecipeForFeedback = currentRecommendations[index];
    currentRating = 0;
    
    // Display recipe in modal
    elements.modalRecipe.innerHTML = `
        <h3>${selectedRecipeForFeedback.title}</h3>
        <p><strong>Mood:</strong> ${selectedRecipeForFeedback.mood}</p>
        <p><strong>Match Score:</strong> ${(selectedRecipeForFeedback.score * 100).toFixed(1)}%</p>
    `;
    
    // Reset rating
    document.querySelectorAll('.stars .fas').forEach(star => {
        star.classList.remove('active');
    });
    elements.ratingText.textContent = 'Select a rating';
    elements.feedbackText.value = '';
    
    // Show modal
    elements.feedbackModal.classList.remove('hidden');
}

// Rate recipe
function rateRecipe(rating) {
    currentRating = rating;
    
    // Update stars
    const stars = document.querySelectorAll('.stars .fas');
    stars.forEach((star, index) => {
        if (index < rating) {
            star.classList.add('active');
        } else {
            star.classList.remove('active');
        }
    });
    
    // Update rating text
    const ratingTexts = [
        'Poor',
        'Fair',
        'Good',
        'Very Good',
        'Excellent'
    ];
    elements.ratingText.textContent = ratingTexts[rating - 1] || 'Select a rating';
}

// Close modal
function closeModal() {
    elements.feedbackModal.classList.add('hidden');
    selectedRecipeForFeedback = null;
    currentRating = 0;
}

// Submit feedback
async function submitFeedback() {
    if (!selectedRecipeForFeedback || currentRating === 0) {
        showMessage('Please select a rating first.', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                recipe_title: selectedRecipeForFeedback.title,
                predicted_mood: selectedRecipeForFeedback.mood,
                rating: currentRating,
                feedback: elements.feedbackText.value
            })
        });
        
        if (response.ok) {
            showMessage('Thank you for your feedback! It helps improve our recommendations.', 'success');
            closeModal();
        } else {
            throw new Error('Failed to submit feedback');
        }
    } catch (error) {
        showMessage(`Error submitting feedback: ${error.message}`, 'error');
    }
}

// Save recipe (local storage)
function saveRecipe(index) {
    if (!currentRecommendations[index]) return;
    
    const recipe = currentRecommendations[index];
    const savedRecipes = JSON.parse(localStorage.getItem('savedRecipes') || '[]');
    
    // Check if already saved
    if (!savedRecipes.some(r => r.title === recipe.title)) {
        savedRecipes.push({
            ...recipe,
            savedAt: new Date().toISOString()
        });
        localStorage.setItem('savedRecipes', JSON.stringify(savedRecipes));
        showMessage('Recipe saved to your collection!', 'success');
    } else {
        showMessage('Recipe already saved.', 'info');
    }
}

// Share recipe
function shareRecipe(index) {
    if (!currentRecommendations[index]) return;
    
    const recipe = currentRecommendations[index];
    const shareText = `Check out this recipe: ${recipe.title}\n\nIngredients: ${recipe.ingredients}\n\nMood match: ${(recipe.score * 100).toFixed(1)}%`;
    
    if (navigator.share) {
        navigator.share({
            title: recipe.title,
            text: shareText,
            url: window.location.href
        });
    } else {
        // Fallback: copy to clipboard
        navigator.clipboard.writeText(shareText)
            .then(() => showMessage('Recipe copied to clipboard!', 'success'))
            .catch(() => showMessage('Failed to copy recipe.', 'error'));
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    // Check API status on load
    checkApiStatus();
    
    // Auto-check API status every 30 seconds
    setInterval(checkApiStatus, 30000);
    
    // Add example button
    addExampleButton();
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to get recommendations
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            getRecommendations();
        }
        
        // Escape to close modal
        if (e.key === 'Escape' && !elements.feedbackModal.classList.contains('hidden')) {
            closeModal();
        }
    });
});

// Add example button
function addExampleButton() {
    const examples = [
        {
            text: "I'm so happy today! Just got great news and want to celebrate with something fun and colorful!",
            ingredients: "berries, honey, yogurt, oats, banana"
        },
        {
            text: "Had a really stressful day at work. Need something simple and comforting that won't take much effort.",
            ingredients: "pasta, butter, garlic, olive oil, tomato"
        },
        {
            text: "Feeling lonely and missing home. Want something warm and nostalgic like my grandma used to make.",
            ingredients: "chicken, rice, carrot, onion, broth"
        }
    ];
    
    const exampleContainer = document.createElement('div');
    exampleContainer.className = 'examples';
    exampleContainer.innerHTML = `
        <h3><i class="fas fa-lightbulb"></i> Try These Examples:</h3>
        <div class="example-buttons">
            ${examples.map((ex, i) => `
                <button class="btn-small" onclick="loadExample(${i})">
                    Example ${i + 1}
                </button>
            `).join('')}
        </div>
    `;
    
    const inputSection = document.querySelector('.input-section');
    inputSection.insertBefore(exampleContainer, inputSection.querySelector('.advanced-settings'));
}

// Load example
function loadExample(index) {
    const examples = [
        {
            text: "I'm so happy today! Just got great news and want to celebrate with something fun and colorful!",
            ingredients: "berries, honey, yogurt, oats, banana"
        },
        {
            text: "Had a really stressful day at work. Need something simple and comforting that won't take much effort.",
            ingredients: "pasta, butter, garlic, olive oil, tomato"
        },
        {
            text: "Feeling lonely and missing home. Want something warm and nostalgic like my grandma used to make.",
            ingredients: "chicken, rice, carrot, onion, broth"
        }
    ];
    
    if (examples[index]) {
        elements.moodText.value = examples[index].text;
        elements.charCount.textContent = examples[index].text.length;
        elements.ingredients.value = examples[index].ingredients;
        currentMood = null; // Reset quick mood
        showMessage(`Loaded example ${index + 1}. Click "Get Recommendations" to see results.`, 'success');
    }
}