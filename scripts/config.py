import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OUTSCRAPER_API_KEY = os.getenv("OUTSCRAPER_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # Haiku = cheaper model
CLAUDE_MODEL_BATCH = "claude-sonnet-4-5-20250929" # Batches = cheaper rate

# Prompt for extracting dish characteristics
DISH_EXTRACTION_PROMPT = """
You are extracting structured dish information from restaurant reviews to power a food recommendation system.

The following characteristic taxonomy is used across all tasks below. Use the following as examples of valid characteristics, 
but do not limit yourself strictly to these terms. Extract any objective, sensory descriptor that describes how the dish 
looks, tastes, feels, or is prepared:
- Flavor: (e.g. spicy, sweet, salty, tangy, savory, smoky)
- Texture: (e.g. crispy, creamy, tender, crunchy, silky)
- Temperature: (e.g. hot, cold, warm)
- Portion: (e.g. generous, small, shareable)
- Preparation: (e.g. well-seasoned, undercooked, al dente)

*Task 1: Parse dish information*
For each dish explicitly named in the review, extract:
- dish_name: the name of the dish (correct obvious typos)
- sentiment: positive, negative, neutral, or mixed
- characteristics: ONLY objective, descriptive attributes of the dish itself. See the characteristic taxonomy above.
    - For each characteristic, extract both the clean sensory descriptor (e.g. "silky", "salty") AND the original phrase 
      from the review it was derived from (e.g. "silky rice noodles", "way too salty")

Do NOT extract:
- Ingredients or components BY THEMSELVES 
(e.g. "rice noodles", "broth", "onions", "dried shrimp") - these describe what is in the dish, not how it tastes or feels
- Subjective opinions or quality judgments, whether single words or phrases 
(e.g. "delicious", "amazing", "good flavor", "cooked well", "nicely done", "worth the hype")

Correct obvious typos in dish names (e.g., "friend fish" → "fried fish")

If a review mentions general food ("the food was great") without naming specific dishes, return an empty array
If a dish is mentioned but has no qualifying characteristics, return an empty characteristics array

*Task 2: Parse restaurant-level signals*
Extract any language that suggests who this restaurant is well-suited for based on taste preferences. This includes both:
- Explicit recommendations (e.g. "great for people who like salty food")
- Implicit signals where the reviewer's conclusion suggests a taste profile fit (e.g. a reviewer who finds the food too salty 
for their above-average salt tolerance implies the restaurant suits people who prefer very salty food)

Restrict to taste-profile-based signals anchored to the characteristic taxonomy above. Do not extract occasion or demographic 
recommendations (e.g. "great for dates", "good for families").

If no such signal exists, return an empty array.

*Task 3: Parse user information*
If the reviewer explicitly describes their own taste preferences, extract them. Restrict strictly to:
- Cuisine preferences they mention (e.g. "I love Chinese food")
- Characteristic preferences anchored to the taxonomy above (e.g. "I usually prefer spicier food")
- Comparative statements about their own palate (e.g. "I eat saltier food than most people")

If a palate statement implies a characteristic preference, extract that characteristic into characteristic_preferences as well 
(e.g. "I typically eat saltier foods than my peers" should populate palate_statements AND add "salty" to characteristic_preferences).

If no self-profiling language exists, return empty arrays for all fields.
"""

# Structured output schema for dish extraction
DISH_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "dishes_mentioned": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dish_name": {
                        "type": "string",
                        "description": "Name of the dish as mentioned in the review"
                    },
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral", "mixed"],
                        "description": "Overall sentiment about this specific dish"
                    },
                    "characteristics": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "characteristic": {
                                    "type": "string",
                                    "description": "Clean, reusable descriptive attributes based on the taxonomy"
                                },
                                "source_phrase": {
                                    "type": "string",
                                    "description": "The original phrase from the review that the characteristic was derived from (e.g. 'silky rice noodles', 'way too salty')"
                                }
                            },
                            "required": ["characteristic", "source_phrase"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["dish_name", "sentiment", "characteristics"],
                "additionalProperties": False
            }
        },
        "restaurant_signals": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Taste-profile-based signals about who the restaurant suits"
        },
        "reviewer_profile": {
            "type": "object",
            "properties": {
                "cuisine_preferences": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Cuisines the reviewer explicitly says they enjoy"
                },
                "characteristic_preferences": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Taste/texture preferences anchored to the taxonomy"
                },
                "palate_statements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Comparative statements about their own palate"
                }
            },
            "required": ["cuisine_preferences", "characteristic_preferences", "palate_statements"],
            "additionalProperties": False
        }
    },
    "required": ["dishes_mentioned", "restaurant_signals", "reviewer_profile"],
    "additionalProperties": False
}


# Restaurant category list for SerpAPI calls
RESTAURANT_CATEGORIES = {
    'american',
    'bagel',
    'bar',
    'bbq',
    'bistro',
    'bodega',
    'boba',
    'brasserie',
    'brazilian',
    'breakfast',
    'brunch',
    'bubble tea',
    'buffet',
    'burger',
    'cafe',
    'cantina',
    'caribbean',
    'casual dining',
    'chinese',
    'chophouse',
    'cocktail bar',
    'coffee shop',
    'creperie',
    'deli',
    'dessert',
    'dim sum',
    'diner',
    'donut',
    'dumpling',
    'eatery',
    'ethiopian',
    'fast food',
    'fine dining',
    'food',
    'food hall',
    'food truck',
    'french',
    'fusion',
    'gastropub',
    'gelateria',
    'german',
    'greek',
    'grill',
    'hawaiian',
    'hot pot',
    'hungarian',
    'ice cream',
    'indian',
    'italian',
    'izakaya',
    'japanese',
    'juice bar',
    'kitchen',
    'korean',
    'latin',
    'lebanese',
    'lounge',
    'mediterranean',
    'mexican',
    'middle eastern',
    'noodle',
    'noodle bar',
    'omakase',
    'osteria',
    'patisserie',
    'persian',
    'peruvian',
    'pho',
    'pizzeria',
    'pub',
    'ramen',
    'restaurant',
    'salvadoran',
    'sandwich shop',
    'seafood',
    'shabu',
    'shawarma',
    'smoothie',
    'soul food',
    'spanish',
    'sports bar',
    'steakhouse',
    'supper club',
    'sushi',
    'tapas',
    'taqueria',
    'tavern',
    'tea house',
    'thai',
    'trattoria',
    'turkish',
    'vegan',
    'vegetarian',
    'vietnamese',
    'wine bar',
    'winery',
    'yakiniku',
}

# Data directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PARSED_DATA_DIR = DATA_DIR / "processed"
PARSED_REVIEWS_DIR = DATA_DIR / "parsed_reviews"
USER_HISTORIES_DIR = DATA_DIR / "user_histories"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
PARSED_DATA_DIR.mkdir(exist_ok=True)
PARSED_REVIEWS_DIR.mkdir(exist_ok=True)
USER_HISTORIES_DIR.mkdir(exist_ok=True)