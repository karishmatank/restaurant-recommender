import os
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OUTSCRAPER_API_KEY = os.getenv("OUTSCRAPER_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # Haiku = cheaper model
CLAUDE_MODEL_BATCH = "claude-sonnet-4-5-20250929" # Batches = cheaper rate

# Prompt for extracting dish characteristics
DISH_EXTRACTION_PROMPT = """
You are extracting structured dish information from restaurant reviews to power a food recommendation system.

For each dish explicitly named in the review, extract:
- dish_name: the name of the dish (correct obvious typos)
- sentiment: positive, negative, neutral, or mixed
- characteristics: ONLY objective, descriptive attributes of the dish itself

For characteristics, restrict yourself strictly to:
- Flavor: (e.g. spicy, sweet, salty, tangy, savory, smoky)
- Texture: (e.g. crispy, creamy, tender, crunchy, silky)
- Temperature: (e.g. hot, cold, warm)
- Portion: (e.g. generous, small, shareable)
- Preparation: (e.g. well-seasoned, undercooked, al dente)

Do NOT include subjective opinions or quality judgments as characteristics (e.g. "delicious", "amazing", "worth the hype", "really good").
Correct obvious typos in dish names (e.g., "friend fish" → "fried fish")

If a review mentions general food ("the food was great") without naming specific dishes, return an empty array
If a dish is mentioned but has no qualifying characteristics, return an empty characteristics array 
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
                        "items": {"type": "string"},
                        "description": "Descriptive attributes about the dish (taste, texture, presentation, etc.)"
                    }
                },
                "required": ["dish_name", "sentiment", "characteristics"],
                "additionalProperties": False
            }
        }
    },
    "required": ["dishes_mentioned"],
    "additionalProperties": False
}

# Data directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PARSED_DATA_DIR = DATA_DIR / "processed"
PARSED_REVIEWS_DIR = DATA_DIR / "parsed_reviews"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
PARSED_DATA_DIR.mkdir(exist_ok=True)
PARSED_REVIEWS_DIR.mkdir(exist_ok=True)