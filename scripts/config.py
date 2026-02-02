import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OUTSCRAPER_API_KEY = os.getenv("OUTSCRAPER_API_KEY")

# Data directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PARSED_DATA_DIR = DATA_DIR / "processed"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
PARSED_DATA_DIR.mkdir(exist_ok=True)