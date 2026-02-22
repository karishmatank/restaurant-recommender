import pandas as pd
from config import PARSED_DATA_DIR, PARSED_REVIEWS_DIR

def load_processed_reviews():
    """
    Load the most recent processed reviews data.
    
    Returns:
        DataFrame with processed reviews
    """
    json_files = list(PARSED_DATA_DIR.glob("reviews_*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No processed review files found in {PARSED_DATA_DIR}")
    
    # Get the most recent file
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    
    print(f"Loading reviews from: {latest_file}")
    df = pd.read_json(latest_file)
    
    return df

def load_parsed_reviews():
    """
    Load the most recent parsed reviews data from the parsed_reviews folder.
    
    Returns:
        DataFrame with parsed reviews including dish information
    """
    
    # Look for JSON files from batch processing
    json_files = list(PARSED_REVIEWS_DIR.glob("batch_parsed_reviews_*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No parsed review files found in {PARSED_REVIEWS_DIR}")
    
    # Get the most recent file
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    
    print(f"Loading parsed reviews from: {latest_file}")
    df = pd.read_json(latest_file)
    
    return df