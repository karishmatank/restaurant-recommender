import pandas as pd
from config import PARSED_DATA_DIR, PARSED_REVIEWS_DIR

def get_timestamp_from_filename(p):
    """
    Get the most recent file by timestamp in filename (more reliable than mtime)
    Filename format: batch_parsed_reviews_0_2156_1773246596.json
    """
    try:
        return int(p.stem.split('_')[-1])
    except:
        return 0

def load_processed_reviews():
    """
    Load the most recent processed reviews data.
    Preserves author_id as string to avoid precision loss from float conversion.
    
    Returns:
        DataFrame with processed reviews
    """
    json_files = list(PARSED_DATA_DIR.glob("reviews_*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No processed review files found in {PARSED_DATA_DIR}")
    
    # Get the most recent file
    latest_file = max(json_files, key=get_timestamp_from_filename)
    
    print(f"Loading reviews from: {latest_file}")
    df = pd.read_json(latest_file, dtype={'author_id': str})
    
    return df

def load_parsed_reviews():
    """
    Load the most recent parsed reviews data from the parsed_reviews folder.
    Preserves author_id as string to avoid precision loss from float conversion.
    
    Returns:
        DataFrame with parsed reviews including dish information
    """
    
    # Look for JSON files from batch processing
    json_files = list(PARSED_REVIEWS_DIR.glob("batch_parsed_reviews_*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No parsed review files found in {PARSED_REVIEWS_DIR}")
    
    latest_file = max(json_files, key=get_timestamp_from_filename)
    
    print(f"Loading parsed reviews from: {latest_file}")
    df = pd.read_json(latest_file, dtype={'author_id': str})
    
    return df


def fix_author_ids_in_parsed_reviews(parsed_reviews_file=None):
    """    
    This function corrects author_id values within the batch parsed reviews
    that were converted to floats during validation testing
    by replacing them with the correct string values from the processed reviews.
    
    Args:
        parsed_reviews_file: Path to specific parsed reviews file to fix.
                           If None, fixes the most recent file.
    
    Returns:
        DataFrame with corrected author_id values
    """
    
    # Load the correct processed reviews with string author_ids
    print("Loading processed reviews with correct author_ids...")
    processed_df = load_processed_reviews()
    print(processed_df.dtypes)
    print(processed_df.head())
    
    # If no specific file provided, use the most recent
    if parsed_reviews_file is None:
        json_files = list(PARSED_REVIEWS_DIR.glob("batch_parsed_reviews_*.json"))
        if not json_files:
            raise FileNotFoundError(f"No parsed review files found in {PARSED_REVIEWS_DIR}")
        parsed_reviews_file = max(json_files, key=lambda p: p.stat().st_mtime)
    
    print(f"Loading parsed reviews from: {parsed_reviews_file}")
    # Load without dtype to get the current state
    parsed_df = pd.read_json(parsed_reviews_file)
    
    print(parsed_df.dtypes)
    
    # Combine files based on same place_id, place_name, rating, and review_text
    parsed_df = parsed_df.merge(
        processed_df[['place_id', 'place_name', 'rating', 'review_text', 'author_id']],
        how='left',
        on=['place_id', 'place_name', 'rating', 'review_text']
    )
    parsed_df = parsed_df.drop(columns=['author_id_x']).rename(columns={'author_id_y': 'author_id'})
    
    # Save the corrected file
    corrected_file = parsed_reviews_file.parent / f"corrected_{parsed_reviews_file.name}"
    
    print(f"\nSaving corrected file to: {corrected_file}")
    parsed_df.to_json(corrected_file, orient='records', indent=2)
    
    # Also save as CSV
    csv_file = corrected_file.with_suffix('.csv')
    parsed_df.to_csv(csv_file, index=False)
    print(f"Saved CSV to: {csv_file}")
    
    return

# if __name__ == "__main__":
#     fix_author_ids_in_parsed_reviews()