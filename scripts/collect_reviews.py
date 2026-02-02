"""
Script to collect Google Maps reviews from Outscraper API for restaurant validation experiment.

This script uses the official Outscraper Python SDK to:
1. Fetch reviews for a list of restaurant Place IDs
2. Save raw JSON responses
3. Parse and extract: author_id, review_text, rating, timestamp
4. Load data into pandas DataFrame for analysis
"""

import json
import time
import pandas as pd
from outscraper import OutscraperClient
from config import RAW_DATA_DIR, OUTSCRAPER_API_KEY, PARSED_DATA_DIR, DATA_DIR
from inputs import RESTAURANT_PLACE_IDS


def fetch_reviews(client, place_id, limit=250):
    """
    Fetch reviews for a specific Google Maps Place ID using Outscraper SDK.
    
    Args:
        client: OutscraperClient instance
        place_id: Google Maps Place ID (e.g., "ChIJN1t_tDeuEmsRUsoyG83frY4")
        limit: Maximum number of reviews to fetch (default: 250)
    
    Returns:
        List of dictionaries containing place data with reviews
    """
    print(f"Fetching reviews for Place ID: {place_id}...")
    
    # Use google_maps_reviews method from SDK
    # See: https://app.outscraper.cloud/api-docs
    results = client.google_maps_reviews(
        query=place_id,
        reviews_limit=limit,
        language="en",
        region="us",
        sort='newest'
    )
    
    return results


def save_raw_response(place_id, data):
    """
    Save raw JSON response to file.
    
    Args:
        place_id: Google Maps Place ID
        data: Raw SDK response data (list of results)
    
    Returns:
        Path to saved file
    """
    timestamp = int(time.time())
    filename = f"{place_id}_{timestamp}.json"
    filepath = RAW_DATA_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved raw response to: {filepath}")


def save_processed_data(df):
    """
    Save processed dataframe
    
    Args:
        df: DataFrame with processed data
    
    """
    timestamp = int(time.time())
    output_file = PARSED_DATA_DIR / f"reviews_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved parsed data to: {output_file}")
    
    # Also save as JSON for easier inspection
    json_file = PARSED_DATA_DIR / f"reviews_{timestamp}.json"
    df.to_json(json_file, orient='records', indent=2, date_format='iso')
    print(f"Saved parsed data to: {json_file}")


def parse_reviews(data, place_id):
    """
    Parse reviews from SDK response and extract required fields.
    
    Args:
        data: SDK response data (list of place results)
        place_id: Google Maps Place ID (for fallback)
    
    Returns:
        List of parsed review dictionaries
    """
    parsed_reviews = []
    
    # SDK returns a list of results (usually just one for a specific Place ID)
    for result in data:
        # Extract reviews
        reviews_data = result.get("reviews_data", [])
        
        for review in reviews_data:
            parsed_review = {
                "place_id": result.get("place_id", place_id),
                "google_id": result.get("google_id", ""),
                "place_name": result.get("name", "Unknown"),
                "author_id": review.get("author_id"),
                "author_link": review.get("author_link"),
                "review_text": review.get("review_text", ""),
                "rating": review.get("review_rating"),
                "timestamp": review.get("review_timestamp"),
                "datetime_utc": review.get("review_datetime_utc"),
                "author_name": review.get("author_title", ""),
                "review_link": review.get("review_link", "")
            }
            parsed_reviews.append(parsed_review)
    
    return parsed_reviews


def collect_reviews_for_restaurants(place_ids, limit=250, delay=1.0):
    """
    Collect reviews for multiple restaurants and return as DataFrame.
    
    Args:
        place_ids: List of Google Maps Place IDs
        limit: Maximum number of reviews per restaurant
        delay: Delay between API calls (seconds)
    
    Returns:
        pandas DataFrame with all reviews
    """
    if not OUTSCRAPER_API_KEY:
        raise ValueError("OUTSCRAPER_API_KEY not found in environment variables. Please set it in .env file.")
    
    # Initialize Outscraper client with API key
    # Using OutscraperClient as shown in https://app.outscraper.cloud/api-docs
    client = OutscraperClient(api_key=OUTSCRAPER_API_KEY)
    
    all_reviews = []
    
    for i, place_id in enumerate(place_ids, 1):
        print(f"\n[{i}/{len(place_ids)}] Processing Place ID: {place_id}")
        
        try:
            # Fetch reviews using SDK
            raw_data = fetch_reviews(client, place_id, limit=limit)
            
            # Save raw response
            save_raw_response(place_id, raw_data)
            
            # Parse reviews
            parsed = parse_reviews(raw_data, place_id)
            all_reviews.extend(parsed)
            
            print(f"Collected {len(parsed)} reviews")
            
            # Rate limiting - delay between requests (optional with SDK, but good practice)
            time.sleep(delay)
        
        except Exception as e:
            print(f"***Error: {e}")
            raise e
    
    # Create DataFrame
    df = pd.DataFrame(all_reviews)
    
    # Save parsed data
    if len(df) > 0:
        save_processed_data(df)
    
    return df

def print_summary(df):
    """
    Prints summary of dataframe
    
    Args:
        df: DataFrame with parsed data
    """
    print("\n" + "="*60)
    print("COLLECTION SUMMARY")
    print("="*60)
    print(f"Total reviews collected: {len(df)}")
    if len(df) > 0:
        print(f"\nReviews per restaurant:")
        print(df.groupby("place_name")["author_id"].count())
        print(f"\nUnique reviewers: {df['author_id'].nunique()}")
        print(f"\nSample data:")
        print(df.head())
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")


def main():
    """Main function to run the data collection."""
    if not RESTAURANT_PLACE_IDS:
        print("ERROR: Please add restaurant Place IDs to RESTAURANT_PLACE_IDS list in the script.")
        return
    
    if not OUTSCRAPER_API_KEY:
        print("ERROR: OUTSCRAPER_API_KEY not found. Please create a .env file with:")
        print("OUTSCRAPER_API_KEY=your_api_key_here")
        return
    
    print(f"Starting data collection for {len(RESTAURANT_PLACE_IDS)} restaurants...")
    print(f"Data will be saved to: {DATA_DIR}")
    
    # Collect reviews
    df = collect_reviews_for_restaurants(
        place_ids=RESTAURANT_PLACE_IDS,
        delay=1.0   # 1 second delay between requests
    )
    
    # Print summary
    print_summary(df)


if __name__ == "__main__":
    main()
