"""
Script to collect Google Maps review contributor histories from SerpAPI for restaurant validation experiment.
"""

import os
import json
import time
import pandas as pd
import serpapi
from load_reviews_utils import load_parsed_reviews
from config import SERPAPI_API_KEY, USER_HISTORIES_DIR, RESTAURANT_CATEGORIES


def sort_review_helpfulness():
    """
    Sort reviews based on how helpful the content is.
    We'll sort reviews based on the number of characteristics parsed.
    While we can use number of dishes, we care more about characteristics for this exercise.
    We can have only one dish but spoken about in a high-quality manner vs 
    50 dishes with no characteristics.

    We will use this ranking to help identify author IDs for whom we are interested in
    extracting more review data for user profile building.
    
    """
    print("\n****** Ranking reviews based on helpfulness ******")

    try:
        df = load_parsed_reviews()
        print(f"\nLoaded {len(df)} reviews with text")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Count total number of characteristics parsed
    count_num_characteristics = lambda x: sum(
        len(dish['characteristics']) for dish in x['dishes_mentioned']
    )
    df['num_characteristics'] = df['parsed_dishes'].apply(count_num_characteristics)

    # Sort reviews
    df = df.sort_values(by='num_characteristics', ascending=False)
    
    return df


def get_top_users(max_users):
    """
    Get top author_ids, ranked based on their review helpfulness
    """

    # Get sorted reviews
    df = sort_review_helpfulness()

    # Some users may have more than one review in our validation dataset
    # We want to count the "best" review only rather than add up metrics across reviews
    df = df.groupby('author_id')[['num_characteristics']].max().reset_index()

    # Cut out authors who gave no characteristics
    author_ids = df[df['num_characteristics'] > 0]['author_id'].head(max_users)

    print(f"\nSelected top {len(author_ids)} users based on review helpfulness")
    
    return author_ids.tolist()


def is_restaurant_related(review_data):
    """
    Check if a review is restaurant-related based on keywords and category.
    
    Args:
        review_data: Dictionary containing review information
    
    Returns:
        Boolean indicating if review is restaurant-related
    """
    # Check place type
    try:
        place_type = review_data['place_info']['type'].lower()
    except KeyError:
        place_type = ''
    
    if any(keyword in place_type for keyword in RESTAURANT_CATEGORIES):
        return True
    
    return False


def fetch_user_reviews_serpapi(contributor_id, delay=1.0):
    """
    Fetch all reviews for a specific Google Maps contributor using SerpAPI.
    
    Uses the newer serpapi package (not the deprecated google-search-results).
    Engine: google_maps_contributor_reviews
    
    Args:
        contributor_id: Google Maps contributor ID (author_id from your data)
        delay: Delay between requests (seconds)
    
    Returns:
        List of review dictionaries
    """
    if not SERPAPI_API_KEY:
        raise ValueError("SERPAPI_API_KEY not found in environment variables")
    
    print(f"  Fetching reviews for contributor: {contributor_id}")
    
    # Initialize SerpAPI client (newer package)
    client = serpapi.Client(api_key=SERPAPI_API_KEY)
    
    all_reviews = []
    
    try:
        # Search parameters - using correct engine and parameter name
        results = client.search(
            engine="google_maps_contributor_reviews",
            contributor_id=contributor_id,
            hl="en",
            num=200  # Max results (SerpAPI limits to 200 total per contributor)
        )
        
        # Extract reviews
        reviews = results.get("reviews", [])
        
        if reviews:
            print(f"    Found {len(reviews)} reviews")
            all_reviews.extend(reviews)
        else:
            print(f"    No reviews found")
        
        time.sleep(delay)  # Rate limiting
        
    except serpapi.SerpApiClientException as e:
        print(f"    SerpAPI error: {e}")
    except Exception as e:
        print(f"    Error: {e}")
    
    print(f"  Total reviews collected: {len(all_reviews)}")
    return all_reviews


def filter_restaurant_reviews(reviews):
    """
    Filter reviews to only include restaurant-related ones.
    
    Args:
        reviews: List of review dictionaries
    
    Returns:
        Filtered list of restaurant reviews
    """
    restaurant_reviews = [r for r in reviews if is_restaurant_related(r)]
    print(f"  Filtered to {len(restaurant_reviews)} restaurant reviews (from {len(reviews)} total)")
    return restaurant_reviews


def save_user_history(contributor_id, reviews):
    """
    Save user review history to JSON file.
    
    Args:
        contributor_id: Google Maps contributor ID
        reviews: List of review dictionaries
    """
    timestamp = int(time.time())
    filename = f"user_{contributor_id}_{timestamp}.json"
    filepath = USER_HISTORIES_DIR / filename
    
    data = {
        "contributor_id": contributor_id,
        "collection_timestamp": timestamp,
        "total_reviews": len(reviews),
        "reviews": reviews
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved to: {filepath}")
    return filepath


def track_user_review_history(max_users=20, delay=1.0):
    """
    Track the full review history for the specified number of users.
    We find the users to track review history for by ranking how helpful their reviews were.
    We'll use SerpAPI to do this.

    Args:
        max_users: Maximum number of users to collect histories for
        delay: Delay between API requests (seconds)
    
    Returns:
        DataFrame with summary statistics
    """
    print("\n" + "="*60)
    print("COLLECTING USER REVIEW HISTORIES")
    print("="*60)
    
    # Get top author_ids (these are contributor_ids in SerpAPI terms)
    contributor_ids = get_top_users(max_users)
    
    if not contributor_ids:
        print("No users found to collect histories for")
        return
    
    # Collect data for each user
    summary_data = []
    
    for idx, contributor_id in enumerate(contributor_ids, 1):
        print(f"\n[{idx}/{len(contributor_ids)}] Processing contributor: {contributor_id}")
        
        try:
            # Fetch all reviews
            all_reviews = fetch_user_reviews_serpapi(contributor_id, delay=delay)
            
            # Filter to restaurant reviews only
            restaurant_reviews = filter_restaurant_reviews(all_reviews)
            
            # Save filtered reviews
            if restaurant_reviews:
                filepath = save_user_history(contributor_id, restaurant_reviews)
                
                summary_data.append({
                    'contributor_id': contributor_id,
                    'total_reviews': len(all_reviews),
                    'restaurant_reviews': len(restaurant_reviews),
                    'filepath': str(filepath)
                })
            else:
                print(f"No restaurant reviews found for {contributor_id}")
        
        except Exception as e:
            print(f"Error processing {contributor_id}: {e}")
    
    # Create summary DataFrame
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        timestamp = int(time.time())
        summary_file = USER_HISTORIES_DIR / f"collection_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print("\n" + "="*60)
        print("COLLECTION SUMMARY")
        print("="*60)
        print(f"Users processed: {len(summary_df)}")
        print(f"Total restaurant reviews: {summary_df['restaurant_reviews'].sum()}")
        print(f"Average reviews per user: {summary_df['restaurant_reviews'].mean():.1f}")
        print(f"\nSummary saved to: {summary_file}")
        
        return summary_df
    
    return None


def main():
    """
    Main function to collect user review histories.
    """
    if not SERPAPI_API_KEY:
        print("ERROR: SERPAPI_API_KEY not found. Please add to .env file:")
        print("SERPAPI_API_KEY=your_api_key_here")
        print("\nGet your API key from: https://serpapi.com/")
        return
    
    # Collect histories for top 20 users
    MAX_USERS = 1 # TODO: CHANGE BACK TO 20
    DELAY = 1  # 1 second between requests
    
    summary_df = track_user_review_history(max_users=MAX_USERS, delay=DELAY)
    
    if summary_df is not None:
        print("\nUser history collection complete!")


if __name__ == "__main__":
    main()
