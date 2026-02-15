"""
Claude API Batch integration for parsing restaurant reviews.

This script uses the Anthropic Batch API to process large volumes of reviews
asynchronously with 50% cost reduction and no rate limits.
"""

import json
import time
import pandas as pd
from textwrap import dedent
from anthropic import Anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL_BATCH, DISH_EXTRACTION_SCHEMA, PARSED_DATA_DIR, PARSED_REVIEWS_DIR


def create_review_parsing_prompt(review_text):
    """
    Create the prompt for Claude to parse a restaurant review.
    
    Args:
        review_text: The review text to parse
    
    Returns:
        Formatted prompt string
    """
    prompt = dedent(f"""
        From the review below, extract: 
        1. ALL dishes mentioned by name (e.g., "carbonara", "margherita pizza", "Caesar salad") 
        2. For EACH dish, determine the sentiment (positive, negative, neutral, or mixed) 
        3. For EACH dish, extract any characteristics mentioned (e.g., flavor descriptors like "spicy", "creamy", "tangy"; texture like "crispy", "tender"; portion comments like "generous", "small"; or preparation like "al dente", "well-done")
        Guidelines: 
        * If a review mentions general food ("the food was great") without naming specific dishes, return an empty array 
        * If a dish is mentioned but has no descriptive characteristics, return an empty characteristics array 
        * Correct obvious typos in dish names (e.g., "friend fish" → "fried fish")
        * Extract characteristics as short phrases 
        * Common characteristics include: taste (spicy, sweet, savory, bland), texture (crispy, creamy, crunchy, tender), temperature (hot, cold), freshness, portion size, presentation
        Review: {review_text}
    """)
    
    return prompt


def create_batch_requests(reviews_df, start_index=0, batch_size=None):
    """
    Create batch API requests for reviews.
    
    Args:
        reviews_df: DataFrame with review data
        start_index: Index to start processing from
        batch_size: Number of reviews to process (None = all reviews)
    
    Returns:
        List of batch request objects
    """
    # Filter out reviews without text
    reviews_with_text = reviews_df[reviews_df['review_text'].notna()].copy()
    
    if batch_size is not None:
        end_index = min(start_index + batch_size, len(reviews_with_text))
        reviews_to_process = reviews_with_text.iloc[start_index:end_index]
    else:
        reviews_to_process = reviews_with_text.iloc[start_index:]
    
    print(f"Creating batch requests for {len(reviews_to_process)} reviews...")
    
    batch_requests = []
    
    for idx, row in reviews_to_process.iterrows():
        prompt = create_review_parsing_prompt(row['review_text'])
        
        # Create individual request in batch format
        request = {
            "custom_id": f"review_{idx}",  # Unique ID to match results back
            "params": {
                "model": CLAUDE_MODEL_BATCH,
                "max_tokens": 2048,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": DISH_EXTRACTION_SCHEMA
                    }
                }
            }
        }
        
        batch_requests.append(request)
    
    return batch_requests


def submit_batch(client, batch_requests):
    """
    Submit a batch of requests to the Anthropic API.
    
    Args:
        client: Anthropic client instance
        batch_requests: List of batch request objects
    
    Returns:
        Batch ID for tracking
    """
    print(f"\nSubmitting batch with {len(batch_requests)} requests...")
    
    # Create the batch
    message_batch = client.messages.batches.create(
        requests=batch_requests
    )
    
    print(f"Batch submitted successfully!")
    print(f"Batch ID: {message_batch.id}")
    print(f"Status: {message_batch.processing_status}")
    
    return message_batch.id


def poll_batch_status(client, batch_id, poll_interval=60):
    """
    Poll the batch status until completion.
    
    Args:
        client: Anthropic client instance
        batch_id: The batch ID to poll
        poll_interval: Seconds between status checks (default: 60)
    
    Returns:
        Final batch status information
    """
    print(f"\nPolling batch status (checking every {poll_interval} seconds)...")
    
    while True:
        batch_status = client.messages.batches.retrieve(batch_id)
        
        status = batch_status.processing_status
        request_counts = batch_status.request_counts
        
        print(f"\nStatus: {status}")
        print(f"  Succeeded: {request_counts.succeeded}")
        print(f"  Errored: {request_counts.errored}")
        print(f"  Expired: {request_counts.expired}")
        print(f"  Canceled: {request_counts.canceled}")
        print(f"  Processing: {request_counts.processing}")
        
        if status == "ended":
            print("\nBatch processing completed!")
            return batch_status
        elif status in ["canceling", "canceled"]:
            print(f"\nBatch was canceled")
            return batch_status
        
        # Wait before next poll
        print(f"  Waiting {poll_interval} seconds before next check...")
        time.sleep(poll_interval)


def retrieve_batch_results(client, batch_id):
    """
    Retrieve results from a completed batch.
    
    Args:
        client: Anthropic client instance
        batch_id: The batch ID
    
    Returns:
        List of batch results
    """
    print(f"\nRetrieving batch results...")
    
    results = []
    
    # The API returns results as an iterable
    for result in client.messages.batches.results(batch_id):
        results.append(result)
    
    print(f"Retrieved {len(results)} results")
    
    return results


def process_batch_results(results, reviews_df, start_index=0):
    """
    Process batch results and create DataFrame.
    
    Args:
        results: List of batch result objects
        reviews_df: Original reviews DataFrame
        start_index: Starting index used in batch
    
    Returns:
        DataFrame with parsed results
    """
    print(f"\nProcessing batch results...")
    
    parsed_results = []
    
    for result in results:
        # Extract custom_id to match back to original review
        custom_id = result.custom_id
        review_idx = int(custom_id.split('_')[1])
        
        # Get original review data
        row = reviews_df.iloc[review_idx]
        
        # Process the result
        parsed_data = None
        
        if result.result.type == "succeeded":
            # Extract JSON from the message content
            message = result.result.message
            if message.content and len(message.content) > 0:
                response_text = message.content[0].text
                try:
                    parsed_data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error for review {review_idx}: {e}")
                    parsed_data = {"raw_response": response_text, "parse_error": str(e)}
        elif result.result.type == "errored":
            # Access the error - SDK may flatten the nested structure
            error_response = result.result.error
            # Try to access nested error object (as per docs), fallback to flattened
            if hasattr(error_response, 'error'):
                error_obj = error_response.error
                error_type = error_obj.type
                error_message = error_obj.message
            else:
                # SDK might flatten the structure
                error_type = error_response.type
                error_message = error_response.message
            
            print(f"Error for review {review_idx}: {error_type} - {error_message}")
            parsed_data = {"error": error_type, "error_message": error_message}
        
        # Create result record
        parsed_result = {
            'review_index': review_idx,
            'place_id': row.get('place_id'),
            'place_name': row.get('place_name'),
            'author_id': row.get('author_id'),
            'rating': row.get('rating'),
            'review_text': row.get('review_text'),
            'parsed_dishes': parsed_data
        }
        
        parsed_results.append(parsed_result)
    
    # Create DataFrame from results
    results_df = pd.DataFrame(parsed_results)
    
    # Sort by review_index to maintain order
    results_df = results_df.sort_values('review_index').reset_index(drop=True)
    
    # Save results
    timestamp = int(time.time())
    output_file = PARSED_REVIEWS_DIR / f"batch_parsed_reviews_{start_index}_{start_index + len(results_df) - 1}_{timestamp}.json"
    results_df.to_json(output_file, orient='records', indent=2)
    print(f"\nSaved parsed results to: {output_file}")
    
    # Also save as CSV (without nested JSON)
    csv_file = PARSED_REVIEWS_DIR / f"batch_parsed_reviews_{start_index}_{start_index + len(results_df) - 1}_{timestamp}.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"Saved CSV to: {csv_file}")
    
    return results_df


def process_reviews_batch(reviews_df, batch_size=None, start_index=0, poll_interval=60):
    """
    Process reviews using Claude Batch API.
    
    Args:
        reviews_df: DataFrame with review data
        batch_size: Number of reviews to process (None = all reviews)
        start_index: Index to start processing from
        poll_interval: Seconds between status checks (default: 60)
    
    Returns:
        DataFrame with parsed results
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables. Please set it in .env file.")
    
    # Initialize Anthropic client
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Create batch requests
    batch_requests = create_batch_requests(reviews_df, start_index, batch_size)
    
    # Submit batch
    batch_id = submit_batch(client, batch_requests)
    
    # Poll for completion
    poll_batch_status(client, batch_id, poll_interval)
    
    # Retrieve results
    results = retrieve_batch_results(client, batch_id)
    
    # Process and save results
    results_df = process_batch_results(results, reviews_df, start_index)
    
    return results_df


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


def main():
    """
    Main function to parse reviews with Claude Batch API.
    """
    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not found. Please add to .env file:")
        print("ANTHROPIC_API_KEY=your_api_key_here")
        return
    
    # Load reviews
    print("Loading processed reviews...")
    reviews_df = load_processed_reviews()
    print(f"Loaded {len(reviews_df)} total reviews")
    
    # Count reviews with text
    reviews_with_text = reviews_df[reviews_df['review_text'].notna()]
    print(f"Reviews with text: {len(reviews_with_text)}")
    
    # Process all reviews (or specify batch_size to test with a subset)
    BATCH_SIZE = None  # None = process all reviews
    START_INDEX = 0
    POLL_INTERVAL = 300  # Check status every 5 minutes
    
    print(f"\nSubmitting batch request for reviews...")
    if BATCH_SIZE:
        print(f"Processing {BATCH_SIZE} reviews starting from index {START_INDEX}")
    else:
        print(f"Processing all {len(reviews_with_text)} reviews")
    
    parsed_df = process_reviews_batch(
        reviews_df=reviews_df,
        batch_size=BATCH_SIZE,
        start_index=START_INDEX,
        poll_interval=POLL_INTERVAL
    )
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Total reviews processed: {len(parsed_df)}")
    print(f"Successfully parsed: {parsed_df['parsed_dishes'].notna().sum()}")
    print(f"Failed to parse: {parsed_df['parsed_dishes'].isna().sum()}")
    
    # Show a sample
    print("\nSample parsed review:")
    sample = parsed_df[parsed_df['parsed_dishes'].notna()].iloc[0] if len(parsed_df[parsed_df['parsed_dishes'].notna()]) > 0 else None
    if sample is not None:
        print(f"Restaurant: {sample['place_name']}")
        print(f"Rating: {sample['rating']}")
        print(f"Review: {sample['review_text'][:100]}...")
        print(f"Parsed dishes: {json.dumps(sample['parsed_dishes'], indent=2)}")


if __name__ == "__main__":
    main()
