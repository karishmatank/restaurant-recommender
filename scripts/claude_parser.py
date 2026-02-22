"""
Claude API integration for parsing restaurant reviews.

This script uses the Anthropic's Claude API to extract structured information
from restaurant reviews, including dish names, sentiment, and characteristics.
"""

import json
import time
import pandas as pd
from textwrap import dedent
from anthropic import Anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, DISH_EXTRACTION_SCHEMA, PARSED_REVIEWS_DIR, DISH_EXTRACTION_PROMPT
from load_reviews_utils import load_processed_reviews


def create_review_parsing_prompt(review_text):
    """
    Create the prompt for Claude to parse a restaurant review.
    
    Args:
        review_text: The review text to parse
    """
    prompt = dedent(f"""
        {DISH_EXTRACTION_PROMPT}
        
        Review: {review_text}
    """)
    
    return prompt


def parse_review_with_claude(client, review_text, max_retries=3, retry_delay=1.0):
    """
    Parse a single review using Claude API.
    
    Args:
        client: Anthropic client instance
        review_text: The review text to parse
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries (seconds)
    
    Returns:
        Parsed review data as dictionary, or None if parsing failed
    """
    if not review_text or pd.isna(review_text):
        return None
    
    prompt = create_review_parsing_prompt(review_text)
    
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": DISH_EXTRACTION_SCHEMA
                    }
                }
            )
            
            # Extract the JSON response from content
            if message.content and len(message.content) > 0:
                response_text = message.content[0].text
                
                # Parse the JSON response
                try:
                    parsed_data = json.loads(response_text)
                    return parsed_data
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    print(f"Response: {response_text[:200]}...")
                    return {"raw_response": response_text, "parse_error": str(e)}
        
        except Exception as e:
            print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return
    
    return


def parse_reviews_batch(reviews_df, batch_size=None, start_index=0, rate_limit_delay=0.5):
    """
    Parse multiple reviews using Claude API.
    
    Args:
        reviews_df: DataFrame with review data (must have 'review_text' column)
        batch_size: Number of reviews to process (None = all reviews)
        start_index: Index to start processing from
        rate_limit_delay: Delay between API calls (seconds)
    
    Returns:
        DataFrame with original data plus parsed dish information
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables. Please set it in .env file.")
    
    # Anthropic client. Automatically reads ANTHROPIC_API_KEY from environment
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    # Filter out reviews without text
    reviews_with_text = reviews_df[reviews_df['review_text'].notna()].copy()
    
    if batch_size is not None:
        end_index = min(start_index + batch_size, len(reviews_with_text))
        reviews_to_process = reviews_with_text.iloc[start_index:end_index]
    else:
        reviews_to_process = reviews_with_text.iloc[start_index:]
    
    print(f"Processing {len(reviews_to_process)} reviews (index {start_index} to {start_index + len(reviews_to_process) - 1})...")
    
    parsed_results = []
    
    for idx, row in reviews_to_process.iterrows():
        print(f"\n[{idx + 1}/{len(reviews_with_text)}] Parsing review from {row.get('place_name', 'Unknown')}...")
        
        parsed_data = parse_review_with_claude(client, row['review_text'])
        
        # Create result record
        result = {
            'review_index': idx,
            'place_id': row.get('place_id'),
            'place_name': row.get('place_name'),
            'author_id': row.get('author_id'),
            'rating': row.get('rating'),
            'review_text': row.get('review_text'),
            'parsed_dishes': parsed_data
        }
        
        parsed_results.append(result)
        
        if parsed_data:
            num_dishes = len(parsed_data.get('dishes_mentioned', []))
            print(f"Extracted {num_dishes} dish(es)")
        else:
            print(f"Parsing failed")
        
        # Rate limiting
        time.sleep(rate_limit_delay)
    
    # Create DataFrame from results
    results_df = pd.DataFrame(parsed_results)
    
    # Save results
    timestamp = int(time.time())
    output_file = PARSED_REVIEWS_DIR / f"parsed_reviews_{start_index}_{start_index + len(results_df) - 1}_{timestamp}.json"
    results_df.to_json(output_file, orient='records', indent=2)
    print(f"\nSaved parsed results to: {output_file}")
    
    # Also save as CSV (without nested JSON)
    csv_file = PARSED_REVIEWS_DIR / f"parsed_reviews_{start_index}_{start_index + len(results_df) - 1}_{timestamp}.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"Saved CSV to: {csv_file}")
    
    return results_df


def main():
    """
    Main function to parse reviews with Claude.
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
    
    # Parse a batch of reviews
    # Start with a small batch for testing (e.g., first 10 reviews)
    BATCH_SIZE = 10  # Adjust this number
    START_INDEX = 0
    
    print(f"\nParsing batch of {BATCH_SIZE} reviews starting from index {START_INDEX}...")
    
    parsed_df = parse_reviews_batch(
        reviews_df=reviews_df,
        batch_size=BATCH_SIZE,
        start_index=START_INDEX,
        rate_limit_delay=0.5  # Adjust based on your rate limits
    )
    
    # Print summary
    print("\n" + "="*60)
    print("PARSING SUMMARY")
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
