"""
Analysis functions for the validation portion of the restaurant recommender project.
"""

import pandas as pd
from load_reviews_utils import load_processed_reviews, load_parsed_reviews


def analyze_cross_restaurant_reviewers(df):
    """
    Analyze how many reviewers reviewed multiple restaurants.
    
    Args:
        df: DataFrame with reviews data
    
    Returns:
        DataFrame with reviewer statistics
    """
    reviewer_stats = df.groupby("author_id").agg({
        "place_id": "nunique",
        "place_name": lambda x: list(x.unique()),
        "rating": "count"
    }).reset_index()
    
    reviewer_stats.columns = ["author_id", "num_restaurants", "restaurants", "total_reviews"]
    
    # Sort by number of restaurants reviewed
    reviewer_stats = reviewer_stats.sort_values("num_restaurants", ascending=False)
    
    return reviewer_stats

def collaborative_filtering_feasibility():
    """
    Compile metrics to test for collaborative filtering feasibility
    """
    print("\n****** Collaborative filtering feasibility tests ******")

    try:
        df = load_processed_reviews()
        print(f"\nLoaded {len(df)} reviews")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print(f"Unique restaurants: {df['place_id'].nunique()}")

    unique_reviewers = df['author_id'].nunique()
    print(f"Unique reviewers: {unique_reviewers}")
    
    # Analyze cross-restaurant reviewers
    reviewer_stats = analyze_cross_restaurant_reviewers(df)
    multiple_restaurant_reviewers = reviewer_stats[reviewer_stats["num_restaurants"] > 1]
    over_3_restaurant_reviewers = reviewer_stats[reviewer_stats["num_restaurants"] > 3]

    print(f"\nReviewers who reviewed multiple restaurants:")
    print(multiple_restaurant_reviewers)

    print(f"\n% of reviewers who reviews multiple restaurants:")
    print(len(multiple_restaurant_reviewers) / unique_reviewers)

    print(f"\n% of reviewers who show up 3+ times:")
    print(len(over_3_restaurant_reviewers) / unique_reviewers)


def content_based_filtering_feasibility():
    """
    Compile metrics to test for content based filtering feasibility
    """
    print("\n****** Content based filtering feasibility tests ******")

    try:
        df = load_parsed_reviews()
        print(f"\nLoaded {len(df)} reviews with text out of 2500")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    dish_count = lambda x: len(x['dishes_mentioned'])
    dishes_with_characteristics_count = lambda x: sum(len(dish['characteristics']) > 0 for dish in x['dishes_mentioned'])
    
    df['dish_count'] = df['parsed_dishes'].apply(dish_count)
    df['dishes_with_characteristics'] = df['parsed_dishes'].apply(dishes_with_characteristics_count)

    reviews_with_dishes = df[df['dish_count'] > 0]
    reviews_with_dish_characteristics = df[df['dishes_with_characteristics'] > 0]

    print(f"\n% of written reviews that name dishes:")
    print(f"{len(reviews_with_dishes)} reviews")
    print(len(reviews_with_dishes) / len(df))

    print(f"\n% of written reviews that name dish characteristics:")
    print(f"{len(reviews_with_dish_characteristics)} reviews")
    print(len(reviews_with_dish_characteristics) / len(df))

if __name__ == "__main__":
    collaborative_filtering_feasibility()
    content_based_filtering_feasibility()
    
