import sys
import os
# Add the src directory (where main.py is located) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
import numpy as np
import main  

# --- Test for sentiment_label function ---

def test_sentiment_label_positive():
    # For scores above 0.2, expect positive sentiment
    assert main.sentiment_label(0.5) == "😊 Positive"

def test_sentiment_label_neutral():
    # For scores between -0.2 and 0.2, expect neutral sentiment
    assert main.sentiment_label(0.0) == "😐 Neutral"

def test_sentiment_label_negative():
    # For scores below -0.2, expect negative sentiment
    assert main.sentiment_label(-0.5) == "😞 Negative"

# --- Test for generate_goal_recommendations function ---

def test_generate_goal_recommendations_house():
    goal_text = "I want to buy a house"
    content_extra, product_extra = main.generate_goal_recommendations(goal_text)
    expected_content = main.content_recommendations_map.get(
        "Real Estate Interest", "🏠 Blog: Tips for First-Time Home Buyers"
    )
    expected_product_home = main.product_recommendations_map.get(
        "Home Loan", "🏡 Home Loan with Competitive Rates"
    )
    expected_product_savings = main.product_recommendations_map.get(
        "Savings", "💳 High-Yield Savings Account for Home Down Payment"
    )
    assert expected_content in content_extra
    assert expected_product_home in product_extra
    assert expected_product_savings in product_extra

def test_generate_goal_recommendations_retirement():
    goal_text = "I am planning for retirement"
    content_extra, product_extra = main.generate_goal_recommendations(goal_text)
    expected_content = main.content_recommendations_map.get(
        "Retirement Advice", "📹 Video: Planning Your Retirement 101"
    )
    expected_product_retirement = main.product_recommendations_map.get(
        "Retirement Saving", "💼 Retirement Savings Plan (401k/Pension)"
    )
    expected_product_etfs = main.product_recommendations_map.get(
        "ETFs", "📉 Low-Cost ETF Investment Fund"
    )
    assert expected_content in content_extra
    assert expected_product_retirement in product_extra
    assert expected_product_etfs in product_extra

# --- Test for recommend_for_user function ---

def test_recommend_for_user_no_user():
    # Override the global profiles with an empty dataframe for testing non-existent user
    main.profiles = pd.DataFrame(columns=[
        'Customer_Id', 'Interests_List', 'Latest_Intent',
        'Avg_Sentiment', 'Preferences_List', 'Cluster'
    ])
    content_recs, product_recs = main.recommend_for_user(9999)
    assert content_recs == []
    assert product_recs == []

def test_recommend_for_user_with_profile():
    # Create a dummy profiles dataframe for testing recommend_for_user
    data = {
        'Customer_Id': [1],
        'Age': [30],
        'Interests': ["Tech, Finance"],
        'Interests_List': [["Tech", "Finance"]],
        'Preferences_List': [["Stocks", "Savings"]],
        'Gender': ["M"],
        'Gender_Binary': [0],
        'Age_norm': [0.5],
        'Income per year': [50000],
        'Income_norm': [0.5],
        'Total_Spend': [1000.0],
        'Average_Spend': [100.0],
        'Num_Transactions': [10],
        'Purchase_Categories': [["Electronics"]],
        'Latest_Intent': ["Tech Purchase Interest"],
        'Avg_Sentiment': [0.1],
        'Cluster': [0]
    }
    dummy_profiles = pd.DataFrame(data)
    main.profiles = dummy_profiles  # Override global profiles in the main module

    content_recs, product_recs = main.recommend_for_user(1)
    
    expected_content = main.content_recommendations_map.get(
        "Tech Purchase Interest", "💻 Article: Top 10 New Gadgets in FinTech"
    )
    expected_product = main.product_recommendations_map.get(
        "Stocks", "📈 Zero-Commission Stock Trading Account"
    )
    assert expected_content in content_recs
    # Since recommendations might vary based on the logic, check that at least one product recommendation exists
    assert len(product_recs) > 0
