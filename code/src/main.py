import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import pipeline
from collections import Counter
from PIL import Image  # NEW: for image processing

# Configure Streamlit page
st.set_page_config(page_title="Hyper-Personalization Dashboard", layout="wide")

# --- Custom CSS for Enhanced UI ---
custom_css = """
<style>
/* Main app background with gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1e1e2f, #2e2e3e);
    color: #f0f0f0;
}

/* Header styling */
h1 {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-weight: 700;
    color: #3eb489;
}

/* Sidebar styling with gradient */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2e2e3e, #1e1e2f);
    color: #f0f0f0;
}

/* Card styles for recommendation sections */
.card {
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 10px;
    box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
}
.card-green {
    background-color: #3eb489;
    color: #fff;
}
.card-red {
    background-color: #ff4c4c;
    color: #fff;
}
.card-blue {
    background-color: #30ba8f;
    color: #fff;
}

/* Button styling */
.stButton > button {
    background-color: #3eb489;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    color: #fff;
    font-size: 16px;
    cursor: pointer;
}
.stButton > button:hover {
    background-color: #34a380;
}

/* Styling for metrics */
.metric {
    font-size: 20px;
    font-weight: bold;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("🤖 AI-Driven Hyper-Personalization Dashboard")

# --- Step 1: Load Data ---
file_path = "C:/Users/Sanjana Rao/Downloads/Hyper-Personalized-NewDataset.xlsx"
@st.cache_data
def load_data(path):
    profiles_ind = pd.read_excel(path, sheet_name="Customer Profile (Individual)")
    social_data = pd.read_excel(path, sheet_name="Social Media Sentiment")
    transactions = pd.read_excel(path, sheet_name="Transaction History")
    return profiles_ind, social_data, transactions

profiles_ind, social_data, transactions = load_data(file_path)
profiles = profiles_ind.copy()

# --- Step 2: Data Preprocessing and Feature Engineering ---
profiles['Interests_List'] = profiles['Interests'].fillna('').apply(lambda x: [i.strip() for i in x.split(',') if i])
profiles['Preferences_List'] = profiles['Preferences'].fillna('').apply(lambda x: [p.strip() for p in x.split(',') if p])

profiles['Gender_Binary'] = profiles['Gender'].map({'M': 0, 'F': 1})
profiles['Age_norm'] = (profiles['Age'] - profiles['Age'].min()) / (profiles['Age'].max() - profiles['Age'].min())
profiles['Income_norm'] = (profiles['Income per year'] - profiles['Income per year'].min()) / \
                          (profiles['Income per year'].max() - profiles['Income per year'].min())

trans_summary = transactions.groupby('Customer_Id').agg({
    'Product_Id': 'count',
    'Category': lambda x: list(set(x)),
    'Amount': 'sum'
}).rename(columns={'Product_Id': 'Num_Transactions', 'Category': 'Purchase_Categories', 'Amount': 'Total_Spend'})
trans_summary['Average_Spend'] = trans_summary['Total_Spend'] / trans_summary['Num_Transactions']

profiles = profiles.merge(trans_summary, how='left', on='Customer_Id')
profiles['Num_Transactions'] = profiles['Num_Transactions'].fillna(0).astype(int)
profiles['Purchase_Categories'] = profiles['Purchase_Categories'].fillna('').apply(lambda x: x if isinstance(x, list) else [])
profiles['Total_Spend'] = profiles['Total_Spend'].fillna(0.0)
profiles['Average_Spend'] = profiles['Average_Spend'].fillna(0.0)

sentiment_summary = social_data.sort_values(by='Timestamp').groupby('Customer_Id').agg({
    'Sentiment_Score': 'mean',
    'Intent': 'last'
}).rename(columns={'Sentiment_Score': 'Avg_Sentiment', 'Intent': 'Latest_Intent'})
profiles = profiles.merge(sentiment_summary, how='left', on='Customer_Id')
profiles['Avg_Sentiment'] = profiles['Avg_Sentiment'].fillna(0.0)
profiles['Latest_Intent'] = profiles['Latest_Intent'].fillna('')

mlb_interests = MultiLabelBinarizer()
mlb_prefs = MultiLabelBinarizer()
interests_onehot = mlb_interests.fit_transform(profiles['Interests_List'])
prefs_onehot = mlb_prefs.fit_transform(profiles['Preferences_List'])

X_features = np.hstack([
    profiles[['Age_norm', 'Income_norm', 'Gender_Binary']].values,
    interests_onehot,
    prefs_onehot
])
kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
profiles['Cluster'] = kmeans.fit_predict(X_features)

# --- Step 3: Sentiment Analysis Setup ---
try:
    sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
except Exception:
    sentiment_analyzer = None
    st.error("Error loading sentiment analysis model. Check your internet connection and model availability.")

def sentiment_label(score):
    if score > 0.2:
        return "😊 Positive"
    elif score < -0.2:
        return "😞 Negative"
    else:
        return "😐 Neutral"

# --- Step 4: Recommendation Mappings and Function ---
content_recommendations_map = {
    "Real Estate Interest": "🏠 Blog: Latest Trends in the Real Estate Market",
    "Tech Purchase Interest": "💻 Article: Top 10 New Gadgets in FinTech",
    "Cash Flow Concern": "💡 Guide: 5 Tips for Better Cash Flow Management",
    "Retirement Advice": "📹 Video: Planning Your Retirement 101",
    "Loan Interest": "📄 Article: How to Secure Low-Interest Loans",
    "Credit Card Usage": "📊 Infographic: Maximizing Credit Card Rewards",
    "Investment Concern": "📈 Guide: Diversifying Your Investment Portfolio",
    "Digital Banking Interest": "🖥️ Blog: Benefits of Digital Banking Services",
    "Wealth Management": "📑 Whitepaper: Modern Wealth Management Strategies",
    "Crypto Enthusiasm": "📰 Newsletter: Latest Cryptocurrency Market Updates"
}
product_recommendations_map = {
    "Health Insurance": "🩺 Health Insurance Plan (Personalized Coverage)",
    "Gym Subscriptions": "🏋️ Discounted Gym Membership (via Partner Program)",
    "Fitness Apps": "📱 Premium Fitness Tracking App Subscription",
    "Cryptocurrency": "💰 Cryptocurrency Investment Account",
    "Investment Planning": "📊 Personal Investment Advisory Service",
    "Passive Income Streams": "🤖 Automated Passive Investment Portfolio",
    "Home Loan": "🏡 Home Loan with Competitive Interest Rates",
    "Retirement Saving": "💼 Retirement Savings Plan (401k/Pension)",
    "ETFs": "📉 Low-Cost ETF Investment Fund",
    "Tech Conferences": "🎫 Invite to Exclusive Tech Finance Conference",
    "AI Innovations": "🤖 AI-Powered Investment Analysis Tool",
    "Discounts": "💵 Exclusive Cashback and Discounts Program",
    "New Arrivals": "🆕 Early Access to New Financial Product Arrivals",
    "Premium Memberships": "⭐ Premium Banking Membership with VIP Benefits",
    "Savings": "💳 High-Yield Savings Account",
    "Budget Shopping": "🛍️ Personalized Budgeting App with Shopping Deals",
    "Stocks": "📈 Zero-Commission Stock Trading Account",
    "Financial Freedom": "🎯 Personal Finance Coaching Program"
}

def recommend_for_user(user_id):
    try:
        user_profile = profiles[profiles['Customer_Id'] == user_id].iloc[0]
    except IndexError:
        return [], []
    content_recs = []
    product_recs = []
    intent = user_profile['Latest_Intent']
    avg_sent = user_profile['Avg_Sentiment']
    if intent and intent in content_recommendations_map:
        content_recs.append(content_recommendations_map[intent])
    else:
        if user_profile['Interests_List']:
            main_interest = user_profile['Interests_List'][0]
            if main_interest in product_recommendations_map:
                content_recs.append(f"Insights on {main_interest} - {product_recommendations_map[main_interest]}")
            else:
                content_recs.append(f"Latest news and tips about {main_interest}")
    if avg_sent < -0.2:
        content_recs.append("💬 Send retention offer with personalized Support Article: We're here to help with your financial concerns")
    elif avg_sent > 0.8:
        content_recs.append("Congratulations! Check out advanced resources to further your financial journey")
    for pref in user_profile['Preferences_List']:
        rec_product = product_recommendations_map.get(pref, pref)
        if rec_product not in product_recs:
            product_recs.append(rec_product)
        if len(product_recs) >= 3:
            break
    cluster_id = user_profile['Cluster']
    cluster_peers = profiles[profiles['Cluster'] == cluster_id]
    pref_counter = Counter()
    for prefs in cluster_peers['Preferences_List']:
        for p in prefs:
            pref_counter[p] += 1
    for pref, _ in pref_counter.most_common():
        if pref not in user_profile['Preferences_List']:
            suggestion = product_recommendations_map.get(pref, pref)
            if suggestion not in product_recs:
                product_recs.append(suggestion + "  (Popular among similar users)")
            break
    return content_recs, product_recs

# --- Step 5.5: AI Models for Voice and Image Inputs (NEW) ---
try:
    asr_model = pipeline("automatic-speech-recognition", model="openai/whisper-base")
except Exception:
    asr_model = None
    st.warning("Speech recognition model could not be loaded. Voice input will be disabled.")
try:
    image_captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
except Exception:
    image_captioner = None
    st.warning("Image captioning model could not be loaded. Image input will be disabled.")

def generate_goal_recommendations(goal_text):
    goal_text = goal_text.lower()
    content_extra = []
    product_extra = []
    if any(word in goal_text for word in ["house", "home", "real estate"]):
        content_extra.append(content_recommendations_map.get("Real Estate Interest", "🏠 Blog: Tips for First-Time Home Buyers"))
        product_extra.append(product_recommendations_map.get("Home Loan", "🏡 Home Loan with Competitive Rates"))
        product_extra.append(product_recommendations_map.get("Savings", "💳 High-Yield Savings Account for Home Down Payment"))
    elif "retire" in goal_text or "retirement" in goal_text:
        content_extra.append(content_recommendations_map.get("Retirement Advice", "📹 Video: Planning Your Retirement 101"))
        product_extra.append(product_recommendations_map.get("Retirement Saving", "💼 Retirement Savings Plan (401k/Pension)"))
        product_extra.append(product_recommendations_map.get("ETFs", "📉 Low-Cost ETF Investment Fund"))
    elif any(word in goal_text for word in ["travel", "vacation", "holiday", "trip"]):
        content_extra.append("✈️ Guide: Budgeting for Your Dream Vacation")
        product_extra.append(product_recommendations_map.get("Savings", "💳 High-Yield Savings Account for Travel Fund"))
        product_extra.append("🌍 International Index Fund to Grow Travel Savings")
    elif any(word in goal_text for word in ["education", "college", "tuition", "school"]):
        content_extra.append("🎓 Guide: Saving Early for Education Expenses")
        product_extra.append("🎓 Education Savings Plan (College Fund Account)")
        product_extra.append(product_recommendations_map.get("Investment Planning", "📊 Personal Investment Advisory Service"))
    elif any(word in goal_text for word in ["car", "vehicle", "auto"]):
        content_extra.append("🚗 Article: Financial Planning for a Car Purchase")
        product_extra.append("🚗 Auto Loan Offer with Low Interest Rates")
        product_extra.append(product_recommendations_map.get("Savings", "💳 High-Yield Savings Account for Car Fund"))
    elif "debt" in goal_text or "loan" in goal_text or "credit card" in goal_text:
        content_extra.append("💡 Guide: Managing and Reducing Your Debt")
        product_extra.append(product_recommendations_map.get("Loan Interest", "📄 Tips: Securing Low-Interest Loans"))
        product_extra.append("💳 Credit Card Balance Optimization Program")
    elif any(word in goal_text for word in ["invest", "investment", "stock", "portfolio"]):
        content_extra.append(content_recommendations_map.get("Investment Concern", "📈 Guide: Diversifying Your Investment Portfolio"))
        product_extra.append(product_recommendations_map.get("Stocks", "📈 Zero-Commission Stock Trading Account"))
        product_extra.append(product_recommendations_map.get("Passive Income Streams", "🤖 Automated Passive Investment Portfolio"))
    elif "save" in goal_text or "savings" in goal_text:
        content_extra.append("💡 Tips: Effective Saving Strategies to Reach Your Goal")
        product_extra.append(product_recommendations_map.get("Savings", "💳 High-Yield Savings Account"))
        product_extra.append(product_recommendations_map.get("Budget Shopping", "🛍️ Personalized Budgeting App for Expense Tracking"))
    else:
        content_extra.append("📖 Personalized financial tips will be prepared for your goal.")
        product_extra.append("💼 Customized investment portfolio recommendation (AI-generated)")
    return content_extra, product_extra

# --- Step 6: Dashboard UI ---
st.sidebar.markdown("## Select a Customer")
user_ids = profiles['Customer_Id'].unique().tolist()
selected_user = st.sidebar.selectbox("Customer ID", sorted(user_ids))

if selected_user:
    user_profile = profiles[profiles['Customer_Id'] == selected_user].iloc[0]
    st.markdown(f"## User Details: **{selected_user}**")
    col1, col2 = st.columns(2)
    col1.write(f"**Age:** {user_profile['Age']}")
    col1.write(f"**Interests:** {', '.join(user_profile['Interests_List']) if user_profile['Interests_List'] else 'None'}")
    col2.write(f"**Latest Sentiment:** {sentiment_label(user_profile['Avg_Sentiment'])}")
    col2.write(f"**Cluster:** {user_profile['Cluster']}")
    
    st.markdown("---")
    st.markdown("### 💰 Financial Summary")
    show_amounts = st.checkbox("Show Financial Details", value=False)
    fin_col1, fin_col2 = st.columns(2)
    if show_amounts:
        fin_col1.metric("Total Spend", f"${user_profile['Total_Spend']:,.2f}")
        fin_col2.metric("Avg Spend per Txn", f"${user_profile['Average_Spend']:,.2f}")
    else:
        fin_col1.metric("Total Spend", "******")
        fin_col2.metric("Avg Spend per Txn", "******")

    st.markdown("---")
    st.markdown("#### Describe a Financial Goal:")
    
    input_col1, input_col2 = st.columns(2)
    with input_col1:
        audio_file = st.file_uploader("🎙️ Voice Input (upload audio)", type=["wav", "mp3", "m4a"])
    with input_col2:
        image_file = st.file_uploader("🖼️ Image Input (upload image)", type=["png", "jpg", "jpeg"])
    
    content_recs_extra = []
    product_recs_extra = []
    if audio_file is not None:
        if asr_model:
            with st.spinner("Transcribing voice input..."):
                with open("temp_audio.wav", "wb") as f:
                    f.write(audio_file.read())
                transcription = asr_model("temp_audio.wav")
            if transcription:
                if isinstance(transcription, dict) and "text" in transcription:
                    voice_goal_text = transcription["text"]
                elif isinstance(transcription, list) and "text" in transcription[0]:
                    voice_goal_text = transcription[0]["text"]
                else:
                    voice_goal_text = str(transcription)
                st.write(f"**Transcribed Goal:** *{voice_goal_text}*")
                content_recs_extra, product_recs_extra = generate_goal_recommendations(voice_goal_text)
            else:
                st.error("Speech-to-text model is not available. Please try text input instead.")
    elif image_file is not None:
        if image_captioner:
            with st.spinner("Interpreting image..."):
                img = Image.open(image_file)
                caption_result = image_captioner(img)
            if caption_result:
                if isinstance(caption_result, list) and "generated_text" in caption_result[0]:
                    image_goal_text = caption_result[0]["generated_text"]
                elif isinstance(caption_result, dict) and "generated_text" in caption_result:
                    image_goal_text = caption_result["generated_text"]
                else:
                    image_goal_text = str(caption_result)
                st.write(f"**Interpreted Goal:** *{image_goal_text}*")
                content_recs_extra, product_recs_extra = generate_goal_recommendations(image_goal_text)
        else:
            st.error("Image captioning model is not available.")
    
    st.markdown("---")
    content_recs, product_recs = recommend_for_user(selected_user)
    if content_recs:
        st.markdown("### 📰 Content Recommendations")
        for rec in content_recs:
            if "Send retention offer" in rec:
                st.markdown(f'<div class="card card-red">{rec}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="card card-green">{rec}</div>', unsafe_allow_html=True)
    if product_recs:
        st.markdown("### 🛒 Product Recommendations")
        for rec in product_recs:
            st.markdown(f'<div class="card card-blue">{rec}</div>', unsafe_allow_html=True)
    if content_recs_extra or product_recs_extra:
        st.markdown("### 🎯 Recommendations Based on Your Goal")
        if content_recs_extra:
            for rec in content_recs_extra:
                st.markdown(f'<div class="card card-green">{rec}</div>', unsafe_allow_html=True)
        if product_recs_extra:
            for rec in product_recs_extra:
                st.markdown(f'<div class="card card-blue">{rec}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Overall Dataset")
    with st.expander("📊 User Segmentation Overview", expanded=False):
        cluster_counts = profiles['Cluster'].value_counts().sort_index()
        fig, ax = plt.subplots()
        ax.bar(cluster_counts.index.astype(str), cluster_counts.values, color='skyblue')
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of Users")
        ax.set_title("Distribution of Users by Cluster")
        st.pyplot(fig)
    with st.expander("⚠️ At-Risk Customers", expanded=False):
        at_risk_df = profiles[profiles['Avg_Sentiment'] < -0.2][['Customer_Id']]
        at_risk_df['Action'] = '📧 Send retention offer with personalized support'
        at_risk_df = at_risk_df.reset_index(drop=True)
        st.table(at_risk_df)
