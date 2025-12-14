import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Data Analysis", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

st.title("📊 Human Study Data Analysis")

# MongoDB connection
@st.cache_resource
def init_mongodb():
    """Initialize MongoDB connection"""
    try:
        connection_string = os.getenv("MONGODB_CONNECTION_STRING")
        if not connection_string:
            return None
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        return client
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        return None

mongodb_client = init_mongodb()

if mongodb_client is None:
    st.error("⚠️ Please set MONGODB_CONNECTION_STRING in .env file")
    st.stop()

db = mongodb_client["prj-nemo"]
collection = db["human-study-pilot"]

# Sidebar for filters
with st.sidebar:
    st.header("🔍 Filters")
    
    # Date range filter
    st.subheader("Date Range")
    use_date_filter = st.checkbox("Filter by date range", value=False)
    
    if use_date_filter:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=None)
        with col2:
            end_date = st.date_input("End Date", value=None)
    else:
        start_date = None
        end_date = None
    
    # Build query filter
    query_filter = {}
    if use_date_filter and (start_date or end_date):
        timestamp_filter = {}
        if start_date:
            start_datetime = datetime.combine(start_date, datetime.min.time())
            timestamp_filter["$gte"] = start_datetime
        if end_date:
            end_datetime = datetime.combine(end_date, datetime.max.time())
            timestamp_filter["$lte"] = end_datetime
        if timestamp_filter:
            query_filter["timestamp"] = timestamp_filter
    
    st.markdown("---")
    st.subheader("📈 Statistics")
    total_docs = collection.count_documents({})
    filtered_docs = collection.count_documents(query_filter) if query_filter else total_docs
    st.metric("Total Documents", total_docs)
    st.metric("Filtered Documents", filtered_docs)

# Load data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(query_filter):
    """Load data from MongoDB"""
    cursor = collection.find(query_filter)
    data = list(cursor)
    if len(data) > 0:
        df = pd.DataFrame(data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return pd.DataFrame()

df = load_data(query_filter)

if df.empty:
    st.warning("⚠️ No data found for the specified filters.")
    st.stop()

# Display basic info
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Responses", len(df))
with col2:
    st.metric("Unique Participants", df['participant_id'].nunique())
with col3:
    st.metric("Unique Samples", df['sample_key'].nunique())
with col4:
    if 'timestamp' in df.columns:
        date_range = f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
        st.metric("Date Range", date_range)

st.markdown("---")

# Tabs for different analyses
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Rankings", "👥 Participants", "💬 Feedback", "📸 Samples", "📥 Export"])

with tab1:
    st.header("Ranking Analysis")
    
    # Use the filtered dataframe from sidebar
    # Show info about current filter
    if 'timestamp' in df.columns and len(df) > 0:
        st.info(f"📊 Showing rankings for {len(df)} responses (filtered by sidebar date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()})")
    
    methods = ['retrieval', 'pixel', 'change', 'scitx']
    
    # Extract rankings from filtered data (df is already filtered by sidebar)
    ranking_data = []
    for _, row in df.iterrows():
        rankings = row.get('rankings', {})
        for method in methods:
            if method in rankings:
                ranking_data.append({
                    'participant_id': row['participant_id'],
                    'sample_key': row['sample_key'],
                    'method': method,
                    'rank': rankings[method],
                    'timestamp': row.get('timestamp', None)
                })
    
    if ranking_data:
        ranking_df = pd.DataFrame(ranking_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ranking Distribution")
            ranking_pivot = ranking_df.groupby(['method', 'rank']).size().unstack(fill_value=0)
            st.bar_chart(ranking_pivot)
        
        with col2:
            st.subheader("Average Rank")
            avg_ranks = ranking_df.groupby('method')['rank'].mean().sort_values()
            st.bar_chart(avg_ranks)
        
        st.subheader("Ranking Statistics")
        col1, col2 = st.columns(2)
        with col1:
            # Count of each rank by method
            rank_counts = ranking_df.groupby(['method', 'rank']).size().reset_index(name='count')
            st.write("**Rank Counts**")
            st.dataframe(rank_counts.pivot(index='method', columns='rank', values='count').fillna(0), use_container_width=True)
        with col2:
            st.write("**Average Rank (lower is better)**")
            st.dataframe(avg_ranks.to_frame('Average Rank'), use_container_width=True)
            
    else:
        st.info("No ranking data available")

with tab2:
    st.header("Participant Statistics")
    
    participant_stats = df.groupby('participant_id').agg({
        'sample_key': 'count',
        'timestamp': ['min', 'max']
    }).reset_index()
    participant_stats.columns = ['participant_id', 'num_samples', 'first_submission', 'last_submission']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Samples per Participant")
        st.bar_chart(participant_stats.set_index('participant_id')['num_samples'])
    
    with col2:
        st.subheader("Statistics")
        st.metric("Average Samples", f"{participant_stats['num_samples'].mean():.2f}")
        st.metric("Min Samples", participant_stats['num_samples'].min())
        st.metric("Max Samples", participant_stats['num_samples'].max())
    
    st.subheader("Participant Details")
    st.dataframe(participant_stats, use_container_width=True)

with tab3:
    st.header("Feedback Analysis")
    
    feedback_data = []
    for _, row in df.iterrows():
        feedbacks = row.get('ranking_feedback', {})
        rankings = row.get('rankings', {})
        for method in methods:
            feedback = feedbacks.get(method, '')
            if feedback and feedback.strip():
                feedback_data.append({
                    'method': method,
                    'rank': rankings.get(method, None),
                    'feedback_length': len(feedback),
                    'participant_id': row['participant_id'],
                    'sample_key': row['sample_key'],
                    'feedback': feedback
                })
    
    if feedback_data:
        feedback_df = pd.DataFrame(feedback_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Feedback Statistics")
            st.metric("Total Feedback Entries", len(feedback_df))
            st.metric("Average Length", f"{feedback_df['feedback_length'].mean():.1f} characters")
            st.metric("Total Characters", f"{feedback_df['feedback_length'].sum():,}")
        
        with col2:
            st.subheader("Feedback Count by Method")
            feedback_counts = feedback_df.groupby('method').size()
            st.bar_chart(feedback_counts)
        
        st.subheader("Feedback Details")
        # Allow filtering
        selected_method = st.selectbox("Filter by Method", ["All"] + list(feedback_df['method'].unique()))
        if selected_method != "All":
            filtered_feedback = feedback_df[feedback_df['method'] == selected_method]
        else:
            filtered_feedback = feedback_df
        
        st.dataframe(
            filtered_feedback[['method', 'rank', 'feedback_length', 'participant_id', 'sample_key', 'feedback']],
            use_container_width=True,
            height=400
        )
    else:
        st.info("No feedback data available")

with tab4:
    st.header("Sample-level Analysis")
    
    sample_stats = df.groupby('sample_key').agg({
        'participant_id': 'count',
        'true_class': 'first',
        'predicted_class': 'first'
    }).reset_index()
    sample_stats.columns = ['sample_key', 'num_responses', 'true_class', 'predicted_class']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Responses per Sample")
        st.bar_chart(sample_stats.set_index('sample_key')['num_responses'].head(20))
    
    with col2:
        st.subheader("Statistics")
        st.metric("Total Samples", len(sample_stats))
        st.metric("Average Responses", f"{sample_stats['num_responses'].mean():.2f}")
        st.metric("Most Evaluated", sample_stats.loc[sample_stats['num_responses'].idxmax(), 'sample_key'])
    
    st.subheader("Sample Details")
    st.dataframe(sample_stats.sort_values('num_responses', ascending=False), use_container_width=True)

with tab5:
    st.header("Data Export")
    
    # Export to CSV
    export_data = []
    for _, row in df.iterrows():
        rankings = row.get('rankings', {})
        feedbacks = row.get('ranking_feedback', {})
        for method in methods:
            export_data.append({
                'participant_id': row['participant_id'],
                'sample_key': row['sample_key'],
                'timestamp': row.get('timestamp', ''),
                'true_class': row.get('true_class', ''),
                'predicted_class': row.get('predicted_class', ''),
                'method': method,
                'rank': rankings.get(method, None),
                'feedback': feedbacks.get(method, ''),
                'cei_score': row.get('cei_scores', {}).get(method, None),
                'explanation_order': row.get('explanation_order', {}).get(method, None)
            })
    
    export_df = pd.DataFrame(export_data)
    
    st.subheader("Export Data")
    st.dataframe(export_df.head(100), use_container_width=True)
    
    # Download button
    csv = export_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"human_study_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    st.subheader("Summary")
    st.json({
        "total_participants": df['participant_id'].nunique(),
        "total_samples": df['sample_key'].nunique(),
        "total_responses": len(df),
        "date_range": {
            "start": str(df['timestamp'].min()) if 'timestamp' in df.columns else None,
            "end": str(df['timestamp'].max()) if 'timestamp' in df.columns else None
        }
    })

