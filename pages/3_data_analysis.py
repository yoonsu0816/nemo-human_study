import os
import json
import streamlit as st
from datetime import datetime, date
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Data Analysis", page_icon=":bar_chart:", layout="wide")

# MongoDB connection
@st.cache_resource
def init_mongodb():
    """Initialize MongoDB connection"""
    try:
        if "mongodb" in st.secrets and "connection_string" in st.secrets["mongodb"]:
            connection_string = st.secrets["mongodb"]["connection_string"]
        elif "MONGODB_CONNECTION_STRING" in os.environ:
            connection_string = os.environ["MONGODB_CONNECTION_STRING"]
        else:
            return None
        
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        return client
    except (ConnectionFailure, Exception) as e:
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        return None

# Initialize MongoDB
mongodb_client = init_mongodb()
db = mongodb_client["prj-nemo"] if mongodb_client is not None else None
responses_collection = db["human-study-pilot"] if db is not None else None

st.title(":bar_chart: Human Study Data Viewer")

if responses_collection is None:
    st.error("⚠️ MongoDB connection not available. Please check your connection settings.")
    st.stop()

# Filters at the top of the page
st.header("🔍 Filters")

# Get unique participant IDs and dataset_model combinations
participant_ids = sorted(responses_collection.distinct("participant_id"))

dataset_model_combinations = []
for response in responses_collection.find({}, {"dataset": 1, "target_model": 1}):
    dataset = response.get("dataset", "")
    model = response.get("target_model", "")
    if dataset and model:
        combo = f"{dataset}_{model}"
        if combo not in dataset_model_combinations:
            dataset_model_combinations.append(combo)
dataset_model_combinations = sorted(dataset_model_combinations)

# Filter layout in columns
filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    selected_participants = st.multiselect(
        "Select Participants:",
        options=participant_ids,
        default=participant_ids if len(participant_ids) > 0 else []
    )

with filter_col2:
    selected_dataset_models = st.multiselect(
        "Select Dataset/Model:",
        options=dataset_model_combinations,
        default=dataset_model_combinations if len(dataset_model_combinations) > 0 else []
    )
    use_date_filter = st.checkbox("Filter by Date Range", value=False)
    filter_complete_only = st.checkbox("Complete participants only (15 documents)", value=False)
    
    date_start = None
    date_end = None
    if use_date_filter:
        # Get min and max dates from data
        all_dates = [r.get("timestamp") for r in responses_collection.find({}, {"timestamp": 1}) if r.get("timestamp")]
        if all_dates:
            min_date = min(all_dates).date() if isinstance(min(all_dates), datetime) else min(all_dates)
            max_date = max(all_dates).date() if isinstance(max(all_dates), datetime) else max(all_dates)
        else:
            min_date = date.today()
            max_date = date.today()
        
        date_start, date_end = st.date_input(
            "Select Date Range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

st.markdown("---")

# Build query
query = {}

# Filter by participants (multiple selection)
if selected_participants:
    query["participant_id"] = {"$in": selected_participants}

# Filter by dataset_model combinations
if selected_dataset_models:
    # Split dataset_model combinations and create OR query
    dataset_model_conditions = []
    for combo in selected_dataset_models:
        parts = combo.split('_', 1)
        if len(parts) == 2:
            dataset, model = parts
            dataset_model_conditions.append({
                "dataset": dataset,
                "target_model": model
            })
    
    if dataset_model_conditions:
        if len(dataset_model_conditions) == 1:
            query.update(dataset_model_conditions[0])
        else:
            query["$or"] = dataset_model_conditions

# Add date filter
if use_date_filter and date_start and date_end:
    query["timestamp"] = {
        "$gte": datetime.combine(date_start, datetime.min.time()),
        "$lte": datetime.combine(date_end, datetime.max.time())
    }

# Get data
all_responses = list(responses_collection.find(query))

# Filter by complete participants (15 documents) if requested
if filter_complete_only:
    # Count documents per participant
    participant_counts = {}
    for response in all_responses:
        participant_id = response.get("participant_id")
        if participant_id:
            participant_counts[participant_id] = participant_counts.get(participant_id, 0) + 1
    
    # Get participants with exactly 15 documents
    complete_participants = {
        pid for pid, count in participant_counts.items() 
        if count == 15
    }
    
    # Filter responses to only include complete participants
    all_responses = [
        response for response in all_responses 
        if response.get("participant_id") in complete_participants
    ]

if len(all_responses) == 0:
    st.info("No data found matching the selected filters.")
    st.stop()

# Statistics
st.header("📊 Statistics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Responses", len(all_responses))
with col2:
    unique_participants = len(set(r["participant_id"] for r in all_responses))
    st.metric("Unique Participants", unique_participants)
with col3:
    unique_samples = len(set((r["dataset"], r["target_model"], r["sample_key"]) for r in all_responses))
    st.metric("Unique Samples", unique_samples)
with col4:
    # Count complete participants (15 documents)
    participant_doc_counts = {}
    for response in all_responses:
        participant_id = response.get("participant_id")
        if participant_id:
            participant_doc_counts[participant_id] = participant_doc_counts.get(participant_id, 0) + 1
    complete_count = sum(1 for count in participant_doc_counts.values() if count == 15)
    st.metric("Complete Participants", complete_count)
# with col4:
#     completion_count = sum(1 for r in all_responses if "completion_comment" in r)
#     st.metric("Completed Studies", completion_count)

st.markdown("---")

# View mode selection
view_mode = st.radio(
    "View Mode:",
    ["Summary Table", "Detailed View", "Ranking Analysis", "Export Data"],
    horizontal=True
)

if view_mode == "Summary Table":
    st.header("📋 Summary Table")
    
    # Prepare summary data
    summary_data = []
    for response in all_responses:
        summary_data.append({
            "Participant ID": response.get("participant_id", "N/A"),
            "Dataset": response.get("dataset", "N/A"),
            "Model": response.get("target_model", "N/A"),
            "Sample Key": response.get("sample_key", "N/A"),
            "True Class": response.get("true_class", "N/A"),
            "Predicted Class": response.get("predicted_class", "N/A"),
            "Has Rankings": "Yes" if response.get("rankings") else "No",
            "Has Highlights": "Yes" if response.get("highlight_feedback") else "No",
            "Timestamp": response.get("timestamp", "N/A")
        })
    
    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True, height=400)

elif view_mode == "Detailed View":
    st.header("🔍 Detailed View")
    
    # Select a specific response to view
    response_options = [
        f"{r['participant_id']} - {r['dataset']}/{r['target_model']} - {r['sample_key']}"
        for r in all_responses
    ]
    
    selected_idx = st.selectbox(
        "Select Response to View:",
        options=range(len(response_options)),
        format_func=lambda x: response_options[x]
    )
    
    if selected_idx is not None:
        response = all_responses[selected_idx]
        
        # Display response details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Information")
            st.write(f"**Participant ID:** {response.get('participant_id', 'N/A')}")
            st.write(f"**Dataset:** {response.get('dataset', 'N/A')}")
            st.write(f"**Model:** {response.get('target_model', 'N/A')}")
            st.write(f"**Sample Key:** {response.get('sample_key', 'N/A')}")
            st.write(f"**True Class:** {response.get('true_class', 'N/A')}")
            st.write(f"**Predicted Class:** {response.get('predicted_class', 'N/A')}")
            st.write(f"**Timestamp:** {response.get('timestamp', 'N/A')}")
        
        with col2:
            st.subheader("Completion Information")
            if "completion_code" in response:
                st.write(f"**Completion Code:** {response.get('completion_code', 'N/A')}")
            if "completion_comment" in response:
                st.write(f"**Comment:** {response.get('completion_comment', 'N/A')}")
            if "completion_timestamp" in response:
                st.write(f"**Completion Time:** {response.get('completion_timestamp', 'N/A')}")
        
        st.markdown("---")
        
        # Rankings
        if "rankings" in response and response["rankings"]:
            st.subheader("Rankings")
            rankings = response["rankings"]
            
            for question_type, method_rankings in rankings.items():
                st.write(f"**{question_type.capitalize()}:**")
                ranking_df = pd.DataFrame([
                    {"Method": method, "Rank": rank}
                    for method, rank in method_rankings.items()
                ])
                ranking_df = ranking_df.sort_values("Rank")
                st.dataframe(ranking_df, use_container_width=True, hide_index=True)
        
        # Highlight Feedback
        if "highlight_feedback" in response and response["highlight_feedback"]:
            st.subheader("Highlight Feedback")
            highlights = response["highlight_feedback"]
            
            for method_id, highlight_list in highlights.items():
                with st.expander(f"Method: {method_id}"):
                    if highlight_list:
                        for idx, highlight in enumerate(highlight_list):
                            st.write(f"**Highlight {idx + 1}:**")
                            st.write(f"- Text: {highlight.get('text', 'N/A')}")
                            st.write(f"- Tag: {highlight.get('tag', 'N/A')}")
                            st.write(f"- Start: {highlight.get('start_idx', 'N/A')}, End: {highlight.get('end_idx', 'N/A')}")
                    else:
                        st.write("No highlights")
        
        # Original Explanations
        if "original_explanations" in response:
            st.subheader("Original Explanations")
            explanations = response["original_explanations"]
            for method_id, explanation in explanations.items():
                with st.expander(f"Method: {method_id}"):
                    st.write(explanation)
        
        # Explanation Order
        if "explanation_order" in response:
            st.subheader("Explanation Order")
            order = response["explanation_order"]
            order_df = pd.DataFrame([
                {"Method": method, "Display Position": pos}
                for method, pos in order.items()
            ])
            order_df = order_df.sort_values("Display Position")
            st.dataframe(order_df, use_container_width=True, hide_index=True)

elif view_mode == "Ranking Analysis":
    st.header("📊 Ranking Analysis")
    
    # Collect all rankings
    criteria_list = ["overall", "factuality", "verbosity", "specificity", "actionability"]
    method_names = ["retrieval", "pixel", "change", "scitx"]
    method_display_names = {
        "retrieval": "Retrieval-based",
        "pixel": "Pixel Attribution",
        "change": "Change of Caption",
        "scitx": "Ours (SciTx)"
    }
    
    # Aggregate rankings by criteria
    rankings_data = {criteria: {method: {1: 0, 2: 0, 3: 0, 4: 0} for method in method_names} for criteria in criteria_list}
    
    for response in all_responses:
        if "rankings" in response and response["rankings"]:
            for criteria, method_rankings in response["rankings"].items():
                if criteria in rankings_data:
                    for method, rank in method_rankings.items():
                        if method in rankings_data[criteria] and rank in rankings_data[criteria][method]:
                            rankings_data[criteria][method][rank] += 1
    
    # Display visualizations for each criteria
    for criteria in criteria_list:
        st.subheader(f"{criteria.capitalize()} Rankings")
        
        # Prepare data for visualization
        chart_data = []
        for method in method_names:
            for rank in [1, 2, 3, 4]:
                count = rankings_data[criteria][method][rank]
                chart_data.append({
                    "Method": method_display_names[method],
                    "Rank": f"Rank {rank}",
                    "Count": count
                })
        
        if chart_data:
            df_chart = pd.DataFrame(chart_data)
            
            # Create stacked bar chart
            fig = px.bar(
                df_chart,
                x="Method",
                y="Count",
                color="Rank",
                title=f"{criteria.capitalize()} - Distribution of Rankings by Method",
                color_discrete_map={
                    "Rank 1": "#2ecc71",  # Green for best
                    "Rank 2": "#3498db",  # Blue
                    "Rank 3": "#f39c12",  # Orange
                    "Rank 4": "#e74c3c"   # Red for worst
                },
                barmode="stack"
            )
            fig.update_layout(
                xaxis_title="Method",
                yaxis_title="Number of Rankings",
                legend_title="Rank",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Create summary table
            summary_table = []
            for method in method_names:
                total = sum(rankings_data[criteria][method].values())
                if total > 0:
                    avg_rank = sum(rank * count for rank, count in rankings_data[criteria][method].items()) / total
                    rank_1_pct = (rankings_data[criteria][method][1] / total) * 100
                    summary_table.append({
                        "Method": method_display_names[method],
                        "Total Responses": total,
                        "Rank 1 Count": rankings_data[criteria][method][1],
                        "Rank 1 %": f"{rank_1_pct:.1f}%",
                        "Average Rank": f"{avg_rank:.2f}"
                    })
            
            if summary_table:
                summary_df = pd.DataFrame(summary_table)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
    
    # Overall summary across all criteria
    st.subheader("Overall Summary")
    
    # Calculate overall statistics
    overall_stats = []
    for method in method_names:
        total_rank_1 = sum(rankings_data[criteria][method][1] for criteria in criteria_list)
        total_responses = sum(sum(rankings_data[criteria][method].values()) for criteria in criteria_list)
        
        if total_responses > 0:
            overall_stats.append({
                "Method": method_display_names[method],
                "Total Rank 1": total_rank_1,
                "Total Responses": total_responses,
                "Rank 1 Rate": f"{(total_rank_1 / total_responses * 100):.1f}%"
            })
    
    if overall_stats:
        overall_df = pd.DataFrame(overall_stats)
        overall_df = overall_df.sort_values("Total Rank 1", ascending=False)
        st.dataframe(overall_df, use_container_width=True, hide_index=True)
        
        # Visualize overall rank 1 distribution
        fig_overall = px.bar(
            overall_df,
            x="Method",
            y="Total Rank 1",
            title="Total Rank 1 Counts Across All Criteria",
            color="Method",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_overall.update_layout(
            xaxis_title="Method",
            yaxis_title="Total Rank 1 Counts",
            height=400
        )
        st.plotly_chart(fig_overall, use_container_width=True)

elif view_mode == "Export Data":
    st.header("💾 Export Data")
    
    # Export options
    export_format = st.radio(
        "Export Format:",
        ["JSON", "CSV (Summary)"],
        horizontal=True
    )
    
    if export_format == "JSON":
        # Export as JSON
        json_data = json.dumps(all_responses, indent=2, default=str)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"human_study_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        # Show preview
        with st.expander("Preview JSON"):
            st.code(json_data[:2000] + "..." if len(json_data) > 2000 else json_data)
    
    elif export_format == "CSV (Summary)":
        # Export summary as CSV
        summary_data = []
        for response in all_responses:
            summary_data.append({
                "participant_id": response.get("participant_id", "N/A"),
                "dataset": response.get("dataset", "N/A"),
                "target_model": response.get("target_model", "N/A"),
                "sample_key": response.get("sample_key", "N/A"),
                "true_class": response.get("true_class", "N/A"),
                "predicted_class": response.get("predicted_class", "N/A"),
                "has_rankings": "Yes" if response.get("rankings") else "No",
                "has_highlights": "Yes" if response.get("highlight_feedback") else "No",
                "timestamp": response.get("timestamp", "N/A")
            })
        
        df = pd.DataFrame(summary_data)
        csv_data = df.to_csv(index=False)
        
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"human_study_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        st.dataframe(df, use_container_width=True)

