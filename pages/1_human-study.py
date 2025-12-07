import os
import json
import streamlit as st
import random
import glob
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError

st.set_page_config(page_title="Human Study", page_icon=":pencil2:", layout="wide", initial_sidebar_state="collapsed")

dataset = "imagenet-r"
target_model = "clip"

# MongoDB connection
@st.cache_resource
def init_mongodb():
    """Initialize MongoDB connection"""
    try:
        # Get connection string from Streamlit secrets or environment variable
        # You can set this in Streamlit secrets: st.secrets["mongodb"]["connection_string"]
        # Or use environment variable: os.getenv("MONGODB_CONNECTION_STRING")
        
        if "mongodb" in st.secrets and "connection_string" in st.secrets["mongodb"]:
            connection_string = st.secrets["mongodb"]["connection_string"]
        elif "MONGODB_CONNECTION_STRING" in os.environ:
            connection_string = os.environ["MONGODB_CONNECTION_STRING"]
        else:
            # Fallback: you can set a default connection string here
            # For local MongoDB: "mongodb://localhost:27017/"
            return None
        
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command('ping')
        return client
    except (ConnectionFailure, Exception) as e:
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        return None

# Initialize MongoDB
mongodb_client = init_mongodb()
db = mongodb_client["prj-nemo"] if mongodb_client is not None else None
responses_collection = db["human-study-pilot"] if db is not None else None

# Check if participant_id is set
if 'participant_id' not in st.session_state or not st.session_state.participant_id:
    st.warning("⚠️ Please enter your Participant ID first.")
    if st.button("Go to Registration Page"):
        st.switch_page("main.py")
    st.stop()

participant_id = st.session_state.participant_id

# Load human-study-data
human_study_data_dir = "./human-study-data"
output_dir = os.path.join(human_study_data_dir, "outputs", f"{dataset}_{target_model}")
samples_dir = os.path.join(human_study_data_dir, "data", "nemo", dataset, "samples")

# Load result files
result_retrieval = json.load(open(os.path.join(output_dir, "error_retrieval.json"), "r"))
result_retrieval_cei = json.load(open(os.path.join(output_dir, "error_retrieval_cei.json"), "r"))    
result_pixel = json.load(open(os.path.join(output_dir, "pixel_attribution.json"), "r"))
result_pixel_cei = json.load(open(os.path.join(output_dir, "pixel_attribution_cei.json"), "r"))
result_change = json.load(open(os.path.join(output_dir, "change_of_caption.json"), "r"))
result_change_cei = json.load(open(os.path.join(output_dir, "change_of_caption_cei.json"), "r"))
result_scitx = json.load(open(os.path.join(output_dir, "scitx.json"), "r"))
result_scitx_cei = json.load(open(os.path.join(output_dir, "scitx_cei.json"), "r"))

# Get available keys
available_keys = list(result_retrieval.keys())

# Initialize session state
if 'current_sample_idx' not in st.session_state:
    st.session_state.current_sample_idx = 0
if 'ratings' not in st.session_state:
    st.session_state.ratings = {}
if 'explanation_order' not in st.session_state:
    st.session_state.explanation_order = {}
if 'ranking_feedback' not in st.session_state:
    st.session_state.ranking_feedback = {}

# Get current sample
if st.session_state.current_sample_idx >= len(available_keys):
    st.success("🎉 모든 샘플 평가를 완료했습니다!")
    st.stop()

selected_key = available_keys[st.session_state.current_sample_idx]

st.title(":pencil2: Evaluation of Explanations for Model's Error")

# Display progress at the top
progress_col1, progress_col2 = st.columns([3, 1])
with progress_col1:
    progress_value = (st.session_state.current_sample_idx + 1) / len(available_keys)
    st.progress(progress_value)
with progress_col2:
    st.markdown(f"**Sample {st.session_state.current_sample_idx + 1} / {len(available_keys)}**")

# st.markdown("---")



# Get image path
image_files = glob.glob(os.path.join(samples_dir, f"{selected_key}_*.jpg"))
if image_files:
    image_path = image_files[0]
else:
    st.error(f"Image not found for key {selected_key}")
    image_path = None

if image_path:
    # Display image and error info
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.image(image_path)
    
    with col2:
        true_cls_name = result_pixel[selected_key]['true_cls_name']
        prediction_cls_name = result_pixel[selected_key]['prediction_cls_name']
        st.markdown("")
        st.markdown("")
        st.markdown(f"### Error Information")
        st.markdown(f"**True Class:** {true_cls_name}")
        st.markdown(f"**Predicted Class:** {prediction_cls_name}")
        st.markdown(f"#### '{true_cls_name}' is misclassified as '{prediction_cls_name}'.")
    
    st.markdown("---")
    
    # Evaluation section
    st.markdown("## Evaluate Explanations")
    st.markdown("Please rank the explanations from best (1st) to worst (4th).")
    # st.markdown("**Note:** Explanations are shown in random order without method names.")
    # st.markdown("**Tip:** You can directly edit each explanation to add annotations or comments.")
    
    # Add CSS to make disabled text areas look better (works in both light and dark mode)
    st.markdown("""
    <style>
    textarea[disabled] {
        opacity: 1 !important;
        cursor: default !important;
        -webkit-text-fill-color: inherit !important;
    }
    textarea[disabled]:focus {
        box-shadow: none !important;
    }
    /* Light mode */
    @media (prefers-color-scheme: light) {
        textarea[disabled] {
            background-color: #f8f9fa !important;
            color: #262730 !important;
            border: 1px solid #e0e0e0 !important;
        }
    }
    /* Dark mode */
    @media (prefers-color-scheme: dark) {
        textarea[disabled] {
            background-color: #1e1e1e !important;
            color: #fafafa !important;
            border: 1px solid #3a3a3a !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    rating_key = f"{participant_id}_{selected_key}"
    
    # Prepare explanations with method names (for saving later)
    explanations_data = [
        ("retrieval", "Retrieval-based", result_retrieval[selected_key]['explanation']),
        ("pixel", "Pixel Attribution", result_pixel[selected_key]['explanation']),
        ("change", "Change of Caption", result_change[selected_key]['explanation']),
        ("scitx", "Ours (SciTx)", result_scitx[selected_key]['explanation']),
    ]
    
    # Get or create random order for this sample
    if selected_key not in st.session_state.explanation_order:
        # Create shuffled list of indices
        order = list(range(len(explanations_data)))
        random.shuffle(order)
        st.session_state.explanation_order[selected_key] = order
    else:
        order = st.session_state.explanation_order[selected_key]
    
    # Display explanations in random order - HORIZONTALLY
    shuffled_explanations = [explanations_data[i] for i in order]
    
    # Initialize rankings in session state for this sample
    rankings_key = f"rankings_{rating_key}"
    if rankings_key not in st.session_state:
        st.session_state[rankings_key] = {}
    
    rankings = st.session_state[rankings_key]
    
    # Reset all rankings button
    reset_col1, reset_col2 = st.columns([12, 1])
    with reset_col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state[rankings_key] = {}
            st.rerun()
    
    # Create 4 columns for horizontal layout
    cols = st.columns(4, gap="medium")
    
    for idx, (method_id, method_name, original_explanation) in enumerate(shuffled_explanations):
        with cols[idx]:
            st.markdown(f"### Explanation {idx + 1}")
            
            # Display explanation in text box (read-only with improved disabled style)
            st.text_area(
                "",
                value=original_explanation,
                key=f"explanation_{selected_key}_{idx}",
                height=300,
                disabled=True,
                label_visibility="collapsed"
            )
            
            # Ranking selection with buttons
            st.markdown("**Select Rank:**")
            rank_buttons = st.columns(4, gap="small")
            
            # Get current rank for this explanation
            current_rank = rankings.get(method_id, None)
            
            # Get ranks already used by other explanations
            used_ranks = {v for k, v in rankings.items() if k != method_id}
            
            for rank_num in range(1, 5):
                with rank_buttons[rank_num - 1]:
                    # Check if this rank is selected for current explanation
                    is_selected = current_rank == rank_num
                    # Check if this rank is used by another explanation
                    is_used_by_other = rank_num in used_ranks
                    
                    # Button label with visual indicator
                    if is_selected:
                        button_label = f"✓ {rank_num}"
                        button_type = "primary"
                    elif is_used_by_other:
                        button_label = f"{rank_num}"
                        button_type = "secondary"
                    else:
                        button_label = f"{rank_num}"
                        button_type = "secondary"
                    
                    # Disable if used by another explanation (but allow if it's the current selection)
                    disabled = is_used_by_other and not is_selected
                    
                    if st.button(button_label, key=f"rank_btn_{selected_key}_{idx}_{rank_num}", 
                                disabled=disabled, type=button_type, use_container_width=True):
                        rankings[method_id] = rank_num
                        st.session_state[rankings_key] = rankings
                        st.rerun()
            
            # Display current selection below buttons
            if current_rank:
                rank_suffix = {1: "st (Best)", 2: "nd", 3: "rd", 4: "th (Worst)"}[current_rank]
                st.markdown(f"**Selected: {current_rank}{rank_suffix}**")
            
            # Feedback input for this explanation
            feedback_key = f"{rating_key}_{method_id}_feedback"
            if feedback_key not in st.session_state.ranking_feedback:
                st.session_state.ranking_feedback[feedback_key] = ""
            
            feedback_text = st.text_area(
                "Which part of this explanation makes it good?",
                value=st.session_state.ranking_feedback[feedback_key],
                key=f"feedback_{selected_key}_{idx}",
                help="Describe which part of this explanation makes it good or bad, and why you ranked it this way.",
                placeholder="'part': 'reason'",
                height=150
            )
            st.session_state.ranking_feedback[feedback_key] = feedback_text
    
    st.markdown("---")
    
    # Validation: Check if all rankings are assigned and unique
    if len(rankings) == len(explanations_data) and len(set(rankings.values())) == len(explanations_data):
        all_ranked = True
    else:
        all_ranked = False
    
    # Display current rankings summary
    # if rankings:
    #     st.markdown("### Current Rankings Summary")
    #     sorted_rankings = sorted(rankings.items(), key=lambda x: x[1])
    #     table_header = "| Rank | Explanation ID |\n|------|----------------|\n"
    #     table_rows = []
    #     for method_id, rank in sorted_rankings:
    #         # Find which explanation number this method_id corresponds to
    #         exp_idx = order.index([i for i, (mid, _, _) in enumerate(explanations_data) if mid == method_id][0])
    #         table_rows.append(f"| {rank} | Explanation {exp_idx + 1} |")
    #     st.markdown(table_header + "\n".join(table_rows))
    
    # st.markdown("---")
    
    # Save function
    def save_current_sample():
        # Map rankings back to method names for saving
        method_rankings = {}
        for method_id, rank in rankings.items():
            method_rankings[method_id] = rank
        
        # Collect ranking feedback
        ranking_feedback_dict = {}
        for method_id, _, _ in explanations_data:
            feedback_key = f"{rating_key}_{method_id}_feedback"
            if feedback_key in st.session_state.ranking_feedback:
                ranking_feedback_dict[method_id] = st.session_state.ranking_feedback[feedback_key]
        
        # Convert explanation_order from list to dictionary with method_id as key
        # order is like [0, 2, 1, 3] where each number is the index in explanations_data
        # Convert to {"retrieval": 0, "pixel": 2, "change": 1, "scitx": 3}
        explanation_order_dict = {}
        for display_position, original_index in enumerate(order):
            method_id = explanations_data[original_index][0]  # Get method_id from explanations_data
            explanation_order_dict[method_id] = display_position
        
        # Prepare document for MongoDB
        document = {
            "participant_id": participant_id,
            "sample_key": selected_key,
            "timestamp": datetime.utcnow(),
            "true_class": true_cls_name,
            "predicted_class": prediction_cls_name,
            "rankings": method_rankings,
            "ranking_feedback": ranking_feedback_dict,
            "original_explanations": {mid: exp for mid, _, exp in explanations_data},
            "explanation_order": explanation_order_dict,
            "cei_scores": {
                "retrieval": result_retrieval_cei[selected_key]['cei'],
                "pixel": result_pixel_cei[selected_key]['cei'],
                "change": result_change_cei[selected_key]['cei'],
                "scitx": result_scitx_cei[selected_key]['cei'],
            },
            "dataset": dataset,
            "target_model": target_model
        }
        
        # Save to MongoDB
        if responses_collection is not None:
            try:
                # Use upsert to update if exists, insert if new
                # Unique index on (participant_id, sample_key) should be created in MongoDB
                filter_query = {
                    "participant_id": participant_id,
                    "sample_key": selected_key
                }
                responses_collection.update_one(
                    filter_query,
                    {"$set": document},
                    upsert=True
                )
            except Exception as e:
                st.error(f"Failed to save to MongoDB: {str(e)}")
                return False
        
        # Also save to JSON as backup
        try:
            results_dir = os.path.join(human_study_data_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            results_file = os.path.join(results_dir, f"{participant_id}_ratings.json")
            
            # Load existing results if any
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    all_ratings = json.load(f)
            else:
                all_ratings = {}
            
            all_ratings[rating_key] = document
            
            with open(results_file, "w") as f:
                json.dump(all_ratings, f, indent=2)
        except Exception as e:
            st.warning(f"Failed to save JSON backup: {str(e)}")
        
        return True
    
    # Navigation buttons
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    
    is_last_sample = st.session_state.current_sample_idx >= len(available_keys) - 1
    
    with nav_col1:
        # Previous button
        if st.button("← Previous", disabled=(st.session_state.current_sample_idx == 0), use_container_width=True):
            if st.session_state.current_sample_idx > 0:
                st.session_state.current_sample_idx -= 1
                st.rerun()
    
    with nav_col2:
        # Empty space for centering
        pass
    
    with nav_col3:
        # Next button (right-aligned)
        if is_last_sample:
            if st.button("Submit", type="primary", use_container_width=True):
                st.info("Thank you for your participation!")
                st.stop()
        else:
            if st.button("Next →", type="primary", use_container_width=True):
                if not all_ranked:
                    st.warning("Please rank all explanations before proceeding.")
                else:
                    save_current_sample()  # Auto-save before moving to next
                    st.session_state.current_sample_idx += 1
                    st.rerun()