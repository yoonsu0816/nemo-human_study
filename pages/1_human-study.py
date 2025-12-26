import os
import json
import streamlit as st
import random
import glob
import base64
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from text_highlighter import text_highlighter

st.set_page_config(page_title="Human Study", page_icon=":pencil2:", layout="wide", initial_sidebar_state="collapsed")

dataset_model_list = [
    "imagenet-r_vit",
    "imagenetd_siglip",
    "objectnet_vit",
]

test_mode = st.session_state.get("test_mode", False)

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
if not test_mode:
    if 'participant_id' not in st.session_state or not st.session_state.participant_id:
        st.warning("⚠️ Please enter your Participant ID first.")
        if st.button("Go to Registration Page"):
            st.switch_page("main.py")
        st.stop()
else:
    # In test mode, set dummy participant ID if not exists
    if 'participant_id' not in st.session_state or not st.session_state.participant_id:
        st.session_state.participant_id = "test_user"
    st.info("🧪 Test Mode: Participant ID check and MongoDB saving are disabled")

participant_id = st.session_state.participant_id

if 'all_samples' not in st.session_state:
    st.session_state.all_samples = []  # [(dataset, target_model, sample_key), ...]
if 'results' not in st.session_state:
    st.session_state.results = {}
    
human_study_data_dir = "./human-study-data"

# Define allowed sample IDs for each dataset (Study 1)
allowed_samples = {
    "imagenet-r": ["20472", "3950", "9582", "20688", "29327"],
    "imagenetd": ["566", "1066", "2447", "2771", "2377"],
    "objectnet": ["2731", "18075", "14720", "15661", "3172"]
}

# Load all samples from all dataset_model combinations
if len(st.session_state.all_samples) == 0:
    for dataset_model in dataset_model_list:
        parts = dataset_model.split('_', 1)  # 첫 번째 '_'만으로 split
        if len(parts) == 2:
            dataset = parts[0]
            target_model = parts[1]
        else:
            st.error(f"Invalid dataset_model: {dataset_model}")
            continue
        
        # Get allowed sample IDs for this dataset
        allowed_ids = allowed_samples.get(dataset, [])
        if not allowed_ids:
            st.warning(f"No allowed samples defined for dataset: {dataset}")
            continue
        
        # Load data for this dataset_model combination
        output_dir = os.path.join(human_study_data_dir, "outputs", f"{dataset}_{target_model}")
        
        # Check if output_dir exists
        if os.path.exists(output_dir):
            try:
                # Load result files
                st.session_state.results[dataset_model] = {
                    "retrieval": json.load(open(os.path.join(output_dir, "subset_error_retrieval.json"), "r")),
                    "pixel": json.load(open(os.path.join(output_dir, "subset_pixel_attribution.json"), "r")),
                    "change": json.load(open(os.path.join(output_dir, "subset_change_of_caption.json"), "r")),
                    "scitx": json.load(open(os.path.join(output_dir, "subset_scitx.json"), "r"))
                }
                
                # Get available keys and filter by allowed IDs
                available_keys = list(st.session_state.results[dataset_model]["retrieval"].keys())
                # Filter to only include allowed sample IDs
                filtered_keys = [key for key in available_keys if key in allowed_ids]
                
                # Add filtered samples from this dataset_model to all_samples
                for key in filtered_keys:
                    st.session_state.all_samples.append((dataset, target_model, key))
                
                # Warn if some allowed IDs are not found
                missing_ids = set(allowed_ids) - set(available_keys)
                if missing_ids:
                    st.warning(f"Some allowed sample IDs not found for {dataset_model}: {missing_ids}")
            except Exception as e:
                st.warning(f"Failed to load data for {dataset_model}: {str(e)}")
        else:
            st.warning(f"Output directory not found: {output_dir}")
    
    # Shuffle all samples to randomize the order
    if len(st.session_state.all_samples) > 0:
        random.shuffle(st.session_state.all_samples)

# Initialize session state
if 'current_sample_idx' not in st.session_state:
    st.session_state.current_sample_idx = 0

if 'explanation_order' not in st.session_state:
    st.session_state.explanation_order = {}
if 'highlight_feedback' not in st.session_state:
    st.session_state.highlight_feedback = {}  # {rating_key_method_id: [{"text": "...", "rating": 1/-1, "detailed_feedback": "..."}]}
if 'highlight_keys' not in st.session_state:
    st.session_state.highlight_keys = {}  # {rating_key_method_id: counter}
if 'study_completed' not in st.session_state:
    st.session_state.study_completed = False

# Get current sample
# if st.session_state.current_sample_idx >= len(st.session_state.all_samples):
#     st.success("🎉 모든 샘플 평가를 완료했습니다!")
#     st.stop()

current_dataset, current_target_model, selected_key = st.session_state.all_samples[st.session_state.current_sample_idx]
samples_dir = os.path.join(human_study_data_dir, "data", "nemo", current_dataset, "samples")

# Ranking function
def render_ranking_question(
    question_type: str,  # "factuality", "verbosity", "specificity", "actionability"
    question_number: str,  # "1", "2", "3", "4"
    question_title: str,  # "Factuality", "Verbosity", etc.
    question_text: str,  # 질문 내용
    selected_key: str,
    shuffled_explanations: list,
    rating_key: str,
    explanations_data: list
):
    """
    Args:
        question_type: 질문 타입 (factuality, verbosity, specificity, actionability)
        question_number: 질문 번호 ("1", "2", "3", "4")
        question_title: 질문 제목 ("Factuality", "Verbosity", etc.)
        question_text: 질문 내용
        selected_key: 현재 샘플 키
        shuffled_explanations: 섞인 explanation 리스트
        rating_key: rating 키 (participant_id_sample_key)
        explanations_data: 원본 explanation 데이터
    """
    rankings_key = f"rankings_{rating_key}_{question_type}"
    if rankings_key not in st.session_state:
        st.session_state[rankings_key] = {}
    
    rankings = st.session_state[rankings_key]
    
    st.markdown(f"#### {question_number}. {question_title}")
    
    reset_col1, reset_col2 = st.columns([12, 1])
    
    with reset_col1:
        st.markdown(f"**[{question_title}]** {question_text}")
    with reset_col2:
        if st.button("🔄 Reset", key=f"reset_{question_type}_{selected_key}", use_container_width=True):
            st.session_state[rankings_key] = {}
            st.rerun()
    
    col1, div1, col2, div2, col3, div3, col4 = st.columns([1, 0.05, 1, 0.05, 1, 0.05, 1])
    cols = [col1, col2, col3, col4]
    
    for idx, (method_id, method_name, original_explanation) in enumerate(shuffled_explanations):
        with cols[idx]:
            # Ranking selection with buttons
            st.markdown(f"**Rank Explanation {idx + 1} :**")
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
                    
                    # 각 항목별로 고유한 키 사용
                    if st.button(button_label, key=f"rank_btn_{selected_key}_{question_type}_{idx}_{rank_num}", 
                                disabled=disabled, type=button_type, use_container_width=True):
                        rankings[method_id] = rank_num
                        st.session_state[rankings_key] = rankings
                        st.rerun()
        
        if idx < len(shuffled_explanations) - 1:
            if idx == 0:
                with div1:
                    st.markdown("<div style='border-left: 1px solid rgba(128, 128, 128, 0.2); height: 80px; margin: 0; padding: 0;'></div>", unsafe_allow_html=True)
            elif idx == 1:
                with div2:
                    st.markdown("<div style='border-left: 1px solid rgba(128, 128, 128, 0.2); height: 80px; margin: 0; padding: 0;'></div>", unsafe_allow_html=True)
            elif idx == 2:
                with div3:
                    st.markdown("<div style='border-left: 1px solid rgba(128, 128, 128, 0.2); height: 80px; margin: 0; padding: 0;'></div>", unsafe_allow_html=True)
    
    # Validation: Check if all rankings are assigned and unique
    all_ranked = (len(rankings) == len(explanations_data) and 
                  len(set(rankings.values())) == len(explanations_data))
    
    return all_ranked, rankings

ranking_questions = [
    {
        "type": "overall",
        "number": "1",
        "title": "Overall Helpfulness",
        "text": "Q. Which explanation do you find more **helpful** compared to the others? Select Rank (1st to 4th):"
    },
    {
        "type": "factuality",
        "number": "2",
        "title": "Factuality",
        "text": "Q. Which explanation do you find more **factually accurate and reliable** compared to the others? Select Rank (1st to 4th):"
    },
    {
        "type": "verbosity",
        "number": "3",
        "title": "Verbosity",
        "text": "Q. Which explanation do you find more **concise and information-dense** compared to the others? Select Rank (1st to 4th):"
    },
    {
        "type": "specificity",
        "number": "4",
        "title": "Specificity",
        "text": "Q. Which explanation do you find **provides more specific and relecant details** about the model's error compared to the others? Select Rank (1st to 4th):"
    },
    {
        "type": "actionability",
        "number": "5",
        "title": "Actionability",
        "text": "Q. Which explanation do you find **provides more actionable and useful insights** for improving the model's performance compared to the others? Select Rank (1st to 4th):"
    }
]

# Save current sample to MongoDB and JSON
def save_current_sample():
    # Collect rankings
    rankings_dict = {}
    for q in ranking_questions:
        rankings_key = f"rankings_{rating_key}_{q['type']}"
        if rankings_key in st.session_state:
            method_rankings = {}
            for method_id, rank in st.session_state[rankings_key].items():
                method_rankings[method_id] = rank
            rankings_dict[q["type"]] = method_rankings

    # Collect highlight feedback   
    highlight_feedback_dict = {}
    for method_id, _, _ in explanations_data:
        highlight_feedback_key = f"{rating_key}_{method_id}"
        if highlight_feedback_key in st.session_state.highlight_feedback:
            highlight_feedback_dict[method_id] = st.session_state.highlight_feedback[highlight_feedback_key]
    
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
        "rankings": rankings_dict,
        "original_explanations": {mid: exp for mid, _, exp in explanations_data},
        "highlight_feedback": highlight_feedback_dict,
        "explanation_order": explanation_order_dict,
        "dataset": current_dataset,
        "target_model": current_target_model
    }
    
    # Save to MongoDB
    if test_mode:
        st.info("🧪 Test Mode: MongoDB saving is disabled")
        return True
    elif responses_collection is not None:
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

st.title(":pencil2: Evaluation of Explanations for Model's Error")

# Check if study is completed
if st.session_state.study_completed:
    # Get completion code from secrets.toml
    completion_code = "COMPLETION_CODE_NOT_SET"
    # completion_code = st.secrets.get("completion_code", "study1_completion_code", "COMPLETION_CODE_NOT_SET")
    if "completion_code" in st.secrets and "study1_completion_code" in st.secrets["completion_code"]:
        completion_code = st.secrets["completion_code"]["study1_completion_code"]
    if completion_code == "COMPLETION_CODE_NOT_SET":
        st.warning("⚠️ Completion code not found in secrets.toml. Please set it in the configuration file.")
    
    # Display completion screen
    st.success("🎉 Thank you for completing the study!")
    
    st.markdown("---")
    
    # Display completion code
    st.markdown("### Your Completion Code")
    st.markdown(f"""
    <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
        <h2 style="color: #2e7d32; margin: 0;">{completion_code}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Please copy this code and submit it on Prolific to receive your payment.**")
    
    st.markdown("---")
    
    # Free form comment section
    st.markdown("### Optional Feedback")
    st.markdown("We would appreciate any feedback or comments about your experience with this study.")
    
    comment = st.text_area(
        "Your comments:",
        value="",
        height=150,
        key="completion_comment_input",
        placeholder="Please share any thoughts, suggestions, or issues you encountered during the study."
    )
    
    if st.button("Submit Comment", type="primary"):
        # Save comment to MongoDB only
        if not test_mode and responses_collection is not None:
            try:
                responses_collection.update_one(
                    {"participant_id": participant_id},
                    {
                        "$set": {
                            "completion_comment": comment,
                            "completion_code": completion_code,
                            "completion_timestamp": datetime.utcnow()
                        }
                    },
                    upsert=True
                )
                st.success("✅ Thank you for your feedback! Your comment has been saved.")
            except Exception as e:
                st.error(f"Failed to save comment: {str(e)}")
        elif test_mode:
            st.info("🧪 Test Mode: Comment saving is disabled")
        else:
            st.warning("⚠️ MongoDB connection not available. Comment could not be saved.")
    
    st.stop()

# Display progress at the top
progress_col1, progress_col2 = st.columns([3, 1])
with progress_col1:
    progress_value = (st.session_state.current_sample_idx + 1) / len(st.session_state.all_samples)
    st.progress(progress_value)
with progress_col2:
    st.markdown(f"**Sample {st.session_state.current_sample_idx + 1} / {len(st.session_state.all_samples)}**")

# Instructions
st.markdown("## :pushpin: Instructions")
st.markdown("#### Text annotation")
st.markdown("""
<div style="font-size: 1.2em;">
<ul>
<li>You should mark both the parts that are good and the parts that are not good in the given explanation.</li>
<li>Select the Good or Bad button above the explanation, then drag to select the relevant text to apply the marking.</li>
<li>To remove a marking, click on the highlighted text, and the marking will be removed.</li>
<li>You must make at least one marking for each explanation.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("#### Ranking survey")
st.markdown("""
<div style="font-size: 1.2em;">
<ul>
<li>There are five questions in which you need to rank the explanations: Overall Helpfulness, Factuality, Verbosity, Specificity, and Actionability.</li>
<li>Rank the explanations according to each question.</li>
<li>You can reset the current ranking by clicking the “Reset” button at the top-right of each question.</li>
</ul>
</div>
""", unsafe_allow_html=True)
st.markdown("---")
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
        # st.image(image_path, use_container_width=False)
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
            img_ext = os.path.splitext(image_path)[1][1:]  # Get extension without dot
            st.markdown(
                f'<div style="height: 400px; display: flex; align-items: center; justify-content: center; overflow: hidden; background-color: transparent;">'
                f'<img src="data:image/{img_ext};base64,{img_base64}" style="max-height: 100%; max-width: 100%; object-fit: contain;" />'
                f'</div>',
                unsafe_allow_html=True
            )
    
    with col2:
        true_cls_name = st.session_state.results[current_dataset + "_" + current_target_model]["pixel"][selected_key]['true_cls_name']
        prediction_cls_name = st.session_state.results[current_dataset + "_" + current_target_model]["pixel"][selected_key]['prediction_cls_name']
        st.markdown("")
        st.markdown("")
        st.markdown(f"### Error Information")
        st.markdown(f'<div style="font-size: 1.2em;"><ul><li><strong>True Class:</strong> {true_cls_name}</li><li><strong>Predicted Class:</strong> {prediction_cls_name}</li></ul></div>', unsafe_allow_html=True)
        st.markdown(f"#### '{true_cls_name}' is misclassified as '{prediction_cls_name}'.")
    
    st.markdown("---")
    
    # Evaluation section
    st.markdown("## Evaluate Explanations")
    rating_key = f"{participant_id}_{current_dataset}_{current_target_model}_{selected_key}"
    
    # Prepare explanations with method names (for saving later)
    explanations_data = [
        ("retrieval", "Retrieval-based", st.session_state.results[current_dataset + "_" + current_target_model]["retrieval"][selected_key]['explanation']),
        ("pixel", "Pixel Attribution", st.session_state.results[current_dataset + "_" + current_target_model]["pixel"][selected_key]['explanation']),
        ("change", "Change of Caption", st.session_state.results[current_dataset + "_" + current_target_model]["change"][selected_key]['explanation']),
        ("scitx", "Ours (SciTx)", st.session_state.results[current_dataset + "_" + current_target_model]["scitx"][selected_key]['explanation']),
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
    
    # Create 4 columns for horizontal layout
    cols = st.columns(4, gap="medium")
    current_highlights = {}
    
    for idx, (method_id, method_name, original_explanation) in enumerate(shuffled_explanations):
        with cols[idx]:
            st.markdown(f"### Explanation {idx + 1}")
        
            highlight_key_base = f"{rating_key}_{method_id}"
            if highlight_key_base not in st.session_state.highlight_keys:
                st.session_state.highlight_keys[highlight_key_base] = 0
            
            highlight_key = f"{highlight_key_base}_{st.session_state.highlight_keys[highlight_key_base]}"
            
            # text_highlighter로 교체
            result = text_highlighter(
                text=original_explanation,
                labels=[("🟢 Good", "rgba(74, 222, 128, 0.3)"), ("❌ Bad", "rgba(239, 68, 68, 0.3)")],
                key=highlight_key,
            )
            current_highlights[method_id] = result if result else []
    
    # st.markdown("---")
    
    all_questions_ranked = {}
    for q in ranking_questions:
        all_ranked, rankings = render_ranking_question(
            question_type=q["type"],
            question_number=q["number"],
            question_title=q["title"],
            question_text=q["text"],
            selected_key=selected_key,
            shuffled_explanations=shuffled_explanations,
            rating_key=rating_key,
            explanations_data=explanations_data
        )
        all_questions_ranked[q["type"]] = all_ranked
    
    
    
    st.divider()
    # Validation: Check if all rankings are assigned and unique
    all_questions_ranked = all(all_questions_ranked.values()) if all_questions_ranked else False
    
    # Validation: Check if all explanations have at least one highlight feedback
    all_highlights_complete = True
    missing_highlights = []
    for method_id, _, _ in explanations_data:
        if method_id not in current_highlights or len(current_highlights[method_id]) == 0:
            all_highlights_complete = False
            missing_highlights.append(method_id)
    
    all_complete = all_questions_ranked and all_highlights_complete
    
    # Navigation buttons
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
    
    is_last_sample = st.session_state.current_sample_idx >= len(st.session_state.all_samples) - 1
    
    def sync_highlight_feedback(current_highlights, rating_key, explanations_data):
        """현재 하이라이트 상태를 highlight_feedback에 동기화"""
        for method_id, _, _ in explanations_data:
            highlight_feedback_key = f"{rating_key}_{method_id}"
            
            # 현재 result를 기반으로 highlight_feedback 업데이트
            result = current_highlights.get(method_id, [])
            
            # highlight_feedback를 result와 동기화
            synced_feedback = []
            for highlight_item in result:
                text = highlight_item.get("text", "")
                tag = highlight_item.get("tag", "").split(" ")[-1]
                start_idx = highlight_item.get("start")
                end_idx = highlight_item.get("end")
                
                if text:
                    synced_feedback.append({
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "text": text,
                        "tag": tag
                    })
            
            # highlight_feedback 업데이트
            st.session_state.highlight_feedback[highlight_feedback_key] = synced_feedback
    
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
                if not all_highlights_complete:
                    st.warning("Please highlight at least one part (Good or Bad) for all explanations.")
                if not all_questions_ranked:
                    st.warning("Please rank all explanations for all questions before proceeding.")
                if all_complete:
                    sync_highlight_feedback(current_highlights, rating_key, explanations_data)
                    save_current_sample()
                    st.session_state.study_completed = True
                    st.session_state.current_sample_idx += 1
                    st.rerun()                
        else:
            if st.button("Next →", type="primary", use_container_width=True):
                if not all_highlights_complete:
                    st.warning("Please highlight at least one part (Good or Bad) for all explanations.")
                if not all_questions_ranked:
                    st.warning("Please rank all explanations for all questions before proceeding.")
                if all_complete:
                    sync_highlight_feedback(current_highlights, rating_key, explanations_data)
                    save_current_sample()  # Auto-save before moving to next
                    st.session_state.current_sample_idx += 1
                    st.rerun()