import os
import json
import streamlit as st
import random
import glob
import base64
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from pymongo import ReturnDocument
from text_highlighter import text_highlighter

st.set_page_config(page_title="Re-evaluation: Study O", page_icon=":pencil2:", layout="wide", initial_sidebar_state="collapsed")

# Prolific ID input at the top
st.title(":pencil2: Re-evaluation: Study O")

# Prolific ID input
if 'prolific_id' not in st.session_state:
    st.session_state.prolific_id = ""

prolific_id = st.text_input(
    "Enter your Prolific ID:",
    value=st.session_state.prolific_id,
    key="prolific_id_input",
    placeholder="Please enter your Prolific ID to continue"
)

if not prolific_id:
    st.warning("⚠️ Please enter your Prolific ID to continue.")
    st.stop()

st.session_state.prolific_id = prolific_id
participant_id = prolific_id

test_mode = st.session_state.get("test_mode", False)

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

if 'all_samples' not in st.session_state:
    st.session_state.all_samples = []
if 'results' not in st.session_state:
    st.session_state.results = {}
    
human_study_data_dir = "./human-study-data"

# Only load objectnet samples with target ID
dataset_model = "objectnet_vit"
dataset = "objectnet"
target_model = "vit"
target_sample_id = "2336"

# Load samples
if len(st.session_state.all_samples) == 0:
    output_dir = os.path.join(human_study_data_dir, "outputs", f"{dataset}_{target_model}")
    
    if os.path.exists(output_dir):
        try:
            # Load result files
            st.session_state.results[dataset_model] = {
                "retrieval": json.load(open(os.path.join(output_dir, "subset_error_retrieval.json"), "r")),
                "pixel": json.load(open(os.path.join(output_dir, "subset_pixel_attribution.json"), "r")),
                "change": json.load(open(os.path.join(output_dir, "subset_change_of_caption.json"), "r")),
                "scitx": json.load(open(os.path.join(output_dir, "subset_scitx.json"), "r"))
            }
            
            # Get available keys and filter for target sample
            available_keys = list(st.session_state.results[dataset_model]["retrieval"].keys())
            filtered_keys = [key for key in available_keys if key == target_sample_id]
            
            # Add filtered samples
            for key in filtered_keys:
                st.session_state.all_samples.append((dataset, target_model, key))
            
            if len(filtered_keys) == 0:
                st.error("Sample not found. Please contact the study administrator.")
                st.stop()
        except Exception as e:
            st.warning(f"Failed to load data for {dataset_model}: {str(e)}")
    else:
        st.error(f"Output directory not found: {output_dir}")
        st.stop()

# Initialize session state
if 'current_sample_idx' not in st.session_state:
    st.session_state.current_sample_idx = 0

if 'explanation_order' not in st.session_state:
    st.session_state.explanation_order = {}
if 'highlight_feedback' not in st.session_state:
    st.session_state.highlight_feedback = {}
if 'highlight_keys' not in st.session_state:
    st.session_state.highlight_keys = {}
if 'study_completed' not in st.session_state:
    st.session_state.study_completed = False

# Check if study is completed
if st.session_state.current_sample_idx >= len(st.session_state.all_samples) and st.session_state.study_completed:
    # Get completion code from secrets.toml
    completion_code = "COMPLETION_CODE_NOT_SET"
    if "completion_code" in st.secrets and "re_eval_completion_code" in st.secrets["completion_code"]:
        completion_code = st.secrets["completion_code"]["re_eval_completion_code"]
    elif "completion_code" in st.secrets and "study1_completion_code" in st.secrets["completion_code"]:
        completion_code = st.secrets["completion_code"]["study1_completion_code"]
    
    if completion_code == "COMPLETION_CODE_NOT_SET":
        st.warning("⚠️ Completion code not found in secrets.toml.")
    
    st.success("🎉 Thank you for completing the re-evaluation!")
    st.markdown("---")
    st.markdown("### Your Completion Code")
    st.markdown(f"""
    <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
        <h2 style="color: #2e7d32; margin: 0;">{completion_code}</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Please copy this code and submit it on Prolific.**")
    st.markdown("---")
    
    comment = st.text_area(
        "Optional Feedback:",
        value="",
        height=150,
        key="completion_comment_input"
    )
    
    if st.button("Submit Comment", type="primary"):
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
                st.success("✅ Thank you for your feedback!")
            except Exception as e:
                st.error(f"Failed to save comment: {str(e)}")
    
    st.stop()

# Get current sample
if st.session_state.current_sample_idx >= len(st.session_state.all_samples):
    st.success("🎉 모든 샘플 평가를 완료했습니다!")
    st.stop()

current_dataset, current_target_model, selected_key = st.session_state.all_samples[st.session_state.current_sample_idx]
samples_dir = os.path.join(human_study_data_dir, "data", "nemo", current_dataset, "samples")

# Ranking function (same as original)
def render_ranking_question(
    question_type: str,
    question_number: str,
    question_title: str,
    question_text: str,
    selected_key: str,
    shuffled_explanations: list,
    rating_key: str,
    explanations_data: list
):
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
            st.markdown(f"**Rank Explanation {idx + 1} :**")
            rank_buttons = st.columns(4, gap="small")
            
            current_rank = rankings.get(method_id, None)
            used_ranks = {v for k, v in rankings.items() if k != method_id}
            
            for rank_num in range(1, 5):
                with rank_buttons[rank_num - 1]:
                    is_selected = current_rank == rank_num
                    is_used_by_other = rank_num in used_ranks
                    
                    if is_selected:
                        button_label = f"✓ {rank_num}"
                        button_type = "primary"
                    else:
                        button_label = f"{rank_num}"
                        button_type = "secondary"
                    
                    disabled = is_used_by_other and not is_selected
                    
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
        "text": "Q. Which explanation do you find **provides more specific and relevant details** about the model's error compared to the others? Select Rank (1st to 4th):"
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
    rankings_dict = {}
    for q in ranking_questions:
        rankings_key = f"rankings_{rating_key}_{q['type']}"
        if rankings_key in st.session_state:
            method_rankings = {}
            for method_id, rank in st.session_state[rankings_key].items():
                method_rankings[method_id] = rank
            rankings_dict[q["type"]] = method_rankings

    highlight_feedback_dict = {}
    for method_id, _, _ in explanations_data:
        highlight_feedback_key = f"{rating_key}_{method_id}"
        if highlight_feedback_key in st.session_state.highlight_feedback:
            highlight_feedback_dict[method_id] = st.session_state.highlight_feedback[highlight_feedback_key]
    
    explanation_order_dict = {}
    for display_position, original_index in enumerate(order):
        method_id = explanations_data[original_index][0]
        explanation_order_dict[method_id] = display_position
    
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
        "target_model": current_target_model,
        "is_re_evaluation": True,
        "re_evaluation_type": "objectnet_2336"
    }
    
    if test_mode:
        st.info("🧪 Test Mode: MongoDB saving is disabled")
        return True
    elif responses_collection is not None:
        try:
            filter_query = {
                "participant_id": participant_id,
                "sample_key": selected_key,
                "dataset": current_dataset,
                "target_model": current_target_model,
                "true_class": true_cls_name,
                "predicted_class": prediction_cls_name
            }
            previous_document = responses_collection.find_one_and_update(
                filter_query,
                {"$set": document},
                upsert=True,
                return_document=ReturnDocument.BEFORE
            )
            if previous_document:
                document["previous_version"] = previous_document
                responses_collection.update_one(
                    filter_query,
                    {"$set": document}
                )
        except Exception as e:
            st.error(f"Failed to save to MongoDB: {str(e)}")
            return False
        
        try:
            results_dir = os.path.join(human_study_data_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            results_file = os.path.join(results_dir, f"{participant_id}_re_eval_objectnet_2336.json")
            
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

# Display progress
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
<li>You can reset the current ranking by clicking the "Reset" button at the top-right of each question.</li>
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
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
            img_ext = os.path.splitext(image_path)[1][1:]
            st.markdown(
                f'<div style="height: 400px; display: flex; align-items: center; justify-content: center; overflow: hidden; background-color: transparent;">'
                f'<img src="data:image/{img_ext};base64,{img_base64}" style="max-height: 100%; max-width: 100%; object-fit: contain;" />'
                f'</div>',
                unsafe_allow_html=True
            )
    
    with col2:
        true_cls_name = st.session_state.results[dataset_model]["pixel"][selected_key]['true_cls_name']
        prediction_cls_name = st.session_state.results[dataset_model]["pixel"][selected_key]['prediction_cls_name']
        st.markdown("")
        st.markdown("")
        st.markdown(f"### Error Information")
        st.markdown(f'<div style="font-size: 1.2em;"><ul><li><strong>True Class:</strong> {true_cls_name}</li><li><strong>Predicted Class:</strong> {prediction_cls_name}</li></ul></div>', unsafe_allow_html=True)
        st.markdown(f"#### '{true_cls_name}' is misclassified as '{prediction_cls_name}'.")
    
    st.markdown("---")
    
    st.markdown("## Evaluate Explanations")
    rating_key = f"{participant_id}_{current_dataset}_{current_target_model}_{selected_key}"
    
    explanations_data = [
        ("retrieval", "Retrieval-based", st.session_state.results[dataset_model]["retrieval"][selected_key]['explanation']),
        ("pixel", "Pixel Attribution", st.session_state.results[dataset_model]["pixel"][selected_key]['explanation']),
        ("change", "Change of Caption", st.session_state.results[dataset_model]["change"][selected_key]['explanation']),
        ("scitx", "Ours (SciTx)", st.session_state.results[dataset_model]["scitx"][selected_key]['explanation']),
    ]
    
    if selected_key not in st.session_state.explanation_order:
        order = list(range(len(explanations_data)))
        random.shuffle(order)
        st.session_state.explanation_order[selected_key] = order
    else:
        order = st.session_state.explanation_order[selected_key]
    
    shuffled_explanations = [explanations_data[i] for i in order]
    
    rankings_key = f"rankings_{rating_key}"
    if rankings_key not in st.session_state:
        st.session_state[rankings_key] = {}
    
    rankings = st.session_state[rankings_key]
    
    cols = st.columns(4, gap="medium")
    current_highlights = {}
    
    for idx, (method_id, method_name, original_explanation) in enumerate(shuffled_explanations):
        with cols[idx]:
            st.markdown(f"### Explanation {idx + 1}")
        
            highlight_key_base = f"{rating_key}_{method_id}"
            if highlight_key_base not in st.session_state.highlight_keys:
                st.session_state.highlight_keys[highlight_key_base] = 0
            
            highlight_key = f"{highlight_key_base}_{st.session_state.highlight_keys[highlight_key_base]}"
            
            result = text_highlighter(
                text=original_explanation,
                labels=[("🟢 Good", "rgba(74, 222, 128, 0.3)"), ("❌ Bad", "rgba(239, 68, 68, 0.3)")],
                key=highlight_key,
            )
            current_highlights[method_id] = result if result else []
    
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
    all_questions_ranked = all(all_questions_ranked.values()) if all_questions_ranked else False
    
    all_highlights_complete = True
    missing_highlights = []
    for method_id, _, _ in explanations_data:
        if method_id not in current_highlights or len(current_highlights[method_id]) == 0:
            all_highlights_complete = False
            missing_highlights.append(method_id)
    
    all_complete = all_questions_ranked and all_highlights_complete
    
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
    
    is_last_sample = st.session_state.current_sample_idx >= len(st.session_state.all_samples) - 1
    
    def sync_highlight_feedback(current_highlights, rating_key, explanations_data):
        for method_id, _, _ in explanations_data:
            highlight_feedback_key = f"{rating_key}_{method_id}"
            result = current_highlights.get(method_id, [])
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
            
            st.session_state.highlight_feedback[highlight_feedback_key] = synced_feedback
    
    with nav_col1:
        if st.button("← Previous", disabled=(st.session_state.current_sample_idx == 0), use_container_width=True):
            if st.session_state.current_sample_idx > 0:
                st.session_state.current_sample_idx -= 1
                st.rerun()
    
    with nav_col2:
        pass
    
    with nav_col3:
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
                    save_current_sample()
                    st.session_state.current_sample_idx += 1
                    st.rerun()

