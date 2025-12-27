import os
import json
import streamlit as st
import glob

st.set_page_config(page_title="Data Review", page_icon=":mag:", layout="wide", initial_sidebar_state="collapsed")

dataset_model_list = [
    "imagenet-r_vit",
    "imagenetd_siglip",
    "objectnet_vit",
]

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'selected_dataset_model' not in st.session_state:
    st.session_state.selected_dataset_model = dataset_model_list[0] if dataset_model_list else None
if 'selected_sample_key' not in st.session_state:
    st.session_state.selected_sample_key = None

human_study_data_dir = "./human-study-data"

# Load all dataset_model results
for dataset_model in dataset_model_list:
    if dataset_model not in st.session_state.results:
        parts = dataset_model.split('_', 1)  # 첫 번째 '_'만으로 split
        if len(parts) == 2:
            dataset = parts[0]
            target_model = parts[1]
        else:
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
            except Exception as e:
                st.warning(f"Failed to load data for {dataset_model}: {str(e)}")
        else:
            st.warning(f"Output directory not found: {output_dir}")

# Check if we have results
if len(st.session_state.results) == 0:
    st.error("No data found. Please check the data directory.")
    st.stop()

st.title(":mag: Data Review - Explanation Comparison")

# Initialize selected_dataset_model if not set
if st.session_state.selected_dataset_model not in st.session_state.results:
    st.session_state.selected_dataset_model = list(st.session_state.results.keys())[0]

# Selection dropdowns
select_col1, select_col2 = st.columns([1, 1])

with select_col1:
    # Dataset/Model selection
    dataset_model_options = list(st.session_state.results.keys())
    current_index = dataset_model_options.index(st.session_state.selected_dataset_model) if st.session_state.selected_dataset_model in dataset_model_options else 0
    
    selected_dataset_model = st.selectbox(
        "Select Dataset / Model:",
        options=dataset_model_options,
        index=current_index,
        key="dataset_model_select"
    )
    if selected_dataset_model != st.session_state.selected_dataset_model:
        st.session_state.selected_dataset_model = selected_dataset_model
        # Reset sample selection when dataset_model changes
        available_samples = list(st.session_state.results[selected_dataset_model]["retrieval"].keys())
        st.session_state.selected_sample_key = available_samples[0] if available_samples else None
        st.rerun()

with select_col2:
    # Sample selection
    available_samples = list(st.session_state.results[selected_dataset_model]["retrieval"].keys())
    
    # Initialize selected_sample_key if not set or not in available samples
    if st.session_state.selected_sample_key is None or st.session_state.selected_sample_key not in available_samples:
        st.session_state.selected_sample_key = available_samples[0] if available_samples else None
    
    current_sample_index = available_samples.index(st.session_state.selected_sample_key) if st.session_state.selected_sample_key in available_samples else 0
    
    selected_sample_key = st.selectbox(
        "Select Sample:",
        options=available_samples,
        index=current_sample_index,
        key="sample_select"
    )
    if selected_sample_key != st.session_state.selected_sample_key:
        st.session_state.selected_sample_key = selected_sample_key
        st.rerun()

# Parse dataset and model from selected_dataset_model
parts = selected_dataset_model.split('_', 1)
current_dataset = parts[0]
current_target_model = parts[1]
selected_key = st.session_state.selected_sample_key
dataset_model = selected_dataset_model

# Get image path
samples_dir = os.path.join(human_study_data_dir, "data", "nemo", current_dataset, "samples")
image_files = glob.glob(os.path.join(samples_dir, f"{selected_key}_*.jpg"))
if image_files:
    image_path = image_files[0]
else:
    st.error(f"Image not found for key {selected_key}")
    image_path = None

# Get results for current dataset_model
results = st.session_state.results.get(dataset_model, {})
if not results:
    st.error(f"Results not found for {dataset_model}")
    st.stop()

# Get data for current sample
sample_data = {
    "retrieval": results.get("retrieval", {}).get(selected_key, {}),
    "pixel": results.get("pixel", {}).get(selected_key, {}),
    "change": results.get("change", {}).get(selected_key, {}),
    "scitx": results.get("scitx", {}).get(selected_key, {})
}

# Get true_class and predicted_class (from pixel data as reference)
true_cls_name = sample_data["pixel"].get("true_cls_name", "N/A")
prediction_cls_name = sample_data["pixel"].get("prediction_cls_name", "N/A")

# Get available samples for display
available_samples = list(st.session_state.results[selected_dataset_model]["retrieval"].keys())

# Display info
info_col1, info_col2 = st.columns([1, 1])
with info_col1:
    current_index = available_samples.index(selected_key) + 1
    st.markdown(f"**Sample {current_index} / {len(available_samples)}**")
with info_col2:
    st.markdown(f"**Dataset:** {current_dataset} | **Model:** {current_target_model}")

st.markdown("---")

if image_path:
    # Display image and error info
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.image(image_path)
    
    with col2:
        st.markdown("")
        st.markdown("")
        st.markdown(f"### Error Information")
        st.markdown(f'<div style="font-size: 1.2em;"><ul><li><strong>True Class:</strong> {true_cls_name}</li><li><strong>Predicted Class:</strong> {prediction_cls_name}</li></ul></div>', unsafe_allow_html=True)
        st.markdown(f"#### '{true_cls_name}' is misclassified as '{prediction_cls_name}'.")
    
    st.markdown("---")
    
    # Display explanations
    st.markdown("## Explanations by Method")
    
    # Method names and icons
    methods = [
        ("retrieval", "Retrieval-based", "🔍"),
        ("pixel", "Pixel Attribution", "🖼️"),
        ("change", "Change of Caption", "📝"),
        ("scitx", "Ours (SciTx)", "✨")
    ]
    
    # Create 4 columns for horizontal layout
    cols = st.columns(4, gap="medium")
    
    # Display all methods in columns
    for idx, (method_id, method_name, icon) in enumerate(methods):
        with cols[idx]:
            method_data = sample_data.get(method_id, {})
            
            if method_data:
                explanation = method_data.get("explanation", "No explanation available")
                cei = method_data.get("cei", None)
                
                st.markdown(f"### {icon} {method_name}")
                
                if cei is not None:
                    st.metric("CEI Score", f"{cei:.2f}")
                
                st.markdown("#### Explanation:")
                st.info(explanation)
            else:
                st.warning(f"No data available for {method_name}")
    
    st.markdown("---")
else:
    st.error(f"Image not found for sample key: {selected_key}")

