import streamlit as st

st.set_page_config(page_title="Start - Human Study", page_icon=":pencil2:", layout="wide", initial_sidebar_state="collapsed")

st.title(":pencil2: Evaluation of AI Model's Error Explanations")
st.markdown("---")

st.markdown("### 📋 Study Overview")
st.markdown("""
<div style="font-size: 1.2em; line-height: 1.8;">
<p><strong>In this study</strong>, you are evaluating explanations for why an AI model misclassified an image. For each case, you will see:</p>
<ul style="margin-left: 20px;">
<li>🖼️ An image that was misclassified by the model</li>
<li>🏷️ The <strong>true class</strong> (correct label) and <strong>predicted class</strong> (incorrect label)</li>
<li>📝 Four different explanations for the model's error (why the model made the wrong prediction)</li>
</ul>
<br>
<p><strong>Your task is to:</strong></p>
<ol style="margin-left: 20px;">
<li><strong>Annotate</strong> each explanation by marking both <span style="color: #4ade80;">🟢 good</span> and <span style="color: #ef4444;">❌ bad</span> parts</li>
<li><strong>Rank</strong> the explanations across five criteria: 
    <ul style="margin-left: 20px; margin-top: 5px;">
    <li>Overall Helpfulness</li>
    <li>Factuality</li>
    <li>Verbosity</li>
    <li>Specificity</li>
    <li>Actionability</li>
    </ul>
</li>
</ol>
<br>
<p>Before starting the evaluation, please enter your <strong>Prolific ID</strong> below.</p>
</div>
""", unsafe_allow_html=True)


# Test mode toggle
# test_mode = st.checkbox("🧪 Test Mode (Skip participant ID and MongoDB)", value=False)
test_mode = True
st.session_state.test_mode = test_mode

# Check if participant_id is already set
if 'participant_id' not in st.session_state:
    st.session_state.participant_id = ""

if not test_mode:
    participant_id = st.text_input(
        "Please enter your Prolific ID :red[*]",
        value=st.session_state.participant_id,
        label_visibility="visible",
        placeholder="Prolific ID"
    )
    
    if participant_id:
        st.session_state.participant_id = participant_id
else:
    # In test mode, set a dummy participant ID
    st.session_state.participant_id = "test_user"
    st.info("🧪 Test Mode: Using dummy participant ID")

if st.session_state.participant_id:
    if st.button("Start Evaluation", type="primary", use_container_width=True):
        st.switch_page("pages/1_human-study.py")