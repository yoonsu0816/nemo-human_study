import streamlit as st

st.set_page_config(page_title="Start - Human Study", page_icon=":pencil2:", initial_sidebar_state="collapsed")

st.title(":pencil2: Evaluation of Explanations for Model's Error")

st.markdown("---")

# Test mode toggle
test_mode = st.checkbox("🧪 Test Mode (Skip participant ID and MongoDB)", value=False)
st.session_state.test_mode = test_mode

# Check if participant_id is already set
if 'participant_id' not in st.session_state:
    st.session_state.participant_id = ""

if not test_mode:
    participant_id = st.text_input(
        "Enter your Participant ID (Your name):",
        value=st.session_state.participant_id,
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