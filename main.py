import streamlit as st

st.set_page_config(page_title="Start - Human Study", page_icon=":pencil2:", initial_sidebar_state="collapsed")

st.title(":pencil2: Evaluation of Explanations for Model's Error")

st.markdown("---")

# Check if participant_id is already set
if 'participant_id' not in st.session_state:
    st.session_state.participant_id = ""

participant_id = st.text_input(
    "Enter your Participant ID (Your name):",
    value=st.session_state.participant_id,
    # placeholder="e.g., P01, P02, etc.",
    # help="Please enter your unique participant ID to begin the study."
)

if participant_id:
    st.session_state.participant_id = participant_id
    
    # st.success(f"Participant ID set: **{participant_id}**")
    # st.markdown("---")
    
    if st.button("Start Evaluation", type="primary", use_container_width=True):
        st.switch_page("pages/1_human-study.py")
