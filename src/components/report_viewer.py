from streamlit import st

def display_report(interpretation):
    st.header("Medical Report Interpretation")
    st.write(interpretation)