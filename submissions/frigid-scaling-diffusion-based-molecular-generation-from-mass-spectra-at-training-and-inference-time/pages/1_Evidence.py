import streamlit as st

from src.frigid_repro import generate_evidence


st.title("Evidence Bundle")
st.json(generate_evidence())
