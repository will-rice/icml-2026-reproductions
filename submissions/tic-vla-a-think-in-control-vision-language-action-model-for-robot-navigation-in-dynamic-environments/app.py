import streamlit as st
import glob

st.set_page_config(page_title="TIC-VLA Reproduction", layout="wide")
st.title("TIC-VLA Reproduction Logbook & Benchmark Evidence")

pages = sorted(glob.glob("pages/*.md"))
if pages:
    page = st.sidebar.radio("Select Evidence Page", pages)
    with open(page, "r", encoding="utf-8") as f:
        st.markdown(f.read())
else:
    st.write("Reproducibility evidence logbook.")
