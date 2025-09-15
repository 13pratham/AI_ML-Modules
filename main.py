import os
import streamlit as st
from pathlib import Path

def streamlit_ui():
    modules_path = Path(__file__).parent / "Modules"
    modules_path.mkdir(exist_ok=True)
    modules = os.listdir(modules_path)
    st.title("AI/ML Modules")
    selected_module = st.selectbox("Select a module", ['Select a Module'] + modules)
    if selected_module != 'Select a Module':
        with open(modules_path / selected_module, 'r') as file:
            content = file.read()
        st.write(content)

if __name__ == "__main__":
    streamlit_ui()