import os
import streamlit as st
from pathlib import Path

def streamlit_ui():
    modules_path = Path(__file__).parent / 'Modules'
    modules_path.mkdir(exist_ok=True)
    modules = os.listdir(modules_path)
    st.set_page_config(page_title='AI_ML Modules', layout='wide')
    st.title('AI_ML Module Selection')
    selected_module = st.selectbox("Select Module", modules)
    with open(modules_path / selected_module, 'r') as file:
        content = file.read()
    st.write(content)

if __name__=='__main__':
    streamlit_ui()