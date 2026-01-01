import streamlit as st
import pandas as pd
import keras
import pickle
import re
import plotly.express as px
from keras.utils import pad_sequences

# config
st.set_page_config(page_title="Cyberbullying Auditor", layout="wide", page_icon="👁️")

# load artifacts
model = keras.models.load_model('model_bullying.keras')
with open('tokenizer.pickle', 'rb') as h:
    tokenizer = pickle.load(h)
with open('encoder.pickle', 'rb') as h:
    encoder = pickle.load(h)

# utils
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# state
if 'history_df' not in st.session_state:
    st.session_state.history_df = pd.DataFrame(columns=['Comment', 'Classification'])

st.title("Cyberbullying Auditor")

# layout
col1, spacer, col2 = st.columns([1.8, 0.2, 1.2])

with col1:
    # new analysis section
    st.subheader("New Analysis")
    user_input = st.text_input("Comment", label_visibility="collapsed", placeholder="Enter a comment...")
    
    if st.button("Analyze") and user_input:
        # inference
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
        
        prediction = model.predict(padded)
        label = encoder.inverse_transform([prediction.argmax()])[0]
        
        # update state
        new_entry = pd.DataFrame({'Comment': [user_input], 'Classification': [label.upper()]})
        st.session_state.history_df = pd.concat([st.session_state.history_df, new_entry], ignore_index=True)

    # history table
    if not st.session_state.history_df.empty:
        st.divider()
        st.subheader("History")
        
        # dynamic height math: (rows + header) * 35px
        # logic: allow growth up to 5 rows, then lock height to force scroll
        rows = len(st.session_state.history_df)
        max_rows = 5
        # calc height: min(actual_rows, 5) + 1 for header * 35px pixel density
        table_height = (min(rows, max_rows) + 1) * 35 + 3

        st.dataframe(
            st.session_state.history_df, 
            use_container_width=True, 
            height=table_height
        )
        
        # download
        csv = st.session_state.history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Table as CSV",
            data=csv,
            file_name='cyberbullying_report.csv',
            mime='text/csv',
        )

with col2:
    if not st.session_state.history_df.empty:
        # distribution chart
        st.subheader("Distribution")
        
        counts = st.session_state.history_df['Classification'].value_counts().reset_index()
        counts.columns = ['Category', 'Count']
        
        fig = px.pie(counts, values='Count', names='Category', hole=0.6)
        
        fig.update_layout(
            margin=dict(t=40, b=20, l=0, r=0),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown(
    """
    <div style='text-align: center; font-size: 12px; color: gray;'>
        Made by 
        <a href='https://www.linkedin.com/in/tobiascarpintieri/' target='_blank' rel='noopener noreferrer' 
           style='color: gray; text-decoration: none; font-weight: bold;'>
           Tobias Carpintieri
        </a> · 2026
    </div>
    """, 
    unsafe_allow_html=True
)