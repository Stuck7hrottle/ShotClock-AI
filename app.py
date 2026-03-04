import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from analyzer import extract_and_analyze

st.set_page_config(page_title="ShotClock AI", layout="wide")

st.title("🎯 ShotClock AI: Rate of Fire Analyzer")
st.write("Upload a range video to detect shots and calculate splits.")

uploaded_file = st.file_uploader("Upload Video (MP4, MOV)", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Save the uploaded file temporarily
    with open("temp_upload.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.video("temp_upload.mp4")
    
    with st.spinner("Analyzing audio transients..."):
        df = extract_and_analyze("temp_upload.mp4")
    
    if not df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Shot Data")
            st.dataframe(df, use_container_width=True)
            
        with col2:
            st.subheader("Split Consistency")
            # Plotting the splits over time
            fig, ax = plt.subplots()
            ax.plot(df["Shot #"], df["Split (s)"], marker='o', color='#ff4b4b')
            ax.set_ylabel("Split Time (seconds)")
            ax.set_xlabel("Shot Number")
            st.pyplot(fig)
            
        st.success(f"Average Rate of Fire: {int(df['Inst. RPM'].mean())} RPM")
    else:
        st.warning("No shots detected. Try adjusting the sensitivity.")
