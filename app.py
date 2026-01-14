import streamlit as st
import time
import os
import pandas as pd
import plotly.express as px
from backend import ModalXSystem

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="DIU Presentation Grader",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    h1, h2, h3, h4, h5, h6, p, li, span { color: #FAFAFA !important; }
    .stMarkdown, .stText { color: #E0E0E0 !important; }
    
    .main-header {
        background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        border: 1px solid #4b6cb7;
        box-shadow: 0 4px 20px rgba(0,0,0,0.6);
    }
    
    div[data-testid="stMetric"] {
        background-color: #1E2130;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: #0dcaf0;
        box-shadow: 0 5px 15px rgba(13, 202, 240, 0.3);
    }
    div[data-testid="stMetric"] label { color: #AAAAAA !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #FFFFFF !important; }

    .stTextInput input {
        background-color: #1E2130;
        color: white;
        border: 1px solid #444;
    }
    div.stDownloadButton > button {
        background: linear-gradient(90deg, #198754, #20c997);
        color: white !important;
        border: none;
        width: 100%;
        padding: 15px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🎓 DIU Smart Faculty Grader</h1><p>AI-Powered Multi-Modal Presentation Assessment</p></div>', unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("📋 Student Details")
    s_name = st.text_input("Student Name", placeholder="e.g. Muntasir Islam")
    s_id = st.text_input("Student ID", placeholder="e.g. 181-15-XXXX")
    st.markdown("---")
    
    st.header("📂 Evidence Source")
    input_method = st.radio("Choose Source:", ["Upload Video File", "Google Drive / YouTube Link"])
    
    video_path = None
    is_url = False

    if input_method == "Upload Video File":
        uploaded_file = st.file_uploader("Select MP4 Video", type=["mp4"])
        if uploaded_file:
            with open("temp_upload.mp4", "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_path = "temp_upload.mp4"
            st.success(f"Video Uploaded: {uploaded_file.name}")
    else:
        url = st.text_input("Paste Link Here")
        if url:
            video_path = url
            is_url = True
            if "drive.google.com" in url:
                st.caption("✅ Google Drive Link Detected")
            elif "youtube" in url or "youtu.be" in url:
                st.caption("✅ YouTube Link Detected")

    st.markdown("---")
    analyze_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
    st.caption("Powered by ModalX Engine v3.0")

# --- MAIN LOGIC ---
if analyze_btn:
    if not video_path or not s_name:
        st.error("⚠️ Please enter Student Name and Provide a Video first.")
    else:
        # Initialize Backend
        if 'modalx' not in st.session_state:
            with st.spinner("⚡ Booting AI Engine..."):
                st.session_state.modalx = ModalXSystem()

        proc_col1, proc_col2 = st.columns([1, 1])
        
        with proc_col1:
            st.info("🎥 Source Stream")
            if is_url:
                st.markdown(f"**Analyzing Link:** `{url}`")
                st.image("https://upload.wikimedia.org/wikipedia/commons/1/14/Video_icon_2.svg", width=100)
            else:
                st.video(video_path)

        results = None
        
        with proc_col2:
            with st.status("🚀 ModalX Engine Running...", expanded=True) as status:
                st.write("🔄 Initializing Neural Networks...")
                time.sleep(1)
                
                st.write("🧠 Extracting Audio & Transcribing (Whisper)...")
                st.write("👁️ Scanning Facial Landmarks (MediaPipe)...")
                st.write("🎭 Analyzing Micro-Expressions & Tone (CNN)...")
                st.write("📝 Evaluating Content Impact & Vocabulary...")
                
                try:
                    # The backend now handles EVERYTHING (Audio, Visual, Emotion, Content)
                    results = st.session_state.modalx.analyze(video_path, s_name, s_id, is_url)
                    status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                except Exception as e:
                    st.error(f"Engine Error: {e}")
                    status.update(label="❌ Analysis Failed", state="error")

        # --- RESULTS DASHBOARD ---
        if results:
            st.divider()
            st.balloons()

            score = results['score']
            
            # Grade Logic
            grade = "F"; grade_bg = "#dc3545"
            if score >= 80: grade, grade_bg = "A+", "#198754"
            elif score >= 75: grade, grade_bg = "A", "#20c997"
            elif score >= 70: grade, grade_bg = "A-", "#0dcaf0"
            elif score >= 65: grade, grade_bg = "B+", "#ffc107"
            elif score >= 60: grade, grade_bg = "B", "#fd7e14"
            elif score >= 50: grade, grade_bg = "C", "#d63384"

            c1, c2, c3 = st.columns([1.5, 1.5, 3])
            
            with c1:
                st.metric("Final Weighted Score", f"{score}/100", delta=f"{score-70} vs Avg")
            
            with c2:
                st.markdown(f"""
                <div style="background-color:{grade_bg}; padding:10px; border-radius:10px; text-align:center; box-shadow: 0 0 15px {grade_bg}80;">
                    <p style="margin:0; font-size:0.9rem; color:white;">Official Grade</p>
                    <h1 style="margin:0; font-size: 2.5rem; color:white; font-weight:800;">{grade}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with c3:
                st.markdown(f"### 👤 {s_name}")
                st.caption(f"Student ID: {s_id}")
                st.info("Grading Logic: Audio (30%), Visual (30%), Emotion (20%), Content (20%)")

            st.divider()

            # --- TABS ---
            tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "🎭 Emotional Intelligence", "📝 Content & Report"])
            
            # TAB 1: CORE METRICS
            with tab1:
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.subheader("🗣️ Audio Intelligence")
                    audio = results['metrics']['audio']
                    st.write(f"**Speaking Pace** ({audio['wpm']} WPM)")
                    st.progress(float(min(audio['wpm']/160, 1.0)))
                    st.write(f"**Pitch Variation** ({audio['physics']['pitch_variation']})")
                    st.progress(float(min(audio['physics']['pitch_variation']/50, 1.0)))
                    st.metric("Pause Ratio", f"{audio['physics']['pause_ratio']}%")

                with col_b:
                    visual = results['metrics']['visual']
                    if visual.get('is_slide_mode', False):
                        st.subheader("🖼️ Slide Design AI")
                        st.info("Mode: Slide Presentation")
                        slides = results['metrics']['slides']
                        st.metric("Word Density", f"{slides['avg_words_per_slide']} words/slide")
                        st.metric("Slide Readability", f"{int(slides['readability_score'])}/100")
                    else:
                        st.subheader("👁️ Behavioral AI")
                        st.info("Mode: Face Presentation")
                        st.write(f"**Eye Contact** ({visual['eye_contact_score']}%)")
                        st.progress(float(visual['eye_contact_score']/100))
                        st.write(f"**Posture Stability** ({visual['posture_score']}%)")
                        st.progress(float(visual['posture_score']/100))

            # TAB 2: EMOTION ANALYZER
            with tab2:
                emo_data = results.get('emotion_data', {})
                if emo_data and emo_data.get('times'):
                    st.subheader("📈 Emotional Flow Over Time")
                    
                    e_times = emo_data['times']
                    e_emotions = emo_data['emotions']
                    e_summary = emo_data['summary']
                    
                    # Timeline Graph
                    df = pd.DataFrame({"Time (s)": e_times, "Emotion": e_emotions})
                    emotion_order = sorted(list(set(e_emotions)))
                    
                    fig = px.scatter(
                        df, x="Time (s)", y="Emotion", color="Emotion",
                        size=[15]*len(df), template="plotly_dark",
                        category_orders={"Emotion": emotion_order}
                    )
                    fig.update_traces(mode='lines+markers', line=dict(width=1, color='gray'))
                    fig.update_layout(height=350, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary
                    c_pie, c_stat = st.columns([1, 1])
                    with c_pie:
                        st.markdown("##### Emotion Distribution")
                        fig_pie = px.pie(names=list(e_summary.keys()), values=list(e_summary.values()), hole=0.4, template="plotly_dark")
                        fig_pie.update_layout(paper_bgcolor="#0E1117")
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with c_stat:
                        dom_emotion = max(e_summary, key=e_summary.get)
                        st.metric("Dominant Tone", dom_emotion.upper())
                        if dom_emotion in ['happy', 'neutral', 'surprise']:
                            st.success("Positive tone detected. Good confidence.")
                        else:
                            st.warning("Negative/Nervous tone detected.")
                else:
                    st.info("Emotion analysis unavailable for this file.")

            # TAB 3: CONTENT & REPORT
            with tab3:
                c_col1, c_col2 = st.columns([2, 1])
                
                with c_col1:
                    st.subheader("🧠 Content Intelligence")
                    audio = results['metrics']['audio']
                    
                    col_i1, col_i2 = st.columns(2)
                    col_i1.metric("Content Score", f"{int(audio.get('content_score', 0))}/100")
                    col_i2.metric("Power Words Used", audio.get('impact_words', 0))
                    
                    st.markdown("### 🤖 AI Recommendations")
                    for item in results['feedback']:
                        st.warning(f"👉 {item}")
                        
                    with st.expander("View Full Transcript"):
                        st.text(audio['transcript'])

                with c_col2:
                    st.markdown("### 📥 Official Report")
                    st.write("Download the verified PDF report containing all graphs and scores.")
                    
                    if results['report']:
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=results['report'],
                            file_name=f"ModalX_Report_{s_id}.pdf",
                            mime="application/pdf"
                        )

            # Cleanup
            if video_path and os.path.exists(video_path) and not is_url:
                os.remove(video_path)
