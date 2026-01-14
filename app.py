import streamlit as st
import time
import os
import pandas as pd
import plotly.express as px
from backend import ModalXSystem
from emotion_engine import EmotionAnalyzer

st.set_page_config(
    page_title="DIU Presentation Grader",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    [data-testid="stSidebar"] {
        background-color: #12151e;
        border-right: 1px solid #333;
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
        box-shadow: 0 4px 12px rgba(32, 201, 151, 0.4);
        transition: transform 0.2s;
    }
    div.stDownloadButton > button:hover {
        transform: scale(1.02);
    }
    
    .stStatusWidget {
        background-color: #1E2130;
        border: 1px solid #444;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🎓 DIU Smart Faculty Grader</h1><p>AI-Powered Multi-Modal Presentation Assessment</p></div>', unsafe_allow_html=True)

@st.cache_resource
def load_emotion_engine():
    return EmotionAnalyzer()

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
    st.caption("Powered by ModalX Engine v2.1")

if analyze_btn:
    if not video_path or not s_name:
        st.error("⚠️ Please enter Student Name and Provide a Video first.")
    else:
        if 'modalx' not in st.session_state:
            with st.spinner("⚡ Booting AI Engine..."):
                st.session_state.modalx = ModalXSystem()

        try:
            emotion_engine = load_emotion_engine()
        except Exception as e:
            st.error(f"Could not load Emotion Model: {e}")
            st.stop()

        proc_col1, proc_col2 = st.columns([1, 1])
        
        with proc_col1:
            st.info("🎥 Source Stream")
            if is_url:
                st.markdown(f"**Analyzing Link:** `{url}`")
                st.image("https://upload.wikimedia.org/wikipedia/commons/1/14/Video_icon_2.svg", width=100)
            else:
                st.video(video_path)

        results = None
        emotion_results = None
        
        with proc_col2:
            with st.status("🚀 ModalX Engine Running...", expanded=True) as status:
                st.write("🔄 Initializing Neural Networks...")
                time.sleep(1)
                
                st.write("🧠 Running General Assessment (Whisper + CV)...")
                try:
                    results = st.session_state.modalx.analyze(video_path, s_name, s_id, is_url)
                except Exception as e:
                    st.error(f"General Engine Error: {e}")
                
                if video_path and os.path.exists(video_path) and not is_url:
                    st.write("🎭 Analyzing Emotional Tones (CNN-1D)...")
                    try:
                        e_times, e_emotions, e_summary = emotion_engine.predict(video_path)
                        emotion_results = {
                            "times": e_times, 
                            "emotions": e_emotions, 
                            "summary": e_summary
                        }
                    except Exception as e:
                        st.warning(f"Emotion Analysis skipped: {e}")
                elif is_url:
                    st.warning("⚠️ Emotion Graph unavailable for external URLs (Download required)")
                
                if results:
                    status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                else:
                    status.update(label="❌ Analysis Failed", state="error")

        if results:
            st.divider()
            st.balloons()

            score = results['score']
            
            grade = "F"
            grade_bg = "#dc3545"
            if score >= 80: grade, grade_bg = "A+", "#198754"
            elif score >= 75: grade, grade_bg = "A", "#20c997"
            elif score >= 70: grade, grade_bg = "A-", "#0dcaf0"
            elif score >= 65: grade, grade_bg = "B+", "#ffc107"
            elif score >= 60: grade, grade_bg = "B", "#fd7e14"
            elif score >= 50: grade, grade_bg = "C", "#d63384"

            c1, c2, c3 = st.columns([1.5, 1.5, 3])
            
            with c1:
                st.metric("Final Score", f"{score}/100", delta=f"{score-70} vs Avg")
            
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
                st.info("Grading Logic: Weighted average of Speech Clarity (60%) and Visual Engagement (40%).")

            st.divider()

            tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "🎭 Emotional Intelligence", "📝 Transcript & Feedback"])
            
            with tab1:
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.subheader("🗣️ Audio Intelligence")
                    audio = results['metrics']['audio']
                    
                    st.write(f"**Speaking Pace** ({audio['wpm']} WPM)")
                    st.progress(float(min(audio['wpm']/160, 1.0)))
                    
                    st.write(f"**Tonal Variation** (Score: {audio['physics']['pitch_variation']})")
                    st.progress(float(min(audio['physics']['pitch_variation']/50, 1.0)))
                    
                    st.write(f"**Confidence (Volume)** (Score: {audio['physics']['volume_score']})")
                    st.progress(float(min(audio['physics']['volume_score']/100, 1.0)))
                    
                    if audio['filler_count'] > 3:
                        st.warning(f"⚠️ High Filler Words Detected: {audio['filler_count']}")
                    else:
                        st.success(f"✅ Low Filler Words: {audio['filler_count']}")

                with col_b:
                    visual = results['metrics']['visual']
                    
                    if visual.get('is_slide_mode', False):
                        st.subheader("🖼️ Slide Design AI")
                        st.info("Scanner Mode: Slide Presentation")
                        slides = results['metrics']['slides']
                        
                        st.metric("Word Density", f"{slides['avg_words_per_slide']} words/slide")
                        st.write("**Readability Score**")
                        st.progress(float(min(slides['readability_score']/100, 1.0)))
                        st.write(f"**Slide Transitions Detected:** {slides['slide_changes']}")
                    else:
                        st.subheader("👁️ Behavioral AI")
                        st.info("Scanner Mode: Presenter Face")
                        
                        st.write(f"**Eye Contact Consistency** ({visual['eye_contact_score']}%)")
                        st.progress(float(visual['eye_contact_score']/100))
                        
                        st.write(f"**Posture Stability** ({visual['posture_score']}%)")
                        st.progress(float(visual['posture_score']/100))

            with tab2:
                if emotion_results and emotion_results['times']:
                    st.subheader("📈 Emotional Flow Over Time")
                    
                    e_times = emotion_results['times']
                    e_emotions = emotion_results['emotions']
                    e_summary = emotion_results['summary']
                    
                    df = pd.DataFrame({"Time (s)": e_times, "Emotion": e_emotions})
                    emotion_order = sorted(list(set(e_emotions)))
                    
                    fig = px.scatter(
                        df, x="Time (s)", y="Emotion", color="Emotion",
                        size=[15]*len(df), template="plotly_dark",
                        category_orders={"Emotion": emotion_order},
                        title="Speaker Emotion Timeline"
                    )
                    fig.update_traces(mode='lines+markers', line=dict(width=1, color='gray'))
                    fig.update_layout(height=400, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    c_pie, c_dom = st.columns([1, 1])
                    with c_pie:
                        st.markdown("##### Emotion Distribution")
                        fig_pie = px.pie(
                            names=list(e_summary.keys()), 
                            values=list(e_summary.values()),
                            hole=0.4, template="plotly_dark"
                        )
                        fig_pie.update_layout(paper_bgcolor="#0E1117")
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with c_dom:
                        dom_emotion = max(e_summary, key=e_summary.get)
                        st.markdown("##### Analysis")
                        st.metric("Dominant Tone", dom_emotion.upper())
                        
                        if dom_emotion in ['happy', 'neutral', 'surprise', 'surprised']:
                            st.success("The speaker maintains a **Positive/Confident** tone.")
                        elif dom_emotion in ['fear', 'sad']:
                            st.warning("The speaker seems **Nervous or Low Energy**. Needs more enthusiasm.")
                        elif dom_emotion in ['angry', 'disgust', 'anger']:
                            st.error("The speaker sounds **Aggressive/Frustrated**. Needs a softer tone.")
                else:
                    st.info("Emotion analysis is not available for this file type or URL.")

            with tab3:
                fb_col1, fb_col2 = st.columns([2, 1])
                
                with fb_col1:
                    st.markdown("### 🤖 AI Recommendations")
                    for item in results['feedback']:
                        st.warning(f"👉 {item}")
                    
                    st.markdown("### 📄 Speech Transcript")
                    st.text_area("Full Transcript", results['metrics']['audio']['transcript'], height=150)

                with fb_col2:
                    st.markdown("### 📥 Official Report")
                    st.write("Download the verified PDF report for faculty submission.")
                    
                    if results['report']:
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=results['report'],
                            file_name=f"ModalX_Report_{s_id}.pdf",
                            mime="application/pdf"
                        )

            if video_path and os.path.exists(video_path) and not is_url:
                os.remove(video_path)
