"""
ModalX v2 - Deep Learning Presentation Grader
Streamlit Dashboard for AI-Powered Presentation Assessment
"""

import streamlit as st
import time
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from backend import ModalXSystemV2

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ModalX v2.0 - Deep Learning Grader",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #0a0f1a; color: #FAFAFA; }
    h1, h2, h3, h4, h5, h6, p, li, span { color: #FAFAFA !important; }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #6B8DD6 100%);
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
    }
    
    .model-card {
        background: linear-gradient(145deg, #1a1f2e 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
        transform: translateY(-2px);
    }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1a1f2e 0%, #0d1117 100%);
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        border-color: #667eea;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }
    
    .score-excellent { color: #10b981 !important; }
    .score-good { color: #3b82f6 !important; }
    .score-average { color: #f59e0b !important; }
    .score-poor { color: #ef4444 !important; }
    
    div.stDownloadButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white !important;
        border: none;
        width: 100%;
        padding: 15px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    div.stDownloadButton > button:hover {
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.5);
        transform: translateY(-2px);
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('''
<div class="main-header">
    <h1 style="margin:0; font-size:2.5rem;">🧠 ModalX v2.0</h1>
    <p style="margin:10px 0 0 0; font-size:1.1rem; opacity:0.9;">
        Deep Learning Presentation Assessment System
    </p>
    <p style="margin:5px 0 0 0; font-size:0.85rem; opacity:0.7;">
        Transformer Emotion • ST-GCN Gestures • BERT Content • ViT Slides
    </p>
</div>
''', unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("📋 Student Details")
    s_name = st.text_input("Student Name", placeholder="e.g. Muntasir Islam")
    s_id = st.text_input("Student ID", placeholder="e.g. 181-15-XXXX")
    
    st.markdown("---")
    
    st.header("🤖 Deep Learning Models")
    
    st.markdown("""
    <div class="model-card">
        <b>🎭 Emotion</b><br/>
        <small>Transformer + Attention</small>
    </div>
    <div class="model-card">
        <b>👤 Facial AU</b><br/>
        <small>ResNet-50 + LSTM</small>
    </div>
    <div class="model-card">
        <b>🙌 Gesture</b><br/>
        <small>ST-GCN Network</small>
    </div>
    <div class="model-card">
        <b>🎙️ Prosody</b><br/>
        <small>CNN-BiLSTM</small>
    </div>
    <div class="model-card">
        <b>📝 Content</b><br/>
        <small>DistilBERT</small>
    </div>
    <div class="model-card">
        <b>🖼️ Slides</b><br/>
        <small>Vision Transformer</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.header("📂 Video Source")
    input_method = st.radio("Choose Source:", ["Upload Video", "URL (YouTube/Drive)"])
    
    video_path = None
    is_url = False
    
    if input_method == "Upload Video":
        uploaded_file = st.file_uploader("Select MP4", type=["mp4", "webm", "mov"])
        if uploaded_file:
            with open("temp_upload.mp4", "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_path = "temp_upload.mp4"
            st.success(f"✅ {uploaded_file.name}")
    else:
        url = st.text_input("Paste URL Here")
        if url:
            video_path = url
            is_url = True
            if "drive.google.com" in url:
                st.caption("✅ Google Drive Link")
            elif "youtube" in url or "youtu.be" in url:
                st.caption("✅ YouTube Link")
    
    st.markdown("---")
    analyze_btn = st.button("🚀 Analyze with Deep Learning", type="primary", use_container_width=True)
    st.caption("Powered by ModalX v2.0 | 6 DL Models")

# --- MAIN ANALYSIS ---
if analyze_btn:
    if not video_path or not s_name:
        st.error("⚠️ Please enter Student Name and provide a video.")
    else:
        # Initialize system
        if 'modalx_v2' not in st.session_state:
            with st.spinner("⚡ Loading 6 Deep Learning Models..."):
                st.session_state.modalx_v2 = ModalXSystemV2(weights_dir="weights")
        
        col_video, col_status = st.columns([1, 1])
        
        with col_video:
            st.subheader("🎥 Source Video")
            if is_url:
                st.markdown(f"**Analyzing:** `{url[:50]}...`")
                st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/2560px-YouTube_full-color_icon_%282017%29.svg.png", width=100)
            else:
                st.video(video_path)
        
        results = None
        
        with col_status:
            with st.status("🧠 Deep Learning Analysis...", expanded=True) as status:
                steps = [
                    ("🔊", "Transcribing with Whisper...", 1),
                    ("🎭", "Analyzing emotions (Transformer)...", 2),
                    ("👤", "Detecting facial Action Units (ResNet-50)...", 2),
                    ("🙌", "Classifying gestures (ST-GCN)...", 2),
                    ("🎙️", "Evaluating prosody (CNN-BiLSTM)...", 1),
                    ("📝", "Scoring content (DistilBERT)...", 1),
                    ("🖼️", "Grading slides (Vision Transformer)...", 2),
                    ("📄", "Generating PDF report...", 1)
                ]
                
                for emoji, text, delay in steps:
                    st.write(f"{emoji} {text}")
                    time.sleep(delay * 0.5)  # Reduced for demo
                
                try:
                    results = st.session_state.modalx_v2.analyze(video_path, s_name, s_id, is_url)
                    status.update(label="✅ Deep Learning Analysis Complete!", state="complete", expanded=False)
                except Exception as e:
                    st.error(f"Analysis Error: {e}")
                    status.update(label="❌ Analysis Failed", state="error")
        
        # --- RESULTS ---
        if results:
            st.divider()
            st.balloons()
            
            score = results['score']
            mode = results.get('mode', 'unknown')
            
            # Grade calculation
            def get_grade_info(s):
                if s >= 85: return "A+", "#10b981", "Exceptional"
                elif s >= 80: return "A", "#22c55e", "Excellent"
                elif s >= 75: return "A-", "#3b82f6", "Very Good"
                elif s >= 70: return "B+", "#6366f1", "Good"
                elif s >= 65: return "B", "#8b5cf6", "Above Average"
                elif s >= 60: return "B-", "#a855f7", "Average"
                elif s >= 55: return "C+", "#f59e0b", "Below Average"
                elif s >= 50: return "C", "#f97316", "Needs Work"
                else: return "F", "#ef4444", "Insufficient"
            
            grade, grade_color, grade_desc = get_grade_info(score)
            
            # Top metrics row
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                st.metric("Final Score", f"{score}/100", delta=f"{score-70} vs avg")
            
            with c2:
                st.markdown(f"""
                <div style="background:{grade_color}; padding:15px; border-radius:12px; text-align:center;">
                    <p style="margin:0; font-size:0.9rem; color:white;">Grade</p>
                    <h1 style="margin:0; font-size:2.5rem; color:white;">{grade}</h1>
                    <small style="color:rgba(255,255,255,0.8);">{grade_desc}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with c3:
                st.metric("Mode", mode.replace("_", " ").title())
            
            with c4:
                st.metric("Student", s_name)
            
            st.divider()
            
            # Tabs for detailed results
            tabs = st.tabs(["📊 Metrics", "🎭 Emotions", "📝 Content", "🤖 AI Feedback", "📄 Report"])
            
            # TAB 1: Metrics
            with tabs[0]:
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.subheader("🎙️ Audio Intelligence")
                    audio = results['metrics']['audio']
                    
                    st.metric("Words Per Minute", f"{audio['wpm']} WPM")
                    st.progress(min(audio['wpm'] / 180, 1.0))
                    
                    if 'prosody' in audio and audio['prosody']:
                        prosody = audio['prosody']
                        col_p1, col_p2 = st.columns(2)
                        with col_p1:
                            st.metric("Pitch Dynamism", f"{prosody.get('pitch_dynamism', 'N/A'):.0f}")
                        with col_p2:
                            st.metric("Fluency Index", f"{prosody.get('fluency_index', 'N/A'):.0f}")
                
                with col_b:
                    if mode == "face_presentation":
                        st.subheader("👤 Visual Behavior")
                        visual = results['metrics']['visual']
                        
                        if visual.get('gesture'):
                            gesture = visual['gesture']
                            st.metric("Movement Intensity", f"{gesture.get('avg_movement_intensity', 0)*100:.0f}%")
                        
                        if visual.get('au'):
                            au = visual['au']
                            st.metric("Engagement", f"{au.get('avg_engagement', 0)*100:.0f}%")
                            st.metric("Confidence", f"{au.get('avg_confidence', 0)*100:.0f}%")
                    else:
                        st.subheader("🖼️ Slide Design")
                        slides = results['metrics'].get('slides')
                        if slides:
                            st.metric("Slides Analyzed", slides.get('slides_analyzed', 0))
                            avg = slides.get('avg_scores', {})
                            if avg:
                                st.metric("Avg Word Count", f"{avg.get('avg_word_count', 0):.0f}")
                                st.metric("Visual Balance", f"{avg.get('avg_visual_balance', 0):.0f}%")
            
            # TAB 2: Emotions
            with tabs[1]:
                emo_data = results.get('emotion_data', {})
                
                if emo_data.get('times') and emo_data.get('emotions'):
                    st.subheader("📈 Emotional Flow")
                    
                    df = pd.DataFrame({
                        "Time (s)": emo_data['times'],
                        "Emotion": emo_data['emotions']
                    })
                    
                    fig = px.scatter(
                        df, x="Time (s)", y="Emotion",
                        color="Emotion",
                        template="plotly_dark",
                        height=400
                    )
                    fig.update_traces(mode='lines+markers')
                    fig.update_layout(paper_bgcolor="#0a0f1a", plot_bgcolor="#0a0f1a")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Emotion distribution
                    col_pie, col_stats = st.columns([1, 1])
                    
                    with col_pie:
                        summary = emo_data.get('summary', {})
                        if summary:
                            fig_pie = px.pie(
                                names=list(summary.keys()),
                                values=list(summary.values()),
                                hole=0.4,
                                template="plotly_dark"
                            )
                            fig_pie.update_layout(paper_bgcolor="#0a0f1a")
                            st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col_stats:
                        st.metric("Emotion Score", f"{emo_data.get('score', 0):.0f}/100")
                        
                        if summary:
                            dominant = max(summary, key=summary.get)
                            st.metric("Dominant Emotion", dominant.upper())
                            
                            if dominant in ['happy', 'neutral', 'calm']:
                                st.success("Positive emotional presence detected")
                            else:
                                st.warning("Consider projecting more positive energy")
                else:
                    st.info("Emotion timeline not available for this video")
            
            # TAB 3: Content
            with tabs[2]:
                content = results['metrics'].get('content', {})
                
                if content:
                    col_c1, col_c2 = st.columns(2)
                    
                    with col_c1:
                        st.subheader("📊 Content Metrics")
                        st.metric("Vocabulary Level", content.get('vocabulary_level', 'N/A').upper())
                        st.metric("Word Count", content.get('word_count', 0))
                        st.metric("Power Words", content.get('power_word_count', 0))
                        st.metric("Transitions", content.get('transition_count', 0))
                    
                    with col_c2:
                        st.subheader("📈 Quality Scores")
                        scores = {
                            'Argument': content.get('argument_score', 50),
                            'Structure': content.get('structure_score', 50),
                            'Engagement': content.get('engagement_score', 50),
                            'Professionalism': content.get('professionalism_score', 50)
                        }
                        
                        for name, score in scores.items():
                            st.write(f"**{name}:** {score:.0f}/100")
                            st.progress(score / 100)
                    
                    with st.expander("📜 Transcript Preview"):
                        st.text(results['metrics']['audio'].get('transcript', 'No transcript')[:1000])
            
            # TAB 4: Feedback
            with tabs[3]:
                st.subheader("🤖 AI-Generated Recommendations")
                
                feedback = results.get('feedback', [])
                if feedback:
                    for i, item in enumerate(feedback, 1):
                        if any(word in item.lower() for word in ['great', 'excellent', 'good']):
                            st.success(f"✅ {item}")
                        elif any(word in item.lower() for word in ['warning', 'reduce', 'avoid', 'too']):
                            st.warning(f"⚠️ {item}")
                        else:
                            st.info(f"💡 {item}")
                else:
                    st.success("No major issues detected! Great presentation.")
            
            # TAB 5: Report
            with tabs[4]:
                st.subheader("📄 Download Official Report")
                
                col_dl1, col_dl2 = st.columns([2, 1])
                
                with col_dl1:
                    st.write("Your comprehensive PDF report contains:")
                    st.markdown("""
                    - ✅ Final weighted score and grade
                    - ✅ Score breakdown by category
                    - ✅ Emotional timeline graph
                    - ✅ AI recommendations
                    - ✅ Deep learning model insights
                    """)
                
                with col_dl2:
                    if results.get('report'):
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=results['report'],
                            file_name=f"ModalX_v2_Report_{s_id}.pdf",
                            mime="application/pdf"
                        )
            
            # Cleanup
            if video_path and os.path.exists(video_path) and not is_url:
                os.remove(video_path)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align:center; opacity:0.6; font-size:0.85rem;">
    ModalX v2.0 | NL Circuits Team | ModalX-AI Challenge DIU<br/>
</div>
""", unsafe_allow_html=True)
