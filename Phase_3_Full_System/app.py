import streamlit as st
import time
import os
from backend import ModalXSystem

st.set_page_config(
    page_title="DIU Presentation Grader",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    h1, h2, h3, h4, h5, h6, p, li, span {
        color: #FAFAFA !important;
    }
    .stMarkdown, .stText {
        color: #E0E0E0 !important;
    }
    
    .main-header {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #4b6cb7;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #444;
        padding: 15px;
        border-radius: 10px;
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: scale(1.02);
        border-color: #4b6cb7;
    }
    div[data-testid="stMetric"] label { color: #AAAAAA !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #FFFFFF !important; }

    .stTextInput input {
        background-color: #262730;
        color: white;
        border: 1px solid #444;
    }
    
    div.stDownloadButton > button {
        background: linear-gradient(45deg, #198754, #20c997);
        color: white !important;
        border: none;
        width: 100%;
        padding: 15px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(25, 135, 84, 0.4);
    }
    div.stDownloadButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    [data-testid="stSidebar"] {
        background-color: #161a24;
        border-right: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🎓 DIU Smart Faculty Grader</h1><p>Automated Presentation Assessment System</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📋 Student Details")
    s_name = st.text_input("Student Name", placeholder="e.g. Muntasir Islam")
    s_id = st.text_input("Student ID", placeholder="e.g. 181-15-XXXX")
    st.markdown("---")
    
    st.header("📂 Evidence Upload")
    input_method = st.radio("Source:", ["Upload Video File", "Google Drive / YouTube Link"])
    
    video_path = None
    is_url = False

    if input_method == "Upload Video File":
        uploaded_file = st.file_uploader("Select Presentation Video (MP4)", type=["mp4"])
        if uploaded_file:
            with open("temp_upload.mp4", "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_path = "temp_upload.mp4"
            st.video(uploaded_file)
    else:
        url = st.text_input("Paste Link (Google Drive or YouTube)")
        if url:
            video_path = url
            is_url = True
            if "drive.google.com" in url:
                st.caption("✅ Google Drive Link Detected")
            elif "youtube" in url or "youtu.be" in url:
                st.caption("✅ YouTube Link Detected")

    st.markdown("---")
    analyze_btn = st.button("🚀 Generate Grading Report", type="primary")

if analyze_btn:
    if not video_path or not s_name:
        st.error("⚠️ Please enter Student Name and Provide a Video first.")
    else:
        if 'modalx' not in st.session_state:
            with st.spinner("Initializing Grading Engine..."):
                st.session_state.modalx = ModalXSystem()

        progress = st.progress(0)
        status = st.empty()
        
        status.text("🔍 Analyzing Verbal Delivery...")
        progress.progress(30)
        time.sleep(0.5)
        
        status.text("👁️ Assessing Body Language & Eye Contact...")
        progress.progress(70)
        
        try:
            results = st.session_state.modalx.analyze(video_path, s_name, s_id, is_url)
            progress.progress(100)
            status.empty()
            
            if not results:
                st.error("Analysis Failed. Check the link or file permissions.")
            else:
                score = results['score']
                
                grade = "F"
                grade_bg = "#dc3545"
                if score >= 80: grade, grade_bg = "A+", "#198754"
                elif score >= 75: grade, grade_bg = "A", "#20c997"
                elif score >= 70: grade, grade_bg = "A-", "#0dcaf0"
                elif score >= 65: grade, grade_bg = "B+", "#ffc107"
                elif score >= 60: grade, grade_bg = "B", "#fd7e14"
                elif score >= 50: grade, grade_bg = "C", "#d63384"

                c1, c2, c3 = st.columns([1, 1, 2])
                with c1:
                    st.metric("Final Score", f"{score}/100")
                with c2:
                    st.markdown(f"""
                    <div style="background-color:{grade_bg}; padding:15px; border-radius:10px; text-align:center; box-shadow: 0 0 15px {grade_bg};">
                        <h4 style="margin:0; color:white !important; font-size:1rem;">Grade</h4>
                        <h1 style="margin:0; font-size: 3rem; color:white !important; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">{grade}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    st.info(f"**Assessment for:** {s_name} ({s_id})")
                    st.caption("Graded on: Pacing, Pitch, Fillers, Eye Contact, Posture.")

                st.divider()

                st.subheader("📊 Rubric Evaluation")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🗣️ Verbal Delivery")
                    audio = results['metrics']['audio']
                    
                    st.write(f"**Pace:** {audio['wpm']} WPM")
                    st.progress(min(audio['wpm']/150, 1.0))
                    
                    st.write(f"**Intonation:** {audio['physics']['pitch_variation']}")
                    st.progress(min(audio['physics']['pitch_variation']/50, 1.0))
                    
                    st.metric("Filler Words", audio['filler_count'])
                    
                with col2:
                    st.markdown("### 👁️ Non-Verbal")
                    visual = results['metrics']['visual']
                    
                    st.write(f"**Eye Contact:** {visual['eye_contact_score']}%")
                    st.progress(visual['eye_contact_score']/100)
                    
                    st.write(f"**Posture:** {visual['posture_score']}%")
                    st.progress(visual['posture_score']/100)

                st.divider()

                c_left, c_right = st.columns([2, 1])
                with c_left:
                    with st.expander("📝 View Transcript (Plagiarism Check)"):
                        st.text_area("Full Text", results['metrics']['audio']['transcript'], height=200)
                
                with c_right:
                    st.markdown("### 📥 Save Report")
                    if results['report']:
                        st.download_button(
                            label="Download Official PDF Result",
                            data=results['report'],
                            file_name=f"Evaluation_{s_id}.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.error("Report generation failed.")

            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                print(f"🗑️ Cleaned up temporary video: {video_path}")

        except Exception as e:
            st.error(f"Error: {e}")
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
