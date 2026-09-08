import streamlit as st
from workflow import run_study_workflow, WorkflowError

st.set_page_config(
    page_title="AI Study Pack Generator",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Study Pack Generator")
st.write("A multi-stage AI workflow that plans, generates, assesses, reviews, and refines personalized study material.")

with st.sidebar:
    st.header("⚙️ Study Settings")
    subject = st.text_input("Subject", placeholder="e.g., Civil Engineering")
    topic = st.text_input("Topic", placeholder="e.g., Reinforced Concrete")
    level = st.selectbox(
        "Education Level",
        ["School", "College", "University", "Competitive Exam"],
        index=2,
    )
    exam_type = st.selectbox(
        "Study Purpose",
        ["General Learning", "Semester Exam", "Competitive Exam", "Interview Preparation"],
    )
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
    language = st.selectbox("Language", ["English", "Urdu"], index=0)
    question_count = st.slider("Practice Questions", 5, 25, 10)
    generate = st.button("🚀 Generate Study Pack", type="primary", use_container_width=True)

st.info("Enter your subject and topic, configure the study settings, then generate the pack.")

if generate:
    if not subject.strip() or not topic.strip():
        st.error("Please enter both Subject and Topic.")
        st.stop()

    user_input = {
        "subject": subject.strip(),
        "topic": topic.strip(),
        "level": level,
        "exam_type": exam_type,
        "difficulty": difficulty,
        "language": language,
        "question_count": question_count,
    }

    progress = st.progress(0)
    status = st.empty()

    try:
        with st.spinner("Running the multi-stage AI workflow..."):
            status.info("🔄 Stage 1/5 — Planning...")
            progress.progress(10)
            result = run_study_workflow(
                user_input,
                progress_callback=lambda stage, pct: (
                    status.info(stage),
                    progress.progress(pct),
                ),
            )

        progress.progress(100)
        status.success("✅ All five workflow stages completed successfully.")

        st.success("🎉 Your personalized study pack is ready!")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["🗺️ Plan", "📖 Content", "📝 Assessment", "🔍 Review", "🏆 Final Pack"]
        )

        with tab1:
            st.markdown(result["plan"])

        with tab2:
            st.markdown(result["content"])

        with tab3:
            st.markdown(result["assessment"])

        with tab4:
            st.markdown(result["review"])

        with tab5:
            st.markdown(result["final_pack"])
            st.download_button(
                "⬇️ Download Final Study Pack",
                data=result["final_pack"],
                file_name="ai_study_pack.md",
                mime="text/markdown",
                use_container_width=True,
            )

    except WorkflowError as exc:
        progress.empty()
        status.error("❌ Workflow stopped.")
        st.error(str(exc))
    except Exception as exc:
        progress.empty()
        status.error("❌ Unexpected application error.")
        st.error(f"{type(exc).__name__}: {exc}")

st.divider()
st.caption("AI Study Pack Generator • Multi-stage workflow: Plan → Generate → Assess → Review → Refine")
