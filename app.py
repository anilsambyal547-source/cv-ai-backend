"""
Streamlit Dashboard for CV AI System.
Features:
- Upload CVs (PDF)
- Search with filters (skills, experience, education, seniority, boolean)
- Candidate detail view with fit breakdown and AI insights
- Shortlist management
- Email candidates (single or shortlist) with cooldown warning and force option
- Navigation buttons for better UX
"""
import sys
from pathlib import Path
import tempfile
import streamlit as st
import pandas as pd
from datetime import datetime
import traceback

# Add project root to path (same as main.py)
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
from config import config
from app.storage.vector_store import vector_store
from app.retrieval.retriever import HybridCVRetriever
from app.ingest.extractor import extract_structured_data
from app.ingest.indexer import index_cv
from app.jd_parser import parse_job_description
from app.insights.generator import generate_insights
from app.email.sender import get_sender
from app.email.tracker import EmailTracker

# Page config
st.set_page_config(
    page_title="CV AI Recruiter",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "retriever" not in st.session_state:
    with st.spinner("Loading retriever..."):
        cross_encoder = None
        try:
            from sentence_transformers import CrossEncoder
            cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)
        except Exception as e:
            st.warning(f"Cross-encoder not available: {e}")
        st.session_state.retriever = HybridCVRetriever(
            collection=vector_store.collection,
            embedder=vector_store.embedder,
            bm25_index_path=config.BM25_INDEX_PATH,
            cross_encoder=cross_encoder,
        )
        st.session_state.retriever.build_bm25_index()
if "shortlist" not in st.session_state:
    st.session_state.shortlist = []  # list of candidate IDs
if "selected_candidate" not in st.session_state:
    st.session_state.selected_candidate = None
if "email_tracker" not in st.session_state:
    st.session_state.email_tracker = EmailTracker(config.EMAIL_TRACKER_DB)
if "sender" not in st.session_state:
    st.session_state.sender = get_sender(config.CREDENTIALS_PATH, config.TOKEN_PATH)

# Sidebar navigation
st.sidebar.title("CV AI Recruiter")
page = st.sidebar.radio(
    "Navigate",
    ["Upload", "Search", "Shortlist", "About"],
)

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def parse_priority_skills(priority_str: str) -> dict:
    """Convert comma-separated skill:weight pairs to dict."""
    result = {}
    for part in priority_str.split(','):
        part = part.strip()
        if ':' in part:
            skill, weight = part.split(':', 1)
            try:
                weight = float(weight)
            except ValueError:
                weight = 1.0
        else:
            skill = part
            weight = 1.0
        result[skill.strip().lower()] = weight
    return result

def candidate_to_dict(candidate_data: dict) -> dict:
    """Convert Chroma metadata to a flat dict for display."""
    meta = candidate_data.get("metadata", {})
    return {
        "id": candidate_data.get("id", ""),
        "name": meta.get("name", "Unknown"),
        "email": meta.get("email", ""),
        "phone": meta.get("phone", ""),
        "skills": meta.get("skills", ""),
        "years_experience": meta.get("years_experience", 0),
        "education_level": meta.get("education_level", ""),
        "seniority": meta.get("seniority", ""),
        "score": candidate_data.get("score", 0.0),
        "fit_breakdown": candidate_data.get("fit_breakdown", {}),
        "missing_skills": candidate_data.get("missing_skills", []),
    }

# ----------------------------------------------------------------------
# Upload Page
# ----------------------------------------------------------------------
if page == "Upload":
    st.title("📤 Upload CVs")
    st.markdown("Upload one or more PDF files to index them.")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Ingest Selected"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            try:
                # Extract and index
                data = extract_structured_data(tmp_path)
                # Use filename stem as base_id (remove extension)
                base_id = Path(uploaded_file.name).stem
                index_cv(base_id, data)
                st.success(f"✅ Indexed {uploaded_file.name}")
            except Exception as e:
                st.error(f"❌ Failed to index {uploaded_file.name}: {e}")
                st.exception(e)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            progress_bar.progress((i + 1) / len(uploaded_files))
        status_text.text("Ingestion complete!")
        st.session_state.retriever.build_bm25_index(force_rebuild=True)

# ----------------------------------------------------------------------
# Search Page
# ----------------------------------------------------------------------
elif page == "Search":
    st.title("🔍 Search Candidates")

    # Search form
    with st.form("search_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Search query", value="python developer")
        with col2:
            limit = st.number_input("Max results", min_value=1, max_value=50, value=10)

        with st.expander("Advanced Filters"):
            col_a, col_b = st.columns(2)
            with col_a:
                min_exp = st.number_input("Min experience", min_value=0, value=0, step=1)
                filter_skills = st.text_input("Must-have skills (comma-separated)", help="Candidates must have all these skills")
                filter_edu = st.selectbox("Education", ["", "bachelor", "master", "phd"])
            with col_b:
                filter_seniority = st.selectbox("Seniority", ["", "junior", "mid", "senior", "lead", "principal"])
                boolean_query = st.text_input("Boolean expression", help="e.g., python AND (docker OR kubernetes)")
                priority_skills = st.text_input("Skill priority (skill:weight)", help="e.g., python:2.0,docker:1.5")

        submitted = st.form_submit_button("Search")

    if submitted:
        # Parse filters
        filter_skills_set = None
        if filter_skills:
            filter_skills_set = {s.strip() for s in filter_skills.split(',') if s.strip()}
        skill_priority = None
        if priority_skills:
            skill_priority = parse_priority_skills(priority_skills)

        with st.spinner("Searching..."):
            try:
                results = st.session_state.retriever.hybrid_search(
                    query=query,
                    top_k=limit,
                    alpha=0.5,
                    min_experience=min_exp if min_exp > 0 else None,
                    filter_skills=filter_skills_set,
                    filter_edu=filter_edu if filter_edu else None,
                    filter_seniority=filter_seniority if filter_seniority else None,
                    boolean_query=boolean_query if boolean_query else None,
                    skill_priority=skill_priority,
                    use_rerank=True,
                    expand_query=True,
                )
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.exception(e)
                results = []

        if not results:
            st.warning("No candidates found.")
        else:
            st.success(f"Found {len(results)} candidates.")

            # Prepare dataframe for display
            data = []
            for r in results:
                meta = r["metadata"]
                data.append({
                    "ID": r["id"][:8] + "...",
                    "Name": meta.get("name", "N/A"),
                    "Score": round(r["score"], 3),
                    "Experience": meta.get("years_experience", 0),
                    "Skills": meta.get("skills", "")[:50],
                    "Education": meta.get("education_level", ""),
                    "Seniority": meta.get("seniority", ""),
                })
            df = pd.DataFrame(data)

            # Display with clickable rows
            st.dataframe(
                df,
                column_config={
                    "ID": st.column_config.TextColumn("ID"),
                    "Name": st.column_config.TextColumn("Name"),
                    "Score": st.column_config.NumberColumn("Score", format="%.3f"),
                    "Experience": st.column_config.NumberColumn("Exp"),
                    "Skills": st.column_config.TextColumn("Skills"),
                    "Education": st.column_config.TextColumn("Edu"),
                    "Seniority": st.column_config.TextColumn("Seniority"),
                },
                hide_index=True,
                use_container_width=True,
            )

            # Let user select a candidate to view details
            selected_id = st.selectbox(
                "Select candidate to view details",
                options=[r["id"] for r in results],
                format_func=lambda x: next((r["metadata"]["name"] for r in results if r["id"] == x), x),
            )
            if selected_id:
                selected = next(r for r in results if r["id"] == selected_id)
                st.session_state.selected_candidate = selected
                st.rerun()

    # Display selected candidate details if any
    if st.session_state.selected_candidate:
        cand = st.session_state.selected_candidate
        meta = cand["metadata"]
        st.divider()
        st.subheader(f"📄 {meta.get('name', 'Candidate')}")

        # Navigation buttons
        col_back, col_search, _ = st.columns([1, 1, 4])
        with col_back:
            if st.button("← Back to Search"):
                st.session_state.selected_candidate = None
                st.rerun()
        with col_search:
            if st.button("New Search"):
                st.session_state.selected_candidate = None
                # Optionally clear search form? Not needed.

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Email:** {meta.get('email', 'N/A')}")
            st.markdown(f"**Phone:** {meta.get('phone', 'N/A')}")
            st.markdown(f"**Experience:** {meta.get('years_experience', 0)} years")
            st.markdown(f"**Education:** {meta.get('education_level', 'N/A')}")
            st.markdown(f"**Seniority:** {meta.get('seniority', 'N/A')}")
            skills = meta.get('skills', '').split(',')
            st.markdown("**Skills:** " + ", ".join(skills[:10]) + ("..." if len(skills) > 10 else ""))
        with col2:
            st.markdown("**Fit Breakdown:**")
            breakdown = cand.get("fit_breakdown", {})
            for k, v in breakdown.items():
                st.markdown(f"- {k}: {v:.3f}")
            missing = cand.get("missing_skills", [])
            if missing:
                st.markdown("**Missing Skills:** " + ", ".join(missing))

        # AI Insights
        if st.button("Generate AI Insights"):
            with st.spinner("Generating insights..."):
                try:
                    insights = generate_insights(meta, job_description=None)
                except Exception as e:
                    st.error(f"❌ Failed to generate insights: {e}")
                    st.exception(e)
                    insights = {"summary": "Error", "strengths": [], "weaknesses": [], "interview_questions": []}
            if insights["summary"] != "Error":
                st.markdown("**Summary:**")
                st.write(insights["summary"])
                if insights["strengths"]:
                    st.markdown("**Strengths:**")
                    for s in insights["strengths"]:
                        st.write(f"- {s}")
                if insights["weaknesses"]:
                    st.markdown("**Weaknesses:**")
                    for w in insights["weaknesses"]:
                        st.write(f"- {w}")
                if insights["interview_questions"]:
                    st.markdown("**Interview Questions:**")
                    for q in insights["interview_questions"]:
                        st.write(f"- {q}")

        # Shortlist and Email actions
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if cand["id"] not in st.session_state.shortlist:
                if st.button("➕ Add to Shortlist"):
                    st.session_state.shortlist.append(cand["id"])
                    st.rerun()
            else:
                st.info("Already in shortlist")
        with col_b:
            # Check cooldown before showing email button
            tracker = st.session_state.email_tracker
            email_cooldown = config.EMAIL_COOLDOWN_DAYS
            recently_emailed = tracker.was_emailed_recently(meta.get('email', ''), email_cooldown)
            if recently_emailed:
                st.warning(f"Emailed within last {email_cooldown} days")
                force_send = st.checkbox("Send anyway (override cooldown)")
            else:
                force_send = False

            if st.button("📧 Email Candidate"):
                # Simple email form (subject and template)
                with st.form("email_form"):
                    subject = st.text_input("Subject", value="Job Opportunity")
                    template_variant = st.selectbox("Template", ["default", "followup"])
                    send = st.form_submit_button("Send")
                    if send:
                        try:
                            sender = st.session_state.sender
                            if not config.CREDENTIALS_PATH.exists():
                                st.error(f"❌ Credentials file not found: {config.CREDENTIALS_PATH}")
                            else:
                                sender.authenticate()
                                template_path = config.EMAIL_TEMPLATE_PATH
                                if template_variant != "default":
                                    template_path = template_path.with_stem(f"email_template_{template_variant}")
                                if not template_path.exists():
                                    st.error(f"❌ Template not found: {template_path}")
                                else:
                                    body = template_path.read_text(encoding="utf-8").replace("{name}", meta.get('name', ''))
                                    msg_id = sender.send_message(meta.get('email', ''), subject, body)
                                    if msg_id:
                                        # Log regardless of cooldown if we're sending (force_send overrides)
                                        st.session_state.email_tracker.log_sent(
                                            candidate_id=cand["id"],
                                            candidate_email=meta.get('email', ''),
                                            candidate_name=meta.get('name', ''),
                                            subject=subject,
                                            template_variant=template_variant,
                                            message_id=msg_id,
                                        )
                                        st.success("✅ Email sent!")
                                    else:
                                        st.error("❌ Failed to send email (no message ID).")
                        except Exception as e:
                            st.error(f"❌ Email error: {e}")
                            st.exception(e)
        with col_c:
            if st.button("❌ Clear Selection"):
                st.session_state.selected_candidate = None
                st.rerun()

# ----------------------------------------------------------------------
# Shortlist Page
# ----------------------------------------------------------------------
elif page == "Shortlist":
    st.title("📋 Shortlist")

    if not st.session_state.shortlist:
        st.info("Your shortlist is empty. Add candidates from the Search page.")
    else:
        # Retrieve full candidate data for shortlisted IDs
        shortlist_candidates = []
        for cid in st.session_state.shortlist:
            try:
                doc = st.session_state.retriever.collection.get(ids=[cid], include=["metadatas", "documents"])
                if doc["ids"]:
                    shortlist_candidates.append({
                        "id": cid,
                        "metadata": doc["metadatas"][0],
                        "document": doc["documents"][0] if doc["documents"] else "",
                    })
                else:
                    st.warning(f"Candidate {cid} not found, removing from shortlist.")
                    st.session_state.shortlist.remove(cid)
            except Exception as e:
                st.error(f"Error fetching candidate {cid}: {e}")

        # Display shortlist
        for i, cand in enumerate(shortlist_candidates):
            meta = cand["metadata"]
            with st.expander(f"{meta.get('name', 'Unknown')} - {meta.get('email', '')}"):
                st.markdown(f"**Experience:** {meta.get('years_experience', 0)} years")
                st.markdown(f"**Skills:** {meta.get('skills', '')}")
                if st.button(f"Remove from shortlist", key=f"remove_{i}"):
                    st.session_state.shortlist.remove(cand["id"])
                    st.rerun()

        # Bulk actions
        if shortlist_candidates:
            st.divider()
            st.subheader("Bulk Actions")
            if st.button("📧 Email All Shortlisted"):
                # Show email form
                with st.form("bulk_email"):
                    subject = st.text_input("Subject", value="Job Opportunity")
                    template_variant = st.selectbox("Template", ["default", "followup"])
                    send = st.form_submit_button("Send Emails")
                    if send:
                        try:
                            sender = st.session_state.sender
                            if not config.CREDENTIALS_PATH.exists():
                                st.error(f"❌ Credentials file not found: {config.CREDENTIALS_PATH}")
                            else:
                                sender.authenticate()
                                template_path = config.EMAIL_TEMPLATE_PATH
                                if template_variant != "default":
                                    template_path = template_path.with_stem(f"email_template_{template_variant}")
                                if not template_path.exists():
                                    st.error(f"❌ Template not found: {template_path}")
                                else:
                                    template_text = template_path.read_text(encoding="utf-8")
                                    success = 0
                                    for cand in shortlist_candidates:
                                        meta = cand["metadata"]
                                        if not meta.get('email'):
                                            st.warning(f"Skipping {meta.get('name')} – no email address.")
                                            continue
                                        body = template_text.replace("{name}", meta.get('name', ''))
                                        msg_id = sender.send_message(meta.get('email', ''), subject, body)
                                        if msg_id:
                                            st.session_state.email_tracker.log_sent(
                                                candidate_id=cand["id"],
                                                candidate_email=meta.get('email', ''),
                                                candidate_name=meta.get('name', ''),
                                                subject=subject,
                                                template_variant=template_variant,
                                                message_id=msg_id,
                                            )
                                            success += 1
                                    st.success(f"✅ Sent {success} emails.")
                        except Exception as e:
                            st.error(f"❌ Bulk email error: {e}")
                            st.exception(e)

# ----------------------------------------------------------------------
# About Page
# ----------------------------------------------------------------------
elif page == "About":
    st.title("ℹ️ About CV AI Recruiter")
    st.markdown("""
    This system uses AI to help you manage and rank CVs.

    **Features:**
    - Upload CVs (PDF) – automatic extraction of structured data.
    - Hybrid search (vector + BM25) with advanced filters.
    - Job description matching.
    - AI‑generated insights (summary, strengths, weaknesses, interview questions).
    - Shortlist management.
    - Email integration (Gmail) with cooldown and override.
    - Built with ChromaDB, Sentence‑Transformers, Gemini, and Streamlit.
    """)
    stats = st.session_state.email_tracker.get_stats()
    st.metric("Emails Sent", stats["sent"])
    st.metric("Scheduled Emails", stats["scheduled"])