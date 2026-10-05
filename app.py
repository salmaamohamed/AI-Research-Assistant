"""
Streamlit UI for the AI Research Assistant workflow defined in main.py.

Run with:
    streamlit run app.py
"""

import os
import sys

import streamlit as st

# main.py logs with non-ASCII characters; keep that from crashing on a cp1252 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from main import MAX_RETRIES, build_graph

st.set_page_config(page_title="AI Research Assistant", page_icon="🔎")
st.title("AI Research Assistant")


@st.cache_resource
def get_graph():
    return build_graph()


def render_step(node: str, update: dict):
    """Show what one node of the graph did."""
    if node == "planner":
        st.markdown("#### 1. Planner")
        st.markdown("Broke the question into research tasks:")
        for i, task in enumerate(update.get("tasks", []), 1):
            st.markdown(f"{i}. {task}")

    elif node == "researcher":
        st.markdown("#### Researcher")
        findings = update.get("findings", [])
        if not findings:
            st.markdown("_No search results were found on this pass._")
        for f in findings:
            st.markdown(f"**Task:** {f['task']}")
            st.markdown(f"**Search query:** `{f['query']}`")
            st.markdown(f["summary"])
            st.caption(
                "Sources: " + " · ".join(f"[{s['title']}]({s['url']})" for s in f["sources"])
            )

    elif node == "critic":
        st.markdown("#### Critic")
        st.markdown(f"**Verdict:** {update.get('critic_verdict', '')}")
        if update.get("critic_feedback"):
            st.markdown(f"**Feedback:** {update['critic_feedback']}")

    elif node == "decision":
        st.markdown("#### Decision")
        if update.get("next_step") == "researcher":
            st.markdown(
                f"Insufficient → back to the researcher "
                f"(retry {update.get('retries', 0)}/{MAX_RETRIES})"
            )
        else:
            st.markdown("Moving on → generating the report")

    elif node == "generator":
        st.markdown("#### Generator")
        st.markdown("Wrote the final report from the collected findings.")

    st.divider()


question = st.text_area(
    "Research question",
    placeholder="e.g. What are the main approaches to building multi-agent LLM systems?",
    height=120,
)
run = st.button("Generate report", type="primary")

report_box = st.container()

if run:
    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("Missing OPENROUTER_API_KEY. Copy .env.example to .env and add your key.")
        st.stop()
    if not question.strip():
        st.warning("Please enter a research question.")
        st.stop()

    steps = []
    report = ""
    flow = st.expander("Reasoning and flow", expanded=False)
    try:
        with st.spinner("Researching..."):
            for chunk in get_graph().stream({"question": question.strip()}, stream_mode="updates"):
                for node, update in chunk.items():
                    update = update or {}
                    steps.append((node, update))
                    with flow:
                        render_step(node, update)
                    report = update.get("report", report)
    except Exception as e:
        st.error(f"The workflow failed: {e}")
        st.stop()

    st.session_state["steps"] = steps
    st.session_state["report"] = report
    with report_box:
        st.markdown(report)

elif "report" in st.session_state:
    with report_box:
        st.markdown(st.session_state["report"])
    with st.expander("Reasoning and flow", expanded=False):
        for node, update in st.session_state["steps"]:
            render_step(node, update)
