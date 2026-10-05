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

STEP_TITLES = {
    "planner": "Planner",
    "researcher": "Researcher",
    "critic": "Critic",
    "decision": "Decision",
    "generator": "Report generator",
}

st.set_page_config(page_title="AI Research Assistant", page_icon="🔎", layout="centered")


@st.cache_resource
def get_graph():
    return build_graph()


def render_step(number: int, node: str, update: dict):
    """Show what one node of the graph did, as its own card."""
    with st.container(border=True):
        st.markdown(f"**Step {number} · {STEP_TITLES.get(node, node)}**")

        if node == "planner":
            st.caption("Broke the question into research tasks")
            for i, task in enumerate(update.get("tasks", []), 1):
                st.markdown(f"{i}. {task}")

        elif node == "researcher":
            findings = update.get("findings", [])
            st.caption("Searched the web and extracted findings for each task")
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
            st.caption("Judged whether the findings are enough to answer the question")
            st.markdown(f"**Verdict:** {update.get('critic_verdict', '')}")
            if update.get("critic_feedback"):
                st.markdown(f"**Feedback:** {update['critic_feedback']}")

        elif node == "decision":
            st.caption("Chose the next step in the flow")
            if update.get("next_step") == "researcher":
                st.markdown(
                    f"Insufficient → back to the researcher "
                    f"(retry {update.get('retries', 0)}/{MAX_RETRIES})"
                )
            else:
                st.markdown("Moving on → generating the report")

        elif node == "generator":
            st.caption("Wrote the final report from the collected findings")


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("AI Research Assistant")
st.caption("Ask a research question. A team of agents plans, searches, reviews and writes the report.")

# --------------------------------------------------------------------------
# Section 1: research question
# --------------------------------------------------------------------------
with st.container(border=True):
    st.subheader("1. Research question")
    question = st.text_area(
        "Research question",
        placeholder="e.g. What are the main approaches to building multi-agent LLM systems?",
        height=120,
        label_visibility="collapsed",
    )
    run = st.button("Generate report", type="primary", use_container_width=True)

    error = ""
    if run and not os.getenv("OPENROUTER_API_KEY"):
        error = "Missing OPENROUTER_API_KEY. Copy .env.example to .env and add your key."
    elif run and not question.strip():
        error = "Please enter a research question."
    if error:
        st.warning(error)

# --------------------------------------------------------------------------
# Section 2: report
# --------------------------------------------------------------------------
with st.container(border=True):
    st.subheader("2. Report")
    report_box = st.container()

# --------------------------------------------------------------------------
# Section 3: reasoning and flow (closed until the user opens it)
# --------------------------------------------------------------------------
with st.container(border=True):
    st.subheader("3. Reasoning and flow")
    flow = st.expander("Show how the agents reached this report", expanded=False)

# --------------------------------------------------------------------------
# Fill the sections: run the workflow, or show the last result
# --------------------------------------------------------------------------
if run and not error:
    st.session_state.pop("report", None)
    st.session_state.pop("steps", None)
    steps = []
    report = ""
    try:
        with report_box, st.spinner("Researching..."):
            for chunk in get_graph().stream({"question": question.strip()}, stream_mode="updates"):
                for node, update in chunk.items():
                    update = update or {}
                    steps.append((node, update))
                    with flow:
                        render_step(len(steps), node, update)
                    report = update.get("report", report)
    except Exception as e:
        report_box.error(f"The workflow failed: {e}")
    else:
        st.session_state["report"] = report
        report_box.markdown(report)
    st.session_state["steps"] = steps

else:
    if "report" in st.session_state:
        report_box.markdown(st.session_state["report"])
    else:
        report_box.caption("Your report will appear here.")

    steps = st.session_state.get("steps", [])
    with flow:
        if not steps:
            st.caption("The agents' steps will appear here once you generate a report.")
        for i, (node, update) in enumerate(steps, 1):
            render_step(i, node, update)
