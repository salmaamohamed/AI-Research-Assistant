"""
AI Research Assistant - a small demo of an Agentic Workflow / Architecture.

Flow:
    planner -> researcher -> critic -> decision
                   ^                      |
                   |__ insufficient ______|   (max 2 retries)
                                          |
                                     sufficient -> generator -> END

Components:
    - One LLM        : meta-llama/llama-3.3-70b-instruct via OpenRouter
    - One tool       : web search (DuckDuckGo, no API key needed)
    - LangGraph State: carries all information between nodes
"""

import json
import os
import re
import sys
from operator import add
from typing import Annotated, TypedDict

from ddgs import DDGS
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

MAX_RETRIES = 3

# --------------------------------------------------------------------------
# LLM (one model for every agent)
# --------------------------------------------------------------------------
llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    temperature=0.2,
)


def ask(system: str, user: str) -> str:
    """Single LLM call helper."""
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return response.content.strip()


def parse_json(text: str):
    """Extract the first JSON object/array from an LLM response."""
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def log(node: str, message: str = ""):
    print(f"\n{'=' * 60}\n▶ NODE: {node.upper()}\n{'=' * 60}")
    if message:
        print(message)


# --------------------------------------------------------------------------
# Tool: web search
# --------------------------------------------------------------------------
@tool
def web_search(query: str) -> list[dict]:
    """Search the web and return a list of {title, url, snippet}."""
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as e:  # network / rate-limit problems should not crash the graph
        print(f"   [web_search] error: {e}")
        return []
    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in results
    ]


# --------------------------------------------------------------------------
# LangGraph State
# --------------------------------------------------------------------------
class ResearchState(TypedDict, total=False):
    question: str
    tasks: list[str]
    findings: Annotated[list[dict], add]  # appended to on every research pass
    critic_verdict: str  # "sufficient" | "insufficient"
    critic_feedback: str
    retries: int
    next_step: str
    report: str


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
def planner(state: ResearchState) -> dict:
    """Planner Agent: break the question into 2-4 research tasks."""
    log("planner", f"Question: {state['question']}")

    text = ask(
        "You are a research planner. Break the user's research question into 2 to 4 "
        "simple, distinct research tasks. Respond ONLY with a JSON array of strings.",
        state["question"],
    )
    tasks = parse_json(text)
    if not isinstance(tasks, list) or not tasks:
        tasks = [state["question"]]  # fallback: research the question directly
    tasks = [str(t) for t in tasks][:4]

    print("Research tasks:")
    for i, t in enumerate(tasks, 1):
        print(f"  {i}. {t}")
    return {"tasks": tasks, "retries": 0}


def researcher(state: ResearchState) -> dict:
    """Research Agent: the LLM decides what to search, uses the tool, extracts findings."""
    retries = state.get("retries", 0)
    log("researcher", f"Attempt {retries + 1} of {MAX_RETRIES + 1}")

    previous_queries = [f["query"] for f in state.get("findings", [])]
    feedback = state.get("critic_feedback", "")
    new_findings = []

    for task in state["tasks"]:
        # 1) LLM decides the search query (and adapts it if the critic complained)
        prompt = f"Research task: {task}\n"
        if feedback:
            prompt += f"\nA reviewer said the earlier research was insufficient: {feedback}\n"
        if previous_queries:
            prompt += f"Queries already used (do NOT repeat): {previous_queries}\n"
        prompt += "\nWrite ONE specific web search query. Respond with the query only."
        query = ask("You write effective web search queries.", prompt).strip('"\' \n')
        previous_queries.append(query)
        print(f"\n  Task : {task}\n  Query: {query}")

        # 2) Tool call
        results = web_search.invoke({"query": query})
        print(f"  Found {len(results)} results")
        if not results:
            continue

        # 3) LLM extracts the important findings from the raw results
        raw = "\n\n".join(
            f"[{i}] {r['title']}\n{r['snippet']}" for i, r in enumerate(results, 1)
        )
        summary = ask(
            "You are a research analyst. From the search results, extract the key findings "
            "relevant to the task as 3-5 concise bullet points. Use only the given results.",
            f"Task: {task}\n\nSearch results:\n{raw}",
        )
        new_findings.append(
            {
                "task": task,
                "query": query,
                "summary": summary,
                "sources": [{"title": r["title"], "url": r["url"]} for r in results],
            }
        )

    return {"findings": new_findings}


def critic(state: ResearchState) -> dict:
    """Critic Agent: is the collected information sufficient and relevant?"""
    log("critic")

    findings_text = "\n\n".join(
        f"Task: {f['task']}\n{f['summary']}" for f in state.get("findings", [])
    ) or "(no findings)"

    text = ask(
        "You are a strict research critic. Decide whether the findings are sufficient and "
        "relevant to answer the research question, including enough to compare the main "
        "approaches. Respond ONLY with JSON: "
        '{"verdict": "sufficient" or "insufficient", "feedback": "what is missing (short)"}',
        f"Research question: {state['question']}\n\nFindings:\n{findings_text}",
    )
    data = parse_json(text) or {}
    verdict = str(data.get("verdict", "")).lower()
    verdict = "insufficient" if "insufficient" in verdict else "sufficient"
    feedback = data.get("feedback", "")

    print(f"Verdict : {verdict}")
    if feedback:
        print(f"Feedback: {feedback}")
    return {"critic_verdict": verdict, "critic_feedback": feedback}


def decision(state: ResearchState) -> dict:
    """Decision node: loop back to the researcher or move on to the generator."""
    log("decision")
    retries = state.get("retries", 0)

    if state["critic_verdict"] == "insufficient" and retries < MAX_RETRIES:
        print(f"Insufficient -> back to researcher (retry {retries + 1}/{MAX_RETRIES})")
        return {"next_step": "researcher", "retries": retries + 1}

    if state["critic_verdict"] == "insufficient":
        print("Still insufficient, but max retries reached -> generating report anyway")
    else:
        print("Sufficient -> generating report")
    return {"next_step": "generator"}


def generator(state: ResearchState) -> dict:
    """Report Generator: write the final report."""
    log("generator")

    # Number the unique sources so the report can cite them as [n]
    source_ids: dict[str, int] = {}
    source_list: list[dict] = []
    findings_text = []
    for f in state["findings"]:
        ids = []
        for s in f["sources"]:
            if s["url"] and s["url"] not in source_ids:
                source_ids[s["url"]] = len(source_ids) + 1
                source_list.append(s)
            if s["url"]:
                ids.append(source_ids[s["url"]])
        findings_text.append(f"Task: {f['task']}\nSources: {ids}\n{f['summary']}")

    body = ask(
        "You are a research report writer. Write a concise report in Markdown with exactly "
        "these sections: '## Introduction', '## Main Findings', '## Comparison' (use a "
        "table if useful), '## Conclusion'. Use only the provided findings, cite sources "
        "inline like [1], and do NOT write a Sources section.",
        f"Research question: {state['question']}\n\nFindings:\n\n" + "\n\n".join(findings_text),
    )

    sources_md = "\n".join(
        f"{i}. [{s['title']}]({s['url']})" for i, s in enumerate(source_list, 1)
    )
    report = f"# Research Report\n\n**Question:** {state['question']}\n\n{body}\n\n## Sources\n\n{sources_md}\n"
    return {"report": report}


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner)
    graph.add_node("researcher", researcher)
    graph.add_node("critic", critic)
    graph.add_node("decision", decision)
    graph.add_node("generator", generator)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "critic")
    graph.add_edge("critic", "decision")
    graph.add_conditional_edges(
        "decision",
        lambda state: state["next_step"],
        {"researcher": "researcher", "generator": "generator"},
    )
    graph.add_edge("generator", END)

    return graph.compile()


def main():
    if not os.getenv("OPENROUTER_API_KEY"):
        sys.exit("Missing OPENROUTER_API_KEY. Copy .env.example to .env and add your key.")

    question = " ".join(sys.argv[1:]).strip() or input("Enter your research question: ").strip()
    if not question:
        sys.exit("No question provided.")

    app = build_graph()
    final_state = app.invoke({"question": question})

    print(f"\n{'#' * 60}\nFINAL REPORT\n{'#' * 60}\n")
    print(final_state["report"])

    with open("report.md", "w", encoding="utf-8") as f:
        f.write(final_state["report"])
    print("Report saved to report.md")


if __name__ == "__main__":
    main()
