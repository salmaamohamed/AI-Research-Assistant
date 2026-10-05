# AI Research Assistant

An agentic research workflow built with LangGraph, LangChain, OpenRouter, and DuckDuckGo search. It plans a research question, gathers web findings, critiques the results, retries incomplete research, and generates a cited Markdown report.

## Screenshots

### Research question and report

![Research question and generated report](Screenshot%202026-10-05%20142718.png)

### Agent reasoning and research findings

![Agent reasoning and research findings](Screenshot%202026-10-05%20142811.png)

### Critic and decision steps

![Critic and decision steps](Screenshot%202026-10-05%20142829.png)

## Requirements

- Python 3.10 or newer
- An OpenRouter API key

## Setup

Create and activate the virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your-api-key
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

`OPENROUTER_MODEL` and `OPENROUTER_BASE_URL` are optional and default to the values shown above.

## Run the Streamlit app

```powershell
streamlit run app.py
```

Enter a research question in the browser and select **Generate report**.

## Run from the command line

Pass the question as an argument:

```powershell
python main.py "What are the main approaches to building multi-agent LLM systems?"
```

Or run without an argument to enter the question interactively:

```powershell
python main.py
```

The CLI prints the report and saves it as `report.md`.

## Workflow

```text
Planner -> Researcher -> Critic -> Decision
						 ^          |
						 |----------|
					retry if needed
							  |
						  Generator
```

- **Planner:** Breaks the question into focused research tasks.
- **Researcher:** Creates search queries, searches DuckDuckGo, and summarizes findings.
- **Critic:** Checks whether the findings are sufficient and relevant.
- **Decision:** Requests another research pass when necessary, up to three retries.
- **Generator:** Produces a Markdown report with inline citations and a sources list.

## Notes

- DuckDuckGo search does not require a separate API key.
- Keep `.env` and generated reports containing sensitive information out of version control.

