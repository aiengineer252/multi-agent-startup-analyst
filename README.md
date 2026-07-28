# 🚀 Multi-Agent Startup Analyst

**Pitch a startup idea. Get grilled by a team of six AI analysts. Walk away with a real investment memo and a GO / NO-GO verdict — in about two minutes.**

This project is a **multi-agent AI system**. Instead of asking one AI "is my idea good?" (and getting one polite, generic answer), it runs your idea past **five independent specialist agents in parallel** — each with its own persona, expertise, and evaluation criteria — and then a **sixth "Managing Partner" agent** reads all five reports, resolves their disagreements, and writes the final investment memo.

**Bring your own LLM.** It runs on [Claude](https://platform.claude.com/) by default (deepest analysis, plus optional live web search), but the same pipeline works with **OpenAI (GPT)**, **Groq** (blazing fast, generous free tier), or **any OpenAI-compatible endpoint** — Ollama running locally, OpenRouter, Together, Mistral, DeepSeek, and more. Switching is one flag: `--provider groq`.

---

## 📖 Table of Contents

- [Why multi-agent instead of one big prompt?](#-why-multi-agent-instead-of-one-big-prompt)
- [The agent team](#-the-agent-team)
- [How it works (architecture)](#-how-it-works-architecture)
- [Quick start — run it in 5 minutes](#-quick-start--run-it-in-5-minutes)
- [Usage & CLI options](#-usage--cli-options)
- [LLM providers — Claude, GPT, Groq, Ollama & more](#-llm-providers--claude-gpt-groq-ollama--more)
- [What the output looks like](#-what-the-output-looks-like)
- [Cost & model choices](#-cost--model-choices)
- [Project structure](#-project-structure)
- [Extending the system](#-extending-the-system)
- [Troubleshooting](#-troubleshooting)

---

## 🤔 Why multi-agent instead of one big prompt?

You *could* paste an idea into a chatbot and ask for an analysis. Three things go wrong:

1. **Averaging.** One model wearing six hats blends perspectives into mush. A dedicated "Risk Analyst" agent whose *only job* is finding failure modes digs far deeper than a paragraph titled "Risks" inside a mega-prompt.
2. **Politeness bias.** A single assistant tries to be balanced. Separate agents are *designed to argue* — the Market analyst may love the idea while the Financial analyst tears the unit economics apart. The synthesizer is then forced to surface and resolve those contradictions instead of papering over them.
3. **Depth per token.** Each specialist gets a full, focused context window for its slice of the problem. Five deep 1,000-word analyses beat one shallow 1,500-word one.

This is the same reason VC firms have separate diligence tracks (market, competitive, financial, legal, technical) before a partner writes the memo. This project just automates the whole firm.

## 👥 The agent team

| # | Agent | Emoji | What it interrogates | Its verdict |
|---|-------|:-----:|----------------------|-------------|
| 1 | **Market Research Analyst** | 📊 | Who buys this? TAM / SAM / SOM with explicit math, timing ("why now?"), painkiller vs. vitamin, acquisition channels | Market score /10 |
| 2 | **Competitive Intelligence Analyst** | ⚔️ | Direct & indirect competitors, the "do nothing" alternative, your wedge, moats, what happens when an incumbent copies you | Competition score /10 |
| 3 | **Financial & Business Model Analyst** | 💰 | Pricing, CAC / LTV / payback with benchmark assumptions and shown arithmetic, path to first $10k MRR, bootstrap vs. VC | Financial score /10 |
| 4 | **Risk & Regulatory Analyst** | 🛡️ | Top-5 ranked risk table (likelihood × impact), compliance (GDPR/DPDP, licenses), platform dependencies, explicit **kill criteria** | Risk score /10 |
| 5 | **Product & Technical Feasibility Analyst** | 🔧 | The riskiest assumption + cheapest test for it, ruthless MVP scope (IN vs. OUT), stack, effort in engineer-weeks, the genuinely hard parts | Feasibility score /10 |
| 6 | **Managing Partner (Synthesizer)** | 🎯 | Reads all five reports. Builds the scorecard, the bull case, the bear case, rules on analyst disagreements, and issues the verdict | **STRONG GO / GO WITH CHANGES / PIVOT / NO-GO** |

Every specialist is required to end with a one-line verdict and a numeric score, which the Managing Partner pulls into a scorecard — so you always get comparable, quantified output, not just prose.

## 🏗 How it works (architecture)

```
                        ┌───────────────────────────┐
                        │      Your startup idea     │
                        └─────────────┬─────────────┘
                                      │
                     ┌────────────────┼────────────────┐
                     │   STAGE 1 — runs in PARALLEL    │
                     │   (ThreadPoolExecutor, 5 threads)│
                     └────────────────┼────────────────┘
        ┌──────────┬──────────┬───────┴──┬──────────┬──────────┐
        ▼          ▼          ▼          ▼          ▼          │
   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐     │
   │📊Market│ │⚔️Compet.│ │💰Finance│ │🛡️ Risk │ │🔧Product│     │
   └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘     │
       └──────────┴──────────┼──────────┴──────────┘           │
                             ▼                                 │
                 5 independent Markdown reports                │
                             │                                 │
                     ┌───────┴────────┐   STAGE 2 — synthesis  │
                     ▼                                          
              ┌─────────────┐                                  
              │🎯 Managing   │  reads all 5 reports, resolves   
              │  Partner     │  conflicts, writes the memo      
              └──────┬──────┘                                  
                     ▼                                          
      ┌──────────────────────────────┐                         
      │  Investment memo + scorecard  │ → printed to terminal   
      │  + full appendix of reports   │ → saved to reports/*.md │
      └──────────────────────────────┘                         
```

Key implementation details:

- **Provider abstraction.** All model I/O goes through a tiny `LLMProvider` interface (`startup_analyst/providers.py`) with one method: `complete(system, user) -> str`. The orchestrator doesn't know or care which API is behind it — that's why Claude, GPT, Groq, and local models are all interchangeable.
- **One API call per agent.** Each agent = a distinct **system prompt** (its persona and standards) + a structured **task prompt**. There's no shared state between specialists, so no groupthink.
- **True parallelism.** The five specialists run concurrently via `ThreadPoolExecutor`, so a full analysis takes roughly the time of the *slowest single agent* plus the synthesis step — not 6× one call.
- **First-class Claude support.** The Anthropic provider streams every call (long reports never hit HTTP timeouts), enables adaptive extended thinking on supported models, and can use Claude's **server-side web search** tool (`--web`, capped at 3 searches per agent) — correctly resuming `pause_turn` responses that server-side tools can produce.
- **OpenAI-compatible everything else.** GPT, Groq, Ollama, OpenRouter, etc. all speak the same Chat Completions protocol, so one generic provider covers them — the code even handles the `max_tokens` → `max_completion_tokens` API drift between old and new servers automatically.
- **Graceful safety handling.** If a model declines a topic, the run doesn't crash — the section is marked and the rest of the memo still builds.

## ⚡ Quick start — run it in 5 minutes

### Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | Check with `python --version` |
| **An API key for ONE provider** | Any of these works: [Anthropic](https://platform.claude.com/) (default, best quality), [OpenAI](https://platform.openai.com/api-keys), [Groq](https://console.groq.com/keys) (**free tier — great for trying this out**), or a local Ollama install (no key needed). |

### 1. Get the code

```bash
cd "D:\ML Projects\multi-agent-startup-analyst"
```

(Or `git clone` it if you've pushed this to a repo.)

### 2. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

> If PowerShell blocks activation, run:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, then retry.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

That's just three packages: `anthropic` (the official SDK), `python-dotenv` (loads your key from `.env`), and `rich` (pretty terminal output).

### 4. Add your API key

```bash
# Windows PowerShell
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` in any editor and paste the key for whichever provider you'll use (you only need one):

```
ANTHROPIC_API_KEY=sk-ant-api03-...     # for --provider anthropic (default)
# OPENAI_API_KEY=sk-...                # for --provider openai
# GROQ_API_KEY=gsk_...                 # for --provider groq
```

> 🔒 `.env` is in `.gitignore` — your keys will never be committed.

### 5. Run your first analysis

```bash
# Try the bundled example idea (a tiffin-delivery startup) on Claude:
python main.py --file examples/sample_idea.txt

# Or pitch your own idea directly:
python main.py "An AI tool that turns long YouTube videos into Instagram Reels automatically"

# No Anthropic key? Run the same thing free on Groq:
python main.py --file examples/sample_idea.txt --provider groq
```

You'll see the six agents report in as they finish, then the final memo renders in your terminal, and the complete report (memo + all five specialist analyses) is saved to `reports/`.

## 🎛 Usage & CLI options

```
python main.py [IDEA] [options]
```

| Flag | Short | Default | What it does |
|------|:-----:|---------|--------------|
| `IDEA` (positional) | — | — | The startup idea, in quotes. |
| `--file PATH` | `-f` | — | Read the idea from a text file instead (better for long, detailed pitches). |
| `--provider NAME` | `-p` | `anthropic` | Which LLM API: `anthropic`, `openai`, `groq`, or `custom` (any OpenAI-compatible endpoint). |
| `--model NAME` | `-m` | per provider | Model to use. Defaults: anthropic → `claude-opus-4-8`, openai → `gpt-5.1`, groq → `llama-3.3-70b-versatile`, custom → *(you must pass one)*. |
| `--base-url URL` | — | — | Endpoint for `--provider custom` (e.g. `http://localhost:11434/v1` for Ollama). |
| `--api-key-env VAR` | — | per provider | Env-var name holding the key (e.g. `OPENROUTER_API_KEY`), if not the provider's default. |
| `--web` | — | off | Allow agents to run live web searches (max 3 each) for current market/competitor data. **Anthropic provider only.** Slower and adds per-search cost, but much fresher intel. |
| `--out DIR` | `-o` | `reports` | Where to save the Markdown report. |
| `--no-print` | — | off | Don't render the memo in the terminal; just save the file. |

**Examples:**

```bash
# Deepest analysis with live web data (best quality — Claude):
python main.py --file my_pitch.txt --web

# Fast + cheap sanity check on a rough idea (Claude Haiku):
python main.py "Uber for dog walking in tier-2 Indian cities" --model claude-haiku-4-5

# Same pipeline on GPT:
python main.py --file my_pitch.txt --provider openai

# Free & very fast on Groq:
python main.py "AI resume screener for recruiters" --provider groq

# Batch-style: save quietly to a custom folder
python main.py --file idea.txt --out analyses --no-print
```

## 🔌 LLM providers — Claude, GPT, Groq, Ollama & more

The pipeline is provider-agnostic: every agent call goes through a one-method `LLMProvider` interface, so you can swap the brain without touching any agent or orchestration code.

| Provider | Flag | Default model | API key env var | Notes |
|----------|------|---------------|-----------------|-------|
| **Anthropic (Claude)** | `--provider anthropic` *(default)* | `claude-opus-4-8` | `ANTHROPIC_API_KEY` | Best analysis quality. The **only** provider with `--web` live search and adaptive thinking. |
| **OpenAI (GPT)** | `--provider openai` | `gpt-5.1` | `OPENAI_API_KEY` | Any chat model works: `-m gpt-5.1`, `-m gpt-4.1-mini`, etc. |
| **Groq** | `--provider groq` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` | Extremely fast inference + a generous **free tier** — ideal for trying the project at zero cost. |
| **Custom (anything OpenAI-compatible)** | `--provider custom` | — (pass `--model`) | `CUSTOM_API_KEY` (or `--api-key-env`) | Needs `--base-url` (or `CUSTOM_BASE_URL` in `.env`). Covers Ollama, OpenRouter, Together, Mistral, DeepSeek, vLLM, LM Studio... |

**Recipes for the `custom` provider:**

```bash
# Local Ollama (no API key, fully offline):
#   .env:  CUSTOM_API_KEY=ollama        (Ollama accepts any value)
python main.py "my idea" --provider custom \
    --base-url http://localhost:11434/v1 --model llama3.1:8b

# OpenRouter (hundreds of models behind one key):
#   .env:  OPENROUTER_API_KEY=sk-or-...
python main.py "my idea" --provider custom \
    --base-url https://openrouter.ai/api/v1 \
    --api-key-env OPENROUTER_API_KEY \
    --model anthropic/claude-sonnet-5

# Together AI:
#   .env:  TOGETHER_API_KEY=...
python main.py "my idea" --provider custom \
    --base-url https://api.together.xyz/v1 \
    --api-key-env TOGETHER_API_KEY \
    --model meta-llama/Llama-3.3-70B-Instruct-Turbo
```

**Quality note:** the six agents are only as sharp as the model behind them. Big frontier models (Claude Opus/Sonnet, GPT-5.x) produce noticeably deeper memos — real unit-economics math, specific competitors, honest verdicts. Small local models will run the pipeline fine but tend to give shallower, more generic analysis. A good workflow: iterate cheap (Groq/local), decide on frontier (Claude/GPT).

**Tip:** the more context you give, the sharper the analysis. A one-liner works, but a `--file` pitch that includes your target user, monetization plan, and "why now" gets dramatically more specific feedback — the agents will engage with *your* numbers instead of inventing their own.

## 📄 What the output looks like

Each run produces `reports/<idea-slug>_<timestamp>.md` containing:

```markdown
# Startup Analysis Report
> Generated: 2026-07-28 14:32 | Model: claude-opus-4-8 | Agents: 5 + 1

## The Idea
> A mobile app called "MessMate" for Indian college students...

# 🎯 Investment Memo — Managing Partner
1. Executive Summary        ← the 4-sentence version for busy readers
2. Scorecard                ← every specialist's /10 score + weighted overall
3. The Bull Case            ← strongest honest argument FOR
4. The Bear Case            ← strongest honest argument AGAINST
5. Where the analysts disagree  ← contradictions, and the Partner's ruling
6. Recommended next 30 days ← concrete, cheap, risk-retiring actions
7. Final Verdict            ← STRONG GO / GO WITH CHANGES / PIVOT / NO-GO

# Appendix: Specialist Reports
📊 Market · ⚔️ Competition · 💰 Financials · 🛡️ Risk · 🔧 Product
(full detailed analysis from each specialist)
```

The memo also prints beautifully in your terminal via `rich`.

## 💸 Cost & model choices

Each run makes **6 API calls** (5 parallel + 1 synthesis). Rough per-run cost by model:

| Provider / model | Quality | Typical cost/run* | When to use |
|-------|---------|------------------|-------------|
| `claude-opus-4-8` *(default)* | ★★★★★ | ~$0.50–$1.50 | Real decisions. Deepest reasoning, best synthesis. |
| `claude-sonnet-5` | ★★★★☆ | ~$0.20–$0.60 | Great quality at a fraction of the cost. |
| `gpt-5.1` (openai) | ★★★★☆ | ~$0.10–$0.50 | Strong alternative if you're already on OpenAI. |
| `claude-haiku-4-5` | ★★★☆☆ | ~$0.05–$0.15 | Rapid iteration — screening many ideas cheaply. |
| Groq `llama-3.3-70b` | ★★★☆☆ | **free tier** / pennies | Zero-cost screening, near-instant runs. |
| Local Ollama | ★★☆☆☆–★★★☆☆ | $0 (your hardware) | Fully offline & private; quality depends on the model you run. |

\* Estimates assume ~2–5k input and ~1.5–4k output tokens per agent; actual cost varies with idea length and how much the agents write. `--web` adds per-search charges on top.

A sensible workflow: **screen** ideas free on Groq (or cheap on Haiku), then rerun the survivors with Opus + `--web` before committing real time or money.

## 📁 Project structure

```
multi-agent-startup-analyst/
├── main.py                     # CLI entry point (argparse + rich output)
├── startup_analyst/
│   ├── __init__.py
│   ├── agents.py               # ★ The 6 agent personas & task prompts — edit these!
│   ├── providers.py            # ★ LLM provider layer: Anthropic / OpenAI / Groq / custom
│   ├── orchestrator.py         # Parallel execution + synthesis (provider-agnostic)
│   └── report.py               # Markdown assembly + file saving
├── examples/
│   └── sample_idea.txt         # A ready-to-run example pitch
├── reports/                    # Generated analyses land here (git-ignored)
├── requirements.txt            # anthropic, openai, python-dotenv, rich
├── .env.example                # Template for your API key(s)
├── .gitignore                  # Keeps .env and reports/ out of git
└── README.md                   # You are here
```

## 🧩 Extending the system

The design goal is that **all agent behavior lives in one file** — `startup_analyst/agents.py`. No orchestration code needs to change for most customizations:

- **Add a new specialist** (e.g., a *Brand & Marketing Analyst* or *Legal Counsel*): append a dict to the `SPECIALISTS` list with a `key`, `name`, `emoji`, `system` persona, and `prompt` template. It automatically runs in parallel with the others and its report flows into the memo.
- **Change the evaluation lens:** editing a `system` prompt changes how tough, domain-specific, or region-aware (e.g., India-market-focused) that agent is.
- **Tune the verdict format:** the synthesizer's `prompt` in `SYNTHESIZER` controls the memo's sections and the verdict scale.
- **Add a new provider:** subclass `LLMProvider` in `startup_analyst/providers.py` (one method: `complete(system, user) -> str`) or, if the API is OpenAI-compatible, just add a preset dict to `PROVIDER_PRESETS` — no new code at all.
- **Swap the interface:** `StartupAnalyst.analyze()` is a clean library API — wrap it in FastAPI/Flask for a web app, or a Telegram bot, without touching agent logic:

```python
from startup_analyst.providers import create_provider
from startup_analyst.orchestrator import StartupAnalyst

provider = create_provider("anthropic", use_web_search=True)  # or "openai", "groq", ...
analyst = StartupAnalyst(provider)
results = analyst.analyze("my idea")     # list[AgentResult]
memo = results[-1].report                # final memo is always last
```

## 🔧 Troubleshooting

| Symptom | Fix |
|---|---|
| `authentication_error` / 401 | Your key is missing or wrong. Check `.env` exists (not just `.env.example`) and that you set the key matching your `--provider` (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GROQ_API_KEY`). |
| `Provider setup failed: GROQ_API_KEY is not set` (or similar) | You picked a provider but didn't add its key to `.env` — grab one from that provider's console (links in the [providers table](#-llm-providers--claude-gpt-groq-ollama--more)). |
| `Warning: ANTHROPIC_API_KEY is not set` | You skipped step 4 — copy `.env.example` to `.env` and paste your key. |
| `rate_limit_error` / 429 | You hit the provider's requests-per-minute cap (5 parallel calls). Wait ~1 minute, try a smaller model, or switch provider (`--provider groq`). |
| `overloaded_error` / 529 | The API is briefly busy. The SDK auto-retries twice; if it still fails, rerun in a minute. |
| `--web` seems ignored | Live web search only works with `--provider anthropic` — the CLI tells you and continues without it on other providers. |
| Model-not-found error on `custom` | Check the exact model name your endpoint expects (e.g. `ollama list` for Ollama) and that `--base-url` ends with `/v1`. |
| Run seems stuck for 1–3 min | Normal on Opus with `--web` — agents are thinking and searching. Groq/Haiku runs finish much faster. |
| `ModuleNotFoundError: anthropic` / `openai` | Your virtualenv isn't active. Re-run the activation command from Quick start step 2, then `pip install -r requirements.txt`. |
| PowerShell won't activate the venv | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, then retry. |
| Garbled emoji on Windows terminal | Use Windows Terminal (not legacy cmd), or `chcp 65001` first. |

---

## 📜 License

MIT — use it, fork it, ship it.

*Built with the [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) and the [OpenAI Python SDK](https://github.com/openai/openai-python) (for OpenAI/Groq/compatible endpoints). Not investment advice — it's a research tool; validate with real customers.* 🚀
