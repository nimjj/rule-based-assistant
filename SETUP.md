# Setup

How to install, configure and run the CIAP Agent Assist server (intent detection, slot filling and
live MCP tool calls for customer calls).

## Prerequisites

- **Python 3.12 or newer** (`pyproject.toml`: `requires-python = ">=3.12"`) and Git.
- **[Ollama](https://ollama.com)** running locally, with two models pulled:
  ```
  ollama pull gemma3:1b            # LLM escalation: ambiguous / multi-intent / correction turns (ciap/pipeline/llm.py)
  ollama pull gemma4:e2b-it-qat    # responder: turns a tool result into an agent-facing answer (RESPONDER_MODEL_NAME)
  ```
- Internet access on first run: `sentence-transformers` downloads the `all-MiniLM-L6-v2` embedding model
  from Hugging Face once, then caches it.
- Node.js (`npx`) only if you use a stdio MCP connection that launches through it (the `billing` entry in
  `connections.json`).

A machine without a GPU works; the responder takes roughly 15 s for a full Zoho lead record on CPU
(see `RESPONDER_TIMEOUT_SECONDS` below).

## Install

```
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt   # runtime only
pip install -r requirements-dev.txt   # runtime + pytest, httpx, pandas, openpyxl (tests and offline tools)
```

Versions in `requirements.txt` are pinned to what the test suite passes with.

## Configure

Create a `.env` file in the repository root. It is loaded at startup (`ciap/integrations/mcp_client.py`) and is
git-ignored, so secrets stay out of the repo.

### MCP connection secrets

`connections.json` is committed and refers to secrets as `${NAME}`; the values come from `.env` or the
environment. You only need values for the connections you actually bind catalog tools to.

| Connection | Variables |
| --- | --- |
| `zoho` | `ZOHO_MCP_URL` (the URL itself is a secret) |
| `hubspot` | `HUBSPOT_CLIENT_ID`, `HUBSPOT_CLIENT_SECRET` (then sign in, see below) |
| `helpdesk` | `HELPDESK_MCP_URL`, `HELPDESK_API_KEY` |
| `shop` | `SHOP_MCP_URL`, `SHOP_ACCESS_TOKEN` |
| `erp` | `ERP_MCP_URL`, `ERP_CLIENT_ID`, `ERP_CLIENT_SECRET` |
| `salesforce` | `SALESFORCE_MCP_URL`, `SF_CLIENT_ID`, `SF_CLIENT_SECRET`, `SF_REFRESH_TOKEN` |
| `legacy_crm` | `LEGACY_CRM_SSE_URL`, `LEGACY_CRM_TOKEN` |
| `billing` | `BILLING_API_KEY` |
| `internal_orders` | `ORDERS_DB_URL` |

Never put a real URL, key or token directly in `connections.json`.

### Optional tuning

All of these have defaults (`ciap/config.py`); set them only to override.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MCP_CONNECTIONS_PATH` | `connections.json` in the repo root | Connections file to use |
| `MCP_TOKENS_DIR` | `data/oauth` | Saved sign-ins for `mcp_oauth` connections |
| `MCP_CALL_TIMEOUT_SECONDS` | `5` | Longest a turn waits for a remote tool call |
| `RESPONDER_MODEL_NAME` | `gemma4:e2b-it-qat` | Ollama model that writes the agent-facing answer |
| `RESPONDER_TIMEOUT_SECONDS` | `20` | Wait for a focused answer before falling back to a plain listing |
| `RESPONDER_FULL_TIMEOUT_SECONDS` | `40` | Same, for `full` answers |
| `RESPONDER_DEFAULT_MODE` | `focused` | `focused` or `full`, when a request names none |
| `RESPONDER_KEEP_ALIVE_SECONDS` | `-1` | How long Ollama keeps the responder loaded (negative: forever) |
| `RESPONDER_MAX_SCORED_ITEMS` | `50` | List items compared by meaning to the call when pruning |
| `RESPONDER_PRUNE_RESULTS` | `true` | `false` sends tool results to the model unpruned (slower, needs a larger context) |
| `RESPONDER_NUM_CTX` | unset | Responder context window in tokens |
| `CALL_IDLE_TIMEOUT_SECONDS` | `1800` | Idle calls are dropped from memory after this long; `0` disables |
| `SLOT_OVERRIDE_ENABLED` | off | `1`/`true` enables the archived agent slot-correction endpoint |

`DEMO_UI_ENABLED` is currently hard-coded to `1` in `ciap/config.py`, so `/demo` is always served.

### Runtime folders (git-ignored, created as needed)

- `data/catalog.json`: intent/tool catalog edited through the `/v1/intents` endpoints. Delete it to go back to the defaults.
- `data/oauth/<name>.json`: saved `mcp_oauth` sign-ins. These hold tokens; keep them readable only by the server's user.
- `logs/`: contains raw customer utterances. Never commit it.

## Connect to MCP servers

Check a connection before binding catalog tools to it (secrets are never printed):

```
python scripts/check_connections.py            # every connection
python scripts/check_connections.py hubspot    # just one
```

Connections with `"auth": {"type": "mcp_oauth"}` (HubSpot) need a one-time browser sign-in:

```
python scripts/connect.py hubspot
```

The browser must run on the same machine, because the redirect comes back to `localhost`. For a remote server, sign in locally and
copy `data/oauth/hubspot.json` to the server's `data/oauth/`. Run it again whenever a connection reports that it needs signing in.

## Run

Pre-flight first (checks imports, loads the embedding model, warms Ollama, runs one intent -> slot -> tool-call turn):

```
python scripts/demo_check.py
```

Then start the server from the repository root:

```
uvicorn ciap.api.server:app
```

- API docs: http://127.0.0.1:8000/docs (OpenAPI JSON at `/openapi.json`)
- Health: http://127.0.0.1:8000/health (`degraded` if Ollama is unreachable)
- Browser demo: http://127.0.0.1:8000/demo

The server has no authentication and keeps call state in memory (see the header of `ciap/api/server.py`), so it is not production-ready as shipped.

## Tests and offline tools

```
pytest                              # tests/ (needs requirements-dev.txt)
python scripts/demo.py call         # replay scripts/samples/*.txt as simulated calls
python scripts/demo.py live         # type turns yourself
python scripts/demo.py test         # score scripts/intent_test_dataset.xlsx
```

## About this repository

- **Bitbucket (`adept-tech-us/ciap_agent_assist`) is the source of truth.** This GitHub repo is Bitbucket's `master` plus the one
  commit that adds this file (and the `!SETUP.md` exception in `.gitignore`).
- Bitbucket's `.gitignore` keeps Markdown and PDF docs local only, which is why this file does not exist there. The full API documentation
  (`API_DOCUMENTATION.md`, referenced in code comments) is also not in the repo.
- To re-sync after Bitbucket changes:
  ```
  git switch master && git pull                       # origin = Bitbucket
  git switch github-master && git rebase master
  git push --force-with-lease github github-master:master
  ```
- The pre-sync GitHub `master` is preserved as the branch `backup/github-master-e43c406`.
