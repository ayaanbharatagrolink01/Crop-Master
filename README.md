# Crop Master Agronomist (FastAPI + OpenAI gpt-4o-mini)

An AI agronomist for Indian farmers that answers from the Bharat Agrolink Crop Master
workbook (500 crops, 60 pests, 50 diseases, 100 crop–pest/disease protocols).

## Project layout

```
backend/
  app/
    main.py        FastAPI app, API routes, serves the frontend
    agent.py       system prompt + streaming tool-calling loop for gpt-4o-mini
    knowledge.py   loads the Excel file, retrieval, database tools
    config.py      settings from environment / backend/.env
  requirements.txt
  .env.example
frontend/
  index.html       chat UI (single file, served at /)
data/
  crop_master.xlsx your workbook
```

## Run locally

```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # then put your OPENAI_API_KEY in .env
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000. Interactive API docs are at http://localhost:8000/docs.

## How a question is answered

1. **Retrieve.** `knowledge.py` matches crop, pest, disease, state and season names
   (English, Hindi and local names) in the question and the farm profile, and collects
   the matching rows from Crop_Master, Crop_Pest_Map, Crop_Disease_Map, Pest_Master,
   Disease_Master, State_Crop_Map and Season_Master.
2. **Ask the model.** `agent.py` sends gpt-4o-mini the system prompt, the farm profile,
   those rows and the question (plus a photo, if attached).
3. **Look up more if needed.** The model can call up to two rounds of database tools:
   `search_crops`, `get_records` and `search_pests_diseases`.
4. **Stream the answer.** The answer streams back to the browser as Server-Sent Events.
   It ends with a `REFS:` line, which the server turns into the record tags shown under
   the answer.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Status, model name, whether an API key is set |
| GET | `/api/meta` | States, seasons, districts, crop names, counts (for the UI) |
| GET | `/api/record/{id}` | Full row for CRP / PST / DIS / CPM / CDM / CAT / SEA ids |
| GET | `/api/search?q=` | Retrieval only, no LLM (handy for testing) |
| POST | `/api/chat` | `{question, profile, history, image?}` → SSE stream |

The SSE events from `/api/chat` are:

- `status`: progress text
- `delta`: a chunk of answer text
- `reset`: discard the text received so far
- `done`: carries `refs`
- `error`: carries `code` and `text`

## Updating the data

Replace `data/crop_master.xlsx` with a new version of the workbook and restart the
server. Keep the same sheet names and column headers.

## Configuration (`backend/.env`)

- `OPENAI_API_KEY`: required.
- `OPENAI_MODEL`: defaults to `gpt-4o-mini`.
- `ALLOWED_ORIGINS`: set this to your domain in production. The default `*` allows any site.
- `MAX_TOOL_ROUNDS`, `MAX_HISTORY_TURNS`, `MAX_IMAGE_MB`: optional limits.

If you host the frontend on a different domain, add
`<script>window.CROP_MASTER_API="https://your-api"</script>` before the main script
in `index.html`.

## Before going live

- **Rate limiting.** `/api/chat` spends your OpenAI credit, so put rate limiting or
  login in front of it, for example with `slowapi` or at nginx or Cloudflare.
- **Data quality.** 362 of the 500 crop rows are flagged *Indicative*. The agronomist
  says so in its answers, but a QA pass on that data improves answers directly.
