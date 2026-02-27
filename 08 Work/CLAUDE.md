Lokam = multi-tenant call intelligence platform (voice AI + analytics).

  

## Stack

- Backend: FastAPI (Py 3.10+), PostgreSQL, SQLAlchemy, Alembic

- Frontend: React + Vite + TS

- Voice/AI: VAPI, OpenAI

- SMS: Twilio (A2P, consent-gated)

- Infra: AWS, Docker

  

## Architecture (Essentials)

- Event-driven workers: call initiation, pre-call SMS, reports

- Strict multi-tenant isolation: Organization → Rooftops → Data

- Call flow: ServiceRecord → Schedule → SMS Consent → Call → Transcript → AI Insights

- ACS: server/app/acs_enterprise/

- Core folders and files: server/app/models, server/app/services, server/app/core/config.py, server/app/api/v1/endpoints/

  

## Hard Rules

- **SMS consent required** for every outbound call (`granted` only)

- Block calls if `revoked` or `is_dnc = true`

- Always respect rooftop schedules + timezones

- Enforce tenant isolation in all queries

- Use async/await consistently

  

## Dev Rules

- Activate conda activate lokam(/home/zero/Packages/miniconda3/envs/lokam) before backend commands

- Never skip Alembic migrations after model changes

- Verify changes applied before committing

- Commit after major changes

- Main dev branch: `playground`

  

## Feature Work Rules

  

- Feature_Decision_Log.md is the source of truth for intent and trade-offs

- Execution_Tracker.md is the source of truth for progress state

  

Rules:

- Never change these files unless explicitly instructed

- When a design decision is made, summarize it and ask the user to update the log

- Do not re-architect or refactor past decisions without approval

- Continue work from the current IN PROGRESS section only