- [ ] Refactor  
- [ ] Feature  
- [x] Improvement  
- [x] Bug Fix  
- [ ] Optimization  
- [x] Documentation Update  

## Description

- Backend: Adjusted VAPI webhook handling in `server/app/services/webhook_service.py`
  - Map end-of-call `endedReason` containing "voicemail" to call `status` = "Missed".
  - On exception during end-of-call report processing, persist call `status` = "Failed".
  - No functional changes elsewhere.

## Screenshots, Recordings

N/A (backend changes + JSON config metadata)

## QA Instructions

Prereqs:
- Ensure tenant DB is running and a `Call` exists with a known `id`.
- Use the correct `x-vapi-secret` header value.

1) Voicemail → status Missed
- POST to `/api/v1/webhooks/vapi-webhook`:
```bash
curl -X POST http://localhost:8000/api/v1/webhooks/vapi-webhook \
  -H 'Content-Type: application/json' \
  -H 'x-vapi-secret: <SECRET>' \
  -d '{
    "type": "end-of-call-report",
    "call": { "assistantOverrides": { "variableValues": { "call_id": "123" } } },
    "endedReason": "voicemail-detected",
    "artifact": { "messages": [] }
  }'
```
- Verify `calls.status` for id 123 is "Missed".

2) Exception path → status Failed
- POST malformed `messages` to trigger an exception in transcript save (string instead of list):
```bash
curl -X POST http://localhost:8000/api/v1/webhooks/vapi-webhook \
  -H 'Content-Type: application/json' \
  -H 'x-vapi-secret: <SECRET>' \
  -d '{
    "type": "end-of-call-report",
    "call": { "assistantOverrides": { "variableValues": { "call_id": "123" } } },
    "endedReason": "any",
    "artifact": { "messages": "not-a-list" }
  }'
```
- Verify `calls.status` for id 123 is "Failed".

3) Status update mapping sanity (optional)
- For `/api/v1/webhooks/vapi-webhook` with `"type":"status-update"`, verify:
  - `"status":"failed"` → "Failed"
  - `"status":"no-answer"` or `"busy"` → "Missed"

4) Prompts config
- Confirm app loads prompts with new `cost` and `total_tokens` fields without errors.

✅ Tested on N/A (HTTP via curl) on Linux

## I have tested the changes introduced in this pull request

- [ ] Yes  
- [x] No, and this is why: Unable to run full stack locally within this session; provided reproducible curl-based QA steps.

## I have reviewed the changes first to ensure small things like console.logs and debug logic have been removed

- [x] Yes

## Added/updated tests?

_It is encouraged to keep the code coverage percentage at 70% and above._

- [ ] Yes  
- [x] No, and this is why: Focused backend status mapping + JSON metadata; integration tests can be added in a follow-up if desired.