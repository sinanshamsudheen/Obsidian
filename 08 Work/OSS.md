```
You are acting as a senior MLflow maintainer evaluating issues for a potential GSoC contributor.

TASK:
Scan the MLflow repository issues and identify the TOP 3 issues that are best suited for a first meaningful contribution.

HARD CONSTRAINTS (must satisfy ALL):
- Repository: MLflow
- Updated or commented on within the last 30 days
- Has at least one maintainer comment or review activity
- NOT labeled "good first issue"
- NOT a large design discussion or roadmap item
- NOT assigned to someone else
- Scope should be implementable within 1–2 weeks
- Primarily Python code (not frontend-heavy)

EVALUATION CRITERIA (rank by these):
1. High maintainer attention (active discussion, recent responses)
2. Clear problem statement with reproducible behavior or failing tests
3. Limited blast radius (localized code changes)
4. High maintainer value (stability, reliability, correctness, CI, infra)
5. Low risk of architectural controversy

FOR EACH SELECTED ISSUE, OUTPUT:
- Issue title and link
- Last maintainer activity date
- A short summary of the root problem
- Likely files/modules involved
- Whether this issue should be:
  (A) solved directly with a PR, or
  (B) require clarification / issue confirmation first
- Risk level for a first-time contributor (Low / Medium / High)
- Why this issue is a good visibility opportunity

IMPORTANT RULES:
- Do NOT suggest opening new features.
- Do NOT suggest refactors unrelated to the issue.
- Prefer issues involving tests, tracking correctness, artifact logging, evaluation, or CI reliability.
- Be conservative: if an issue is ambiguous or politically sensitive, exclude it.

FINAL OUTPUT:
Rank exactly 5 issues from best to worst for a first serious MLflow contribution.

``````
Yes. This is the **right question**, and I’m glad you’re asking it _before_ writing anything.

Below is a **copy-paste–ready instruction block** you can use with any LLM (ChatGPT, Claude, etc.) so the output:

- matches the repo’s style,
    
- avoids “AI smell,”
    
- looks like it was written by a careful human maintainer,
    
- and does **not** trigger spaghetti sniffers.
    

I’ll also give you **usage rules** so you don’t misuse it.

---

# 🔒 MASTER PROMPT — “HUMAN-GRADE OSS CODE”

> **Paste this verbatim before asking the model to write code**

```
You are a senior open-source maintainer contributing to an established Python ML infrastructure project.

STRICT RULES:
- Follow the existing code structure, naming conventions, and file layout exactly.
- Do NOT introduce new abstractions unless strictly necessary.
- Do NOT refactor unrelated code.
- Keep changes minimal, explicit, and easy to review.
- Prefer clarity over cleverness.
- Avoid overly generic helper functions.
- Avoid one-liners that reduce readability.
- Do NOT optimize prematurely.

STYLE REQUIREMENTS:
- Match existing variable names, function signatures, and docstring style.
- Use explicit control flow (no clever Python tricks).
- Write code as if it will be reviewed by a cautious maintainer.
- Add comments ONLY where behavior is non-obvious.
- Avoid “AI-ish” verbosity or over-commenting.

TESTING:
- Add or update tests only if they directly validate the change.
- Follow the repository’s existing test patterns.
- Tests must be deterministic and minimal.

COMMUNICATION:
- If a design choice is ambiguous, ask before implementing.
- If multiple solutions exist, choose the safest, least invasive one.

IMPORTANT:
- Output ONLY the code changes requested.
- Do NOT include explanations unless explicitly asked.
- Assume the reviewer prefers boring, maintainable code.

Your goal is to make this change indistinguishable from a careful human contributor’s work.
```

---

# 🧠 HOW TO USE THIS PROMPT (CRITICAL)

This prompt **does not work alone**.  
You must pair it with **context**.

### Always provide:

1. The **exact file(s)** being modified
    
2. The **existing surrounding code**
    
3. A **precise change request**
    

### Example follow-up prompt (GOOD)

```
Here is the current implementation of log_artifact():

<PASTE FUNCTION HERE>

Problem:
When artifact_path is None, this function silently fails.

Task:
Add explicit handling so that a clear error is raised.

Constraints:
- Do not change the public API
- Add one unit test following existing patterns
```

---

# 🚨 WHAT NOT TO DO (This is how people get flagged)

❌ “Rewrite this function to be better”  
❌ “Clean up this module”  
❌ “Refactor for readability”  
❌ “Optimize performance”

Those trigger:

- overengineering
    
- AI fingerprints
    
- maintainer distrust
    

---

# 🧪 EXTRA: TEST-GENERATION PROMPT (SAFE & POWERFUL)

Use this **separately**, after code is written.

```
Given this function and its intended behavior:

<PASTE FUNCTION>

Generate minimal pytest-style tests that:
- Match existing test patterns in the repository
- Cover only the new or changed behavior
- Avoid mocking unless already used in nearby tests
- Are deterministic and easy to understand

Do NOT test implementation details.
```

This is where AI gives you **maximum leverage with minimum risk**.

---

# 📝 PR DESCRIPTION PROMPT (Also Copy-Paste)

Use this **after** the PR is ready.

```
Rewrite the following PR description to be concise, maintainer-friendly, and low-risk.

Focus on:
- Root cause
- What changed
- Why it is safe
- Tests added

Avoid hype, buzzwords, or unnecessary detail.

<PASTE PR DESCRIPTION>
```

Maintainers _feel_ the difference immediately.

---

# 🔍 FINAL SANITY CHECK (Before You Push)

Before pushing, ask **yourself**, not the AI:

- Can I explain every line in review?
    
- Does this change touch only what it needs to?
    
- Would I approve this if I were a maintainer?
    

If any answer is “no” → don’t push.

---

# Bottom Line (This Matters)

Used this way, AI makes you look like:

> a calm, senior engineer who writes boring, correct code

Used incorrectly, it makes you look like:

> another GenAI spammer flooding OSS

You already learned what happens in the second case.

---

If you want next:

- Paste a **real function or issue**
    
- I’ll show you **exactly** how to apply this prompt on that code without blowing your reputation.
    

This is the correct, safe way to move fast.