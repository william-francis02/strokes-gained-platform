# Golf Strokes Gained Project — 10-Day Interview-Ready Sprint

**Assumption:** ~10 working days, 3-4 hrs/day = ~30-40 hrs total. If the interview date confirms sooner, cut Days 9-10 first, not Days 1-4 — the foundation matters more than the polish.

**Ground rule:** Claude Code can scaffold boilerplate (Docker, test stubs, FastAPI routing skeletons). It does NOT write your model logic, your stats formulas, or your architecture decisions. Every line in those areas, you write or you rewrite by hand until you understand it. If you can't explain a design choice unprompted, it doesn't go in the demo.

**What "tangible" actually means here:** not a finished platform — a working, deployed-enough slice you can screen-share and defend under questioning, plus a credible roadmap for the rest. Interviewers respond well to "here's what's live, here's what's next and why I sequenced it that way." That's a stronger signal than a broken full stack.

---

## Day 1-2 — Data Lockdown + Stats Foundation
- Lock your data source (synthetic generation is fine and fast if you're honest about it being synthetic — don't pretend it's real ShotLink data)
- Build expected-strokes-to-holeout logic by distance/lie
- Validate against known PGA benchmarks
- **You must be able to explain the math from memory, no notes**

**Checkpoint:** Working stats module, sanity-checked outputs.

---

## Day 3-5 — ML Model + MLFlow
- Train 2-3 model types on strokes gained prediction
- Log every run in MLFlow from the first attempt
- Pick a final model, be ready to justify it against the alternatives using actual logged metrics, not gut feel

**Checkpoint:** MLFlow experiment history with a real comparison story. This is your best "let me show you my process, not just my result" moment in the interview.

---

## Day 6-7 — FastAPI Serving
- Wrap the model in a `/predict` endpoint
- Add input validation — an interviewer poking at your API live with a garbage input and watching it crash is a bad look
- Test via curl/Postman, not just Streamlit

**Checkpoint:** API runs locally, handles bad input gracefully, you can explain the request/response schema without looking it up.

---

## Day 8 — Docker (stretch, don't skip Day 1-7 quality to force this in)
- Containerize the API
- Confirm it runs clean from the container, not just your local Python env

**Checkpoint:** `docker run` works. If you're short on time, this is the first thing to cut — a working local API beats a broken containerized one.

---

## Day 9 — Streamlit Dashboard (minimal, not decorative)
- One page: input a shot scenario, show the predicted strokes gained, maybe a chart of trend data
- Do not spend more than a day here — it's the least technically impressive layer and the easiest trap to over-polish because it's visually satisfying

**Checkpoint:** Something you can screen-share in 60 seconds that looks intentional, not flashy.

---

## Day 10 — Interview Prep, Not Coding
- Write a one-page README: what's built, what's next (Sprint 6-8 from the original plan: K8s, CI/CD, Azure, GenAI agent), and why you sequenced it this way
- Rehearse explaining: your model choice, one bug you hit and fixed, one thing you'd do differently with more time
- Prepare the "what's next" pitch — this is where you sell the roadmap to Option 4 without having built it yet. Say it as intent with a plan, not as a vague someday

**Checkpoint:** You can talk through the whole thing for 5 minutes without a script.

---

## What You Say In The Interview
Be straight about scope, don't inflate it:

> "I built a strokes-gained prediction model trained on [data source], tracked experiments in MLFlow, and served it through a FastAPI endpoint with a Streamlit front end. It's containerized in Docker. The next phase is Kubernetes orchestration, a CI/CD pipeline, Azure deployment, and a GenAI agent layer that uses the model as a tool — I scoped those out but prioritized getting a working, defensible core first given the timeline."

That last sentence does real work: it shows judgment, not just output. Anyone can list tools they touched. Showing you deliberately sequenced scope under a deadline is a stronger signal of how you'll behave on an actual team.

## The Failure Mode To Avoid
Don't let Claude Code produce something that *looks* finished (Docker + K8s + a slick agent) that you can't actually walk through. An interviewer who senses you don't understand your own project will dig until it's obvious — and "I had AI write this part" as an answer to a follow-up question is a worse outcome than "I didn't get to that part yet." Depth on less beats breadth on more, every time, in an interview setting.
