# Golf Strokes Gained Project — 10-Day Interview-Ready Sprint

**Assumption:** ~10 working days, 3-4 hrs/day = ~30-40 hrs total. If the interview date confirms sooner, cut Days 9-10 first, not Days 1-4 — the foundation matters more than the polish.

**Ground rule:** Claude Code can scaffold boilerplate (Docker, test stubs, FastAPI routing skeletons). It does NOT write your model logic, your stats formulas, or your architecture decisions. Every line in those areas, you write or you rewrite by hand until you understand it. If you can't explain a design choice unprompted, it doesn't go in the demo.

**What "tangible" actually means here:** not a finished platform — a working, deployed-enough slice you can screen-share and defend under questioning, plus a credible roadmap for the rest. Interviewers respond well to "here's what's live, here's what's next and why I sequenced it that way." That's a stronger signal than a broken full stack.

---

## Scope Update (Day 2)

Original plan assumed shot-level strokes-gained prediction. Data investigation (see `data/README.md`) confirmed no shot-level or round-level dataset was available — the only real observational data found was tournament-level (player × tournament, `sg_putt`/`sg_arg`/`sg_app`/`sg_ott`/`sg_total`). `putt_baseline.csv` and `shot_baseline.csv` are Broadie's published expected-strokes reference curves, not shot logs — kept as a separate demo module, not the training data. `OWGR_Ranking.csv` was evaluated and rejected — single-week snapshot, no time-series coverage, can't feature a multi-year model.

**Revised scope:** predict tournament outcome (finish position / made_cut) from a player's category-level strokes gained profile. Same MLOps stack, different target variable. This is a real, defensible ML problem — it directly answers "does short game or ball-striking separate winners from the field more?"

This pivot is your strongest interview talking point in the whole project. Don't bury it — lead with it when asked "walk me through your project."

---

## Day 1-2 — Data Lockdown + Baseline Reference Module
- ~~Lock your data source~~ **DONE:** ASA All PGA Raw Data (2015-2022, 36,864 rows, player × tournament grain), confirmed via read-only inspection
- Build a small, separate module using `putt_baseline.csv` / `shot_baseline.csv` — compute expected-strokes-to-holeout by distance/lie, validate against known benchmarks. This demonstrates you understand the underlying strokes-gained methodology, even though it's not what feeds the main model
- Write `data/README.md` documenting: source, grain, missing-data policy for the 20.84% withdrawn/DQ'd rows (drop from the finish-position regression; exclude — not lump into missed-cut — for the made_cut classifier), and why OWGR was evaluated and rejected
- **You must be able to explain the pivot and the baseline math from memory, no notes**

**Checkpoint:** Working baseline module + documented, cleaned tournament-level dataset with an explicit missing-data policy.

---

## Day 3-5 — ML Model + MLFlow
- Target variable: finish position (regression) or made_cut (classification) — pick one as primary, the other as stretch
- Features: `sg_putt`, `sg_arg`, `sg_app`, `sg_ott` (and derived combinations if useful)
- Train 2-3 model types, log every run in MLFlow from the first attempt
- Pick a final model, be ready to justify it against alternatives using logged metrics, not gut feel
- Bonus talking point: which SG category has the strongest relationship with winning? That's an actual insight, not just a working pipeline

**Checkpoint:** MLFlow experiment history with a real comparison story, plus one sentence you can say confidently about which stat matters most.

---

## Day 6-7 — FastAPI Serving
- Wrap the model in a `/predict` endpoint (input: a player's `sg_putt`/`sg_arg`/`sg_app`/`sg_ott`, output: predicted finish position or cut probability)
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
- One page: input a player's SG category stats, show predicted finish/cut probability, maybe a chart comparing SG categories across historical winners
- Do not spend more than a day here — it's the least technically impressive layer and the easiest trap to over-polish because it's visually satisfying

**Checkpoint:** Something you can screen-share in 60 seconds that looks intentional, not flashy.

---

## Day 10 — Interview Prep, Not Coding
- Write a one-page README: what's built, the data pivot and why, what's next (Sprint 6-8 from the original plan: K8s, CI/CD, Azure, GenAI agent), and why you sequenced it this way
- Rehearse explaining: the data pivot, your model choice, one bug you hit and fixed, one thing you'd do differently with more time
- Prepare the "what's next" pitch — this is where you sell the roadmap to Option 4 without having built it yet. Say it as intent with a plan, not as a vague someday

**Checkpoint:** You can talk through the whole thing for 5 minutes without a script — including the pivot, without it sounding like an excuse.

---

## What You Say In The Interview

> "I built a model predicting PGA Tour finish position from a player's category-level strokes gained stats, trained on 2015-2022 tournament data. I originally scoped this as shot-level prediction, but when I investigated the available data I found no public shot-level dataset existed at the volume I needed, so I pivoted to tournament-level outcome prediction — same MLOps stack, adjusted target based on what the data actually supported. I tracked experiments in MLFlow, served it through FastAPI, containerized it in Docker. Next phase is Kubernetes, CI/CD, Azure, and a GenAI agent layer."

That pivot sentence in the middle is now your strongest line in the whole pitch — it's the one part of this story an interviewer can't get from any other bootcamp grad's identical golf-score-predictor project.

## The Failure Mode To Avoid
Don't let Claude Code produce something that *looks* finished (Docker + K8s + a slick agent) that you can't actually walk through. An interviewer who senses you don't understand your own project will dig until it's obvious — and "I had AI write this part" as an answer to a follow-up question is a worse outcome than "I didn't get to that part yet." Depth on less beats breadth on more, every time, in an interview setting.
