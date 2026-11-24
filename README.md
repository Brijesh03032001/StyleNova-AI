<div align="center">

![StyleNova AI Banner](public/images/styleNova.png)

# StyleNova AI
### Vision-Language AI · Adaptive Recommendation · Full-Stack ML System

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-C13B1F?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenAI CLIP](https://img.shields.io/badge/OpenAI_CLIP-412991?style=for-the-badge)](https://github.com/openai/CLIP)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-A85C00?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org)

**🏆 Top 3 — Clozyt Hackathon**

**A fashion recommender that fuses OpenAI CLIP vision-language embeddings with a per-user feedback loop: a query is embedded with CLIP, blended against a preference vector learned from each user's own swipes, and served through FastAPI to a Next.js swipe interface.**

</div>

---

## Highlights

- Encoded product images and text into a shared **512-dimensional CLIP embedding space** (ViT-B/32, trained on 400M image-text pairs), so a query like *"bohemian flowy dress"* retrieves visually similar items **even when catalog tags do not match**.
- Built a **3-component hybrid score** combining TF-IDF content similarity, scikit-learn user-based **kNN** collaborative filtering and CLIP visual similarity, because single-signal recommenders overfit fashion behavior.
- Designed **3 adaptive weighting stages** for cold start — shifting from **0.6 / 0.2 / 0.2** before 5 feedback events to **0.3 / 0.5 / 0.2** after 20 events — while anchoring the CLIP visual weight at **0.2** throughout.
- Updated the preference vector after **every swipe** with exponential moving average learning at **λ = 0.3**, adapting to recent taste without collapsing into an echo chamber.
- Bootstrapped new users with a **6-step preference quiz** covering categories, colors, brands, styles, sizing and budget, before any collaborative data exists.
- Precomputed **512-d** product embeddings as SQLite BLOBs at catalog ingest, cutting online recommendation latency from roughly **2 seconds** per cold query to **under 200 ms**.
- Replaced hard blacklists with **soft exclusion** — the last **3 passes** and items disliked **2+ times** — preserving discovery diversity while suppressing genuinely unwanted items.
- Built **optimistic UI** updates in Zustand that remove the swiped card instantly while FastAPI recomputes weights asynchronously, cutting perceived interaction latency from about **450 ms** to **under 100 ms**, with rollback on failure.
- Modeled `SwipeEvent`, `PreferenceVector` and `QuizResponse` as indexed Prisma tables, keeping the recent-swipe query behind the EMA update under about **35 ms** at **1,000 events per user**.
- Integrated **GPT-4o-mini** to generate a natural-language reason per recommendation from recent swipe patterns and item attributes, adding under about **250 ms** — turning a similarity score into an explanation.
- Exposed the preference-weighting system as an **LLM function-calling tool**, so typed feedback like *"less formal, brighter colors"* calls `adjust_preference_weights` directly, cutting the swipes needed to escape a mismatched cluster from about **12** to **4**.
- Implemented velocity-based **fling detection** in Framer Motion, tuned against roughly **75 test swipes**, so a fast flick commits even under the distance threshold.

---

## Architecture

```mermaid
flowchart TB
    subgraph Ingest["Catalog ingest — offline"]
        IMG["Product images + text"] --> CLIP["CLIP ViT-B/32"]
        CLIP --> EMB[("512-d embeddings<br/>SQLite BLOBs")]
    end

    subgraph Serve["Serving — online, under 200ms"]
        Q["Query: text, image, brand or category"] --> QE["CLIP encode query"]
        QE --> HYB["Hybrid scorer"]
        EMB --> HYB
        TFIDF["TF-IDF content similarity"] --> HYB
        KNN["scikit-learn user kNN"] --> HYB
        PV[("Preference vector<br/>per user")] --> HYB
        HYB --> SOFT["Soft exclusion<br/>last 3 passes · 2+ dislikes"]
        SOFT --> RANK["Ranked recommendations"]
    end

    RANK --> API["FastAPI + Pydantic"]
    API --> UI["Next.js 15 swipe UI<br/>Zustand optimistic updates"]
    API --> EXPL["GPT-4o-mini<br/>natural-language reason"]
    UI --> TOOL["adjust_preference_weights<br/>LLM function call"]
    TOOL --> PV
```

## The feedback loop

```mermaid
sequenceDiagram
    participant U as User
    participant UI as Next.js (Zustand)
    participant A as FastAPI
    participant P as Preference vector
    participant D as Prisma / SQLite

    U->>UI: 6-step preference quiz
    UI->>A: bootstrap weights (0.6 / 0.2 / 0.2)
    loop each swipe
        U->>UI: swipe or fling
        UI-->>U: card removed instantly, under 100ms perceived
        UI->>A: POST /swipe
        A->>D: write SwipeEvent (indexed)
        A->>P: EMA update, lambda = 0.3
        P->>P: shift weights toward 0.3 / 0.5 / 0.2 after 20 events
        A-->>UI: next batch + GPT-4o-mini explanation
    end
    U->>A: "less formal, brighter colors"
    A->>P: adjust_preference_weights (function call)
    P-->>A: corrected cluster in ~4 swipes instead of ~12
```

---

## Measured results

| Metric | Value |
|---|---|
| Precision@10 | **0.73** |
| Diversity | **0.68** |
| Catalog coverage | **89%** |
| Quiz completion | **84%** |
| Swipe-through (10+ swipes) | **67%** |
| Avg session duration | **3.2 min** |
| Online recommendation latency | **< 200 ms** |
| Perceived swipe latency | **< 100 ms** |

## Adaptive weighting

| Stage | Feedback events | Content · Collaborative · Visual |
|---|---|---|
| Cold start | < 5 | **0.6 / 0.2 / 0.2** |
| Warming | 5–20 | interpolated |
| Warm | > 20 | **0.3 / 0.5 / 0.2** |

## Tech stack

**Vision-language** OpenAI CLIP ViT-B/32 · PyTorch · cosine similarity · L2 normalization
**Recommendation** scikit-learn kNN · TF-IDF · EMA preference learning · soft exclusion
**Backend** Python · FastAPI · Pydantic · NumPy · Pandas · Prisma · SQLite
**Frontend** Next.js 15 · TypeScript · Zustand · Framer Motion · Zod
**LLM** GPT-4o-mini explanations · function calling

## Quick start

```bash
git clone https://github.com/Brijesh03032001/StyleNova-AI.git
cd StyleNova-AI
```

```bash
pip install -r requirements.txt
python scripts/precompute_embeddings.py      # CLIP → SQLite BLOBs
uvicorn api.main:app --port 8000
```

```bash
npm install && npm run dev                   # http://localhost:3000
```

---

**Built by [Brijesh Kumar](https://github.com/Brijesh03032001)** · [Portfolio](https://brijeshbuilds.com) · [LinkedIn](https://linkedin.com/in/brijeshkumar03)
