# YC Startup Landscape Analysis
### Trends, Domains, and the Impact of the AI Era

This analysis explores the current startup landscape within **Y Combinator**, focusing on:
- Emerging trends
- High-performing domains
- How the AI era has reshaped startup formation and success

I conducted two primary analyses:
1. **Trend prediction based on startup descriptions**
2. **Changes observed since the AI era (post-2022)**

---

## I. Predicting Startup Trends

### Methodology
To identify trends, I used **TF-IDF Vectorization** on startup descriptions to generate clusters based on semantic similarity.
After extracting representative keywords for each cluster, I **manually labeled the domains**.

The objective was to understand **which startup domains scale better**, using observed success rates as a proxy for performance.

> ⚠️ Note: Startup success rates are inherently biased. Many startups choose to exit early, sell, or get acquired before reaching large scale.

---

### Key Findings

#### 🔝 Top-Performing Domains
The three domains with the highest observed success rates are:

1. **AI Platforms & Automation**
2. **General Software / Infrastructure**
3. **B2B SaaS & Management Tools**

- **Success rate range:** `0.189 – 0.248`
- These domains consistently demonstrate stronger scalability and long-term viability.

#### 📉 Consumer & Marketplaces
- This is the **largest cluster by number of companies**
- Dominated by **early-stage startups**
- **Low scaling success**, despite high participation

This suggests that while consumer startups are popular to build, they face significantly greater challenges in reaching meaningful scale.

---

## II. The AI Era (Post-2022)

### YC Batch Dynamics
- YC has become **more selective**
- Overall batch sizes are **smaller**
- Capital and attention are increasingly concentrated in fewer, more focused bets

---

### Shifting Domain Trends

#### 📈 Rapid Growth
- **AI Agents**
- **AI Voice Applications**
- **AI Ops & Automation**

These categories show sharp increases, especially in recent batches.

#### 📉 Declining Areas
- Marketplaces
- SMB tools

#### ➖ Stable Domains
- AI Platforms & Infrastructure
  These remain consistently strong year over year.

---

### Core Insight
**AI agents and AI voice startups are currently dominating the race.**

This reflects a deeper shift:
> Software is no longer just a tool — it is becoming a *worker*.

AI-native startups are not merely improving workflows; they are **replacing or augmenting human labor directly**, which fundamentally changes how software creates value.

---

## Final Takeaway

The YC startup landscape is increasingly defined by:
- **AI-native companies**
- **Automation over enablement**
- **Scalability through intelligence, not headcount**

The winners are building systems that *act*, not just *assist*.

# An App that validate startup idea
I also make an app that validate startups based on their descriptions

https://yc-startup-validator.streamlit.app/

---

# Phase 2 — Production ML Serving

The analysis didn't stop at insights. The model got deployed.

Phase 2 wraps the classifier in a full production ML system — REST API, experiment tracking, drift monitoring, and an automated retraining pipeline. It's the difference between a notebook result and something that keeps running.

→ [See Phase 2 README](phase2/README.md)

**What's running:**
- `POST /predict` — score any startup description in real time
- MLflow — track and compare every training run
- Drift dashboard — watch prediction distributions shift over time
- Retraining pipeline — fetch new data, retrain, promote if metrics improve

---

# Project Wrap-up

## What This Project Built

This is a two-phase ML system — analysis first, then production serving.

**Phase 1** was pure analysis: scrape YC companies, cluster them by description using TF-IDF, label domains by hand, and measure which ones actually scale. The output was a ranked view of what types of startups succeed — and what's noise.

**Phase 2** turned that model into a running system. A FastAPI service scores startup descriptions in real time, MLflow tracks every training run, a SQLite database logs every prediction, and a Streamlit dashboard watches for drift. A retraining pipeline ties it together — fetch new data, retrain, compare metrics, promote if better.

## What ML System This Represents

This is a classic **supervised text classification pipeline** with a full MLOps loop:

- **Model**: Logistic Regression on TF-IDF features
- **Training**: Tracked and versioned with MLflow
- **Serving**: REST API with real-time inference
- **Observability**: Prediction logging + drift detection
- **Lifecycle**: Automated retraining with model promotion

In industry terms, this is the core pattern behind any production NLP classifier — spam filters, content moderation, lead scoring, support ticket routing. The model is simple, but the infrastructure around it is what makes it production-grade.

## Takeaways

1. **The model is the easy part.** TF-IDF + LogisticRegression trains in seconds and performs well. The hard work is everything around it — serving, monitoring, retraining.

2. **AI Platforms and B2B SaaS dominate YC.** Success rates of 19–25% vs single digits for consumer startups. The data is clear: build for businesses, not consumers.

3. **AI agents are eating the post-2022 batch.** This isn't a trend — it's a structural shift. Software is becoming a worker, not just a tool.

4. **MLOps is infrastructure, not magic.** Protobuf version conflicts, port collisions, three services that need to coordinate — none of that goes away with better tooling. You still have to understand what's running and why.

The winners in this dataset build systems that act. This project does the same.
