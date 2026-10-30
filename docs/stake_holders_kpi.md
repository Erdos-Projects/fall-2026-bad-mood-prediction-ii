# Problem Statement, Stakeholders, and KPIs

## Problem We Are Trying to Solve

- **Challenge:** Daily mood can fluctuate, and wearable measurements alone may not capture these changes reliably. Can information available on a given day help estimate how someone may feel the following day?
- **Research question:** Can we predict the **probability of seven next-day moods** (ALERT, HAPPY, NEUTRAL, RESTED/RELAXED, SAD, TENSE/ANXIOUS, and TIRED) using wearable activity, heart-rate and sleep measures, demographics, and current-day self-reported moods?
- **Approach:** Train seven independent binary classifiers because multiple mood labels can occur together. Evaluate predictions on later observations from participants represented in training, rather than claiming generalization to new participants.
- **Potential value:** Explore whether routinely collected data could support future mood-monitoring and well-being research; predictions are **not clinical assessments or diagnoses**.

## Stakeholders

- **Researchers in digital health and behavioral science:** Can study mood persistence and the added predictive value of wearable and self-reported features.
- **Wearable and wellness application developers:** Can assess whether next-day mood estimates are informative enough to justify further product research and validation.
- **Healthcare and mental-health professionals (potential downstream stakeholders):** May be interested in validated research findings, but these models are **not suitable for clinical decisions** without substantial additional study.

## Key Performance Indicators (KPIs)

- **Primary — Discrimination (ROC-AUC):** How well does each classifier rank positive next-day mood observations above negative ones? **Higher is better**; 0.5 represents chance-level ranking.
- **Primary — Probability quality (Brier score):** How accurate are the predicted probabilities? **Lower is better**. 
- **Secondary — Binary classification (F1-score):** How well do precision and recall balance at **mood-specific thresholds selected using development cross-validation**? Higher is better; F1 does not maximize precision and recall separately.
- **Generalization check:** Report all KPIs on the **held-out, participant-wise chronological test set** (461 observations), without retuning on test results.