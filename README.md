# REACT 2026 Datathon - 2nd Runners Up Solution (Team COiN Lab)

This repository contains the source code and solution overview for **Team COiN Lab**, placing as the **2nd Runners Up** (Private PR-AUC: `0.56084`) in the Temporal Fraud Detection task of the REACT 2026 Datathon, organized by the IEEE Southeast University Student Branch.

## Team Members
* **Md. Abdur Rahman**
* **Md. Aayat Hossain Mridha**

![Team COiN Lab](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F31499789%2Fd97ed70da7928e9bd73dc7a23c702e62%2FWhatsApp%20Image%202026-09-11%20at%206.29.51%20PM.jpeg?generation=1789226596444852&alt=media)

## Overview: A CPU-Only Temporal Fraud Detection System

Our goal was to build a highly competitive tabular model that reconstructs behavioral signals without suffering from target, temporal, or test-set leakage. We adopted a strict self-imposed constraint: **no stage may require a GPU**. 

The entire pipeline trains and scores in **~73 minutes on four CPU cores**.

### Key Components

1. **Causal Feature Engineering (250 Features):**
   * Constructed strictly from earlier rows using primitives like group-wise `cumcount`, `cumsum`, running extrema, and half-open trailing windows.
   * **Categories included:** Amount normality, Customer Tempo, Familiarity/Recency, Relationship Structure, and Movement plausibility.

2. **Fraud-Rate Encodings & Staleness Mitigation:**
   * Calculated expanding and exponentially decayed target encodings for entities.
   * **Staleness Probe:** We quantified the exact performance loss (`-0.0183` AP) when encodings freeze and go up to 62 days stale in the test window. To mitigate this, we included a "label-free" model in the blend that uses zero target encodings, immune to staleness.

3. **Explicit Rule Mining:**
   * A scan of highest-gain features found single thresholds isolating fraud at 60-70x the base rate. We explicitly supply these as 12 binary rules, covering 39.9% of total fraud.

4. **Modeling & Validation (Rank Blend):**
   * Two walk-forward folds without shuffling: `horizon` (matches forecast distance) and `recent` (matches recent behavior). 
   * A 4-member rank-averaged blend invariant to calibration:
     * **LightGBM** (full matrix)
     * **LightGBM** (label-free, all encodings removed)
     * **CatBoost** (depth 6)
     * **TabMLP** (384-192-96 multilayer perceptron trained on CPU via PyTorch)
   * Blend weights optimized via Coordinate Ascent on the validation selection score.

## Repository Structure

* `REACT DATATHON Final Submission.ipynb` - The complete end-to-end pipeline (Data Loading, Feature Engineering, Rule Mining, Modeling, and Inference).
* `Code Report.pdf` - Our detailed methodology paper.

## How to Run

1. **Install Requirements:**
   ```bash
   pip install pandas numpy scipy scikit-learn lightgbm catboost torch
