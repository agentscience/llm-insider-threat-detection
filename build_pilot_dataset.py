# -*- coding: utf-8 -*-
"""
Build a small insider-threat pilot dataset from CERT email.csv + psychometric.csv.

- Objective: Select as many "highly suspicious" (1) samples as possible, whilst retaining
             a portion of "clearly normal" (0) samples.
- Additional requirement: Mandate the inclusion of samples from the "Top 10 most active users"
                          (a number of both positive and negative samples).
- Method: Heuristic rules + Isolation Forest unsupervised anomaly scores + user-level
          psychometric risk weighting.
- Output: small_insider_pilot.csv (containing a 'label' column: 1=suspicious, 0=normal).
"""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def build_dataset():
    """
    Main function to execute the dataset building pipeline.
    """
    # ========= Configurable Parameters (modify as required) =========
    EMAIL_CSV  = "email.csv"
    PSYCHO_CSV = "psychometric.csv"
    OUTPUT_CSV = "small_insider_pilot.csv"

    # Target sampling size (the script will attempt to meet this, but the actual count may vary slightly).
    TARGET_POS = 200      # Target number of suspicious (1) samples in the small dataset.
    TARGET_NEG = 200      # Target number of normal (0) samples in the small dataset.

    # Top 10 user mandatory inclusion policy.
    TOPK = 10
    TOP_POS_PER_USER = 6  # Number of suspicious samples to select from each top user.
    TOP_NEG_PER_USER = 4  # Number of normal samples to select from each top user.

    # Unsupervised anomaly proportion assumption (for Isolation Forest).
    IFOREST_CONTAM = 0.01

    # Rule thresholds (quantiles).
    Q_SIZE  = 0.95        # High quantile threshold for email size.
    Q_RECIP = 0.95        # High quantile threshold for the number of recipients.

    # Definition of "non-working hours".
    OFF_HOURS = (0, 7, 18, 23)  # 00:00-07:00 & 18:00-23:00
    WEEKEND = {5, 6} # Saturday, Sunday

    # Mapping of psychometric fields (used if present).
    PSYCHO_POS_RISK_HIGHER = {"neuroticism", "risk", "impulsiveness"}      # High value -> High risk.
    PSYCHO_POS_RISK_LOWER  = {"conscientiousness", "agreeableness"}      # Low value -> High risk (negative z-score).

    # ========= 1) Load Data =========
    print("Loading data...")
    email  = pd.read_csv(EMAIL_CSV)
    psycho = pd.read_csv(PSYCHO_CSV)

    def has(df, col): return col in df.columns

    # Basic cleaning.
    print("Preprocessing emails...")
    if has(email, "date"):
        email["date"] = pd.to_datetime(email["date"], errors="coerce")
    else:
        raise ValueError("The 'email.csv' file is missing the 'date' column.")

    for c in ["cc", "bcc", "to", "from", "content", "pc", "user", "id"]:
        if has(email, c):
            email[c] = email[c].fillna("").astype(str)
        else:
            # If a column is missing, create an empty one for robustness.
            email[c] = "" 

    if not has(email, "size"):        email["size"] = 0
    if not has(email, "attachments"): email["attachments"] = 0

    # ========= 2) Tally Top 10 Users (by activity) =========
    user_counts = email["user"].value_counts()
    top_users = list(user_counts.head(TOPK).index)
    print(f"Top {TOPK} users by email count:", top_users)

    # ========= 3) Feature Engineering (email level) =========
    print("Performing feature engineering...")

    def count_semicolon_list(s: str) -> int:
        s = s.strip()
        if not s:
            return 0
        return s.count(";") + 1

    email["num_to"]  = email["to"].apply(count_semicolon_list)
    email["num_cc"]  = email["cc"].apply(count_semicolon_list)
    email["num_bcc"] = email["bcc"].apply(count_semicolon_list)
    email["num_recipients"] = email["num_to"] + email["num_cc"] + email["num_bcc"]

    email["hour"] = email["date"].dt.hour
    email["day_of_week"] = email["date"].dt.dayofweek

    email["is_offhour"] = email["hour"].apply(lambda h: int((h>=OFF_HOURS[0] and h<=OFF_HOURS[1]) or (h>=OFF_HOURS[2] and h<=OFF_HOURS[3])))
    email["is_weekend"] = email["day_of_week"].isin(WEEKEND).astype(int)
    email["has_attach"] = (email["attachments"] > 0).astype(int)
    email["has_bcc"]    = (email["num_bcc"] > 0).astype(int)

    size_thr  = email["size"].quantile(Q_SIZE)
    recip_thr = email["num_recipients"].quantile(Q_RECIP)
    email["is_big_size"]   = (email["size"] >= size_thr).astype(int)
    email["is_many_recip"] = (email["num_recipients"] >= recip_thr).astype(int)

    SENSITIVE_TERMS = [
        "confidential","secret","client list","password","credential","intellectual property",
        "finance","payroll","source code","design doc","leak","external","nda","contract"
    ]
    pattern = re.compile("|".join([re.escape(t) for t in SENSITIVE_TERMS]), flags=re.IGNORECASE)
    email["has_sensitive_kw"] = email["content"].apply(lambda x: int(bool(pattern.search(x))))

    # ========= 4) Unsupervised Anomaly Score (Isolation Forest) =========
    print("Scoring with Isolation Forest...")
    feat_cols = ["size", "attachments", "num_recipients", "hour", "day_of_week"]
    X = email[feat_cols].copy().fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iforest = IsolationForest(
        n_estimators=200,
        contamination=IFOREST_CONTAM,
        random_state=42,
        n_jobs=-1
    )
    iforest.fit(X_scaled)
    email["iforest_pred"]  = iforest.predict(X_scaled)           # 1 for normal / -1 for anomaly.
    email["iforest_score"] = -iforest.score_samples(X_scaled)    # Higher score -> More anomalous.

    # ========= 5) Merge Psychometric Data (user level) =========
    print("Merging psychometric data...")
    # Robustly find the key for merging.
    join_key = "user"
    if join_key not in psycho.columns:
        candidates = [c for c in psycho.columns if c.lower() in {"user","userid","employee","employee_id","empid"}]
        if candidates:
            join_key = candidates[0]
        else:
            # If no key is found, do not merge psychometric features.
            print("Warning: Could not find a valid user key in psychometric.csv. Skipping this feature.")
            psycho = pd.DataFrame() 
    
    if not psycho.empty:
        email = email.merge(psycho, how="left", left_on="user", right_on=join_key, suffixes=("", "_psy"))

    psy_cols = set(c.lower() for c in email.columns)
    risk_parts = []
    
    for c in PSYCHO_POS_RISK_HIGHER:
        if c in psy_cols:
            col = [cc for cc in email.columns if cc.lower()==c][0]
            z = (email[col] - email[col].mean()) / (email[col].std() + 1e-6)
            risk_parts.append(z.fillna(0).clip(-3,3))

    for c in PSYCHO_POS_RISK_LOWER:
        if c in psy_cols:
            col = [cc for cc in email.columns if cc.lower()==c][0]
            z = (email[col] - email[col].mean()) / (email[col].std() + 1e-6)
            risk_parts.append((-z).fillna(0).clip(-3,3))
    
    email["user_psy_risk"] = np.vstack(risk_parts).mean(axis=0) if risk_parts else 0.0


    # ========= 6) Combine to Create Total Risk Score =========
    print("Combining risk scores...")
    rule_score = (
        1.0 * email["is_offhour"] +
        0.6 * email["is_weekend"] +
        1.0 * email["has_attach"] +
        0.8 * email["has_bcc"] +
        1.2 * email["is_big_size"] +
        1.0 * email["is_many_recip"] +
        1.2 * email["has_sensitive_kw"]
    )

    rule_z = (rule_score - rule_score.mean()) / (rule_score.std() + 1e-6)
    if_z   = (email["iforest_score"] - email["iforest_score"].mean()) / (email["iforest_score"].std() + 1e-6)

    email["total_risk"] = (1.2 * rule_z + 1.0 * if_z + 0.8 * email["user_psy_risk"]).astype(float)

    # ========= 7) Sample Selection: Fulfil Top 10 user coverage first, then top up the total =========
    print("Selecting samples, ensuring Top 10 user coverage...")
    picked_idx = set()
    pos_list, neg_list = [], []

    def pick_from_df(df, need, exclude_idx):
        """Select the top 'need' items, skipping already selected indices."""
        out = []
        for idx, row in df.iterrows():
            if idx in exclude_idx: 
                continue
            out.append(idx)
            if len(out) >= need:
                break
        return out

    # 7.1) First, select samples for each top user.
    for u in top_users:
        sub = email[email["user"]==u]
        if sub.empty: 
            continue

        sub_pos = sub.sort_values("total_risk", ascending=False)
        pos_idx = pick_from_df(sub_pos, TOP_POS_PER_USER, picked_idx)
        pos_list.extend(pos_idx)
        picked_idx.update(pos_idx)

        sub_neg = sub.sort_values("total_risk", ascending=True)
        neg_idx = pick_from_df(sub_neg, TOP_NEG_PER_USER, picked_idx)
        neg_list.extend(neg_idx)
        picked_idx.update(neg_idx)

    print(f"Samples selected from top users: positives={len(pos_list)}, negatives={len(neg_list)}")

    # 7.2) Globally top up the remaining quota.
    remaining_pos = max(0, TARGET_POS - len(pos_list))
    remaining_neg = max(0, TARGET_NEG - len(neg_list))

    global_pos_pool = email.sort_values("total_risk", ascending=False)
    more_pos_idx = pick_from_df(global_pos_pool, remaining_pos, picked_idx)
    pos_list.extend(more_pos_idx)
    picked_idx.update(more_pos_idx)

    global_neg_pool = email.sort_values("total_risk", ascending=True)
    more_neg_idx = pick_from_df(global_neg_pool, remaining_neg, picked_idx)
    neg_list.extend(more_neg_idx)

    print(f"Final selection count: positives={len(pos_list)} / target {TARGET_POS}, negatives={len(neg_list)} / target {TARGET_NEG}")

    # ========= 8) Assemble and Save =========
    pos_pick = email.loc[pos_list].copy()
    pos_pick["label"] = 1
    neg_pick = email.loc[neg_list].copy()
    neg_pick["label"] = 0

    small = pd.concat([pos_pick, neg_pick], axis=0).sample(frac=1.0, random_state=42)

    export_cols = [
        "id","date","user","pc","from","to","cc","bcc","content",
        "size","attachments","num_to","num_cc","num_bcc","num_recipients",
        "hour","day_of_week","is_offhour","is_weekend","has_attach","has_bcc",
        "is_big_size","is_many_recip","has_sensitive_kw",
        "iforest_score","user_psy_risk","total_risk","label"
    ]
    export_cols = [c for c in export_cols if c in small.columns]
    small = small[export_cols].reset_index(drop=True)

    small.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved: {OUTPUT_CSV} | Shape: {small.shape}")
    print("Label counts:\n", small["label"].value_counts(dropna=False))
    print("Coverage of top users in the small dataset:")
    print(small["user"].value_counts().loc[[u for u in top_users if u in small['user'].values]])
    print("\nTop 5 suspicious examples:")
    print(small[small["label"]==1].head(5)[["date","user","to","attachments","size","num_recipients","total_risk"]])


if __name__ == "__main__":
    build_dataset()
