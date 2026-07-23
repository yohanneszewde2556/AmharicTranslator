"""
Extracts useful Amharic-English parallel translation pairs from data/CRISPR-Global/
"""

import pandas as pd
import json
import os
import re

CRISPR_DIR = "data/CRISPR-Global"
geez_pattern = re.compile(r'[\u1200-\u137F]')

pairs = []

# 1. crispr_terminology_multilingual.csv
term_path = os.path.join(CRISPR_DIR, "crispr_terminology_multilingual.csv")
if os.path.exists(term_path):
    df = pd.read_csv(term_path)
    for _, row in df.iterrows():
        en = str(row.get("english", "")).strip()
        am = str(row.get("amharic", "")).strip()
        def_en = str(row.get("definition_en", "")).strip()
        
        if en and am and geez_pattern.search(am):
            pairs.append({"amharic": am, "english": en, "source": "crispr_terminology_term"})
        # Check if definition has any parallel (definition is in English)

# 2. crispr_educational_content_multilingual.json
edu_path = os.path.join(CRISPR_DIR, "crispr_educational_content_multilingual.json")
if os.path.exists(edu_path):
    with open(edu_path, "r", encoding="utf-8", errors="ignore") as f:
        edu_data = json.load(f)
    
    for item in edu_data.get("educational_content", []):
        explanations = item.get("explanations", {})
        
        en_exp = explanations.get("english", {})
        am_exp = explanations.get("amharic", {})
        
        if isinstance(en_exp, str) and isinstance(am_exp, str):
            if en_exp.strip() and am_exp.strip() and geez_pattern.search(am_exp):
                pairs.append({"amharic": am_exp.strip(), "english": en_exp.strip(), "source": "crispr_edu_explanation"})
        elif isinstance(en_exp, dict) and isinstance(am_exp, dict):
            # Short explanation
            en_short = str(en_exp.get("short_explanation", "")).strip()
            am_short = str(am_exp.get("short_explanation", "")).strip()
            if en_short and am_short and geez_pattern.search(am_short):
                pairs.append({"amharic": am_short, "english": en_short, "source": "crispr_edu_short"})

            # Detailed explanation
            en_detail = str(en_exp.get("detailed_explanation", "")).strip()
            am_detail = str(am_exp.get("detailed_explanation", "")).strip()
            if en_detail and am_detail and geez_pattern.search(am_detail):
                pairs.append({"amharic": am_detail, "english": en_detail, "source": "crispr_edu_detail"})

            # Key points list
            en_pts = en_exp.get("key_points", [])
            am_pts = am_exp.get("key_points", [])
            if isinstance(en_pts, list) and isinstance(am_pts, list):
                for p_en, p_am in zip(en_pts, am_pts):
                    p_en_str = str(p_en).strip()
                    p_am_str = str(p_am).strip()
                    if p_en_str and p_am_str and geez_pattern.search(p_am_str):
                        pairs.append({"amharic": p_am_str, "english": p_en_str, "source": "crispr_edu_keypoint"})

# 3. Sentence alignment / splitting for detailed paragraphs
aligned_pairs = []
for p in pairs:
    am_text = p["amharic"]
    en_text = p["english"]
    src = p["source"]
    
    # If the text is a multi-sentence paragraph, split into sentence units by period / Ethiopic fullstop
    if "።" in am_text and "." in en_text and len(am_text.split("።")) > 1:
        am_sents = [s.strip() for s in am_text.split("።") if s.strip()]
        en_sents = [s.strip() for s in en_text.split(".") if s.strip()]
        if len(am_sents) == len(en_sents):
            for sa, se in zip(am_sents, en_sents):
                aligned_pairs.append({"amharic": sa + "።", "english": se + ".", "source": src + "_sent"})
        else:
            aligned_pairs.append({"amharic": am_text, "english": en_text, "source": src})
    else:
        aligned_pairs.append({"amharic": am_text, "english": en_text, "source": src})

extracted_df = pd.DataFrame(aligned_pairs).drop_duplicates(subset=["amharic"]).drop_duplicates(subset=["english"])
print(f"Extracted total {len(extracted_df)} clean parallel pairs from CRISPR-Global.")
print("\nSample Extracted Pairs:")
for idx, row in extracted_df.head(10).iterrows():
    print(f"[{row['source']}]")
    print(f"  AM: {row['amharic']}")
    print(f"  EN: {row['english']}\n")

# Save clean extracted CSV
out_path = "data/v2/filtered/crispr_global_clean.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
extracted_df.to_csv(out_path, index=False)
print(f"Saved extracted parallel pairs to: {out_path}")
