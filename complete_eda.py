"""
Complete EDA -- Business Entity Resolution Challenge
Covers all three areas (name, address+country, ground-truth/matching) in one
run, restricted to the TRAIN split via split_assignments.tsv, and produces
printed findings + saved plots + reusable TSV artifacts.

Requires: pandas, numpy, matplotlib. scikit-learn is optional (used only for
TF-IDF name-similarity; script degrades gracefully without it).

Run on a sample first if this is your first run (set SAMPLE_MODE = True
below) to sanity-check everything before the full ~9M-row run.
"""

import re
import random
import difflib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ======================================================================
# CONFIG
# ======================================================================
TRAIN_DIR = "student_resource/dataset/train"
SPLIT_ASSIGNMENTS_PATH = "/Users/jahnvee/Documents/Amazon ML/student_resource/dataset/train/split_assignments.tsv" 
SPLIT_TO_USE = "train"                 # "train" | "val" | "holdout"
SAMPLE_MODE = False                    # True = load only a slice, for a quick first run
SAMPLE_ROWS_IF_SAMPLE_MODE = 50000

OUT_DIR = Path("eda_outputs")
PLOT_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

N_TRUE_PAIRS = 3000
N_RANDOM_PAIRS = 3000
DUP_ADDRESS_TOP_N = 20
DANGEROUS_SIM_THRESHOLD = 0.6   # random pairs above this combined sim = flag as risky
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ======================================================================
# 0. LOADING + SPLIT RESTRICTION
# ======================================================================

def read_tsv(path, nrows=None):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=nrows)


def load_split_entity_ids(path, split_name):
    sa = pd.read_csv(path, sep="\t", dtype=str)
    sa = sa[sa["split"] == split_name]
    ids = {
        "S1": set(sa.loc[sa["source"] == "S1", "entity_id"]),
        "S2": set(sa.loc[sa["source"] == "S2", "entity_id"]),
        "S3": set(sa.loc[sa["source"] == "S3", "entity_id"]),
    }
    print(f"Split '{split_name}' entity counts -> "
          f"S1: {len(ids['S1'])}, S2: {len(ids['S2'])}, S3: {len(ids['S3'])}")
    return ids


def restrict_to_split(s1, s2, s3, gt, split_ids):
    s1_f = s1[s1["entity_id"].isin(split_ids["S1"])].reset_index(drop=True)
    s2_f = s2[s2["entity_id"].isin(split_ids["S2"])].reset_index(drop=True)
    s3_f = s3[s3["entity_id"].isin(split_ids["S3"])].reset_index(drop=True)
    gt_f = gt[gt["source1_entity_id"].isin(split_ids["S1"])].copy()

    def keep_in_split(match_str):
        ids = [m.strip() for m in match_str.split(",") if m.strip()]
        kept = [m for m in ids
                if (m.startswith("S2-") and m in split_ids["S2"])
                or (m.startswith("S3-") and m in split_ids["S3"])]
        return ",".join(kept)

    gt_f["matched_entity_ids"] = gt_f["matched_entity_ids"].apply(keep_in_split)
    print(f"After restricting to split: S1={len(s1_f)}, S2={len(s2_f)}, "
          f"S3={len(s3_f)}, ground_truth={len(gt_f)}")
    return s1_f, s2_f, s3_f, gt_f


# ======================================================================
# SHARED TEXT HELPERS
# ======================================================================

NULL_PLACEHOLDER_PATTERN = re.compile(r"<\s*null\s*>|\bnull\b|\bn/?a\b|\bnone\b|\bnil\b", re.IGNORECASE)
LANDMARK_PATTERN = re.compile(r"\bnear\b|\bopposite\b|\bbehind\b|\bnext to\b", re.IGNORECASE)
DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")
PIN_PATTERN = re.compile(r"\b\d{5,6}\b")


def strip_null_placeholders(text):
    cleaned = NULL_PLACEHOLDER_PATTERN.sub(" ", text)
    cleaned = re.sub(r"\s*,\s*,", ",", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" ,")


def is_effectively_missing(text):
    if text.strip() == "":
        return True
    return strip_null_placeholders(text).strip() == ""


STREET_TYPE_ABBREVIATIONS = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "apt": "apartment", "bldg": "building", "ln": "lane", "dr": "drive",
    "ct": "court", "hwy": "highway", "sq": "square", "pl": "place",
    "fl": "floor", "no": "number",
}
US_STATE_ABBREVIATIONS = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "washington dc",
}
# street-type wins where keys collide (ct, fl, in, ...) -- known limitation, see notes
ADDRESS_ABBREVIATIONS = {**US_STATE_ABBREVIATIONS, **STREET_TYPE_ABBREVIATIONS}

NAME_SUFFIX_ABBREVIATIONS = {
    "corp": "corporation", "co": "company", "ltd": "limited", "pvt": "private",
    "inc": "incorporated", "llc": "llc", "llp": "llp", "intl": "international",
    "mfg": "manufacturing", "assoc": "associates", "grp": "group",
}


def normalize_text(text, abbrev_map):
    a = strip_null_placeholders(text).lower()
    a = a.replace("&", " and ")
    a = re.sub(r"[^\w\s]", " ", a)
    a = re.sub(r"\s+", " ", a).strip()
    tokens = [abbrev_map.get(tok, tok) for tok in a.split()]
    return " ".join(tokens)


def jaccard(tokens_a, tokens_b):
    a, b = set(tokens_a), set(tokens_b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def char_similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def scan_frequent_short_tokens(series, sample_n=20000, max_len=4, min_count=10):
    sample = series.sample(min(sample_n, len(series)), random_state=SEED)
    tokens = []
    for t in sample:
        t = strip_null_placeholders(t)
        t = re.sub(r"[^\w\s]", " ", t.lower())
        tokens.extend(t.split())
    counts = Counter(tokens)
    cands = {t: c for t, c in counts.items() if len(t) <= max_len and c >= min_count and not t.isdigit()}
    return sorted(cands.items(), key=lambda x: -x[1])


# ======================================================================
# PLOT HELPERS
# ======================================================================

def plot_overlaid_hist(series_dict, title, xlabel, filename, bins=40):
    plt.figure(figsize=(7, 4.5))
    for label, series in series_dict.items():
        plt.hist(series, bins=bins, alpha=0.55, label=label, density=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=120)
    plt.close()
    print(f"  [plot saved] {PLOT_DIR / filename}")


def plot_bar(labels, values, title, ylabel, filename, rotate=0):
    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotate)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=120)
    plt.close()
    print(f"  [plot saved] {PLOT_DIR / filename}")


def plot_scatter_by_label(df, x, y, label_col, title, filename, sample_n=4000):
    plt.figure(figsize=(6, 6))
    for label, color in [("true", "tab:green"), ("random", "tab:red")]:
        sub = df[df[label_col] == label]
        if len(sub) > sample_n:
            sub = sub.sample(sample_n, random_state=SEED)
        plt.scatter(sub[x], sub[y], s=8, alpha=0.35, label=label, color=color)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=120)
    plt.close()
    print(f"  [plot saved] {PLOT_DIR / filename}")


# ======================================================================
# SECTION A -- BUSINESS NAME EDA
# ======================================================================

def analyze_name_basics(df, source_name):
    print(f"\n--- [{source_name}] Name missingness / duplicates / length / tokens ---")
    name = df["business_name"]
    is_missing = name.apply(is_effectively_missing)
    print(f"Missing name rate: {is_missing.mean():.3%} ({is_missing.sum()} / {len(df)})")

    dup_counts = name[~is_missing].value_counts()
    dup_rate = 1 - (dup_counts.index.nunique() / max(len(name[~is_missing]), 1))
    print(f"Exact-duplicate name rate (non-unique names): {dup_rate:.2%}")
    print("Top duplicated raw names:")
    print(dup_counts.head(10).to_string())

    non_missing = name[~is_missing]
    token_counts = non_missing.str.split().str.len()
    print("\nToken count distribution:")
    print(token_counts.describe().to_string())

    return {"missing_rate": is_missing.mean(), "token_counts": token_counts,
            "dup_counts": dup_counts}


def analyze_name_normalization(df, source_name):
    print(f"\n--- [{source_name}] Name normalization / abbreviations / suffixes ---")
    candidates = scan_frequent_short_tokens(df["business_name"], sample_n=20000)
    print("Top short/frequent tokens (check against NAME_SUFFIX_ABBREVIATIONS):")
    for tok, cnt in candidates[:20]:
        flag = "  <- known" if tok in NAME_SUFFIX_ABBREVIATIONS else ""
        print(f"  {tok:6s} {cnt:6d}{flag}")

    sample = df["business_name"][df["business_name"].str.strip() != ""].sample(
        min(5, len(df)), random_state=SEED
    )
    print("\nBefore / after normalization examples:")
    for n in sample:
        print(f"  RAW : {n}")
        print(f"  NORM: {normalize_text(n, NAME_SUFFIX_ABBREVIATIONS)}")


# ======================================================================
# SECTION B -- ADDRESS + COUNTRY EDA
# ======================================================================

def analyze_address_basics(df, source_name):
    print(f"\n--- [{source_name}] Address missingness / length / tokens ---")
    addr = df["business_address"]
    is_missing = addr.apply(is_effectively_missing)
    print(f"True missing address rate: {is_missing.mean():.3%} ({is_missing.sum()} / {len(df)})")

    has_embedded_null = addr.apply(
        lambda a: bool(NULL_PLACEHOLDER_PATTERN.search(a)) and not is_effectively_missing(a)
    )
    print(f"Embedded NULL in one field (address otherwise present): "
          f"{has_embedded_null.mean():.3%}")

    non_missing = addr[~is_missing].apply(strip_null_placeholders)
    token_counts = non_missing.str.split().str.len()
    print("Token count distribution:")
    print(token_counts.describe().to_string())
    print(f"Addresses with <=3 tokens: {(token_counts <= 3).mean():.2%}")

    landmark_only = non_missing[
        non_missing.apply(lambda a: bool(LANDMARK_PATTERN.search(a)) and len(a.split()) <= 5)
    ]
    print(f"Landmark-only addresses: {len(landmark_only)} "
          f"({len(landmark_only) / max(len(non_missing), 1):.2%})")

    has_pin = addr.apply(lambda a: bool(PIN_PATTERN.search(a)))
    print(f"Addresses with a plausible PIN/ZIP: {has_pin.mean():.2%}")

    has_deva = addr.apply(lambda a: bool(DEVANAGARI_PATTERN.search(a)))
    print(f"Addresses with Devanagari script: {has_deva.sum()} ({has_deva.mean():.3%})")

    return {"missing_rate": is_missing.mean(), "token_counts": token_counts,
            "pin_rate": has_pin.mean(), "devanagari_rate": has_deva.mean()}


def analyze_address_duplicates(df, source_name):
    print(f"\n--- [{source_name}] Address duplicate frequency ---")
    norm_addr = df["business_address"].apply(lambda a: normalize_text(a, ADDRESS_ABBREVIATIONS))
    counts = norm_addr[norm_addr != ""].value_counts()
    dup_rate = 1 - (counts.index.nunique() / max((norm_addr != "").sum(), 1))
    print(f"Fraction of non-unique normalized addresses: {dup_rate:.2%}")
    print(f"Top {DUP_ADDRESS_TOP_N} most repeated normalized addresses:")
    print(counts.head(DUP_ADDRESS_TOP_N).to_string())
    return counts


def analyze_country(s1, s2, s3):
    print("\n--- Country distributions ---")
    dist = {}
    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        vc = df["country"].value_counts(dropna=False)
        print(f"\n[{name}] country value counts:")
        print(vc.to_string())
        dist[name] = vc
    return dist


# ======================================================================
# SECTION C -- PAIR SAMPLING + COMBINED FEATURES (used by name/address/matching)
# ======================================================================

def build_pair_samples(s1, s2, s3, gt, n_true, n_random):
    """One shared pass: builds TRUE match pairs and RANDOM non-match pairs
    with name, address, and country for both sides. All downstream
    similarity analysis (name, address, combined/matching) reuses this."""
    s1_idx = s1.set_index("entity_id")
    s2_idx = s2.set_index("entity_id")
    s3_idx = s3.set_index("entity_id")

    def lookup(entity_id):
        if entity_id.startswith("S2-"):
            row = s2_idx.loc[entity_id] if entity_id in s2_idx.index else None
        elif entity_id.startswith("S3-"):
            row = s3_idx.loc[entity_id] if entity_id in s3_idx.index else None
        else:
            row = None
        return row

    rows = []
    for _, r in gt.iterrows():
        s1_id = r["source1_entity_id"]
        if s1_id not in s1_idx.index:
            continue
        s1_row = s1_idx.loc[s1_id]
        matches = [m.strip() for m in r["matched_entity_ids"].split(",") if m.strip()]
        for m in matches:
            other = lookup(m)
            if other is None:
                continue
            rows.append(("true", s1_id, m, s1_row["business_name"], other["business_name"],
                         s1_row["business_address"], other["business_address"],
                         s1_row["country"], other["country"]))
        if len(rows) >= n_true:
            break
    true_rows = rows[:n_true]

    pool = pd.concat([
        s2[["entity_id", "business_name", "business_address", "country"]],
        s3[["entity_id", "business_name", "business_address", "country"]],
    ], ignore_index=True)
    s1_sample = s1.sample(min(n_random, len(s1)), random_state=SEED)
    pool_sample = pool.sample(min(n_random, len(pool)), random_state=SEED).reset_index(drop=True)

    random_rows = []
    for i, (_, r1) in enumerate(s1_sample.iterrows()):
        r2 = pool_sample.iloc[i % len(pool_sample)]
        random_rows.append(("random", r1["entity_id"], r2["entity_id"], r1["business_name"],
                            r2["business_name"], r1["business_address"], r2["business_address"],
                            r1["country"], r2["country"]))

    cols = ["label", "s1_id", "other_id", "name_a", "name_b", "addr_a", "addr_b",
            "country_a", "country_b"]
    return pd.DataFrame(true_rows + random_rows, columns=cols)


def compute_pair_features(pairs_df):
    df = pairs_df.copy()
    df["name_a_norm"] = df["name_a"].apply(lambda x: normalize_text(x, NAME_SUFFIX_ABBREVIATIONS))
    df["name_b_norm"] = df["name_b"].apply(lambda x: normalize_text(x, NAME_SUFFIX_ABBREVIATIONS))
    df["addr_a_norm"] = df["addr_a"].apply(lambda x: normalize_text(x, ADDRESS_ABBREVIATIONS))
    df["addr_b_norm"] = df["addr_b"].apply(lambda x: normalize_text(x, ADDRESS_ABBREVIATIONS))

    df["name_jaccard"] = df.apply(
        lambda r: jaccard(r["name_a_norm"].split(), r["name_b_norm"].split()), axis=1)
    df["name_charsim"] = df.apply(
        lambda r: char_similarity(r["name_a_norm"], r["name_b_norm"]), axis=1)
    df["addr_jaccard"] = df.apply(
        lambda r: jaccard(r["addr_a_norm"].split(), r["addr_b_norm"].split()), axis=1)
    df["addr_charsim"] = df.apply(
        lambda r: char_similarity(r["addr_a_norm"], r["addr_b_norm"]), axis=1)
    df["country_agree"] = df["country_a"] == df["country_b"]
    df["name_exact_raw"] = df["name_a"].str.strip().str.lower() == df["name_b"].str.strip().str.lower()
    df["name_exact_norm"] = df["name_a_norm"] == df["name_b_norm"]
    df["addr_exact_norm"] = (df["addr_a_norm"] == df["addr_b_norm"]) & (df["addr_a_norm"] != "")

    if HAS_SKLEARN:
        corpus = pd.concat([df["name_a_norm"], df["name_b_norm"]]).tolist()
        vec = TfidfVectorizer().fit(corpus)
        va = vec.transform(df["name_a_norm"])
        vb = vec.transform(df["name_b_norm"])
        # row-wise cosine similarity between matching rows of va/vb
        df["name_tfidf_sim"] = np.array(
            [cosine_similarity(va[i], vb[i])[0, 0] for i in range(va.shape[0])]
        )
    else:
        df["name_tfidf_sim"] = np.nan
        print("  [note] scikit-learn not available -- skipping TF-IDF name similarity.")

    # a simple combined score for the matching-stage discussion
    df["combined_sim"] = (df["name_jaccard"] + df["addr_jaccard"]) / 2
    return df


# ======================================================================
# SECTION D -- GROUND TRUTH / MATCHING EDA
# ======================================================================

def analyze_match_distribution(gt):
    print("\n--- Ground truth: match count distribution ---")
    gt = gt.copy()
    gt["matches"] = gt["matched_entity_ids"].apply(
        lambda x: [m.strip() for m in x.split(",") if m.strip()]
    )
    gt["n_matches"] = gt["matches"].apply(len)
    gt["bucket"] = gt["n_matches"].apply(lambda n: "0" if n == 0 else "1" if n == 1 else "2+")

    bucket_counts = gt["bucket"].value_counts()
    print("0 / 1 / 2+ match distribution:")
    print(bucket_counts.to_string())
    print(f"\nSingleton percentage: {(gt['n_matches'] == 0).mean():.2%}")

    s2_only = gt["matches"].apply(lambda ms: len(ms) > 0 and all(m.startswith("S2-") for m in ms))
    s3_only = gt["matches"].apply(lambda ms: len(ms) > 0 and all(m.startswith("S3-") for m in ms))
    both = gt["matches"].apply(
        lambda ms: any(m.startswith("S2-") for m in ms) and any(m.startswith("S3-") for m in ms)
    )
    print(f"\nS1->S2 only: {s2_only.mean():.2%}   S1->S3 only: {s3_only.mean():.2%}   "
          f"Both S2 and S3: {both.mean():.2%}")

    print("\nMatches-per-S1 distribution:")
    print(gt["n_matches"].describe().to_string())

    return gt, bucket_counts, {"s2_only": s2_only.mean(), "s3_only": s3_only.mean(), "both": both.mean()}


def analyze_true_match_agreement(features_df):
    print("\n--- True-match agreement summary (from shared pair sample) ---")
    true_df = features_df[features_df["label"] == "true"]
    print(f"Exact raw name match rate among true matches: {true_df['name_exact_raw'].mean():.2%}")
    print(f"Exact normalized name match rate among true matches: {true_df['name_exact_norm'].mean():.2%}")
    print(f"Exact normalized address match rate among true matches: {true_df['addr_exact_norm'].mean():.2%}")
    print(f"Country agreement rate among true matches: {true_df['country_agree'].mean():.2%}")
    print("\nTrue-match similarity score summary:")
    print(true_df[["name_jaccard", "name_charsim", "addr_jaccard", "addr_charsim",
                    "name_tfidf_sim"]].describe().to_string())


def flag_dangerous_pairs(features_df, threshold):
    print(f"\n--- Dangerous false-positive risk: RANDOM pairs with combined_sim > {threshold} ---")
    random_df = features_df[features_df["label"] == "random"]
    risky = random_df[random_df["combined_sim"] > threshold]
    print(f"{len(risky)} / {len(random_df)} random (non-match) pairs score above threshold "
          f"({len(risky) / max(len(random_df), 1):.2%}) -- these are the profile of pairs "
          f"most likely to cause a false positive (costly under F_0.5).")
    if len(risky):
        print("\nExamples:")
        cols = ["name_a", "name_b", "addr_a", "addr_b", "combined_sim"]
        print(risky[cols].head(5).to_string(index=False))
    return risky


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("Loading data...")
    nrows = SAMPLE_ROWS_IF_SAMPLE_MODE if SAMPLE_MODE else None
    s1 = read_tsv(f"{TRAIN_DIR}/train_source1.tsv", nrows=nrows)
    s2 = read_tsv(f"{TRAIN_DIR}/train_source2.tsv", nrows=nrows)
    s3 = read_tsv(f"{TRAIN_DIR}/train_source3.tsv", nrows=nrows)
    gt = read_tsv(f"{TRAIN_DIR}/train_ground_truth.tsv", nrows=nrows)

    print(f"\nRestricting to '{SPLIT_TO_USE}' split via {SPLIT_ASSIGNMENTS_PATH} ...")
    split_ids = load_split_entity_ids(SPLIT_ASSIGNMENTS_PATH, SPLIT_TO_USE)
    s1, s2, s3, gt = restrict_to_split(s1, s2, s3, gt, split_ids)

    # ---------- Section A: name ----------
    name_stats = {}
    for src_name, df in [("Source1", s1), ("Source2", s2), ("Source3", s3)]:
        name_stats[src_name] = analyze_name_basics(df, src_name)
        analyze_name_normalization(df, src_name)

    plot_overlaid_hist(
        {k: v["token_counts"] for k, v in name_stats.items()},
        "Business name token count by source", "token count", "name_token_counts.png"
    )
    plot_bar(
        list(name_stats.keys()), [v["missing_rate"] for v in name_stats.values()],
        "Missing name rate by source", "missing rate", "name_missing_rate.png"
    )

    # ---------- Section B: address + country ----------
    addr_stats = {}
    for src_name, df in [("Source1", s1), ("Source2", s2), ("Source3", s3)]:
        addr_stats[src_name] = analyze_address_basics(df, src_name)
        analyze_address_duplicates(df, src_name)

    plot_overlaid_hist(
        {k: v["token_counts"] for k, v in addr_stats.items()},
        "Address token count by source", "token count", "address_token_counts.png"
    )
    plot_bar(
        list(addr_stats.keys()), [v["pin_rate"] for v in addr_stats.values()],
        "PIN/ZIP presence rate by source", "rate", "pin_presence_rate.png"
    )

    analyze_country(s1, s2, s3)

    # ---------- Section C: shared pair sampling + features ----------
    print("\nBuilding shared true/random pair sample for similarity analysis...")
    pairs = build_pair_samples(s1, s2, s3, gt, N_TRUE_PAIRS, N_RANDOM_PAIRS)
    features = compute_pair_features(pairs)
    features.to_csv(OUT_DIR / "pair_features_sample.tsv", sep="\t", index=False)
    print(f"Saved pair-level features -> {OUT_DIR / 'pair_features_sample.tsv'}")

    plot_overlaid_hist(
        {lbl: features.loc[features.label == lbl, "name_jaccard"] for lbl in ["true", "random"]},
        "Name similarity (Jaccard): true vs random pairs", "jaccard", "name_jaccard_hist.png"
    )
    plot_overlaid_hist(
        {lbl: features.loc[features.label == lbl, "addr_jaccard"] for lbl in ["true", "random"]},
        "Address similarity (Jaccard): true vs random pairs", "jaccard", "addr_jaccard_hist.png"
    )
    if HAS_SKLEARN:
        plot_overlaid_hist(
            {lbl: features.loc[features.label == lbl, "name_tfidf_sim"] for lbl in ["true", "random"]},
            "Name similarity (TF-IDF cosine): true vs random pairs", "cosine similarity",
            "name_tfidf_hist.png"
        )
    plot_scatter_by_label(
        features, "name_jaccard", "addr_jaccard", "label",
        "Name vs Address similarity -- true (green) vs random (red)",
        "name_vs_addr_scatter.png"
    )

    # ---------- Section D: matching / ground truth ----------
    gt_enriched, bucket_counts, s2s3_split = analyze_match_distribution(gt)
    plot_bar(
        bucket_counts.index.tolist(), bucket_counts.values.tolist(),
        "Match count distribution (0 / 1 / 2+)", "count", "match_count_distribution.png"
    )
    plot_bar(
        list(s2s3_split.keys()), list(s2s3_split.values()),
        "S1->S2 only vs S1->S3 only vs both", "fraction", "s2_s3_match_split.png"
    )

    analyze_true_match_agreement(features)
    flag_dangerous_pairs(features, DANGEROUS_SIM_THRESHOLD)

    # ---------- save reusable artifacts ----------
    pd.DataFrame(list(NAME_SUFFIX_ABBREVIATIONS.items()),
                 columns=["abbreviation", "expansion"]).to_csv(
        OUT_DIR / "name_abbreviation_map.tsv", sep="\t", index=False)
    pd.DataFrame(list(ADDRESS_ABBREVIATIONS.items()),
                 columns=["abbreviation", "expansion"]).to_csv(
        OUT_DIR / "address_abbreviation_map.tsv", sep="\t", index=False)

    print("\nDone. Artifacts saved under eda_outputs/ (TSVs) and "
          "eda_outputs/plots/ (PNGs). Send me the printed console output "
          "plus the plots and I'll guide preprocessing next.")


if __name__ == "__main__":
    main()
