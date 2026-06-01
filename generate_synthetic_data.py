"""
MGNN-IPV High-Fidelity Synthetic Dataset Generator
==================================================
This script generates schema-compliant synthetic datasets to run and test
the MGNN-IPV framework. It creates four Parquet files in the 'data/' directory:
1. patents.parquet
2. startups.parquet
3. citations.parquet
4. ownership.parquet

Usage:
  python generate_synthetic_data.py [--scale {small,large}] [--output-dir data/]
"""

import argparse
import os
import random
import numpy as np
import pandas as pd

# Technical patent terms to generate realistic titles and abstracts
TECH_ADJECTIVES = [
    "High-Performance", "Scalable", "Distributed", "Optimized", "Quantum-Safe",
    "Artificial Intelligence-Based", "Neural-Network-Augmented", "Secure",
    "Low-Latency", "Heterogeneous", "Autonomous", "Adaptive", "Robust"
]
TECH_NOUNS = [
    "Computing Architecture", "Signal Processor", "Data Pipeline", "Cryptographic Protocol",
    "Sensor Fusion System", "Biochemical Assay", "Wireless Transceiver", "Catalytic Converter",
    "Optical Switch", "Thermal Management Core", "Energy Storage Grid", "Nanomaterial Matrix"
]
TECH_VERBS = [
    "for accelerating", "for validating", "for securing", "for monitoring",
    "for modulating", "for optimizing", "for analyzing", "for synthesizing"
]
TECH_TARGETS = [
    "multimodal data streams.", "startup transaction ledgers.", "high-dimensional tensors.",
    "cellular receptor bindings.", "vehicular communication networks.", "lithium-ion battery lifecycles."
]

def generate_patent_text():
    adj = random.choice(TECH_ADJECTIVES)
    noun = random.choice(TECH_NOUNS)
    verb = random.choice(TECH_VERBS)
    target = random.choice(TECH_TARGETS)
    title = f"{adj} {noun} {verb} {target[:-1]}"
    
    abstract = (
        f"This disclosure relates to a {adj.lower()} {noun.lower()} configured {verb} {target} "
        f"The system implements a novel processing topology that increases throughput by at least 40% "
        f"while reducing power consumption. In one embodiment, a plurality of heterogeneous sensors "
        f"interact dynamically via a low-latency feedback network to adjust operational parameters in real time."
    )
    return title, abstract

def main():
    parser = argparse.ArgumentParser(description="Generate MGNN-IPV Synthetic Dataset")
    parser.add_argument("--scale", choices=["small", "large"], default="small",
                        help="small: 1000 patents / 100 startups (CPU testing), large: 142847 patents / 9413 startups (GPU training)")
    parser.add_argument("--output-dir", type=str, default="data/",
                        help="Directory to save the generated parquet files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Set seeds for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    if args.scale == "small":
        n_patents = 1000
        n_startups = 100
        mean_cites_per_patent = 5
    else:
        n_patents = 142847
        n_startups = 9413
        mean_cites_per_patent = 12

    print(f"=== Generating {args.scale.upper()} Scale MGNN-IPV Synthetic Dataset ===")
    print(f"Target directory: {args.output_dir}")
    print(f"Patents: {n_patents}, Startups: {n_startups}")

    # 1. Generate Startups
    print("Generating startups.parquet...")
    startup_ids = [f"ST-{i:06d}" for i in range(n_startups)]
    names = [f"TechCo {i}" for i in range(n_startups)]
    founding_years = np.random.randint(2000, 2022, size=n_startups)
    
    # Distributions based on paper
    stages = np.random.choice(
        ["Seed", "Series A", "Series B", "Series C", "Acquired", "IPO"],
        size=n_startups,
        p=[0.4, 0.3, 0.15, 0.08, 0.05, 0.02]
    )
    sectors = np.random.choice(
        ["Computing", "Biotech", "Energy", "Materials", "Telecom", "Chemistry"],
        size=n_startups,
        p=[0.35, 0.25, 0.15, 0.10, 0.08, 0.07]
    )
    countries = np.random.choice(
        ["USA", "GBR", "IND", "DEU", "CAN", "ISR", "SGP"],
        size=n_startups,
        p=[0.60, 0.12, 0.10, 0.08, 0.04, 0.04, 0.02]
    )
    
    employees = np.random.exponential(scale=45, size=n_startups).astype(int) + 2
    # Prior funding (highly skewed log-normal)
    prior_funding = np.random.lognormal(mean=14, sigma=2, size=n_startups)
    # Scale based on stage
    for i in range(n_startups):
        if stages[i] in ["Seed"]:
            prior_funding[i] *= 0.15
        elif stages[i] in ["Series B", "Series C"]:
            prior_funding[i] *= 5.0
        elif stages[i] in ["Acquired", "IPO"]:
            prior_funding[i] *= 12.0
            
    # funded_next_round (ground truth binary classification target)
    funded_next_round = np.random.binomial(1, 0.27, size=n_startups)

    startups_df = pd.DataFrame({
        "startup_id": startup_ids,
        "name": names,
        "founding_year": founding_years,
        "employees": employees,
        "prior_funding": prior_funding,
        "stage": stages,
        "sector": sectors,
        "country": countries,
        "funded_next_round": funded_next_round
    })
    startups_df.to_parquet(os.path.join(args.output_dir, "startups.parquet"), index=False)
    print("Saved startups.parquet successfully.")

    # 2. Generate Patents
    print("Generating patents.parquet...")
    patent_ids = [f"US-{i:07d}-A" for i in range(n_patents)]
    
    filing_years = np.random.randint(2005, 2024, size=n_patents)
    
    # Skewed structural features
    forward_cites = np.random.negative_binomial(n=1, p=0.2, size=n_patents)
    backward_cites = np.random.negative_binomial(n=2, p=0.15, size=n_patents)
    num_claims = np.random.poisson(lam=14, size=n_patents) + 1
    num_ipc = np.random.poisson(lam=2, size=n_patents) + 1
    family_size = np.random.geometric(p=0.4, size=n_patents)
    
    # Semantic text titles and abstracts
    titles = []
    abstracts = []
    for _ in range(n_patents):
        t, a = generate_patent_text()
        titles.append(t)
        abstracts.append(a)

    # Patent valuation ground truth target (skewed log-normal)
    valuation = np.random.lognormal(mean=11.5, sigma=1.8, size=n_patents)
    # Correlation with forward citations and family size
    valuation = valuation * (1.0 + 0.5 * forward_cites + 0.3 * family_size)

    patents_df = pd.DataFrame({
        "patent_id": patent_ids,
        "title": titles,
        "abstract": abstracts,
        "forward_cites": forward_cites,
        "backward_cites": backward_cites,
        "num_claims": num_claims,
        "num_ipc": num_ipc,
        "filing_year": filing_years,
        "family_size": family_size,
        "valuation": valuation
    })
    patents_df.to_parquet(os.path.join(args.output_dir, "patents.parquet"), index=False)
    print("Saved patents.parquet successfully.")

    # 3. Generate Citation Network (citations.parquet)
    print("Generating citations.parquet...")
    # Citing patent must be filed >= cited patent
    citing_ids = []
    cited_ids = []
    citation_years = []
    
    # Vectorized / fast approximation for citation graph creation
    print("Linking citation network edges...")
    for i in range(n_patents):
        year_i = filing_years[i]
        # Potential cited candidates are patents filed before or in the same year
        possible_candidates = np.where(filing_years <= year_i)[0]
        if len(possible_candidates) > 0:
            n_c = min(random.randint(0, mean_cites_per_patent * 2), len(possible_candidates))
            if n_c > 0:
                chosen = np.random.choice(possible_candidates, size=n_c, replace=False)
                for c in chosen:
                    if i != c:  # No self citations
                        citing_ids.append(patent_ids[i])
                        cited_ids.append(patent_ids[c])
                        citation_years.append(int(year_i))

    citations_df = pd.DataFrame({
        "citing_id": citing_ids,
        "cited_id": cited_ids,
        "year": citation_years
    })
    citations_df.to_parquet(os.path.join(args.output_dir, "citations.parquet"), index=False)
    print(f"Saved citations.parquet successfully. Total citation edges: {len(citations_df)}")

    # 4. Generate Ownership Links (ownership.parquet)
    print("Generating ownership.parquet...")
    # Map patents to startups
    own_pat_ids = []
    own_start_ids = []
    
    # Assign each patent to at least one startup, with some startups having large portfolios
    # Startups have portfolio size drawn from exponential distribution
    portfolio_sizes = np.random.exponential(scale=n_patents / n_startups, size=n_startups).astype(int) + 1
    
    patent_pool = list(patent_ids)
    random.shuffle(patent_pool)
    
    ptr = 0
    for s_idx, size in enumerate(portfolio_sizes):
        s_id = startup_ids[s_idx]
        take = min(size, len(patent_pool) - ptr)
        if take <= 0:
            break
        for p_id in patent_pool[ptr:ptr+take]:
            own_pat_ids.append(p_id)
            own_start_ids.append(s_id)
        ptr += take

    # Assign remaining patents to random startups
    if ptr < len(patent_pool):
        for p_id in patent_pool[ptr:]:
            own_pat_ids.append(p_id)
            own_start_ids.append(random.choice(startup_ids))

    ownership_df = pd.DataFrame({
        "patent_id": own_pat_ids,
        "startup_id": own_start_ids
    })
    ownership_df.to_parquet(os.path.join(args.output_dir, "ownership.parquet"), index=False)
    print(f"Saved ownership.parquet successfully. Linked {len(ownership_df)} patent-firm ownership records.")

    print("\n" + "=" * 50)
    print("MGNN-IPV High-Fidelity Synthetic Dataset successfully generated.")
    print("Parquet files are ready under:", os.path.abspath(args.output_dir))
    print("=" * 50)

if __name__ == "__main__":
    main()
