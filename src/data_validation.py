"""
Data Quality Validation Script
Phase 2: Dataset Design, Data Schema, Skill Taxonomy & Integrity Checks
"""

import sys
from pathlib import Path
import pandas as pd

def validate_datasets(base_dir: Path = None):
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    data_dir = base_dir / "data"
    resumes_path = data_dir / "raw" / "resumes" / "resumes.csv"
    jobs_path = data_dir / "raw" / "jobs" / "jobs.csv"
    pairs_path = data_dir / "raw" / "raw_pairs.csv"
    skills_path = data_dir / "taxonomy" / "skills.csv"

    errors = []
    warnings = []

    print("=" * 65)
    print("AI-Based Resume Screening System: Dataset Quality Validation")
    print("=" * 65)

    # 1. Check file existence
    for path, name in [
        (resumes_path, "Resumes CSV"),
        (jobs_path, "Jobs CSV"),
        (pairs_path, "Raw Pairs CSV"),
        (skills_path, "Skills Taxonomy CSV"),
    ]:
        if not path.exists():
            errors.append(f"Missing essential data file: {name} at {path}")

    if errors:
        print("[-] FATAL: Cannot proceed with validation due to missing files:")
        for err in errors:
            print(f"    - {err}")
        return False

    # 2. Load datasets
    df_resumes = pd.read_csv(resumes_path)
    df_jobs = pd.read_csv(jobs_path)
    df_pairs = pd.read_csv(pairs_path)
    df_skills = pd.read_csv(skills_path)

    # 3. Validate Resumes Dataset
    # Required columns
    req_resume_cols = ["resume_id", "resume_text", "education", "degree", "field_of_study", "experience_years", "skills"]
    for col in req_resume_cols:
        if col not in df_resumes.columns:
            errors.append(f"Resumes dataset missing required column: '{col}'")

    # Duplicate resume IDs
    dup_resumes = df_resumes["resume_id"].duplicated().sum()
    if dup_resumes > 0:
        errors.append(f"Found {dup_resumes} duplicate resume_id(s) in resumes.csv")

    # Missing critical values
    missing_resume_text = df_resumes["resume_text"].isna().sum()
    empty_resume_text = (df_resumes["resume_text"].astype(str).str.strip() == "").sum()
    if missing_resume_text + empty_resume_text > 0:
        errors.append(f"Found {missing_resume_text + empty_resume_text} empty/missing resume_text entry(ies)")

    # 4. Validate Jobs Dataset
    req_job_cols = ["job_id", "job_title", "company", "job_description", "job_category", "required_skills"]
    for col in req_job_cols:
        if col not in df_jobs.columns:
            errors.append(f"Jobs dataset missing required column: '{col}'")

    # Duplicate job IDs
    dup_jobs = df_jobs["job_id"].duplicated().sum()
    if dup_jobs > 0:
        errors.append(f"Found {dup_jobs} duplicate job_id(s) in jobs.csv")

    # Missing critical values
    missing_job_desc = df_jobs["job_description"].isna().sum()
    empty_job_desc = (df_jobs["job_description"].astype(str).str.strip() == "").sum()
    if missing_job_desc + empty_job_desc > 0:
        errors.append(f"Found {missing_job_desc + empty_job_desc} empty/missing job_description entry(ies)")

    # 5. Validate Pairs Dataset
    req_pair_cols = ["pair_id", "resume_id", "job_id", "match_label"]
    for col in req_pair_cols:
        if col not in df_pairs.columns:
            errors.append(f"Pairs dataset missing required column: '{col}'")

    # Duplicate pair IDs
    dup_pair_ids = df_pairs["pair_id"].duplicated().sum()
    if dup_pair_ids > 0:
        errors.append(f"Found {dup_pair_ids} duplicate pair_id(s) in raw_pairs.csv")

    # Duplicate pairs (same resume_id + job_id)
    dup_pair_tuples = df_pairs.duplicated(subset=["resume_id", "job_id"]).sum()
    if dup_pair_tuples > 0:
        errors.append(f"Found {dup_pair_tuples} duplicate (resume_id, job_id) pair combination(s)")

    # Invalid labels
    invalid_labels = df_pairs[~df_pairs["match_label"].isin([0, 1])]
    if len(invalid_labels) > 0:
        errors.append(f"Found {len(invalid_labels)} invalid match_label entries (must be 0 or 1)")

    # Foreign Key Integrity
    resume_ids = set(df_resumes["resume_id"])
    job_ids = set(df_jobs["job_id"])

    missing_resume_fks = df_pairs[~df_pairs["resume_id"].isin(resume_ids)]
    if len(missing_resume_fks) > 0:
        errors.append(f"Found {len(missing_resume_fks)} pair rows referencing non-existent resume_ids")

    missing_job_fks = df_pairs[~df_pairs["job_id"].isin(job_ids)]
    if len(missing_job_fks) > 0:
        errors.append(f"Found {len(missing_job_fks)} pair rows referencing non-existent job_ids")

    # 6. Validate Skills Taxonomy
    dup_skill_ids = df_skills["skill_id"].duplicated().sum()
    if dup_skill_ids > 0:
        errors.append(f"Found {dup_skill_ids} duplicate skill_id(s) in skills.csv")

    dup_skill_names = df_skills["skill_name"].str.lower().duplicated().sum()
    if dup_skill_names > 0:
        errors.append(f"Found {dup_skill_names} duplicate skill_name(s) in skills.csv")

    # 7. Summary & Reporting
    print(f"Resumes Count:         {len(df_resumes)}")
    print(f"Jobs Count:            {len(df_jobs)}")
    print(f"Pairs Count:           {len(df_pairs)}")
    print(f"Skills Taxonomy Count: {len(df_skills)} across {df_skills['category'].nunique()} categories")
    print("-" * 65)

    label_counts = df_pairs["match_label"].value_counts().to_dict()
    matches = label_counts.get(1, 0)
    no_matches = label_counts.get(0, 0)
    print(f"Class Distribution:    Match (1): {matches} ({(matches/len(df_pairs)*100):.1f}%) | No Match (0): {no_matches} ({(no_matches/len(df_pairs)*100):.1f}%)")
    print("-" * 65)
    print(f"Duplicate Resume IDs:  {dup_resumes}")
    print(f"Duplicate Job IDs:     {dup_jobs}")
    print(f"Duplicate Pair IDs:    {dup_pair_ids}")
    print(f"Duplicate Pairs:       {dup_pair_tuples}")
    print(f"Empty/Missing Text:    {missing_resume_text + empty_resume_text + missing_job_desc + empty_job_desc}")
    print(f"FK Integrity Errors:   {len(missing_resume_fks) + len(missing_job_fks)}")
    print("=" * 65)

    if errors:
        print("[-] VALIDATION FAILED: Found the following issues:")
        for err in errors:
            print(f"    - {err}")
        return False
    else:
        print("[+] STATUS: PASSED - All schema, integrity, and label quality checks succeeded.")
        print("=" * 65)
        return True

if __name__ == "__main__":
    success = validate_datasets()
    sys.exit(0 if success else 1)
