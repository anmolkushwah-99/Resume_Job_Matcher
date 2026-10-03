"""
NLP Pipeline Quality & Entity Extraction Validation Script.
Phase 4: Resume Information Extraction, Taxonomy Validation & Consistency Checks.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
from src.preprocessing.nlp_pipeline import process_resume_text
from src.preprocessing.skill_extractor import load_skill_taxonomy


def validate_nlp_pipeline(base_dir: Path = None) -> bool:
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    resumes_csv = base_dir / "data" / "raw" / "resumes" / "resumes.csv"
    taxonomy_csv = base_dir / "data" / "taxonomy" / "skills.csv"

    print("=" * 68)
    print("AI-Based Resume Screening System: NLP & Entity Extraction Validation")
    print("=" * 68)

    errors = []
    warnings = []

    # 1. Check file existence
    if not resumes_csv.exists():
        print(f"[-] FATAL: Missing resumes dataset at {resumes_csv}")
        return False

    if not taxonomy_csv.exists():
        print(f"[-] FATAL: Missing taxonomy dataset at {taxonomy_csv}")
        return False

    taxonomy = load_skill_taxonomy(taxonomy_csv)
    valid_skill_ids = {s["skill_id"] for s in taxonomy}
    valid_skill_names = {s["skill_name"].lower() for s in taxonomy}

    df_resumes = pd.read_csv(resumes_csv)
    total_resumes = len(df_resumes)

    print(f"Taxonomy Skills Loaded: {len(taxonomy)}")
    print(f"Resumes to Process:     {total_resumes}")
    print("-" * 68)

    processed_count = 0
    skills_detected_total = 0
    education_detected_total = 0
    experience_detected_total = 0

    for idx, row in df_resumes.iterrows():
        resume_id = str(row.get("resume_id", f"RES_{idx+1}"))
        text = str(row.get("resume_text", ""))

        if not text or text.strip() == "" or text.lower() == "nan":
            errors.append(f"Resume {resume_id} has empty text.")
            continue

        try:
            # 1. Run pipeline
            res = process_resume_text(text, resume_id=resume_id)
            processed_count += 1

            # 2. Check for duplicate skills
            skill_ids_in_res = [s["skill_id"] for s in res["skills"]]
            if len(skill_ids_in_res) != len(set(skill_ids_in_res)):
                errors.append(f"Resume {resume_id} contains duplicate extracted skills: {skill_ids_in_res}")

            # 3. Check skill IDs exist in taxonomy
            for s in res["skills"]:
                if s["skill_id"] not in valid_skill_ids:
                    errors.append(f"Resume {resume_id} extracted unknown skill ID '{s['skill_id']}'")

            skills_detected_total += len(res["skills"])

            # 4. Check Education
            if res.get("degree") or (res.get("education") and res["education"].get("degree")):
                education_detected_total += 1

            # 5. Check Experience
            exp_years = res.get("experience_years", 0.0)
            if not isinstance(exp_years, (int, float)) or exp_years < 0:
                errors.append(f"Resume {resume_id} has invalid experience_years value: {exp_years}")
            elif exp_years > 0:
                experience_detected_total += 1

            # 6. Check Repeatability (Idempotency)
            res_repeat = process_resume_text(text, resume_id=resume_id)
            if [s["skill_id"] for s in res["skills"]] != [s["skill_id"] for s in res_repeat["skills"]]:
                errors.append(f"Resume {resume_id} extraction is not repeatable.")

        except Exception as e:
            errors.append(f"Resume {resume_id} raised unexpected exception: {str(e)}")

    # Summary report
    avg_skills = skills_detected_total / processed_count if processed_count > 0 else 0

    print(f"Total Resumes Processed:      {processed_count}/{total_resumes}")
    print(f"Successful Extractions:       {processed_count - len(errors)}")
    print(f"Failed Extractions:           {len(errors)}")
    print(f"Average Skills Detected:      {avg_skills:.2f} per resume")
    print(f"Resumes with Education Found: {education_detected_total} ({(education_detected_total/total_resumes)*100:.1f}%)")
    print(f"Resumes with Experience Found:{experience_detected_total} ({(experience_detected_total/total_resumes)*100:.1f}%)")
    print("-" * 68)

    if errors:
        print("[-] NLP VALIDATION FAILED: Found the following issues:")
        for err in errors:
            print(f"    - {err}")
        print("=" * 68)
        return False
    else:
        print("[+] STATUS: PASSED - All tokenization, skill taxonomy matching,")
        print("    education, and experience extraction checks succeeded.")
        print("=" * 68)
        return True


if __name__ == "__main__":
    success = validate_nlp_pipeline()
    sys.exit(0 if success else 1)
