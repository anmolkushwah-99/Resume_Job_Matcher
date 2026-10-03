import logging
from pathlib import Path
import pandas as pd
from flask import Blueprint, request, jsonify, current_app, render_template
from src.extraction.resume_service import (
    process_resume_upload,
    ResumeValidationError,
    EmptyPDFError,
    ResumeStorageError,
    ResumeDatabaseError,
)
from src.preprocessing.nlp_service import (
    process_resume_by_id,
    ResumeNotFoundError as NLPResumeNotFoundError,
    ResumeProcessingError,
)
from src.matching.match_service import (
    calculate_and_store_match,
    calculate_matches_for_resume,
    get_resume_skills_from_db_or_nlp,
    get_job_skills_from_db_or_csv,
    ResumeNotFoundError as MatchResumeNotFoundError,
    JobNotFoundError,
    MatchServiceError,
)
from src.models.svm_service import (
    predict_and_store_svm_match,
    predict_svm_matches_for_resume,
)
from src.models.ann_service import (
    predict_and_store_ann_match,
    predict_ann_matches_for_resume,
)
from src.models.lstm_service import (
    predict_and_store_lstm_match,
    predict_lstm_matches_for_resume,
)
from src.database.mysql_client import test_mysql_connection, get_db_cursor
from src.matching.final_match_engine import FinalMatchEngine, _get_job_metadata, _get_all_job_ids_from_db_or_csv
from src.matching.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


@api_bp.route("/", methods=["GET"])
def index():
    """Root endpoint providing service information and serving frontend if HTML is requested."""
    accept_header = request.headers.get("Accept", "")
    if "text/html" in accept_header and "application/json" not in accept_header:
        return render_template("index.html")

    return jsonify({
        "system": "AI-Based Resume Screening and Job Matching System",
        "version": "1.0.0",
        "phase": "Step 9 Evaluation & Step 10 Final Resume-Job Matching Engine",
        "status": "operational",
        "primary_model": "tfidf_svm",
        "endpoints": {
            "GET /dashboard": "Recruiter & Candidate UI Dashboard (Step 11)",
            "POST /api/resumes/upload": "Upload and extract resume PDF",
            "POST /api/resumes/<resume_id>/process": "Run NLP pipeline and extract structured features",
            "POST /api/matches": "Generic matching endpoint using Step 10 primary model (tfidf_svm)",
            "POST /api/matches/calculate": "Calculate TF-IDF cosine similarity for a single resume-job pair",
            "POST /api/resumes/<resume_id>/matches": "Calculate and rank job matches for a resume",
            "POST /api/matches/svm": "Predict resume-job match class using supervised SVM (Step 10 engine)",
            "POST /api/resumes/<resume_id>/svm-matches": "Predict and rank job matches for a resume using SVM",
            "POST /api/matches/svm/batch": "Batch evaluate resume against multiple jobs with error isolation",
            "POST /api/matches/ann": "Predict resume-job match class using supervised ANN (MLP)",
            "POST /api/resumes/<resume_id>/ann-matches": "Predict and rank job matches for a resume using ANN",
            "POST /api/matches/lstm": "Predict resume-job match class using supervised LSTM (RNN)",
            "POST /api/resumes/<resume_id>/lstm-matches": "Predict and rank job matches for a resume using LSTM",
            "GET /api/resumes": "List all candidate resumes with metadata and skills",
            "GET /api/jobs": "List all benchmark jobs with skill requirements",
            "GET /api/stats": "Dashboard statistics and model readiness metrics",
            "GET /api/matches/history": "Recent match evaluation history",
            "GET /api/health": "Health, MySQL database, and model registry status"
        }
    }), 200


@api_bp.route("/dashboard", methods=["GET"])
@api_bp.route("/app", methods=["GET"])
@api_bp.route("/ui", methods=["GET"])
def dashboard():
    """Serves the Step 11 Recruiter & Candidate Dashboard SPA."""
    return render_template("index.html")


@api_bp.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint checking API, MySQL connectivity, and model registry readiness."""
    db_status = test_mysql_connection()
    models_health = ModelRegistry.check_health()
    is_healthy = db_status.get("status") == "connected" and models_health.get("primary_model_ready", False)
    return jsonify({
        "status": "healthy" if is_healthy else "degraded",
        "database": db_status,
        "models": models_health
    }), 200


@api_bp.route("/api/stats", methods=["GET"])
def get_stats():
    """Returns real aggregate counts for dashboard KPIs."""
    resumes_count = 0
    jobs_count = 0
    matches_count = 0

    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM resumes")
            row = cur.fetchone()
            if row:
                resumes_count = row["cnt"]

            cur.execute("SELECT COUNT(*) AS cnt FROM jobs")
            row = cur.fetchone()
            if row:
                jobs_count = row["cnt"]

            cur.execute("SELECT COUNT(*) AS cnt FROM matches")
            row = cur.fetchone()
            if row:
                matches_count = row["cnt"]
    except Exception as e:
        logger.debug("Could not fetch stats from MySQL: %s", str(e))

    # CSV fallback for benchmark records
    if resumes_count == 0:
        resumes_csv = Path(__file__).resolve().parent.parent / "data" / "raw" / "resumes" / "resumes.csv"
        if resumes_csv.exists():
            try:
                df = pd.read_csv(resumes_csv)
                resumes_count = len(df)
            except Exception:
                resumes_count = 25

    if jobs_count == 0:
        jobs_csv = Path(__file__).resolve().parent.parent / "data" / "raw" / "jobs" / "jobs.csv"
        if jobs_csv.exists():
            try:
                df = pd.read_csv(jobs_csv)
                jobs_count = len(df)
            except Exception:
                jobs_count = 12

    models_health = ModelRegistry.check_health()
    return jsonify({
        "success": True,
        "total_resumes": resumes_count,
        "total_jobs": jobs_count,
        "total_matches_evaluated": matches_count,
        "primary_model": ModelRegistry.get_primary_model_name(),
        "primary_model_type": "Linear Support Vector Classifier (LinearSVC)",
        "decision_rule": "decision_score >= 0.0 -> Match",
        "models_health": models_health
    }), 200


@api_bp.route("/api/resumes", methods=["GET"])
def get_resumes_list():
    """Returns list of all available resumes with extracted skills and metadata."""
    resumes_list = []
    seen_ids = set()

    # 1. Fetch from DB
    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT id, filename, education, degree, field_of_study, experience_years, uploaded_at
                FROM resumes
                ORDER BY uploaded_at DESC
                """
            )
            rows = cur.fetchall()
            for r in rows:
                rid = str(r["id"]).strip()
                skills = get_resume_skills_from_db_or_nlp(rid)
                resumes_list.append({
                    "resume_id": rid,
                    "filename": r.get("filename", f"resume_{rid[:8]}.pdf"),
                    "education": r.get("education") or r.get("degree") or "Not Specified",
                    "experience_years": float(r.get("experience_years") or 0.0),
                    "skills": skills,
                    "skill_count": len(skills),
                    "source": "database",
                    "uploaded_at": str(r.get("uploaded_at", ""))
                })
                seen_ids.add(rid)
    except Exception as e:
        logger.debug("Could not fetch resumes from DB: %s", str(e))

    # 2. Benchmark resumes.csv fallback / augmentation
    resumes_csv = Path(__file__).resolve().parent.parent / "data" / "raw" / "resumes" / "resumes.csv"
    if resumes_csv.exists():
        try:
            df = pd.read_csv(resumes_csv)
            for _, row in df.iterrows():
                rid = str(row["resume_id"]).strip()
                if rid not in seen_ids:
                    skills_raw = str(row.get("skills", ""))
                    skills = [s.strip() for s in skills_raw.split(";") if s.strip() and pd.notna(skills_raw)]
                    exp_val = 0.0
                    try:
                        exp_val = float(row.get("experience_years", 0.0) or 0.0)
                    except (ValueError, TypeError):
                        pass

                    resumes_list.append({
                        "resume_id": rid,
                        "filename": f"{rid}.pdf",
                        "education": str(row.get("degree", "Bachelor's Degree")),
                        "experience_years": exp_val,
                        "skills": skills,
                        "skill_count": len(skills),
                        "source": "benchmark",
                        "uploaded_at": ""
                    })
                    seen_ids.add(rid)
        except Exception:
            pass

    return jsonify({
        "success": True,
        "total_resumes": len(resumes_list),
        "resumes": resumes_list
    }), 200


@api_bp.route("/api/resumes/<resume_id>", methods=["GET"])
def get_resume_detail(resume_id: str):
    """Returns structured details and extracted skills for a single resume."""
    if not resume_id or not resume_id.strip():
        return jsonify({"success": False, "error": "Invalid resume_id parameter."}), 400

    rid = resume_id.strip()
    skills = get_resume_skills_from_db_or_nlp(rid)
    meta = None

    # Check MySQL
    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT id, filename, education, degree, field_of_study, experience_years, uploaded_at
                FROM resumes WHERE id = %s
                """,
                (rid,)
            )
            r = cur.fetchone()
            if r:
                meta = {
                    "resume_id": rid,
                    "filename": r.get("filename", f"resume_{rid[:8]}.pdf"),
                    "education": r.get("education") or r.get("degree") or "Not Specified",
                    "degree": r.get("degree", ""),
                    "field_of_study": r.get("field_of_study", ""),
                    "experience_years": float(r.get("experience_years") or 0.0),
                    "skills": skills,
                    "skill_count": len(skills),
                    "source": "database",
                    "uploaded_at": str(r.get("uploaded_at", ""))
                }
    except Exception:
        pass

    # Check CSV
    if not meta:
        resumes_csv = Path(__file__).resolve().parent.parent / "data" / "raw" / "resumes" / "resumes.csv"
        if resumes_csv.exists():
            try:
                df = pd.read_csv(resumes_csv)
                sub = df[df["resume_id"] == rid]
                if not sub.empty:
                    row = sub.iloc[0]
                    skills_raw = str(row.get("skills", ""))
                    sk = [s.strip() for s in skills_raw.split(";") if s.strip() and pd.notna(skills_raw)]
                    meta = {
                        "resume_id": rid,
                        "filename": f"{rid}.pdf",
                        "education": str(row.get("degree", "Bachelor's Degree")),
                        "degree": str(row.get("degree", "")),
                        "field_of_study": str(row.get("field_of_study", "")),
                        "experience_years": float(row.get("experience_years", 0.0) or 0.0),
                        "skills": sk if not skills else skills,
                        "skill_count": len(sk if not skills else skills),
                        "source": "benchmark",
                        "uploaded_at": ""
                    }
            except Exception:
                pass

    if not meta:
        return jsonify({"success": False, "error": f"Resume '{rid}' not found."}), 404

    return jsonify({"success": True, "resume": meta}), 200


@api_bp.route("/api/jobs", methods=["GET"])
def get_jobs_list():
    """Returns list of all available benchmark jobs with skills and requirements."""
    jobs_list = []
    seen_ids = set()

    # 1. Fetch from MySQL
    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT id, job_title, company, job_description, job_category,
                       minimum_experience, education_requirement
                FROM jobs
                ORDER BY id ASC
                """
            )
            rows = cur.fetchall()
            for r in rows:
                jid = str(r["id"]).strip()
                skills = get_job_skills_from_db_or_csv(jid)
                desc = str(r.get("job_description", ""))
                snippet = desc[:180] + "..." if len(desc) > 180 else desc
                title = r.get("job_title", jid)
                jobs_list.append({
                    "job_id": jid,
                    "title": title,
                    "job_title": title,
                    "company": r.get("company", ""),
                    "category": r.get("job_category", ""),
                    "minimum_experience": float(r.get("minimum_experience") or 0.0),
                    "education_requirement": r.get("education_requirement", ""),
                    "description": desc,
                    "description_snippet": snippet,
                    "skills": skills,
                    "required_skills": skills,
                    "skill_count": len(skills)
                })
                seen_ids.add(jid)
    except Exception as e:
        logger.debug("Could not fetch jobs from DB: %s", str(e))

    # 2. Benchmark jobs.csv fallback
    jobs_csv = Path(__file__).resolve().parent.parent / "data" / "raw" / "jobs" / "jobs.csv"
    if jobs_csv.exists():
        try:
            df = pd.read_csv(jobs_csv)
            for _, row in df.iterrows():
                jid = str(row["job_id"]).strip()
                if jid not in seen_ids:
                    skills = get_job_skills_from_db_or_csv(jid)
                    desc = str(row.get("job_description", ""))
                    snippet = desc[:180] + "..." if len(desc) > 180 else desc
                    title = str(row.get("job_title", jid))
                    jobs_list.append({
                        "job_id": jid,
                        "title": title,
                        "job_title": title,
                        "company": str(row.get("company", "")),
                        "category": str(row.get("job_category", "")),
                        "minimum_experience": float(row.get("minimum_experience", 0.0) or 0.0),
                        "education_requirement": str(row.get("education_requirement", "")),
                        "description": desc,
                        "description_snippet": snippet,
                        "skills": skills,
                        "required_skills": skills,
                        "skill_count": len(skills)
                    })
                    seen_ids.add(jid)
        except Exception:
            pass

    return jsonify({
        "success": True,
        "total_jobs": len(jobs_list),
        "jobs": jobs_list
    }), 200


@api_bp.route("/api/jobs/<job_id>", methods=["GET"])
def get_job_detail(job_id: str):
    """Returns full details and skill requirements for a specific job."""
    if not job_id or not job_id.strip():
        return jsonify({"success": False, "error": "Invalid job_id parameter."}), 400

    jid = job_id.strip()
    meta = _get_job_metadata(jid)
    if not meta or meta.get("title") == jid and "description" not in meta:
        # Check if job exists in CSV
        pass

    skills = get_job_skills_from_db_or_csv(jid)
    desc = ""
    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute("SELECT job_description FROM jobs WHERE id = %s", (jid,))
            r = cur.fetchone()
            if r:
                desc = r.get("job_description", "")
    except Exception:
        pass

    if not desc:
        jobs_csv = Path(__file__).resolve().parent.parent / "data" / "raw" / "jobs" / "jobs.csv"
        if jobs_csv.exists():
            try:
                df = pd.read_csv(jobs_csv)
                sub = df[df["job_id"] == jid]
                if not sub.empty:
                    desc = str(sub.iloc[0].get("job_description", ""))
            except Exception:
                pass

    if not meta and not desc:
        return jsonify({"success": False, "error": f"Job '{jid}' not found."}), 404

    job_data = {
        "job_id": jid,
        "title": meta.get("title", jid),
        "company": meta.get("company", ""),
        "category": meta.get("category", ""),
        "minimum_experience": meta.get("minimum_experience", 0.0),
        "education_requirement": meta.get("education_requirement", ""),
        "description": desc,
        "skills": skills,
        "skill_count": len(skills)
    }
    return jsonify({"success": True, "job": job_data}), 200


@api_bp.route("/api/matches/history", methods=["GET"])
def get_matches_history():
    """Returns recent evaluation history records from MySQL matches table."""
    history = []
    try:
        with get_db_cursor(commit=False) as cur:
            cur.execute(
                """
                SELECT m.id, m.resume_id, m.job_id, m.overall_score, m.skill_score,
                       m.model_name, m.created_at, j.job_title, j.company
                FROM matches m
                LEFT JOIN jobs j ON m.job_id = j.id
                ORDER BY m.created_at DESC
                LIMIT 50
                """
            )
            rows = cur.fetchall()
            for r in rows:
                score = float(r.get("overall_score") or 0.0)
                is_match = bool(score >= 0.0)
                history.append({
                    "match_id": r["id"],
                    "resume_id": r["resume_id"],
                    "job_id": r["job_id"],
                    "job_title": r.get("job_title") or r["job_id"],
                    "company": r.get("company", ""),
                    "model_name": r["model_name"],
                    "decision_score": round(score, 4),
                    "is_match": is_match,
                    "skill_score": float(r.get("skill_score") or 0.0),
                    "created_at": str(r.get("created_at", ""))
                })
    except Exception as e:
        logger.debug("Could not fetch match history: %s", str(e))

    return jsonify({
        "success": True,
        "total_records": len(history),
        "history": history
    }), 200




@api_bp.route("/api/resumes/upload", methods=["POST"])
def upload_resume():
    """
    Resume PDF Upload and Extraction Endpoint (Step 3).

    Expects multipart/form-data with:
    - file: Resume PDF file (required)
    - user_id: UUID of user (optional)

    Returns JSON response:
    - 201 on success (with resume_id, filename, text_length)
    - 400 on invalid file/request
    - 413 on file size > 10MB
    - 422 on unreadable PDF or no extractable text
    - 500 on storage or database failure
    """
    # 1. Verify file part presence
    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No file part in request. Please provide a file with key 'file'."
        }), 400

    file = request.files["file"]

    # 2. Verify filename presence
    if not file or not file.filename or file.filename.strip() == "":
        return jsonify({
            "success": False,
            "error": "No file selected for upload or empty filename."
        }), 400

    user_id = request.form.get("user_id")

    # 3. Process upload through service
    try:
        result = process_resume_upload(file_input=file, user_id=user_id)
        return jsonify(result), 201

    except ResumeValidationError as rve:
        logger.warning("Validation error during resume upload: %s", rve.message)
        return jsonify({
            "success": False,
            "error": rve.message
        }), rve.status_code

    except EmptyPDFError as epe:
        logger.warning("Empty PDF error during resume upload: %s", epe.message)
        return jsonify({
            "success": False,
            "error": epe.message
        }), 422

    except ResumeStorageError as rse:
        logger.error("Storage error during resume upload: %s", rse.message)
        return jsonify({
            "success": False,
            "error": "Failed to store resume document in storage. Please try again later."
        }), 500

    except ResumeDatabaseError as rde:
        logger.error("Database error during resume upload: %s", rde.message)
        return jsonify({
            "success": False,
            "error": "Failed to save resume metadata in database. Please try again later."
        }), 500

    except Exception as e:
        logger.exception("Unexpected error processing resume upload")
        return jsonify({
            "success": False,
            "error": "An unexpected server error occurred while processing the resume."
        }), 500


@api_bp.route("/api/resumes/<resume_id>/process", methods=["POST"])
def process_resume_nlp(resume_id: str):
    """
    NLP Preprocessing and Entity Extraction Endpoint (Step 4).

    Fetches the resume by ID, extracts structured features (skills, education,
    experience, sections, tokens), updates MySQL, and returns a privacy-safe summary.

    Returns JSON response:
    - 200 on success
    - 400 on invalid request
    - 404 if resume_id not found
    - 422 if resume cannot be processed
    - 500 on server/database error
    """
    if not resume_id or not resume_id.strip():
        return jsonify({
            "success": False,
            "error": "Invalid or missing resume_id parameter."
        }), 400

    try:
        result = process_resume_by_id(resume_id.strip())
        return jsonify(result), 200

    except NLPResumeNotFoundError as rnfe:
        logger.warning("Resume not found: %s", str(rnfe))
        return jsonify({
            "success": False,
            "error": str(rnfe)
        }), 404

    except ResumeProcessingError as rpe:
        logger.warning("Resume processing error: %s", str(rpe))
        return jsonify({
            "success": False,
            "error": str(rpe)
        }), 422

    except ValueError as ve:
        return jsonify({
            "success": False,
            "error": str(ve)
        }), 400

    except Exception as e:
        logger.exception("Unexpected error during resume NLP processing")
        return jsonify({
            "success": False,
            "error": "An unexpected server error occurred while processing resume NLP features."
        }), 500


@api_bp.route("/api/matches/calculate", methods=["POST"])
def match_resume_job():
    """
    TF-IDF Cosine Similarity Baseline Matching Endpoint (Step 5).

    Calculates the textual similarity between a candidate resume and a benchmark job,
    persists the score to MySQL, and returns a privacy-safe evaluation response.

    Expects JSON body:
    {
        "resume_id": "RES001" or UUID,
        "job_id": "JOB001" or UUID
    }

    Returns JSON response:
    - 200 on success
    - 400 on invalid or missing request body/parameters
    - 404 if resume_id or job_id is not found
    - 500 on internal calculation/database error
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": "Request body must be a valid JSON object containing 'resume_id' and 'job_id'."
        }), 400

    resume_id = data.get("resume_id")
    job_id = data.get("job_id")

    if not resume_id or not str(resume_id).strip():
        return jsonify({
            "success": False,
            "error": "Missing or empty 'resume_id' parameter."
        }), 400

    if not job_id or not str(job_id).strip():
        return jsonify({
            "success": False,
            "error": "Missing or empty 'job_id' parameter."
        }), 400

    try:
        result = calculate_and_store_match(
            resume_id=str(resume_id).strip(),
            job_id=str(job_id).strip(),
            model_name="tfidf_cosine_baseline",
            persist=True
        )
        return jsonify(result), 200

    except MatchResumeNotFoundError as rne:
        logger.warning("Resume not found for matching: %s", str(rne))
        return jsonify({
            "success": False,
            "error": str(rne)
        }), 404

    except JobNotFoundError as jne:
        logger.warning("Job not found for matching: %s", str(jne))
        return jsonify({
            "success": False,
            "error": str(jne)
        }), 404

    except ValueError as ve:
        return jsonify({
            "success": False,
            "error": str(ve)
        }), 400

    except Exception as e:
        logger.exception("Unexpected error calculating resume-job match")
        return jsonify({
            "success": False,
            "error": "An unexpected server error occurred while calculating resume-job match."
        }), 500


@api_bp.route("/api/resumes/<resume_id>/matches", methods=["POST", "GET"])
def get_resume_all_matches(resume_id: str):
    """
    Computes TF-IDF Cosine Similarity for a given resume against all available
    benchmark jobs, returning the matches ordered from highest to lowest similarity.

    Returns JSON response:
    - 200 on success
    - 400 on invalid resume_id
    - 404 if resume_id is not found
    - 500 on server error
    """
    if not resume_id or not resume_id.strip():
        return jsonify({
            "success": False,
            "error": "Invalid or missing resume_id parameter."
        }), 400

    limit = request.args.get("limit", type=int)

    try:
        result = calculate_matches_for_resume(
            resume_id=resume_id.strip(),
            model_name="tfidf_cosine_baseline",
            limit=limit,
            persist=True
        )
        return jsonify(result), 200

    except MatchResumeNotFoundError as rne:
        logger.warning("Resume not found for matches list: %s", str(rne))
        return jsonify({
            "success": False,
            "error": str(rne)
        }), 404

    except JobNotFoundError as jne:
        logger.warning("No jobs found for matches: %s", str(jne))
        return jsonify({
            "success": False,
            "error": str(jne)
        }), 404

    except Exception as e:
        logger.exception("Unexpected error calculating all matches for resume")
        return jsonify({
            "success": False,
            "error": "An unexpected server error occurred while generating resume job matches."
        }), 500


@api_bp.route("/api/matches/svm", methods=["POST"])
def match_resume_job_svm():
    """
    Supervised SVM Resume-Job Matching Endpoint (Step 6 / Step 10).

    Evaluates a single resume against a target job using the Step 9-validated
    TF-IDF LinearSVC model, returning decision score, match classification,
    and skill gap explanation while persisting to MySQL.

    Expects JSON body:
    {
        "resume_id": "RES001" or UUID,
        "job_id": "JOB001" or UUID
    }
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": "Request body must be a valid JSON object containing 'resume_id' and 'job_id'."
        }), 400

    resume_id = data.get("resume_id")
    job_id = data.get("job_id")

    if not resume_id or not str(resume_id).strip():
        return jsonify({
            "success": False,
            "error": "Missing or empty 'resume_id' parameter."
        }), 400

    if not job_id or not str(job_id).strip():
        return jsonify({
            "success": False,
            "error": "Missing or empty 'job_id' parameter."
        }), 400

    try:
        engine = FinalMatchEngine.get_instance()
        result = engine.match_one(
            resume_id=str(resume_id).strip(),
            job_id=str(job_id).strip(),
            persist=True
        )
        result["model"] = result["model_name"]
        result["shared_skills"] = result["matched_skills"]
        return jsonify(result), 200

    except MatchResumeNotFoundError as rne:
        logger.warning("Resume not found for SVM matching: %s", str(rne))
        return jsonify({"success": False, "error": str(rne)}), 404

    except JobNotFoundError as jne:
        logger.warning("Job not found for SVM matching: %s", str(jne))
        return jsonify({"success": False, "error": str(jne)}), 404

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400

    except Exception as e:
        logger.exception("Unexpected error calculating SVM match")
        return jsonify({
            "success": False,
            "error": "An unexpected server error occurred while calculating SVM match."
        }), 500


@api_bp.route("/api/resumes/<resume_id>/svm-matches", methods=["POST", "GET"])
def get_resume_all_svm_matches(resume_id: str):
    """
    Evaluates and ranks benchmark jobs for a resume using the Step 10 Final Match Engine.
    Supports top_k, match_only, and specific job_ids filters.
    """
    if not resume_id or not resume_id.strip():
        return jsonify({
            "success": False,
            "error": "Invalid or missing resume_id parameter."
        }), 400

    req_json = request.get_json(silent=True) or {}
    top_k = req_json.get("top_k")
    if top_k is None:
        top_k = request.args.get("top_k", type=int) or request.args.get("limit", type=int)

    match_only_raw = req_json.get("match_only")
    if match_only_raw is None:
        match_only_raw = request.args.get("match_only", "")
    match_only = bool(match_only_raw is True or str(match_only_raw).lower() in ("true", "1"))

    job_ids = req_json.get("job_ids")
    if isinstance(job_ids, str):
        job_ids = [j.strip() for j in job_ids.split(",") if j.strip()]

    try:
        engine = FinalMatchEngine.get_instance()
        result = engine.match_many(
            resume_id=resume_id.strip(),
            job_ids=job_ids,
            top_k=top_k,
            match_only=match_only,
            persist=True
        )
        # Compatibility fields
        result["model_info"] = result["model"]
        result["model_name"] = result["model"]["name"]
        result["model"] = result["model"]["name"]
        result["matches"] = result["results"]
        return jsonify(result), 200


    except MatchResumeNotFoundError as rne:
        logger.warning("Resume not found for SVM matches list: %s", str(rne))
        return jsonify({"success": False, "error": str(rne)}), 404

    except JobNotFoundError as jne:
        logger.warning("No jobs found for SVM matches: %s", str(jne))
        return jsonify({"success": False, "error": str(jne)}), 404

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400

    except Exception as e:
        logger.exception("Unexpected error calculating all SVM matches for resume")
        return jsonify({
            "success": False,
            "error": "An unexpected server error occurred while generating resume SVM matches."
        }), 500


@api_bp.route("/api/matches/svm/batch", methods=["POST"])
def match_resume_jobs_svm_batch():
    """
    Batch SVM Matching Endpoint (Step 10).
    Evaluates one resume against multiple jobs with per-job error isolation.
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": "Request body must be a valid JSON object containing 'resume_id' and 'job_ids'."
        }), 400

    resume_id = data.get("resume_id")
    job_ids = data.get("job_ids")

    if not resume_id or not str(resume_id).strip():
        return jsonify({
            "success": False,
            "error": "Missing or empty 'resume_id' parameter."
        }), 400

    if job_ids is None or not isinstance(job_ids, list) or len(job_ids) == 0:
        return jsonify({
            "success": False,
            "error": "Parameter 'job_ids' must be a non-empty list of job IDs."
        }), 400

    try:
        engine = FinalMatchEngine.get_instance()
        result = engine.match_batch(
            resume_id=str(resume_id).strip(),
            job_ids=[str(j).strip() for j in job_ids if str(j).strip()],
            persist=True
        )
        return jsonify(result), 200

    except MatchResumeNotFoundError as rne:
        return jsonify({"success": False, "error": str(rne)}), 404
    except Exception as e:
        logger.exception("Error during batch SVM matching")
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/api/matches", methods=["POST"])
def match_generic():
    """
    Generic Production Matching Endpoint (Step 10).
    Defaults to the validated primary model (tfidf_svm).
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": "Request body must be a valid JSON object."
        }), 400

    model = data.get("model", "tfidf_svm")
    if model != "tfidf_svm":
        return jsonify({
            "success": False,
            "error": f"Model '{model}' is not the primary production matching engine. Use 'tfidf_svm' or model-specific routes."
        }), 400

    resume_id = data.get("resume_id")
    job_id = data.get("job_id")
    job_ids = data.get("job_ids")

    if not resume_id or not str(resume_id).strip():
        return jsonify({
            "success": False,
            "error": "Missing or empty 'resume_id' parameter."
        }), 400

    engine = FinalMatchEngine.get_instance()
    try:
        if job_id:
            res = engine.match_one(str(resume_id).strip(), str(job_id).strip(), persist=True)
            res["model"] = res["model_name"]
            res["shared_skills"] = res["matched_skills"]
            return jsonify(res), 200
        else:
            top_k = data.get("top_k") or data.get("limit")
            match_only = bool(data.get("match_only", False))
            res = engine.match_many(
                resume_id=str(resume_id).strip(),
                job_ids=job_ids,
                top_k=top_k,
                match_only=match_only,
                persist=True
            )
            res["model_info"] = res.get("model")
            res["model_name"] = "tfidf_svm"
            res["model"] = "tfidf_svm"
            res["matches"] = res.get("results", [])
            return jsonify(res), 200

    except MatchResumeNotFoundError as rne:
        return jsonify({"success": False, "error": str(rne)}), 404
    except JobNotFoundError as jne:
        return jsonify({"success": False, "error": str(jne)}), 404
    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        logger.exception("Error in generic match endpoint")
        return jsonify({"success": False, "error": "Internal server error occurred."}), 500


@api_bp.route("/api/matches/ann", methods=["POST"])

def match_resume_job_ann():
    """
    Supervised ANN / MLP Resume-Job Matching Endpoint (Step 7).

    Predicts binary match classification (0 = Non-Match, 1 = Match) and
    computes the continuous sigmoid score for a resume-job pair using the trained
    TF-IDF Keras ANN model, persisting the result to MySQL.

    Expects JSON body:
    {
        "resume_id": "RES001" or UUID,
        "job_id": "JOB001" or UUID
    }

    Returns JSON response:
    - 200 on success
    - 400 on invalid or missing request body/parameters
    - 404 if resume_id or job_id is not found
    - 500 on internal calculation/database error
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": "Request body must be a valid JSON object containing 'resume_id' and 'job_id'."
        }), 400

    resume_id = data.get("resume_id")
    job_id = data.get("job_id")

    if not resume_id or not str(resume_id).strip():
        return jsonify({
            "success": False,
            "error": "Missing or empty 'resume_id' parameter."
        }), 400

    if not job_id or not str(job_id).strip():
        return jsonify({
            "success": False,
            "error": "Missing or empty 'job_id' parameter."
        }), 400

    try:
        result = predict_and_store_ann_match(
            resume_id=str(resume_id).strip(),
            job_id=str(job_id).strip(),
            model_name="tfidf_ann",
            persist=True
        )
        return jsonify(result), 200

    except MatchResumeNotFoundError as rne:
        logger.warning("Resume not found for ANN matching: %s", str(rne))
        return jsonify({
            "success": False,
            "error": str(rne)
        }), 404

    except JobNotFoundError as jne:
        logger.warning("Job not found for ANN matching: %s", str(jne))
        return jsonify({
            "success": False,
            "error": str(jne)
        }), 404

    except ValueError as ve:
        return jsonify({
            "success": False,
            "error": str(ve)
        }), 400

    except Exception as e:
        logger.exception("Unexpected error calculating ANN match")
        return jsonify({
            "success": False,
            "error": "An unexpected server error occurred while calculating ANN match."
        }), 500


@api_bp.route("/api/resumes/<resume_id>/ann-matches", methods=["POST", "GET"])
def get_resume_all_ann_matches(resume_id: str):
    """
    Computes supervised ANN predictions for a given resume against all available
    benchmark jobs, returning the matches ordered from highest to lowest ANN score.

    Returns JSON response:
    - 200 on success
    - 400 on invalid resume_id
    - 404 if resume_id is not found
    - 500 on server error
    """
    if not resume_id or not resume_id.strip():
        return jsonify({
            "success": False,
            "error": "Invalid or missing resume_id parameter."
        }), 400

    limit = request.args.get("limit", type=int)

    try:
        result = predict_ann_matches_for_resume(
            resume_id=resume_id.strip(),
            model_name="tfidf_ann",
            limit=limit,
            persist=True
        )
        return jsonify(result), 200

    except MatchResumeNotFoundError as rne:
        logger.warning("Resume not found for ANN matches list: %s", str(rne))
        return jsonify({
            "success": False,
            "error": str(rne)
        }), 404

    except JobNotFoundError as jne:
        logger.warning("No jobs found for ANN matches: %s", str(jne))
        return jsonify({
            "success": False,
            "error": str(jne)
        }), 404

    except Exception as e:
        logger.exception("Unexpected error calculating all ANN matches for resume")
        return jsonify({
            "success": False,
            "error": "An unexpected server error occurred while generating resume ANN matches."
        }), 500


@api_bp.route("/api/matches/lstm", methods=["POST"])
def match_resume_job_lstm():
    """
    Supervised LSTM / RNN Sequence Resume-Job Matching Endpoint (Step 8).

    Predicts binary match classification (0 = Non-Match, 1 = Match) and
    computes the continuous sigmoid score for a resume-job pair using the trained
    Embedding + LSTM model, persisting the result to MySQL.

    Expects JSON body:
    {
        "resume_id": "RES001" or UUID,
        "job_id": "JOB001" or UUID
    }

    Returns JSON response:
    - 200 on success
    - 400 on invalid or missing request body/parameters
    - 404 if resume_id or job_id is not found
    - 500 on internal calculation/database error
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": "Request body must be a valid JSON object containing 'resume_id' and 'job_id'."
        }), 400

    resume_id = data.get("resume_id")
    job_id = data.get("job_id")

    if not resume_id or not str(resume_id).strip():
        return jsonify({
            "success": False,
            "error": "Missing or empty 'resume_id' parameter."
        }), 400

    if not job_id or not str(job_id).strip():
        return jsonify({
            "success": False,
            "error": "Missing or empty 'job_id' parameter."
        }), 400

    try:
        result = predict_and_store_lstm_match(
            resume_id=str(resume_id).strip(),
            job_id=str(job_id).strip(),
            model_name="embedding_lstm",
            persist=True
        )
        return jsonify(result), 200

    except MatchResumeNotFoundError as rne:
        logger.warning("Resume not found for LSTM matching: %s", str(rne))
        return jsonify({
            "success": False,
            "error": str(rne)
        }), 404

    except JobNotFoundError as jne:
        logger.warning("Job not found for LSTM matching: %s", str(jne))
        return jsonify({
            "success": False,
            "error": str(jne)
        }), 404

    except ValueError as ve:
        return jsonify({
            "success": False,
            "error": str(ve)
        }), 400

    except Exception as e:
        logger.exception("Unexpected error calculating LSTM match")
        return jsonify({
            "success": False,
            "error": "An unexpected server error occurred while calculating LSTM match."
        }), 500


@api_bp.route("/api/resumes/<resume_id>/lstm-matches", methods=["POST", "GET"])
def get_resume_all_lstm_matches(resume_id: str):
    """
    Computes supervised LSTM predictions for a given resume against all available
    benchmark jobs, returning the matches ordered from highest to lowest LSTM score.

    Returns JSON response:
    - 200 on success
    - 400 on invalid resume_id
    - 404 if resume_id is not found
    - 500 on server error
    """
    if not resume_id or not resume_id.strip():
        return jsonify({
            "success": False,
            "error": "Invalid or missing resume_id parameter."
        }), 400

    limit = request.args.get("limit", type=int)

    try:
        result = predict_lstm_matches_for_resume(
            resume_id=resume_id.strip(),
            model_name="embedding_lstm",
            limit=limit,
            persist=True
        )
        return jsonify(result), 200

    except MatchResumeNotFoundError as rne:
        logger.warning("Resume not found for LSTM matches list: %s", str(rne))
        return jsonify({
            "success": False,
            "error": str(rne)
        }), 404

    except JobNotFoundError as jne:
        logger.warning("No jobs found for LSTM matches: %s", str(jne))
        return jsonify({
            "success": False,
            "error": str(jne)
        }), 404

    except Exception as e:
        logger.exception("Unexpected error calculating all LSTM matches for resume")
        return jsonify({
            "success": False,
            "error": "An unexpected server error occurred while generating resume LSTM matches."
        }), 500




