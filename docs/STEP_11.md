# STEP 11 — RECRUITER / CANDIDATE DASHBOARD & FRONTEND INTEGRATION
## AI-Based Resume Screening and Job Matching System

---

## 1. Objective

The objective of **Step 11** is to build a responsive, user-friendly, and professional Single Page Application (SPA) dashboard for recruiters and candidates directly on top of the validated **Step 10** REST APIs.

### Absolute Constraints Preserved:
1. **Zero ML Retraining / Modification**: The machine-learning pipeline, `tfidf_svm` (LinearSVC, $C=1.0$, balanced class weight), and TF-IDF feature generator remain frozen.
2. **Deterministic Ranking Semantics**: Preserved strict server-side ranking (`decision_score DESC` with tie-breaker `job_id ASC`).
3. **Decision Score Integrity**: The signed LinearSVC decision score is displayed as a signed decimal (e.g. `+1.4521`), **strictly never formatted as a probability or fake percentage**.
4. **Single-Request Multi-Job Matching**: Matches for all benchmark jobs are computed in a single batch API call (`POST /api/resumes/<id>/svm-matches`), eliminating multi-request network overhead.
5. **Full Backward Compatibility**: All 218 previous backend unit/integration tests continue passing with zero regressions.

---

## 2. Frontend Architecture & Technology Stack

- **Architecture**: Single Page Application (SPA) with client-side hash routing (`#dashboard`, `#resumes`, `#jobs`, `#matching`, `#history`, `#about`).
- **Core Technology**: Vanilla HTML5, modern CSS3 (Custom Properties & Tokens), and ES6+ JavaScript.
- **Serving Mechanism**: Served natively by Flask via `render_template('index.html')` and static routing (`/static/css/`, `/static/js/`).
- **Typography & Icons**: Google Fonts (`Outfit`, `Inter`, `JetBrains Mono`) and FontAwesome 6 icon library.
- **Design Language**: Tailored dark-slate theme with clean contrast ratios, responsive glassmorphism containers, and accessible color-blind friendly badges.

```
app/
├── static/
│   ├── css/
│   │   ├── variables.css      # Design tokens, color palette, typography & elevation
│   │   ├── style.css          # Base resets, app container, sidebar, topbar, modals & toasts
│   │   ├── dashboard.css      # Metric cards, quick match box, data tables & activity list
│   │   └── matching.css       # Match toolbar, result cards, score gauges & comparison drawer
│   └── js/
│       ├── api.js             # Centralized API service wrapping fetch with error handling
│       ├── ui.js              # Toast notifications, modal handlers, badges & formatters
│       ├── matching.js        # Multi-job matching executor, client filtering & comparison logic
│       └── app.js             # Application orchestrator, hash routing & form submissions
└── templates/
    └── index.html             # Master responsive HTML5 layout and modal containers
```

---

## 3. Implemented Views & Navigation

| View / Section | Hash Route | Description |
|---|---|---|
| **Dashboard** | `#dashboard` | 4 metric KPI cards (Resumes, Jobs, Matches, Model), quick matching launcher, system architecture breakdown, and recent activity feed. |
| **Candidate Resumes** | `#resumes` | Data table of all candidate profiles with extracted skills chips, education summary, and direct "View Details" & "Match" action buttons. |
| **Available Jobs** | `#jobs` | Data table of benchmark job roles with required competencies chips and "View Details" modal launcher. |
| **Find Matching Jobs** | `#matching` | Primary workflow view: resume selector, Top-K filter, Match/Non-Match toggle, keyword search, summary overview banner, and ranked result cards. |
| **Match History** | `#history` | Historical resume-job evaluations retrieved from MySQL `matches` table. |
| **How Matching Works** | `#about` | Step-by-step pipeline walkthrough and academic explainability guidelines. |

---

## 4. Reusable UI Components

1. **`DecisionScore`**: Formats the LinearSVC hyperplane distance with explicit polarity signs (`+1.4521` or `-0.3120`), highlighting positive vs negative separation without misleading probability wording.
2. **`MatchStatus`**: Renders `✓ MATCH` (green pill) for `is_match=True` and `○ NON-MATCH` (neutral pill) for `is_match=False` based solely on the server response.
3. **`SkillChips`**: Categorized badges for `matched` competencies (green with checkmark) and `missing` competencies (dashed outline).
4. **`SkillOverlap`**: Auxiliary overlap percentage (`|Resume ∩ Job| / |Job|`), explicitly labeled as keyword overlap rather than model confidence.
5. **`ComparisonModal`**: Side-by-side comparison drawer supporting up to 3 selected jobs with ranked positions, decision scores, and skill gap tables.
6. **`Toast` & `ModalBackdrop`**: Non-blocking toast notifications for success/error alerts and accessible modal dialogs with backdrop click/escape closing.

---

## 5. API Integration & Service Layer

All frontend network requests are encapsulated in `app/static/js/api.js`:

| Client Method | HTTP Request | Description |
|---|---|---|
| `ApiService.getHealth()` | `GET /api/health` | System health, database connection, and model readiness. |
| `ApiService.getStats()` | `GET /api/stats` | Live aggregate metrics for resumes, jobs, and evaluations. |
| `ApiService.getResumes()` | `GET /api/resumes` | Candidate resumes list with skills and education. |
| `ApiService.getResume(id)` | `GET /api/resumes/<id>` | Single resume structured NLP metadata. |
| `ApiService.uploadResume(file, id)` | `POST /api/resumes/upload` | Multipart PDF resume upload and automatic NLP extraction. |
| `ApiService.getJobs()` | `GET /api/jobs` | Benchmark jobs with required competencies. |
| `ApiService.getJob(id)` | `GET /api/jobs/<id>` | Full job description and requirements. |
| `ApiService.matchResumeMultiJob(id, opts)` | `POST /api/resumes/<id>/svm-matches` | Multi-job batch matching with `top_k` and `match_only` options. |
| `ApiService.getMatchHistory()` | `GET /api/matches/history` | Historical match records persisted in MySQL. |

---

## 6. Primary End-to-End Demonstration Workflow

```
[Open Application] → http://127.0.0.1:5000/dashboard
        ↓
[Upload Resume] → Select candidate PDF in Modal → Processed via pypdf & spaCy
        ↓
[Select Candidate] → Choose from dropdown in "Find Matching Jobs" view
        ↓
[Execute Multi-Job Match] → Single request to POST /api/resumes/<id>/svm-matches (~83ms)
        ↓
[Ranked Results Rendered] →
    #1 Job Role (Decision Score: +1.4521, MATCH, 75% Skill Overlap)
       ✓ Matched Skills: [Python, Machine Learning, SQL]
       ○ Missing Skills: [Docker]
    #2 Job Role (Decision Score: +0.9832, MATCH, 60% Skill Overlap)
       ...
        ↓
[Inspect Details & Compare] → Open detail modal or select up to 3 jobs for side-by-side analysis
```

---

## 7. Security, Privacy & Error Handling

1. **No Client-Side Secrets**: MySQL credentials, database hostnames, and internal filesystem paths are strictly kept on the server.
2. **No Model File Exposure**: `.joblib` model files and TF-IDF vocabulary matrices are never downloaded to the browser.
3. **Sanitized User Output**: All dynamic text injected into the DOM is escaped via `UI.escapeHtml` to prevent Cross-Site Scripting (XSS).
4. **PII Protection**: Raw phone numbers, physical addresses, and unparsed candidate PII are omitted from general card listings.
5. **Graceful Network Error Handling**: Server errors (400, 404, 500) and network disconnects trigger user-friendly toast alerts rather than unhandled browser crashes.

---

## 8. Verification & Regression Testing

### Test Suite Execution
```powershell
.venv\Scripts\python.exe -m pytest -q
```

### Exact Results:
- **Backend Baseline (Steps 1–10):** `218 / 218 PASSED` (Zero Regressions)
- **Step 11 Frontend Routes & API Tests:** `8 / 8 PASSED`
- **Total Combined Suite:** **`226 / 226 PASSED` (100% Pass Rate)**
- **Execution Time:** ~164 seconds

---

## 9. Files Created & Modified

### Files Created:
1. `app/static/css/variables.css`: Global CSS design tokens, HSL palette, typography, and elevation scales.
2. `app/static/css/style.css`: Base layout, sidebar navigation, topbar, buttons, modals, and toasts.
3. `app/static/css/dashboard.css`: Metric cards, quick match box, status indicators, and data tables.
4. `app/static/css/matching.css`: Match toolbar, ranked result cards, decision score displays, and comparison grid.
5. `app/static/js/api.js`: Centralized API service layer wrapping fetch with error handling.
6. `app/static/js/ui.js`: UI helper utilities, toasts, modal controllers, and score formatters.
7. `app/static/js/matching.js`: Match controller, multi-job executor, search filters, and comparison drawer.
8. `app/static/js/app.js`: Master application orchestrator, hash routing, and event bindings.
9. `app/templates/index.html`: Master HTML5 application template with modals and views.
10. `tests/test_frontend_routes.py`: Unit and integration tests for frontend routes and supporting APIs.
11. `docs/STEP_11.md`: Step 11 architecture, workflow, and verification documentation.

### Files Modified:
1. `app/routes.py`: Added `GET /api/resumes`, `GET /api/jobs`, `GET /api/stats`, `GET /api/matches/history`, and UI aliases (`/dashboard`, `/app`, `/ui`).
2. `README.md`: Updated with Step 11 capabilities, user flows, and run instructions.

---

## 10. Step 12 Readiness

With Step 11 complete, the system provides a functional and validated UI over the frozen Step 10 matching engine. Future production hardening (Step 12) may include:
- Role-based authentication & JWT session authorization (Recruiter vs Candidate logins).
- Webhook/Email notifications when high-scoring candidates match open requisitions.
- Automated model monitoring and drift detection dashboards.
- Production containerization (Docker, Gunicorn, NGINX reverse proxy).
