-- =====================================================================
-- MySQL Schema Definition
-- AI-Based Resume Screening and Job Matching System
-- Database: resume_job
-- =====================================================================

-- Set character set
SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ---------------------------------------------------------------------
-- 1. USERS TABLE
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id CHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    role ENUM('candidate', 'recruiter', 'admin') NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_users_email (email),
    INDEX idx_users_role (role)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------
-- 2. RESUMES TABLE
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS resumes (
    id CHAR(36) PRIMARY KEY,
    user_id CHAR(36) NULL,
    filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    extracted_text LONGTEXT NULL,
    education VARCHAR(100) NULL,
    degree VARCHAR(100) NULL,
    field_of_study VARCHAR(150) NULL,
    experience_years DECIMAL(4, 1) DEFAULT 0.0,
    uploaded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_resumes_user_id (user_id),
    INDEX idx_resumes_uploaded_at (uploaded_at),
    CONSTRAINT fk_resumes_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT chk_resumes_exp CHECK (experience_years >= 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------
-- 3. JOBS TABLE
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS jobs (
    id CHAR(36) PRIMARY KEY,
    recruiter_id CHAR(36) NULL,
    job_title VARCHAR(255) NOT NULL,
    company VARCHAR(255) NOT NULL,
    job_description LONGTEXT NOT NULL,
    job_category VARCHAR(100) NOT NULL,
    minimum_experience DECIMAL(4, 1) DEFAULT 0.0,
    education_requirement VARCHAR(255) NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_jobs_recruiter_id (recruiter_id),
    INDEX idx_jobs_category (job_category),
    INDEX idx_jobs_created_at (created_at),
    CONSTRAINT fk_jobs_recruiter FOREIGN KEY (recruiter_id) REFERENCES users(id) ON DELETE SET NULL,
    CONSTRAINT chk_jobs_exp CHECK (minimum_experience >= 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------
-- 4. SKILLS TABLE (Master Taxonomy)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS skills (
    id CHAR(36) PRIMARY KEY,
    skill_name VARCHAR(100) UNIQUE NOT NULL,
    category VARCHAR(100) NOT NULL,
    aliases TEXT NULL,
    INDEX idx_skills_name (skill_name),
    INDEX idx_skills_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------
-- 5. RESUME_SKILLS JUNCTION TABLE
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS resume_skills (
    resume_id CHAR(36) NOT NULL,
    skill_id CHAR(36) NOT NULL,
    PRIMARY KEY (resume_id, skill_id),
    INDEX idx_resume_skills_skill (skill_id),
    CONSTRAINT fk_rs_resume FOREIGN KEY (resume_id) REFERENCES resumes(id) ON DELETE CASCADE,
    CONSTRAINT fk_rs_skill FOREIGN KEY (skill_id) REFERENCES skills(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------
-- 6. JOB_SKILLS JUNCTION TABLE
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS job_skills (
    job_id CHAR(36) NOT NULL,
    skill_id CHAR(36) NOT NULL,
    is_required BOOLEAN NOT NULL DEFAULT true,
    PRIMARY KEY (job_id, skill_id),
    INDEX idx_job_skills_skill (skill_id),
    INDEX idx_job_skills_required (is_required),
    CONSTRAINT fk_js_job FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    CONSTRAINT fk_js_skill FOREIGN KEY (skill_id) REFERENCES skills(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------
-- 7. MATCHES TABLE (Prediction & Scoring Results)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS matches (
    id CHAR(36) PRIMARY KEY,
    resume_id CHAR(36) NOT NULL,
    job_id CHAR(36) NOT NULL,
    similarity_score DECIMAL(5, 4) NULL,
    skill_score DECIMAL(5, 4) NULL,
    overall_score DECIMAL(5, 4) NULL,
    model_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_matches_resume (resume_id),
    INDEX idx_matches_job (job_id),
    INDEX idx_matches_overall (overall_score),
    INDEX idx_matches_created_at (created_at),
    CONSTRAINT fk_matches_resume FOREIGN KEY (resume_id) REFERENCES resumes(id) ON DELETE CASCADE,
    CONSTRAINT fk_matches_job FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

SET FOREIGN_KEY_CHECKS = 1;
