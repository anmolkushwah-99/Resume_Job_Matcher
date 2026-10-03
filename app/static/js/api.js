/**
 * Centralized API Service Layer
 * AI-Based Resume Screening and Job Matching System
 */

const API_BASE = ''; // Same-origin relative path

class ApiService {
    /**
     * Internal unified fetch wrapper
     */
    static async request(endpoint, options = {}) {
        const url = `${API_BASE}${endpoint}`;
        const defaultHeaders = {
            'Accept': 'application/json'
        };

        // If body is not FormData, default to application/json
        if (options.body && !(options.body instanceof FormData)) {
            defaultHeaders['Content-Type'] = 'application/json';
        }

        const config = {
            ...options,
            headers: {
                ...defaultHeaders,
                ...(options.headers || {})
            }
        };

        try {
            const response = await fetch(url, config);
            let data = null;
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                data = await response.json();
            } else {
                data = { message: await response.text() };
            }

            if (!response.ok) {
                const errorMessage = data?.error || data?.message || `Request failed with status ${response.status}`;
                throw new Error(errorMessage);
            }

            return data;
        } catch (error) {
            console.error(`API Error [${endpoint}]:`, error);
            throw error;
        }
    }

    // --- Health & Stats ---
    static async getHealth() {
        return this.request('/api/health');
    }

    static async getStats() {
        return this.request('/api/stats');
    }

    // --- Resumes ---
    static async getResumes() {
        return this.request('/api/resumes');
    }

    static async getResume(resumeId) {
        return this.request(`/api/resumes/${encodeURIComponent(resumeId)}`);
    }

    static async uploadResume(file, resumeId = null) {
        const formData = new FormData();
        formData.append('file', file);
        if (resumeId) {
            formData.append('resume_id', resumeId);
        }
        return this.request('/api/resumes/upload', {
            method: 'POST',
            body: formData
        });
    }

    // --- Jobs ---
    static async getJobs() {
        return this.request('/api/jobs');
    }

    static async getJob(jobId) {
        return this.request(`/api/jobs/${encodeURIComponent(jobId)}`);
    }

    // --- Matching Engine (Step 10 Endpoints) ---
    /**
     * Run multi-job matching for a selected resume against all or top_k available jobs
     * Uses POST /api/resumes/<resume_id>/svm-matches
     */
    static async matchResumeMultiJob(resumeId, { top_k = null, match_only = false } = {}) {
        const payload = {};
        if (top_k !== null && top_k !== undefined) {
            payload.top_k = parseInt(top_k, 10);
        }
        if (match_only) {
            payload.match_only = true;
        }

        return this.request(`/api/resumes/${encodeURIComponent(resumeId)}/svm-matches`, {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }

    /**
     * Single pair match calculation
     */
    static async matchSinglePair(resumeId, jobId) {
        return this.request('/api/matches/svm', {
            method: 'POST',
            body: JSON.stringify({ resume_id: resumeId, job_id: jobId })
        });
    }

    // --- Match History ---
    static async getMatchHistory(limit = 50) {
        return this.request(`/api/matches/history?limit=${limit}`);
    }
}

window.ApiService = ApiService;
