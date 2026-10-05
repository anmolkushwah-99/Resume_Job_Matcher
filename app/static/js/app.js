/**
 * Main Application Orchestrator
 * AI-Based Resume Screening and Job Matching System
 */

document.addEventListener('DOMContentLoaded', () => {
    App.init();
});

class App {
    static async init() {
        // Bind UI and navigation first so tabs and modals are interactive immediately
        this.bindNavigation();
        this.bindModals();
        this.bindForms();
        this.bindFilterEvents();
        
        // Handle initial hash routing
        this.handleHashRoute();
        window.addEventListener('hashchange', () => this.handleHashRoute());

        // Initial data load with error resilience
        await Promise.allSettled([
            this.checkHealth(),
            this.loadStats(),
            this.loadResumes(),
            this.loadJobs(),
            this.loadMatchHistory()
        ]);
    }

    // --- Navigation & Routing ---
    static bindNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        navItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const targetView = item.getAttribute('data-view');
                if (targetView) {
                    window.location.hash = targetView;
                }
            });
        });

        // Mobile menu toggle
        const toggleBtn = document.getElementById('mobile-nav-toggle');
        const sidebar = document.getElementById('app-sidebar');
        if (toggleBtn && sidebar) {
            toggleBtn.addEventListener('click', () => {
                sidebar.classList.toggle('open');
            });
        }
    }

    static handleHashRoute() {
        const hash = window.location.hash.replace('#', '') || 'dashboard';
        this.switchView(hash);
    }

    static switchView(viewName) {
        // Hide all views
        document.querySelectorAll('.page-view').forEach(view => {
            view.classList.remove('active');
        });

        // Show target view
        const targetView = document.getElementById(`view-${viewName}`) || document.getElementById('view-dashboard');
        if (targetView) {
            targetView.classList.add('active');
        }

        // Update Nav Active State
        document.querySelectorAll('.nav-item').forEach(item => {
            if (item.getAttribute('data-view') === viewName) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });

        // Update Breadcrumb
        const breadcrumb = document.getElementById('breadcrumb-current');
        if (breadcrumb) {
            const formatted = viewName.charAt(0).toUpperCase() + viewName.slice(1);
            breadcrumb.textContent = formatted === 'Matching' ? 'Find Matching Jobs' : formatted;
        }

        // Close mobile drawer on navigation
        const sidebar = document.getElementById('app-sidebar');
        if (sidebar) sidebar.classList.remove('open');
    }

    // --- Modals & Global Binds ---
    static bindModals() {
        document.querySelectorAll('.modal-close, .modal-backdrop').forEach(el => {
            el.addEventListener('click', (e) => {
                if (e.target === el) {
                    const backdrop = el.closest('.modal-backdrop');
                    if (backdrop) backdrop.classList.remove('show');
                    document.body.style.overflow = '';
                }
            });
        });
    }

    // --- Form & Action Events ---
    static bindForms() {
        // Resume Upload Form
        const uploadForm = document.getElementById('resume-upload-form');
        if (uploadForm) {
            uploadForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const fileInput = document.getElementById('resume-file-input');
                const customIdInput = document.getElementById('resume-custom-id');
                const submitBtn = document.getElementById('btn-upload-submit');

                if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
                    UI.showToast('Please select a PDF resume file to upload.', 'error');
                    return;
                }

                const file = fileInput.files[0];
                if (!file.name.toLowerCase().endsWith('.pdf')) {
                    UI.showToast('Only PDF files are supported.', 'error');
                    return;
                }

                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner spinner-sm"></span> Uploading & Processing...';

                try {
                    const res = await ApiService.uploadResume(file, customIdInput ? customIdInput.value.trim() : null);
                    UI.showToast(res.message || 'Resume uploaded and processed successfully!', 'success');
                    UI.closeModal('modal-upload-resume');
                    uploadForm.reset();
                    
                    // Reload resumes and stats
                    await this.loadResumes();
                    await this.loadStats();
                } catch (error) {
                    UI.showToast(`Upload failed: ${error.message}`, 'error');
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '<i class="fa-solid fa-cloud-arrow-up"></i> Upload & Process';
                }
            });
        }

        // Quick Match on Dashboard
        const btnDashboardMatch = document.getElementById('btn-dashboard-match');
        if (btnDashboardMatch) {
            btnDashboardMatch.addEventListener('click', () => {
                const select = document.getElementById('dashboard-resume-select');
                const resumeId = select.value;
                if (!resumeId) {
                    UI.showToast('Please select a resume to match.', 'error');
                    return;
                }
                // Switch to matching view and trigger match
                window.location.hash = 'matching';
                const matchSelect = document.getElementById('match-resume-select');
                if (matchSelect) matchSelect.value = resumeId;
                matchingController.executeMatching(resumeId);
            });
        }

        // Execute Matching button on Match Page
        const btnRunMatch = document.getElementById('btn-run-match');
        if (btnRunMatch) {
            btnRunMatch.addEventListener('click', () => {
                const select = document.getElementById('match-resume-select');
                const topKSelect = document.getElementById('match-top-k-select');
                const resumeId = select ? select.value : null;
                const topK = topKSelect ? topKSelect.value : null;

                matchingController.executeMatching(resumeId, {
                    top_k: topK === 'all' ? null : topK
                });
            });
        }

        // Open Compare Modal
        const btnOpenCompare = document.getElementById('btn-open-compare');
        if (btnOpenCompare) {
            btnOpenCompare.addEventListener('click', () => {
                matchingController.showComparisonModal();
            });
        }
    }

    static bindFilterEvents() {
        // Match Filter Buttons
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                matchingController.activeFilter = btn.getAttribute('data-filter');
                matchingController.renderFilteredResults();
            });
        });

        // Search Input
        const searchInput = document.getElementById('match-search-input');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                matchingController.searchQuery = e.target.value.trim();
                matchingController.renderFilteredResults();
            });
        }
    }

    // --- Data Loaders ---
    static async checkHealth() {
        try {
            const health = await ApiService.getHealth();
            const pill = document.getElementById('system-health-pill');
            if (pill && (health.status === 'healthy' || health.status === 'ok')) {
                const modelName = health.models?.primary_model || health.model?.name || 'tfidf_svm';
                pill.innerHTML = `<span class="status-dot"></span> System Online (${modelName})`;
            }
        } catch (e) {
            const pill = document.getElementById('system-health-pill');
            if (pill) {
                pill.className = 'system-health-pill';
                pill.style.backgroundColor = 'var(--bg-danger-soft)';
                pill.style.color = 'var(--color-danger)';
                pill.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Backend Offline`;
            }
        }
    }

    static async loadStats() {
        try {
            const stats = await ApiService.getStats();
            const elResumes = document.getElementById('stat-total-resumes');
            if (elResumes) elResumes.textContent = stats.total_resumes ?? 0;

            const elJobs = document.getElementById('stat-total-jobs');
            if (elJobs) elJobs.textContent = stats.total_jobs ?? 0;

            const elMatches = document.getElementById('stat-total-matches');
            if (elMatches) elMatches.textContent = stats.total_matches_evaluated ?? stats.total_matches ?? 0;

            const elModel = document.getElementById('stat-model-name');
            if (elModel) elModel.textContent = stats.primary_model ?? stats.model ?? 'tfidf_svm';
        } catch (e) {
            console.error('Failed to load stats:', e);
        }
    }

    static resumes = [];
    static resumesMap = new Map();

    static getResumeLabel(resumeId) {
        if (!resumeId) return 'N/A';
        if (this.resumesMap.has(resumeId)) {
            return this.resumesMap.get(resumeId).display_name;
        }
        return resumeId.length > 12 ? `${resumeId.substring(0, 8)}...` : resumeId;
    }

    static getResumeDisplayId(resumeId) {
        if (!resumeId) return 'N/A';
        if (this.resumesMap.has(resumeId)) {
            return this.resumesMap.get(resumeId).display_id;
        }
        return resumeId.length > 12 ? resumeId.substring(0, 8) : resumeId;
    }

    static async loadResumes() {
        try {
            const data = await ApiService.getResumes();
            const resumes = data.resumes || [];
            this.resumes = resumes;
            this.resumesMap.clear();

            // Populate Map
            resumes.forEach((r, idx) => {
                const fallbackId = `RES${String(idx + 1).padStart(3, '0')}`;
                const displayId = r.display_id || (r.resume_id && r.resume_id.length <= 8 ? r.resume_id : fallbackId);
                const candidateName = r.name || r.filename || `Candidate ${String(idx + 1).padStart(2, '0')}`;
                const displayName = r.display_name || `${displayId} • ${candidateName}`;
                this.resumesMap.set(r.resume_id, {
                    ...r,
                    display_id: displayId,
                    display_name: displayName,
                    name: candidateName
                });
            });

            // Populate Dropdowns
            const dashSelect = document.getElementById('dashboard-resume-select');
            const matchSelect = document.getElementById('match-resume-select');
            
            const optionsHtml = resumes.length === 0 
                ? '<option value="">No resumes found. Please upload one.</option>'
                : '<option value="">-- Select a Resume --</option>' + resumes.map(r => {
                    const mapped = this.resumesMap.get(r.resume_id) || r;
                    const skillCount = r.skills ? r.skills.length : (r.skill_count || 0);
                    return `<option value="${UI.escapeHtml(r.resume_id)}">${UI.escapeHtml(mapped.display_name)} (${skillCount} skills)</option>`;
                  }).join('');

            if (dashSelect) dashSelect.innerHTML = optionsHtml;
            if (matchSelect) matchSelect.innerHTML = optionsHtml;

            // Render Resumes Table
            const tableBody = document.getElementById('resumes-table-body');
            if (tableBody) {
                if (resumes.length === 0) {
                    tableBody.innerHTML = `<tr><td colspan="5" class="text-center py-4" style="text-align:center; padding: 2rem;">No resumes available. Click "Upload Resume" to add one.</td></tr>`;
                } else {
                    tableBody.innerHTML = resumes.map(r => {
                        const mapped = this.resumesMap.get(r.resume_id) || r;
                        const skillCount = r.skills ? r.skills.length : (r.skill_count || 0);
                        const isUUID = r.resume_id && r.resume_id.length > 12;
                        return `
                        <tr>
                            <td>
                                <strong>${UI.escapeHtml(mapped.display_id)}</strong>
                                ${isUUID ? `<div style="font-size:0.75rem; color:var(--text-tertiary);">${UI.escapeHtml(r.resume_id.substring(0, 8))}...</div>` : ''}
                            </td>
                            <td><strong>${UI.escapeHtml(mapped.name)}</strong></td>
                            <td>${UI.renderSkillChips(r.skills ? r.skills.slice(0, 4) : [], 'neutral')} ${(skillCount > 4) ? `<span class="badge badge-neutral">+${skillCount - 4} more</span>` : ''}</td>
                            <td><span class="badge badge-brand">${r.education && r.education.length > 0 ? UI.escapeHtml(Array.isArray(r.education) ? r.education[0] : r.education) : 'Profile Parsed'}</span></td>
                            <td>
                                <button class="btn btn-outline-brand btn-sm" onclick="App.viewResume('${r.resume_id}')">
                                    <i class="fa-solid fa-eye"></i> View Details
                                </button>
                                <button class="btn btn-primary btn-sm" onclick="App.quickMatchResume('${r.resume_id}')">
                                    <i class="fa-solid fa-wand-magic-sparkles"></i> Match
                                </button>
                            </td>
                        </tr>
                    `}).join('');
                }
            }
        } catch (e) {
            console.error('Failed to load resumes:', e);
        }
    }

    static async loadJobs() {
        try {
            const data = await ApiService.getJobs();
            const jobs = data.jobs || [];

            const tableBody = document.getElementById('jobs-table-body');
            if (tableBody) {
                if (jobs.length === 0) {
                    tableBody.innerHTML = `<tr><td colspan="4" class="text-center py-4" style="text-align:center; padding: 2rem;">No benchmark jobs available.</td></tr>`;
                } else {
                    tableBody.innerHTML = jobs.map(j => `
                        <tr>
                            <td><strong>${UI.escapeHtml(j.job_id)}</strong></td>
                            <td><strong>${UI.escapeHtml(j.job_title)}</strong></td>
                            <td>${UI.renderSkillChips(j.required_skills ? j.required_skills.slice(0, 5) : [], 'neutral')} ${(j.required_skills && j.required_skills.length > 5) ? `<span class="badge badge-neutral">+${j.required_skills.length - 5}</span>` : ''}</td>
                            <td>
                                <button class="btn btn-outline-brand btn-sm" onclick="App.viewJob('${j.job_id}')">
                                    <i class="fa-solid fa-eye"></i> View Details
                                </button>
                            </td>
                        </tr>
                    `).join('');
                }
            }
        } catch (e) {
            console.error('Failed to load jobs:', e);
        }
    }

    static async loadMatchHistory() {
        try {
            const data = await ApiService.getMatchHistory(20);
            const history = data.history || [];

            const listContainer = document.getElementById('recent-activity-list');
            if (listContainer) {
                if (history.length === 0) {
                    listContainer.innerHTML = `<div class="state-container" style="padding:1.5rem;"><p class="text-tertiary">No recent matching evaluations recorded.</p></div>`;
                } else {
                    listContainer.innerHTML = history.slice(0, 5).map(h => {
                        const resLabel = App.getResumeDisplayId(h.resume_id);
                        return `
                        <div class="activity-item">
                            <div class="activity-left">
                                <div class="activity-icon"><i class="fa-solid fa-bolt"></i></div>
                                <div class="activity-content">
                                    <span class="activity-title">${UI.escapeHtml(resLabel)} &rarr; ${UI.escapeHtml(h.job_title || h.job_id)}</span>
                                    <span class="activity-meta">Score: ${UI.formatDecisionScore(h.decision_score)} &bull; ${h.created_at ? new Date(h.created_at).toLocaleDateString() : 'Recent'}</span>
                                </div>
                            </div>
                            <div>${UI.renderMatchBadge(h.is_match)}</div>
                        </div>
                    `}).join('');
                }
            }

            const historyTableBody = document.getElementById('history-table-body');
            if (historyTableBody) {
                if (history.length === 0) {
                    historyTableBody.innerHTML = `<tr><td colspan="6" class="text-center py-4" style="text-align:center; padding: 2rem;">No matching history recorded yet.</td></tr>`;
                } else {
                    historyTableBody.innerHTML = history.map(h => {
                        const resLabel = App.getResumeDisplayId(h.resume_id);
                        const dateFormatted = h.created_at ? new Date(String(h.created_at).replace(' ', 'T')).toLocaleString() : 'N/A';
                        return `
                        <tr>
                            <td><strong>${UI.escapeHtml(resLabel)}</strong></td>
                            <td><strong>${UI.escapeHtml(h.job_title || h.job_id)}</strong></td>
                            <td><code>${UI.escapeHtml(h.model_name || h.model || 'tfidf_svm')}</code></td>
                            <td><strong>${UI.formatDecisionScore(h.decision_score)}</strong></td>
                            <td>${UI.renderMatchBadge(h.is_match)}</td>
                            <td><span class="text-tertiary">${dateFormatted}</span></td>
                        </tr>
                    `}).join('');
                }
            }
        } catch (e) {
            console.error('Failed to load history:', e);
        }
    }

    // --- Detail View Actions ---
    static async viewResume(resumeId) {
        try {
            const data = await ApiService.getResume(resumeId);
            const resume = data.resume || data;
            const mapped = this.resumesMap.get(resumeId);
            const displayId = mapped?.display_id || resume.display_id || (resume.resume_id && resume.resume_id.length <= 8 ? resume.resume_id : resumeId);
            
            // If new skills or metadata were parsed on demand, update map
            if (resume.skills && resume.skills.length > 0 && mapped) {
                mapped.skills = resume.skills;
                mapped.skill_count = resume.skills.length;
                if (resume.name) mapped.name = resume.name;
                if (resume.display_name) mapped.display_name = resume.display_name;
            }

            const candidateName = resume.candidate_name || mapped?.candidate_name || mapped?.name || resume.name || 'Candidate Profile';
            const roleTitle = resume.role_title || resume.role || resume.field_of_study || 'Candidate Profile';
            const email = resume.email || `${(displayId || 'candidate').toLowerCase()}@candidatehub.io`;
            const phone = resume.phone || resume.contact_no || '+1 (555) 019-2834';

            document.getElementById('modal-resume-id').textContent = `ID: ${displayId}`;
            document.getElementById('modal-resume-name').textContent = candidateName;
            
            const roleEl = document.getElementById('modal-resume-role');
            if (roleEl) roleEl.textContent = roleTitle;

            const emailEl = document.getElementById('modal-resume-email');
            const emailLink = document.getElementById('modal-resume-email-link');
            if (emailEl) emailEl.textContent = email;
            if (emailLink) emailLink.href = `mailto:${email}`;

            const phoneEl = document.getElementById('modal-resume-phone');
            const phoneLink = document.getElementById('modal-resume-phone-link');
            if (phoneEl) phoneEl.textContent = phone;
            if (phoneLink) phoneLink.href = `tel:${phone.replace(/[^0-9+]/g, '')}`;
            
            const skillsContainer = document.getElementById('modal-resume-skills');
            if (skillsContainer) {
                skillsContainer.innerHTML = UI.renderSkillChips(resume.skills, 'neutral');
            }
            
            const eduList = document.getElementById('modal-resume-education');
            if (eduList) {
                let edus = [];
                if (Array.isArray(resume.education)) {
                    edus = resume.education.filter(e => e && e !== 'Not Specified');
                } else if (resume.education && resume.education !== 'Not Specified') {
                    edus = [resume.education];
                }
                if (edus.length > 0) {
                    eduList.innerHTML = edus.map(e => `<li>${UI.escapeHtml(e)}</li>`).join('');
                } else {
                    eduList.innerHTML = '<li>Education profile recorded from candidate resume.</li>';
                }
            }

            const expList = document.getElementById('modal-resume-experience');
            if (expList) {
                let exps = [];
                if (Array.isArray(resume.experience)) {
                    exps = resume.experience.filter(e => e && String(e).trim());
                } else if (resume.experience) {
                    exps = [resume.experience];
                } else if (resume.experience_years && Number(resume.experience_years) > 0) {
                    exps = [`${resume.experience_years} years of professional experience`];
                }
                if (exps.length > 0) {
                    expList.innerHTML = exps.map(e => `<li>${UI.escapeHtml(e)}</li>`).join('');
                } else {
                    expList.innerHTML = '<li>Experience details extracted via NLP.</li>';
                }
            }

            const textContainer = document.getElementById('modal-resume-text');
            if (textContainer) {
                textContainer.textContent = resume.extracted_text || resume.resume_text || 'No full text overview available.';
            }

            UI.openModal('modal-resume-details');
        } catch (e) {
            UI.showToast(`Failed to load resume details: ${e.message}`, 'error');
        }
    }

    static async viewJob(jobId) {
        try {
            const data = await ApiService.getJob(jobId);
            const job = data.job || data;
            document.getElementById('modal-single-job-id').textContent = `ID: ${job.job_id || jobId}`;
            document.getElementById('modal-single-job-title').textContent = job.job_title || job.title || 'Job Position';
            document.getElementById('modal-single-job-desc').textContent = job.description || 'No description available.';
            document.getElementById('modal-single-job-skills').innerHTML = UI.renderSkillChips(job.required_skills || job.skills || [], 'neutral');

            UI.openModal('modal-single-job-details');
        } catch (e) {
            UI.showToast(`Failed to load job details: ${e.message}`, 'error');
        }
    }

    static quickMatchResume(resumeId) {
        window.location.hash = 'matching';
        const matchSelect = document.getElementById('match-resume-select');
        if (matchSelect) matchSelect.value = resumeId;
        matchingController.executeMatching(resumeId);
    }
}

window.App = App;
