/**
 * Main Application Orchestrator
 * AI-Based Resume Screening and Job Matching System
 */

document.addEventListener('DOMContentLoaded', () => {
    App.init();
});

class App {
    static async init() {
        this.bindNavigation();
        this.bindModals();
        this.bindForms();
        this.bindFilterEvents();
        
        // Initial data load
        await this.checkHealth();
        await this.loadStats();
        await this.loadResumes();
        await this.loadJobs();
        await this.loadMatchHistory();

        // Handle initial hash routing
        this.handleHashRoute();
        window.addEventListener('hashchange', () => this.handleHashRoute());
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
            if (pill && health.status === 'ok') {
                pill.innerHTML = `<span class="status-dot"></span> System Online (${health.model?.name || 'SVM'})`;
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
            document.getElementById('stat-total-resumes').textContent = stats.total_resumes ?? 0;
            document.getElementById('stat-total-jobs').textContent = stats.total_jobs ?? 0;
            document.getElementById('stat-total-matches').textContent = stats.total_matches ?? 0;
            document.getElementById('stat-model-name').textContent = stats.model ?? 'tfidf_svm';
        } catch (e) {
            console.error('Failed to load stats:', e);
        }
    }

    static async loadResumes() {
        try {
            const data = await ApiService.getResumes();
            const resumes = data.resumes || [];

            // Populate Dropdowns
            const dashSelect = document.getElementById('dashboard-resume-select');
            const matchSelect = document.getElementById('match-resume-select');
            
            const optionsHtml = resumes.length === 0 
                ? '<option value="">No resumes found. Please upload one.</option>'
                : '<option value="">-- Select a Resume --</option>' + resumes.map(r => `
                    <option value="${UI.escapeHtml(r.resume_id)}">${UI.escapeHtml(r.name || r.resume_id)} (${r.skills ? r.skills.length : 0} skills)</option>
                  `).join('');

            if (dashSelect) dashSelect.innerHTML = optionsHtml;
            if (matchSelect) matchSelect.innerHTML = optionsHtml;

            // Render Resumes Table
            const tableBody = document.getElementById('resumes-table-body');
            if (tableBody) {
                if (resumes.length === 0) {
                    tableBody.innerHTML = `<tr><td colspan="5" class="text-center py-4" style="text-align:center; padding: 2rem;">No resumes available. Click "Upload Resume" to add one.</td></tr>`;
                } else {
                    tableBody.innerHTML = resumes.map(r => `
                        <tr>
                            <td><strong>${UI.escapeHtml(r.resume_id)}</strong></td>
                            <td>${UI.escapeHtml(r.name || 'Candidate')}</td>
                            <td>${UI.renderSkillChips(r.skills ? r.skills.slice(0, 4) : [], 'neutral')} ${(r.skills && r.skills.length > 4) ? `<span class="badge badge-neutral">+${r.skills.length - 4} more</span>` : ''}</td>
                            <td><span class="badge badge-brand">${r.education && r.education.length > 0 ? UI.escapeHtml(r.education[0]) : 'Profile Parsed'}</span></td>
                            <td>
                                <button class="btn btn-outline-brand btn-sm" onclick="App.viewResume('${r.resume_id}')">
                                    <i class="fa-solid fa-eye"></i> View Details
                                </button>
                                <button class="btn btn-primary btn-sm" onclick="App.quickMatchResume('${r.resume_id}')">
                                    <i class="fa-solid fa-wand-magic-sparkles"></i> Match
                                </button>
                            </td>
                        </tr>
                    `).join('');
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
                    listContainer.innerHTML = history.slice(0, 5).map(h => `
                        <div class="activity-item">
                            <div class="activity-left">
                                <div class="activity-icon"><i class="fa-solid fa-bolt"></i></div>
                                <div class="activity-content">
                                    <span class="activity-title">${UI.escapeHtml(h.resume_id)} &rarr; ${UI.escapeHtml(h.job_title || h.job_id)}</span>
                                    <span class="activity-meta">Score: ${UI.formatDecisionScore(h.decision_score)} &bull; ${h.created_at ? new Date(h.created_at).toLocaleDateString() : 'Recent'}</span>
                                </div>
                            </div>
                            <div>${UI.renderMatchBadge(h.is_match)}</div>
                        </div>
                    `).join('');
                }
            }

            const historyTableBody = document.getElementById('history-table-body');
            if (historyTableBody) {
                if (history.length === 0) {
                    historyTableBody.innerHTML = `<tr><td colspan="6" class="text-center py-4" style="text-align:center; padding: 2rem;">No matching history recorded yet.</td></tr>`;
                } else {
                    historyTableBody.innerHTML = history.map(h => `
                        <tr>
                            <td><strong>${UI.escapeHtml(h.resume_id)}</strong></td>
                            <td><strong>${UI.escapeHtml(h.job_title || h.job_id)}</strong></td>
                            <td><code>${UI.escapeHtml(h.model || 'tfidf_svm')}</code></td>
                            <td><strong>${UI.formatDecisionScore(h.decision_score)}</strong></td>
                            <td>${UI.renderMatchBadge(h.is_match)}</td>
                            <td><span class="text-tertiary">${h.created_at ? new Date(h.created_at).toLocaleString() : 'N/A'}</span></td>
                        </tr>
                    `).join('');
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
            document.getElementById('modal-resume-id').textContent = data.resume_id;
            document.getElementById('modal-resume-name').textContent = data.name || 'Candidate Profile';
            document.getElementById('modal-resume-skills').innerHTML = UI.renderSkillChips(data.skills, 'neutral');
            
            const eduList = document.getElementById('modal-resume-education');
            if (data.education && data.education.length > 0) {
                eduList.innerHTML = data.education.map(e => `<li>${UI.escapeHtml(e)}</li>`).join('');
            } else {
                eduList.innerHTML = '<li>No formal education records parsed.</li>';
            }

            const expList = document.getElementById('modal-resume-experience');
            if (data.experience && data.experience.length > 0) {
                expList.innerHTML = data.experience.map(e => `<li>${UI.escapeHtml(e)}</li>`).join('');
            } else {
                expList.innerHTML = '<li>Experience details extracted via NLP.</li>';
            }

            UI.openModal('modal-resume-details');
        } catch (e) {
            UI.showToast(`Failed to load resume details: ${e.message}`, 'error');
        }
    }

    static async viewJob(jobId) {
        try {
            const data = await ApiService.getJob(jobId);
            document.getElementById('modal-single-job-id').textContent = data.job_id;
            document.getElementById('modal-single-job-title').textContent = data.job_title;
            document.getElementById('modal-single-job-desc').textContent = data.description || 'No description available.';
            document.getElementById('modal-single-job-skills').innerHTML = UI.renderSkillChips(data.required_skills, 'neutral');

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
