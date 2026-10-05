/**
 * Matching Controller & Results Renderer
 * AI-Based Resume Screening and Job Matching System
 */

class MatchingController {
    constructor() {
        this.currentResults = [];
        this.currentResumeId = null;
        this.selectedForCompare = new Set();
        this.activeFilter = 'all'; // 'all' | 'match_only' | 'non_match'
        this.searchQuery = '';
    }

    /**
     * Trigger Multi-Job Matching for selected resume
     */
    async executeMatching(resumeId, options = {}) {
        if (!resumeId) {
            UI.showToast('Please select a resume first.', 'error');
            return;
        }

        this.currentResumeId = resumeId;
        const resultsContainer = document.getElementById('match-results-container');
        const emptyState = document.getElementById('match-empty-state');
        const loadingState = document.getElementById('match-loading-state');
        const summaryBanner = document.getElementById('match-summary-banner');

        if (emptyState) emptyState.style.display = 'none';
        if (summaryBanner) summaryBanner.style.display = 'none';
        if (resultsContainer) resultsContainer.innerHTML = '';
        if (loadingState) loadingState.style.display = 'flex';

        try {
            const data = await ApiService.matchResumeMultiJob(resumeId, options);
            
            // Backend returns: { resume_id, total_jobs_evaluated, total_matches, top_k, results: [...] }
            this.currentResults = data.results || [];
            
            if (loadingState) loadingState.style.display = 'none';

            this.renderSummaryBanner(data);
            this.renderFilteredResults();
            const evaluatedCount = data.total_jobs_evaluated ?? this.currentResults.length;
            const matchCount = data.total_predicted_matches ?? data.total_matches ?? this.currentResults.filter(r => r.is_match).length;
            UI.showToast(`Evaluated ${evaluatedCount} jobs. ${matchCount} predicted matches found.`, 'success');
        } catch (error) {
            if (loadingState) loadingState.style.display = 'none';
            if (emptyState) emptyState.style.display = 'flex';
            UI.showToast(`Matching calculation failed: ${error.message}`, 'error');
        }
    }

    /**
     * Render the Top Summary Banner
     */
    renderSummaryBanner(data) {
        const banner = document.getElementById('match-summary-banner');
        if (!banner) return;

        banner.style.display = 'flex';
        const resLabel = window.App ? window.App.getResumeLabel(data.resume_id) : data.resume_id;
        document.getElementById('summary-resume-name').textContent = `Resume: ${resLabel}`;
        document.getElementById('summary-jobs-evaluated').textContent = data.total_jobs_evaluated ?? this.currentResults.length;
        document.getElementById('summary-predicted-matches').textContent = data.total_predicted_matches ?? data.total_matches ?? this.currentResults.filter(r => r.is_match).length;
        document.getElementById('summary-top-score').textContent = this.currentResults.length > 0 ? UI.formatDecisionScore(this.currentResults[0].decision_score) : '0.0000';
    }

    /**
     * Filter and render match cards
     */
    renderFilteredResults() {
        const container = document.getElementById('match-results-container');
        if (!container) return;

        let filtered = [...this.currentResults];

        // Apply Match Filter
        if (this.activeFilter === 'match_only') {
            filtered = filtered.filter(item => item.is_match === true);
        } else if (this.activeFilter === 'non_match') {
            filtered = filtered.filter(item => item.is_match === false);
        }

        // Apply Search Filter
        if (this.searchQuery) {
            const q = this.searchQuery.toLowerCase();
            filtered = filtered.filter(item => 
                (item.job_title && item.job_title.toLowerCase().includes(q)) ||
                (item.job_id && item.job_id.toLowerCase().includes(q))
            );
        }

        if (filtered.length === 0) {
            container.innerHTML = `
                <div class="state-container">
                    <i class="fa-solid fa-filter-circle-xmark state-icon"></i>
                    <h3 class="state-title">No Matching Results Found</h3>
                    <p class="state-description">No evaluated jobs matched the current filter or search criteria.</p>
                </div>
            `;
            return;
        }

        container.innerHTML = filtered.map(item => this.createResultCardHtml(item)).join('');
    }

    /**
     * HTML template for an individual ranked job match card
     */
    createResultCardHtml(item) {
        const isMatch = item.is_match;
        const scoreClass = item.decision_score >= 0 ? 'score-positive' : 'score-negative';
        const isTop = item.rank === 1;
        const isSelected = this.selectedForCompare.has(item.job_id);

        return `
            <div class="match-result-card ${isMatch ? 'is-match-card' : 'is-nonmatch-card'}" id="match-card-${item.job_id}">
                <div class="result-card-header">
                    <div class="result-title-section">
                        <div class="rank-badge ${isTop ? 'top-rank' : ''}">#${item.rank}</div>
                        <div class="result-job-meta">
                            <h4>${UI.escapeHtml(item.job_title || 'Untitled Job')}</h4>
                            <div class="job-id-tag">Job ID: ${UI.escapeHtml(item.job_id)}</div>
                        </div>
                    </div>

                    <div class="result-scores-section">
                        <div class="score-display-box">
                            <span class="score-display-label">Decision Score</span>
                            <span class="score-display-value ${scoreClass}">${UI.formatDecisionScore(item.decision_score)}</span>
                        </div>
                        <div class="overlap-display-box">
                            <span class="score-display-label">Skill Overlap</span>
                            <span class="score-display-value" style="color: var(--text-secondary);">${UI.renderSkillOverlap(item.skill_overlap_ratio)}</span>
                        </div>
                        <div>
                            ${UI.renderMatchBadge(isMatch)}
                        </div>
                    </div>
                </div>

                <div class="result-skills-breakdown">
                    <div>
                        <div class="skill-category-title title-matched">
                            <i class="fa-solid fa-circle-check"></i> Matched Skills (${(item.matched_skills || []).length})
                        </div>
                        <div class="skill-chips">
                            ${UI.renderSkillChips(item.matched_skills, 'matched')}
                        </div>
                    </div>
                    <div>
                        <div class="skill-category-title title-missing">
                            <i class="fa-regular fa-circle-xmark"></i> Missing Skills (${(item.missing_skills || []).length})
                        </div>
                        <div class="skill-chips">
                            ${UI.renderSkillChips(item.missing_skills, 'missing')}
                        </div>
                    </div>
                </div>

                <div class="result-card-actions">
                    <button class="btn btn-secondary btn-sm" onclick="matchingController.toggleCompare('${item.job_id}')">
                        <i class="fa-solid ${isSelected ? 'fa-square-check' : 'fa-square'}"></i> ${isSelected ? 'In Compare' : 'Compare'}
                    </button>
                    <button class="btn btn-outline-brand btn-sm" onclick="matchingController.viewDetails('${item.job_id}')">
                        <i class="fa-solid fa-circle-info"></i> View Details
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * View detailed breakdown modal for a result
     */
    async viewDetails(jobId) {
        const item = this.currentResults.find(r => r.job_id === jobId);
        if (!item) return;

        document.getElementById('modal-job-title').textContent = item.job_title || jobId;
        document.getElementById('modal-job-id').textContent = item.job_id;
        document.getElementById('modal-decision-score').textContent = UI.formatDecisionScore(item.decision_score);
        document.getElementById('modal-match-status').innerHTML = UI.renderMatchBadge(item.is_match);
        document.getElementById('modal-skill-overlap').textContent = UI.renderSkillOverlap(item.skill_overlap_ratio);
        document.getElementById('modal-matched-skills').innerHTML = UI.renderSkillChips(item.matched_skills, 'matched');
        document.getElementById('modal-missing-skills').innerHTML = UI.renderSkillChips(item.missing_skills, 'missing');

        // Fetch full job description if available
        try {
            const jobData = await ApiService.getJob(jobId);
            const job = jobData.job || jobData;
            document.getElementById('modal-job-desc').textContent = job.description || 'No description provided.';
            document.getElementById('modal-job-skills').innerHTML = UI.renderSkillChips(job.required_skills || job.skills || [], 'neutral');
        } catch (e) {
            document.getElementById('modal-job-desc').textContent = 'Unable to load full job description.';
        }

        UI.openModal('modal-match-details');
    }

    /**
     * Toggle item for side-by-side comparison
     */
    toggleCompare(jobId) {
        if (this.selectedForCompare.has(jobId)) {
            this.selectedForCompare.delete(jobId);
        } else {
            if (this.selectedForCompare.size >= 3) {
                UI.showToast('You can compare up to 3 jobs at a time.', 'info');
                return;
            }
            this.selectedForCompare.add(jobId);
        }

        this.updateCompareButton();
        this.renderFilteredResults();
    }

    /**
     * Update Floating Compare Button State
     */
    updateCompareButton() {
        const btn = document.getElementById('btn-open-compare');
        if (!btn) return;

        const count = this.selectedForCompare.size;
        if (count > 0) {
            btn.style.display = 'inline-flex';
            btn.innerHTML = `<i class="fa-solid fa-code-compare"></i> Compare Selected (${count})`;
        } else {
            btn.style.display = 'none';
        }
    }

    /**
     * Open Comparison Modal
     */
    showComparisonModal() {
        const compareItems = this.currentResults.filter(r => this.selectedForCompare.has(r.job_id));
        const grid = document.getElementById('comparison-grid');
        if (!grid) return;

        if (compareItems.length === 0) {
            UI.showToast('No jobs selected for comparison.', 'info');
            return;
        }

        grid.innerHTML = compareItems.map(item => `
            <div class="compare-card">
                <div class="compare-header">
                    <h4>${UI.escapeHtml(item.job_title)}</h4>
                    <span class="text-tertiary" style="font-size:0.75rem; font-family:var(--font-mono);">${UI.escapeHtml(item.job_id)}</span>
                </div>
                <div>
                    <div class="compare-stat-row">
                        <span>Rank:</span>
                        <strong>#${item.rank}</strong>
                    </div>
                    <div class="compare-stat-row">
                        <span>Decision Score:</span>
                        <strong class="${item.decision_score >= 0 ? 'score-positive' : 'score-negative'}">${UI.formatDecisionScore(item.decision_score)}</strong>
                    </div>
                    <div class="compare-stat-row">
                        <span>Match Status:</span>
                        <div>${UI.renderMatchBadge(item.is_match)}</div>
                    </div>
                    <div class="compare-stat-row">
                        <span>Skill Overlap:</span>
                        <strong>${UI.renderSkillOverlap(item.skill_overlap_ratio)}</strong>
                    </div>
                </div>
                <div>
                    <div class="skill-category-title title-matched"><i class="fa-solid fa-check"></i> Matched Skills:</div>
                    <div class="skill-chips">${UI.renderSkillChips(item.matched_skills, 'matched')}</div>
                </div>
                <div>
                    <div class="skill-category-title title-missing"><i class="fa-regular fa-circle"></i> Missing Skills:</div>
                    <div class="skill-chips">${UI.renderSkillChips(item.missing_skills, 'missing')}</div>
                </div>
            </div>
        `).join('');

        UI.openModal('modal-comparison');
    }
}

window.matchingController = new MatchingController();
