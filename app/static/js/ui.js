/**
 * UI Utilities and Components Manager
 * AI-Based Resume Screening and Job Matching System
 */

class UI {
    /**
     * Show a transient toast notification
     */
    static showToast(message, type = 'info', duration = 3500) {
        let container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'toast-container';
            document.body.appendChild(container);
        }

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        const iconClass = type === 'success' ? 'fa-circle-check' : (type === 'error' ? 'fa-circle-exclamation' : 'fa-circle-info');
        toast.innerHTML = `
            <i class="fa-solid ${iconClass}"></i>
            <span>${this.escapeHtml(message)}</span>
        `;

        container.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            toast.style.transition = 'all 0.25s ease';
            setTimeout(() => toast.remove(), 250);
        }, duration);
    }

    /**
     * Open a modal dialog by ID
     */
    static openModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.add('show');
            document.body.style.overflow = 'hidden';
        }
    }

    /**
     * Close a modal dialog by ID
     */
    static closeModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.remove('show');
            document.body.style.overflow = '';
        }
    }

    /**
     * Format decision score strictly as decimal number (never probability)
     */
    static formatDecisionScore(score) {
        if (score === null || score === undefined || isNaN(score)) {
            return 'N/A';
        }
        const num = parseFloat(score);
        return (num > 0 ? '+' : '') + num.toFixed(4);
    }

    /**
     * Render Match / Non-Match Status Badge
     */
    static renderMatchBadge(isMatch) {
        if (isMatch) {
            return `<span class="badge badge-match"><i class="fa-solid fa-check"></i> MATCH</span>`;
        } else {
            return `<span class="badge badge-nonmatch"><i class="fa-solid fa-xmark"></i> NON-MATCH</span>`;
        }
    }

    /**
     * Render Skill Chips
     */
    static renderSkillChips(skills = [], type = 'neutral') {
        if (!skills || skills.length === 0) {
            return '<span class="text-tertiary text-sm italic" style="font-size: 0.8rem; color: var(--text-tertiary);">None identified</span>';
        }

        const className = type === 'matched' ? 'skill-chip matched' : (type === 'missing' ? 'skill-chip missing' : 'skill-chip');
        const icon = type === 'matched' ? '<i class="fa-solid fa-check" style="font-size:0.7rem;"></i> ' : (type === 'missing' ? '<i class="fa-regular fa-circle" style="font-size:0.7rem;"></i> ' : '');

        return skills.map(skill => `<span class="${className}">${icon}${this.escapeHtml(skill)}</span>`).join('');
    }

    /**
     * Render Skill Overlap Percentage with proper label
     */
    static renderSkillOverlap(ratio) {
        if (ratio === null || ratio === undefined || isNaN(ratio)) {
            return 'N/A';
        }
        const percent = Math.round(parseFloat(ratio) * 100);
        return `${percent}%`;
    }

    /**
     * Escape HTML helper for XSS prevention
     */
    static escapeHtml(str) {
        if (str === null || str === undefined) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
}

window.UI = UI;
