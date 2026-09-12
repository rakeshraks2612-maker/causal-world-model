"""
PRISM Dashboard — Industrial SCADA / Palantir Foundry Design System.
High-density, precision-engineered UI for cyber-physical mission control.
Strictly zero cartoonish emojis; clean mathematical, engineering, and architectural typography.
"""

def get_custom_css() -> str:
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600;700&family=Geist:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --bg-main: #07090e;
        --bg-surface: #0c1017;
        --bg-surface-elevated: #121824;
        --bg-surface-subtle: rgba(18, 24, 36, 0.7);
        
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-medium: rgba(255, 255, 255, 0.15);
        --border-active: rgba(59, 130, 246, 0.5);
        
        --color-primary: #3b82f6;
        --color-cyan: #06b6d4;
        --color-emerald: #10b981;
        --color-amber: #f59e0b;
        --color-rose: #f43f5e;
        --color-slate: #64748b;
        
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --text-dim: #475569;
        
        --font-sans: 'Geist', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'Geist Mono', 'JetBrains Mono', monospace;
    }

    /* Base Reset & Background */
    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-sans) !important;
        letter-spacing: -0.01em;
    }

    /* Precision Sidebar */
    div[data-testid="stSidebar"] {
        background-color: var(--bg-surface) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-sans) !important;
        font-weight: 600 !important;
        letter-spacing: -0.025em !important;
        color: var(--text-primary) !important;
    }

    /* High-Density Tab Bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px !important;
        background: var(--bg-surface) !important;
        padding: 4px !important;
        border-radius: 8px !important;
        border: 1px solid var(--border-subtle) !important;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: var(--text-secondary) !important;
        font-family: var(--font-sans) !important;
        font-weight: 500 !important;
        font-size: 0.84rem !important;
        border-radius: 6px !important;
        padding: 6px 14px !important;
        transition: all 0.15s ease !important;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        background: var(--bg-surface-elevated) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        border: 1px solid var(--border-medium) !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4) !important;
    }

    /* SCADA Panel Container */
    .scada-panel {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 18px;
    }

    .scada-panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 10px;
        margin-bottom: 14px;
    }

    .scada-panel-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Industrial Metric Cell */
    .metric-cell {
        background: var(--bg-surface-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 12px 14px;
        position: relative;
    }

    .metric-cell-label {
        font-family: var(--font-sans);
        font-size: 0.70rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
        display: flex;
        justify-content: space-between;
    }

    .metric-cell-value {
        font-family: var(--font-mono);
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }

    .metric-cell-unit {
        font-size: 0.80rem;
        font-weight: 500;
        color: var(--text-secondary);
        margin-left: 2px;
    }

    .metric-cell-footer {
        font-family: var(--font-mono);
        font-size: 0.72rem;
        color: var(--text-secondary);
        margin-top: 6px;
        display: flex;
        justify-content: space-between;
        border-top: 1px solid rgba(255,255,255,0.04);
        padding-top: 4px;
    }

    /* Decision Status Banners */
    .decision-banner {
        border-radius: 8px;
        padding: 20px 22px;
        margin-bottom: 18px;
        border: 1px solid;
    }

    .decision-banner-recommended {
        background: linear-gradient(180deg, rgba(16, 185, 129, 0.08) 0%, rgba(12, 16, 23, 0.95) 100%);
        border-color: rgba(16, 185, 129, 0.35);
    }

    .decision-banner-unsafe {
        background: linear-gradient(180deg, rgba(244, 63, 94, 0.08) 0%, rgba(12, 16, 23, 0.95) 100%);
        border-color: rgba(244, 63, 94, 0.4);
    }

    .decision-banner-abstain {
        background: linear-gradient(180deg, rgba(245, 158, 11, 0.08) 0%, rgba(12, 16, 23, 0.95) 100%);
        border-color: rgba(245, 158, 11, 0.4);
    }

    .decision-banner-noop {
        background: linear-gradient(180deg, rgba(59, 130, 246, 0.08) 0%, rgba(12, 16, 23, 0.95) 100%);
        border-color: rgba(59, 130, 246, 0.35);
    }

    /* Technical Badges & Status Tags */
    .status-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 8px;
        border-radius: 4px;
        font-family: var(--font-mono);
        font-size: 0.70rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border: 1px solid;
    }

    .status-tag-emerald {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border-color: rgba(16, 185, 129, 0.3);
    }

    .status-tag-rose {
        background: rgba(244, 63, 94, 0.12);
        color: #fb7185;
        border-color: rgba(244, 63, 94, 0.35);
    }

    .status-tag-amber {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border-color: rgba(245, 158, 11, 0.3);
    }

    .status-tag-cyan {
        background: rgba(6, 182, 212, 0.12);
        color: #22d3ee;
        border-color: rgba(6, 182, 212, 0.3);
    }

    .status-tag-blue {
        background: rgba(59, 130, 246, 0.12);
        color: #60a5fa;
        border-color: rgba(59, 130, 246, 0.3);
    }

    .status-tag-slate {
        background: rgba(100, 116, 139, 0.15);
        color: #94a3b8;
        border-color: rgba(100, 116, 139, 0.3);
    }

    /* Micro Status Indicators */
    .indicator-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        display: inline-block;
    }
    .indicator-dot-green { background: #10b981; box-shadow: 0 0 6px rgba(16, 185, 129, 0.6); }
    .indicator-dot-red { background: #f43f5e; box-shadow: 0 0 6px rgba(244, 63, 94, 0.6); }
    .indicator-dot-amber { background: #f59e0b; box-shadow: 0 0 6px rgba(245, 158, 11, 0.6); }
    .indicator-dot-blue { background: #3b82f6; box-shadow: 0 0 6px rgba(59, 130, 246, 0.6); }

    /* Headroom Progress Meter */
    .meter-container {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 3px;
        height: 6px;
        width: 100%;
        overflow: hidden;
        margin-top: 8px;
    }

    .meter-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s ease;
    }

    /* Provenance Console */
    .provenance-box {
        background: #090c12;
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 12px 14px;
        font-family: var(--font-mono);
        font-size: 0.78rem;
    }

    .hash-display {
        color: var(--color-cyan);
        word-break: break-all;
        background: rgba(6, 182, 212, 0.06);
        padding: 6px 10px;
        border-radius: 4px;
        border: 1px solid rgba(6, 182, 212, 0.2);
        margin: 6px 0;
    }

    /* Clean Streamlit Overrides */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    .stButton>button {
        font-family: var(--font-sans) !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        border-radius: 6px !important;
        border: 1px solid var(--border-medium) !important;
        background: var(--bg-surface-elevated) !important;
        color: var(--text-primary) !important;
        transition: all 0.15s ease !important;
    }
    .stButton>button:hover {
        border-color: var(--color-primary) !important;
        color: #ffffff !important;
    }
    div[data-testid="stExpander"] {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 6px !important;
    }
    </style>
    """
