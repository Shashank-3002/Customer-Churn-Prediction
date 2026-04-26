# ============================================================
#  AI CUSTOMER RETENTION INTELLIGENCE  —  app.py
#  streamlit run app.py
# ============================================================

import io
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder

matplotlib.use("Agg")

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS  (dark futuristic theme) ──────────────────────
st.markdown("""
<style>
/* ── base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #080c14;
    color: #c8d8f0;
    font-family: 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background: #0b1120;
    border-right: 1px solid #1a2540;
}
[data-testid="stHeader"] { background: transparent; }

/* ── sidebar inputs ── */
.stSlider > div, .stNumberInput > div, .stSelectbox > div {
    background: transparent;
}
label { color: #7a9cc0 !important; font-size: 0.78rem !important; letter-spacing: .04em; }

/* ── metric cards ── */
[data-testid="metric-container"] {
    background: #0d1627;
    border: 1px solid #1e3050;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="metric-container"] label { color: #4a7aaa !important; font-size: 0.72rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e0eeff;
    font-size: 1.6rem !important;
    font-weight: 700;
}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0b1120;
    border-bottom: 1px solid #1a2540;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    color: #4a6a90;
    background: transparent;
    border-radius: 6px 6px 0 0;
    padding: 0.5rem 1.4rem;
    font-size: 0.82rem;
    letter-spacing: .05em;
}
.stTabs [aria-selected="true"] {
    color: #60aaff !important;
    background: #0d1627 !important;
    border-bottom: 2px solid #60aaff;
}

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1a3a6e, #0d2550);
    color: #90c0ff;
    border: 1px solid #2a4a80;
    border-radius: 8px;
    font-size: 0.82rem;
    letter-spacing: .06em;
    padding: 0.55rem 1.6rem;
    transition: all .2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e4080, #122d60);
    border-color: #60aaff;
    color: #c0e0ff;
}

/* ── dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #1a2540; border-radius: 10px; }

/* ── divider ── */
hr { border-color: #1a2540; }

/* ── custom boxes ── */
.risk-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: .08em;
    margin-bottom: 1rem;
}
.risk-high   { background: #2a0a0a; color: #ff6060; border: 1px solid #5a1a1a; }
.risk-medium { background: #2a1e00; color: #ffbb40; border: 1px solid #5a3e00; }
.risk-low    { background: #0a2010; color: #40cc80; border: 1px solid #1a5030; }

.insight-box {
    background: #0d1627;
    border: 1px solid #1e3050;
    border-left: 3px solid #60aaff;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #a0c0e0;
}
.insight-box b { color: #c0deff; }

.stat-row {
    display: flex;
    gap: 12px;
    margin: 1rem 0;
}
.stat-card {
    flex: 1;
    background: #0d1627;
    border: 1px solid #1e3050;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.stat-card .val { font-size: 1.5rem; font-weight: 700; color: #e0eeff; }
.stat-card .lbl { font-size: 0.7rem; color: #4a7aaa; letter-spacing: .06em; margin-top: 2px; }

.section-label {
    color: #4a7aaa;
    font-size: 0.7rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin: 1.5rem 0 0.5rem;
}

.stAlert { background: #0d1627 !important; border-color: #1e3050 !important; }
</style>
""", unsafe_allow_html=True)


# ── LOAD ARTIFACTS ───────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model     = joblib.load("churn_model.pkl")
    scaler    = joblib.load("scaler.pkl")
    features  = joblib.load("feature_names.pkl")
    threshold = joblib.load("threshold.pkl")
    return model, scaler, features, threshold

try:
    model, scaler, FEATURES, THRESHOLD = load_artifacts()
except FileNotFoundError:
    st.error("⚠️  Model files not found. Run `python train_model.py` first.")
    st.stop()


# ── HELPERS ──────────────────────────────────────────────────
def risk_label(p):
    if p > 0.70: return "HIGH RISK",   "risk-high"
    if p > 0.40: return "MEDIUM RISK", "risk-medium"
    return "LOW RISK", "risk-low"

def retention_strategy(p, profile):
    if p > 0.70:
        strategies = []
        if profile["num_products"] == 1:
            strategies.append("Cross-sell a second product (savings/insurance) — single-product customers churn 3× more.")
        if not profile["is_active"]:
            strategies.append("Reactivation campaign: personalised cashback offer for next 3 transactions.")
        if profile["balance"] > 80000:
            strategies.append("Assign a dedicated relationship manager — high-balance customer at serious risk.")
        if profile["age"] > 50:
            strategies.append("Offer premium support tier + fee waiver to reinforce loyalty.")
        if not strategies:
            strategies.append("Immediate outreach: loyalty reward + fee waiver + dedicated support upgrade.")
        return strategies
    if p > 0.40:
        return [
            "Targeted email campaign highlighting unused product benefits.",
            "Offer a limited-time interest rate boost on savings.",
            "Schedule a quarterly account review call.",
        ]
    return [
        "Maintain engagement with monthly loyalty newsletter.",
        "Enroll in referral programme — satisfied customers bring new ones.",
    ]

def shap_waterfall_fig(shap_values, feature_names, base_value):
    """Custom dark-themed SHAP waterfall chart."""
    vals  = list(shap_values)
    names = list(feature_names)

    # Sort by absolute magnitude
    order  = sorted(range(len(vals)), key=lambda i: abs(vals[i]))
    vals   = [vals[i]   for i in order]
    names  = [names[i]  for i in order]

    # Keep top 8 for readability
    if len(vals) > 8:
        rest     = sum(vals[:-8])
        vals     = [rest] + vals[-8:]
        names    = [f"other ({len(names)-8} features)"] + names[-8:]

    colors = ["#ff5555" if v > 0 else "#44bb77" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0d1627")
    ax.set_facecolor("#0d1627")

    bars = ax.barh(names, vals, color=colors, edgecolor="none", height=0.55)

    ax.axvline(0, color="#2a4060", linewidth=1)
    ax.set_xlabel("SHAP value  (impact on churn probability)", color="#4a7aaa", fontsize=8)
    ax.tick_params(colors="#7a9cc0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1a2540")
    ax.xaxis.label.set_color("#4a7aaa")

    # Value labels
    for bar, val in zip(bars, vals):
        x = bar.get_width()
        ax.text(
            x + (0.002 if x >= 0 else -0.002),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center", ha="left" if x >= 0 else "right",
            color="#c0d8f0", fontsize=7,
        )

    fig.tight_layout(pad=1.2)
    return fig

def gauge_fig(prob):
    """Semi-circle gauge for churn probability."""
    fig, ax = plt.subplots(figsize=(3.5, 2), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#080c14")
    ax.set_facecolor("#080c14")

    theta_start = np.pi
    theta_end   = 0
    theta_range = np.linspace(theta_start, theta_end, 200)

    # Background arc
    ax.plot(theta_range, [1]*200, color="#1a2540", linewidth=12, solid_capstyle="butt")

    # Value arc
    fill_end   = theta_start - prob * np.pi
    theta_fill = np.linspace(theta_start, fill_end, 200)
    color = "#ff4444" if prob > 0.70 else ("#ffaa00" if prob > 0.40 else "#33cc66")
    ax.plot(theta_fill, [1]*200, color=color, linewidth=12, solid_capstyle="butt")

    ax.set_ylim(0, 1.4)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_xlim(0, np.pi)
    ax.axis("off")

    ax.text(0, -0.1, f"{prob:.0%}", ha="center", va="center",
            fontsize=22, fontweight="700", color="#e0eeff",
            transform=ax.transData)
    ax.text(0, -0.45, "CHURN PROBABILITY", ha="center", va="center",
            fontsize=6, color="#4a7aaa", letter_spacing_workaround=None,
            transform=ax.transData)
    fig.tight_layout(pad=0)
    return fig


# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 1.5rem;'>
        <div style='font-size:1.3rem;font-weight:700;color:#60aaff;letter-spacing:.08em;'>⬡ CHURN.AI</div>
        <div style='font-size:0.7rem;color:#3a5a80;letter-spacing:.1em;margin-top:2px;'>RETENTION INTELLIGENCE v2</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Customer Profile</div>', unsafe_allow_html=True)

    credit_score     = st.number_input("Credit Score",        300, 900,    650, step=10)
    age              = st.slider(      "Age",                 18,  92,     35)
    tenure           = st.slider(      "Tenure (years)",      0,   10,     5)
    balance          = st.number_input("Account Balance ($)", 0.0, 300000.0, 50000.0, step=500.0)
    estimated_salary = st.number_input("Est. Annual Salary ($)", 0.0, 200000.0, 80000.0, step=500.0)

    st.markdown('<div class="section-label">Product & Activity</div>', unsafe_allow_html=True)

    num_products     = st.slider("Products Held",   1, 4, 2)
    has_cr_card      = st.selectbox("Credit Card",  ["Yes", "No"])
    is_active        = st.selectbox("Active Member",["Yes", "No"])

    st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)

    gender    = st.selectbox("Gender",    ["Male", "Female"])
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

    st.divider()
    predict_btn = st.button("⬡  RUN PREDICTION", use_container_width=True)

# ── BUILD INPUT ──────────────────────────────────────────────
gender_enc  = 1 if gender  == "Male"    else 0
geo_germany = 1 if geography == "Germany" else 0
geo_spain   = 1 if geography == "Spain"   else 0
has_cc_enc  = 1 if has_cr_card == "Yes" else 0
active_enc  = 1 if is_active   == "Yes" else 0

input_arr = np.array([[
    credit_score, gender_enc, age, tenure, balance,
    num_products, has_cc_enc, active_enc, estimated_salary,
    geo_germany, geo_spain,
]])
input_scaled = scaler.transform(input_arr)

profile = {
    "num_products": num_products,
    "is_active":    active_enc,
    "balance":      balance,
    "age":          age,
}


# ── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div style='padding:1.5rem 0 0.5rem;'>
    <div style='font-size:1.7rem;font-weight:700;color:#e0eeff;letter-spacing:.04em;'>
        Customer Retention Intelligence
    </div>
    <div style='font-size:0.8rem;color:#3a6090;letter-spacing:.08em;margin-top:4px;'>
        XGBoost · SHAP Explainability · SMOTE-balanced · F1-tuned threshold
    </div>
</div>
""", unsafe_allow_html=True)
st.divider()


# ── TABS ─────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["⬡  SINGLE PREDICTION", "⬡  BULK ANALYSIS"])


# ═══════════════════════════════════════════════════════════════
#  TAB 1 — SINGLE PREDICTION
# ═══════════════════════════════════════════════════════════════
with tab1:

    if not predict_btn:
        st.markdown("""
        <div style='text-align:center;padding:4rem 0;color:#2a4060;'>
            <div style='font-size:3rem;'>⬡</div>
            <div style='font-size:0.9rem;letter-spacing:.1em;margin-top:1rem;'>
                CONFIGURE CUSTOMER PROFILE IN SIDEBAR<br>THEN CLICK RUN PREDICTION
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── COMPUTE ──────────────────────────────────────────────
    prob      = float(model.predict_proba(input_scaled)[0][1])
    pred      = "Churn" if prob >= THRESHOLD else "Stay"
    risk_lbl, risk_cls = risk_label(prob)
    shap_vals = explainer.shap_values(input_scaled)[0]

    # ── TOP ROW ──────────────────────────────────────────────
    col_gauge, col_metrics = st.columns([1, 2])

    with col_gauge:
        # Probability gauge
        fig_g, ax_g = plt.subplots(figsize=(3.5, 2.2))
        fig_g.patch.set_facecolor("#0d1627")
        ax_g.set_facecolor("#0d1627")

        theta    = np.linspace(np.pi, 0, 300)
        ax_g     = plt.subplot(111, polar=True)
        fig_g.patch.set_facecolor("#0d1627")
        ax_g.set_facecolor("#0d1627")
        ax_g.plot(theta, [1]*300, color="#1a2540", linewidth=16, solid_capstyle="butt")

        fill_ang  = np.linspace(np.pi, np.pi - prob * np.pi, 300)
        bar_color = "#ff4444" if prob > 0.70 else ("#ffaa00" if prob > 0.40 else "#33cc66")
        ax_g.plot(fill_ang, [1]*300, color=bar_color, linewidth=16, solid_capstyle="butt")

        ax_g.set_ylim(0, 1.5)
        ax_g.set_xlim(0, np.pi)
        ax_g.axis("off")
        ax_g.text(np.pi/2, 0.1,  f"{prob:.0%}",          ha="center", fontsize=26, fontweight="700", color="#e0eeff")
        ax_g.text(np.pi/2, -0.4, "CHURN PROBABILITY",    ha="center", fontsize=6.5, color="#4a7aaa")
        fig_g.tight_layout(pad=0.5)
        st.pyplot(fig_g, use_container_width=True)
        plt.close(fig_g)

        st.markdown(f'<div style="text-align:center;"><span class="risk-badge {risk_cls}">{risk_lbl}</span></div>', unsafe_allow_html=True)

    with col_metrics:
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction",       pred)
        m2.metric("Threshold",        f"{THRESHOLD:.2f}")
        m3.metric("Est. LTV at Risk", f"${estimated_salary * 0.03 * max(1, 10-tenure):,.0f}")

        m4, m5, m6 = st.columns(3)
        m4.metric("Credit Score",  credit_score)
        m5.metric("Tenure",        f"{tenure} yrs")
        m6.metric("Products Held", num_products)

        # Quick flags
        flags = []
        if not active_enc:           flags.append("⚠️ Inactive member")
        if num_products == 1:        flags.append("⚠️ Single product only")
        if balance == 0:             flags.append("⚠️ Zero balance")
        if tenure < 2:               flags.append("⚠️ Low tenure")
        if credit_score < 500:       flags.append("⚠️ Low credit score")

        if flags:
            for f in flags[:3]:
                st.markdown(f'<div class="insight-box">{f}</div>', unsafe_allow_html=True)

    st.divider()

    # ── SHAP + STRATEGY ──────────────────────────────────────
    col_shap, col_strat = st.columns([1.1, 1])

    with col_shap:
        st.markdown('<div class="section-label">Feature Impact (SHAP)</div>', unsafe_allow_html=True)
        st.caption("🔴 pushes toward churn  ·  🟢 pushes toward staying")

        fig_w = shap_waterfall_fig(shap_vals, FEATURES, 0)
        st.pyplot(fig_w, use_container_width=True)
        plt.close(fig_w)

        # Top 3 drivers as text
        shap_dict = dict(zip(FEATURES, shap_vals))
        top3 = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        st.markdown('<div class="section-label">Top 3 Risk Drivers</div>', unsafe_allow_html=True)
        for feat, val in top3:
            arrow = "▲" if val > 0 else "▼"
            color = "#ff6060" if val > 0 else "#44cc77"
            st.markdown(
                f'<div class="insight-box">'
                f'<span style="color:{color};font-weight:700;">{arrow} {feat.replace("_"," ").title()}</span>'
                f' — SHAP {val:+.4f}</div>',
                unsafe_allow_html=True
            )

    with col_strat:
        st.markdown('<div class="section-label">Retention Strategy</div>', unsafe_allow_html=True)

        urgency_map = {
            "risk-high":   ("IMMEDIATE ACTION REQUIRED", "#ff4444"),
            "risk-medium": ("PROACTIVE OUTREACH RECOMMENDED", "#ffaa00"),
            "risk-low":    ("MAINTAIN ENGAGEMENT", "#33cc66"),
        }
        urgency_text, urgency_color = urgency_map[risk_cls]
        st.markdown(
            f'<div style="font-size:0.72rem;letter-spacing:.1em;color:{urgency_color};'
            f'font-weight:700;margin-bottom:1rem;">{urgency_text}</div>',
            unsafe_allow_html=True
        )

        strategies = retention_strategy(prob, profile)
        for i, s in enumerate(strategies, 1):
            st.markdown(
                f'<div class="insight-box"><b style="color:#60aaff;">#{i}</b> {s}</div>',
                unsafe_allow_html=True
            )

        st.divider()
        st.markdown('<div class="section-label">Customer Snapshot</div>', unsafe_allow_html=True)

        snap = {
            "Geography":   geography,
            "Gender":      gender,
            "Has Credit Card": has_cr_card,
            "Active Member":   is_active,
            "Balance":    f"${balance:,.0f}",
            "Salary":     f"${estimated_salary:,.0f}",
        }
        for k, v in snap.items():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:4px 0;border-bottom:1px solid #131e30;font-size:0.82rem;">'
                f'<span style="color:#4a7aaa;">{k}</span>'
                f'<span style="color:#c0d8f0;">{v}</span></div>',
                unsafe_allow_html=True
            )


# ═══════════════════════════════════════════════════════════════
#  TAB 2 — BULK CSV ANALYSIS
# ═══════════════════════════════════════════════════════════════
with tab2:

    st.markdown('<div class="section-label">Upload Customer Dataset</div>', unsafe_allow_html=True)
    st.caption("Upload a CSV matching the Churn_Modelling.csv format. Predictions run instantly on your machine — no data is sent anywhere.")

    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if not uploaded:
        st.markdown("""
        <div style='text-align:center;padding:3rem 0;color:#2a4060;'>
            <div style='font-size:2.5rem;'>⬡</div>
            <div style='font-size:0.85rem;letter-spacing:.08em;margin-top:1rem;'>
                DROP A CSV TO SCORE YOUR ENTIRE CUSTOMER BASE
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_raw = pd.read_csv(uploaded)
        st.success(f"✓  {len(df_raw):,} customers loaded")

        with st.spinner("Scoring customers ..."):
            df_work = df_raw.copy()
            drop_c  = [c for c in ["RowNumber","CustomerId","Surname","Exited"] if c in df_work.columns]
            df_work.drop(columns=drop_c, inplace=True, errors="ignore")

            if "Gender" in df_work.columns:
                df_work["Gender"] = LabelEncoder().fit_transform(df_work["Gender"])
            if "Geography" in df_work.columns:
                df_work = pd.get_dummies(df_work, columns=["Geography"], drop_first=True)
            for col in ["Geography_Germany", "Geography_Spain"]:
                if col not in df_work.columns:
                    df_work[col] = 0

            col_order = [
                "CreditScore","Gender","Age","Tenure","Balance",
                "NumOfProducts","HasCrCard","IsActiveMember",
                "EstimatedSalary","Geography_Germany","Geography_Spain",
            ]
            df_feat = df_work[[c for c in col_order if c in df_work.columns]]
            X_sc    = scaler.transform(df_feat.values)
            probs   = model.predict_proba(X_sc)[:, 1]

        preds = ["Churn" if p >= THRESHOLD else "Stay" for p in probs]
        risks = []
        for p in probs:
            if p > 0.70:   risks.append("🔴 High")
            elif p > 0.40: risks.append("🟠 Medium")
            else:          risks.append("🟢 Low")

        df_out = df_raw.copy()
        df_out["Churn_Probability_%"] = (probs * 100).round(1)
        df_out["Prediction"]          = preds
        df_out["Risk_Level"]          = risks
        df_out.sort_values("Churn_Probability_%", ascending=False, inplace=True)

        # ── SUMMARY METRICS ──────────────────────────────────
        n_total  = len(df_out)
        n_churn  = sum(1 for p in preds if p == "Churn")
        n_high   = sum(1 for p in probs if p > 0.70)
        n_medium = sum(1 for p in probs if 0.40 < p <= 0.70)
        avg_prob = probs.mean()

        st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Customers",  f"{n_total:,}")
        s2.metric("Predicted Churn",  f"{n_churn:,}",  delta=f"{n_churn/n_total:.1%} of base")
        s3.metric("🔴 High Risk",     f"{n_high:,}")
        s4.metric("Avg Churn Prob",   f"{avg_prob:.1%}")

        # ── RISK DISTRIBUTION CHART ───────────────────────────
        st.markdown('<div class="section-label">Risk Distribution</div>', unsafe_allow_html=True)

        fig_dist, ax_dist = plt.subplots(figsize=(8, 2.5))
        fig_dist.patch.set_facecolor("#0d1627")
        ax_dist.set_facecolor("#0d1627")

        bins   = np.linspace(0, 1, 41)
        colors_hist = ["#ff4444" if b > 0.70 else ("#ffaa00" if b > 0.40 else "#33cc66")
                        for b in (bins[:-1] + bins[1:]) / 2]
        counts, _ = np.histogram(probs, bins=bins)
        centers   = (bins[:-1] + bins[1:]) / 2

        ax_dist.bar(centers, counts, width=0.024, color=colors_hist, edgecolor="none")
        ax_dist.axvline(THRESHOLD, color="#60aaff", linewidth=1.2, linestyle="--", alpha=0.8)
        ax_dist.text(THRESHOLD + 0.01, counts.max() * 0.9, f"threshold {THRESHOLD:.2f}",
                     color="#60aaff", fontsize=7.5)
        ax_dist.set_xlabel("Churn probability", color="#4a7aaa", fontsize=8)
        ax_dist.set_ylabel("Customers",         color="#4a7aaa", fontsize=8)
        ax_dist.tick_params(colors="#7a9cc0", labelsize=7)
        for spine in ax_dist.spines.values():
            spine.set_edgecolor("#1a2540")
        fig_dist.tight_layout(pad=1)
        st.pyplot(fig_dist, use_container_width=True)
        plt.close(fig_dist)

        # ── RESULTS TABLE ─────────────────────────────────────
        st.markdown('<div class="section-label">Ranked Customer List  (highest risk first)</div>', unsafe_allow_html=True)

        show_cols = [c for c in ["CustomerId","Surname","Age","Geography","Balance",
                                  "NumOfProducts","IsActiveMember",
                                  "Churn_Probability_%","Prediction","Risk_Level"]
                     if c in df_out.columns]
        st.dataframe(
            df_out[show_cols].reset_index(drop=True),
            use_container_width=True,
            height=380,
        )

        # ── DOWNLOAD ──────────────────────────────────────────
        csv_bytes = df_out.to_csv(index=False).encode()
        st.download_button(
            label="⬇  Download Full Results CSV",
            data=csv_bytes,
            file_name="churn_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )