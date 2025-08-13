import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime
import plotly.express as px
import requests
from dotenv import load_dotenv

# -------------------------------
# App & Env
# -------------------------------
st.set_page_config(page_title="Early Warning Loan Agent", layout="wide")
load_dotenv()

st.title("💡 Early Warning Loan Default Agent")
st.caption(
    "Upload CSV → auto‑map behind the scenes → detect partials/bounces/delays → anomaly detection → "
    "risk score & severity tiers → rich portfolio views → customer drilldowns → simulate/send alerts."
)

# -------------------------------
# Sidebar controls (Condensed)
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    section = st.selectbox(
        "Choose Configuration Section",
        ["Severity thresholds", "Anomaly detection", "Alert routing"]
    )

    # Session defaults
    if "bounce_weight" not in st.session_state:
        st.session_state.update(
            bounce_weight=50, max_gap_points=40, delay_points_per_day=2,
            anomaly_points=20, severity_watch=30, severity_action=60,
            contamination=0.20, default_officer_email="credit.officer@example.com",
            teams_webhook=os.getenv("TEAMS_WEBHOOK_URL", ""), dry_run=True
        )

    if section == "Severity thresholds":
        st.session_state.bounce_weight = st.slider("Bounce weight (points)", 10, 80, st.session_state.bounce_weight, 5)
        st.session_state.max_gap_points = st.slider("Max points for EMI gap %", 10, 60, st.session_state.max_gap_points, 5)
        st.session_state.delay_points_per_day = st.slider("Delay points/day (cap 30)", 1, 5, st.session_state.delay_points_per_day, 1)
        st.session_state.anomaly_points = st.slider("Anomaly points", 5, 30, st.session_state.anomaly_points, 1)
        st.session_state.severity_watch = st.slider("WATCH threshold (score ≥)", 10, 80, st.session_state.severity_watch, 5)
        st.session_state.severity_action = st.slider("ACTION threshold (score ≥)", 20, 100, st.session_state.severity_action, 5)

    elif section == "Anomaly detection":
        st.session_state.contamination = st.slider("Anomaly rate (contamination)", 0.01, 0.30, st.session_state.contamination, 0.01)

    elif section == "Alert routing":
        st.session_state.default_officer_email = st.text_input("Officer email (demo)", st.session_state.default_officer_email)
        st.session_state.teams_webhook = st.text_input("Teams Incoming Webhook URL", st.session_state.teams_webhook, type="password")
        st.session_state.dry_run = st.checkbox("Dry run (simulate only)", value=st.session_state.dry_run)

# Local copies
bounce_weight         = st.session_state.bounce_weight
max_gap_points        = st.session_state.max_gap_points
delay_points_per_day  = st.session_state.delay_points_per_day
anomaly_points        = st.session_state.anomaly_points
severity_watch        = st.session_state.severity_watch
severity_action       = st.session_state.severity_action
contamination         = st.session_state.contamination
default_officer_email = st.session_state.default_officer_email
teams_webhook         = st.session_state.teams_webhook
dry_run               = st.session_state.dry_run

# -------------------------------
# Helpers & Schema
# -------------------------------
REQUIRED_LOGICAL_FIELDS = {
    "loan_id":       ["loan_id", "acct_id", "account_id", "loan number", "loan_no", "id"],
    "customer_name": ["customer_name", "borrower", "name", "customer"],
    "emi_due_date":  ["emi_due_date", "due_date", "emi_due", "installment_due_date"],
    "emi_amount":    ["emi_amount", "emi amt", "emi", "installment_amount", "scheduled_amount"],
    "amount_paid":   ["amount_paid", "paid_amount", "amt_paid", "amount paid", "received_amount"],
    "payment_date":  ["payment_date", "paid_on", "posting_date", "posted_date", "payment_dt"],
    "bounce_flag":   ["bounce_flag", "bounced", "cheque_bounce", "return_code", "ach_return", "nsf_flag"],
    "loan_type":     ["loan_type", "product", "product_type", "loan_product", "segment"]
}
REQUIRED_MIN = ["loan_id", "emi_due_date", "emi_amount", "amount_paid"]

@st.cache_data(show_spinner=False)
def load_csv(file_or_path):
    return pd.read_csv(file_or_path, sep=None, engine="python", encoding="utf-8-sig")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"\s+", "_", regex=True)
                  .str.replace("-", "_")
    )
    # Best‑effort synonym remap to canonical names
    remap = {}
    for logical, candidates in REQUIRED_LOGICAL_FIELDS.items():
        for c in candidates:
            if c in df.columns:
                remap[c] = logical
                break
    return df.rename(columns=remap) if remap else df

def coerce_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def parse_date(series):
    return pd.to_datetime(series, errors="coerce")

def build_alert_message(row):
    return (
        f"🚨 Loan Alert: {row['severity']}\n"
        f"Loan: {row['loan_id']} | Customer: {row.get('customer_name','-')}\n"
        f"Loan Type: {row.get('loan_type','Unknown')}\n"
        f"Score: {row['risk_score']}\n"
        f"Reasons: {row['reason_codes']}\n"
        f"EMI Gap: {row['emi_gap']:.0f} | Gap%: {row['emi_gap_pct']:.0%} | "
        f"Days Delay: {row['days_delay']} | Bounce: {int(row['bounce_flag'])}"
    )

def send_to_teams(webhook_url: str, text: str) -> bool:
    try:
        r = requests.post(webhook_url, json={"text": text}, timeout=10)
        return r.status_code in (200, 204)
    except Exception:
        return False

def severity_from_score(score, watch, action):
    if score >= action:
        return "Action"
    if score >= watch:
        return "Watch"
    return "Info"

# -------------------------------
# Tabs
# -------------------------------
tab_data, tab_portfolio, tab_risk, tab_customers, tab_alerts, tab_notify = st.tabs(
    ["📁 Data", "📊 Portfolio", "🗺️ Risk Landscape", "🧑‍💼 Customers", "🚨 Alerts", "📣 Notify"]
)

# ============ DATA TAB ============
with tab_data:
    st.subheader("📄 Raw Data")

    # Strictly require an upload (no sample fallback)
    uploaded = st.file_uploader("Upload Loan Repayment CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to continue.")
        st.stop()

    df_raw = load_csv(uploaded)
    st.dataframe(df_raw.head(50), use_container_width=True, height=220)

    # Normalize + auto-map
    df_norm = normalize_columns(df_raw)

    # === Column Mapping: single dropdown (Hidden by default) ===
    map_view = st.selectbox(
        "Column Mapping",
        ["Hidden", "View auto‑mapping"],
        index=0
    )
    if map_view == "View auto‑mapping":
        # read-only mapping summary
        cols_lower = set(df_norm.columns.str.lower())
        mapping = {
            logical: next((c for c in candidates if c in cols_lower), None)
            for logical, candidates in REQUIRED_LOGICAL_FIELDS.items()
        }
        map_df = pd.DataFrame(
            {"logical_field": list(mapping.keys()),
             "mapped_to": [mapping[k] or "—" for k in mapping.keys()]}
        )
        st.dataframe(map_df, use_container_width=True, height=220)

    # Validate required fields after normalize
    missing_required = [col for col in REQUIRED_MIN if col not in df_norm.columns]
    if missing_required:
        st.error(
            "Your file is missing required columns after auto-mapping: "
            + ", ".join(missing_required)
            + ".\n\nExpected fields include: "
            + ", ".join(REQUIRED_MIN)
            + "."
        )
        st.stop()

    # Build working DF
    def pick(colname, default=np.nan):
        return df_norm[colname] if colname in df_norm.columns else pd.Series([default] * len(df_norm))

    work = pd.DataFrame({
        "loan_id":       pick("loan_id"),
        "customer_name": pick("customer_name", default="Unknown"),
        "loan_type":     pick("loan_type", default="Unknown"),
        "emi_due_date":  pick("emi_due_date"),
        "emi_amount":    pick("emi_amount"),
        "amount_paid":   pick("amount_paid"),
        "payment_date":  pick("payment_date"),
        "bounce_flag":   pick("bounce_flag", default=0),
    })

    # Coerce types & features
    work["emi_amount"] = coerce_numeric(work["emi_amount"])
    work["amount_paid"] = coerce_numeric(work["amount_paid"])
    work["emi_due_date"] = parse_date(work["emi_due_date"])
    work["payment_date"] = parse_date(work["payment_date"])

    bf = work["bounce_flag"].copy().fillna(0)
    if bf.dtype == object:
        work["bounce_flag"] = np.where(bf.astype(str).str.strip().eq(""), 0, 1)
    else:
        work["bounce_flag"] = coerce_numeric(bf).fillna(0).astype(int)

    today = pd.Timestamp(datetime.now().date())
    work["emi_gap"] = (work["emi_amount"] - work["amount_paid"]).fillna(0)

    work["days_delay"] = np.where(
        work["payment_date"].isna(),
        (today - work["emi_due_date"]).dt.days,
        (work["payment_date"] - work["emi_due_date"]).dt.days
    )
    work["days_delay"] = work["days_delay"].fillna(0).astype(int)

    work["emi_gap_pct"] = np.where(
        (work["emi_amount"] > 0) & (work["emi_gap"] > 0),
        work["emi_gap"] / work["emi_amount"],
        0.0
    )

    work["rule_high"] = np.where(
        (work["bounce_flag"] == 1) | (work["emi_gap"] > 0) | (work["days_delay"] > 0),
        1, 0
    )

    # Anomaly detection
    can_do_ml = work[["emi_gap", "days_delay"]].dropna().shape[0] >= 10
    if can_do_ml:
        features = work[["emi_gap", "days_delay"]].astype(float).fillna(0)
        contamination_eff = min(0.3, max(contamination, 2.0 / len(features)))
        model = IsolationForest(contamination=contamination_eff, random_state=42)
        work["anomaly_score_raw"] = model.fit_predict(features)  # -1 anomaly, 1 normal
        work["anomaly_flag"] = np.where(work["anomaly_score_raw"] == -1, "Anomaly", "Normal")
    else:
        work["anomaly_flag"] = "Normal"

    # Reason codes
    def reasons(row):
        r = []
        if row.get("bounce_flag", 0) == 1:
            r.append("Bounced payment")
        if pd.notna(row.get("amount_paid")) and pd.notna(row.get("emi_amount")):
            if float(row["amount_paid"]) == 0:
                r.append("Missed payment")
            elif float(row["amount_paid"]) < float(row["emi_amount"]):
                r.append("Partial payment")
        if row.get("days_delay", 0) > 0:
            r.append(f"Delayed by {int(row['days_delay'])} days")
        if row.get("anomaly_flag") == "Anomaly":
            r.append("Anomalous pattern")
        if not r:
            r.append("No issues")
        return ", ".join(r)

    work["reason_codes"] = work.apply(reasons, axis=1)

    # Risk score & severity
    def risk_score(row):
        score = 0.0
        if row["bounce_flag"] == 1:
            score += bounce_weight
        gap_pct = float(row["emi_gap_pct"])
        score += min(max_gap_points, gap_pct * max_gap_points)
        score += min(30, max(0, int(row["days_delay"])) * delay_points_per_day)
        if row.get("anomaly_flag") == "Anomaly":
            score += anomaly_points
        return round(score, 1)

    work["risk_score"] = work.apply(risk_score, axis=1)
    work["severity"] = work["risk_score"].apply(lambda s: severity_from_score(s, severity_watch, severity_action))

    st.success("Data ready ✅")

# ============ PORTFOLIO TAB ============
with tab_portfolio:
    st.subheader("📊 Portfolio Overview")

    total = len(work); rules_high = int(work["rule_high"].sum())
    anomalies = int((work["anomaly_flag"] == "Anomaly").sum())
    actions = int((work["severity"] == "Action").sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Loans", total)
    m2.metric("High‑Risk (rules)", rules_high)
    m3.metric("Anomalies", anomalies)
    m4.metric("Action Severity", actions)

    cA, cB = st.columns(2)
    fig_sev = px.histogram(
        work, x="severity", color="severity",
        color_discrete_map={"Info":"#6c757d","Watch":"#f39c12","Action":"#e74c3c"},
        category_orders={"severity":["Info","Watch","Action"]},
        title="Severity Distribution"
    )
    cA.plotly_chart(fig_sev, use_container_width=True)

    fig_stack = px.histogram(
        work, x="loan_type", color="severity", barmode="group",
        color_discrete_map={"Info":"#6c757d","Watch":"#f39c12","Action":"#e74c3c"},
        title="Loan Types by Severity"
    )
    cB.plotly_chart(fig_stack, use_container_width=True)

# ============ RISK LANDSCAPE TAB ============
with tab_risk:
    st.subheader("🗺️ Risk Landscape")
    view = st.radio("Choose view", ["Quadrant (Gap% vs Days Delay)", "Density Heatmap", "Sunburst (Loan Type → Severity)"], horizontal=True)

    if view == "Quadrant (Gap% vs Days Delay)":
        fig = px.scatter(
            work, x="days_delay", y="emi_gap_pct",
            color="severity",
            color_discrete_map={"Info":"#6c757d","Watch":"#f39c12","Action":"#e74c3c"},
            size="risk_score",
            hover_data=["loan_id","customer_name","loan_type","reason_codes","risk_score"],
            title="Quadrant: Days Delay vs EMI Gap % (bubble size = risk score)"
        )
        fig.update_yaxes(tickformat=".0%", title_text="EMI Gap (%)")
        fig.update_xaxes(title_text="Days Delay")
        st.plotly_chart(fig, use_container_width=True)

    elif view == "Density Heatmap":
        fig = px.density_heatmap(
            work, x="days_delay", y="emi_gap_pct",
            nbinsx=20, nbinsy=20,
            color_continuous_scale="YlOrRd",
            title="Density Heatmap: Portfolio Hotspots (Days Delay vs EMI Gap %)"
        )
        fig.update_yaxes(tickformat=".0%", title_text="EMI Gap (%)")
        fig.update_xaxes(title_text="Days Delay")
        st.plotly_chart(fig, use_container_width=True)
    else:
        sb_df = work.copy(); sb_df["count"] = 1
        fig = px.sunburst(
            sb_df, path=["loan_type", "severity"], values="count",
            color="severity",
            color_discrete_map={"Info":"#6c757d","Watch":"#f39c12","Action":"#e74c3c"},
            title="Sunburst: Composition by Loan Type → Severity"
        )
        st.plotly_chart(fig, use_container_width=True)

# ============ CUSTOMERS TAB (compact) ============
with tab_customers:
    st.subheader("🧑‍💼 Customer Explorer")

    customers = (
        work["customer_name"].fillna("Unknown").astype(str)
        .replace("", "Unknown").unique().tolist()
    )
    customers = sorted(customers, key=lambda x: (x == "Unknown", x.lower()))

    sel_cust = st.selectbox("Select Customer", customers, index=0)
    cust_df = work[work["customer_name"].astype(str) == str(sel_cust)].copy()

    if cust_df.empty:
        st.info("No records for selected customer.")
    else:
        view_option = st.selectbox(
            "Choose what to view",
            ["Summary Metrics", "Loan Details Table", "Risk Chart"]
        )

        if view_option == "Summary Metrics":
            loans_count = len(cust_df)
            max_sev = cust_df["severity"].value_counts().idxmax()
            avg_score = round(cust_df["risk_score"].mean(), 1)

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Loans", loans_count)
            b2.metric("Top Severity", max_sev)
            b3.metric("Avg Risk Score", avg_score)
            b4.metric("Loan Types", len(set(cust_df["loan_type"])))

        elif view_option == "Loan Details Table":
            show_cols = ["loan_id","loan_type","severity","risk_score","reason_codes","emi_gap","emi_gap_pct","days_delay","bounce_flag","emi_due_date","payment_date"]
            table = cust_df[show_cols].sort_values(["severity","risk_score"], ascending=[False,False]).copy()
            table["emi_gap_pct"] = table["emi_gap_pct"].map(lambda v: f"{v:.0%}")
            st.dataframe(table, use_container_width=True, height=280)

        elif view_option == "Risk Chart":
            fig = px.bar(
                cust_df.sort_values("risk_score", ascending=True),
                x="risk_score", y="loan_id", color="severity",
                color_discrete_map={"Info":"#6c757d","Watch":"#f39c12","Action":"#e74c3c"},
                orientation="h",
                hover_data=["loan_type","reason_codes","emi_gap","days_delay"],
                title=f"Risk Scores by Loan for {sel_cust}"
            )
            st.plotly_chart(fig, use_container_width=True)

# ============ ALERTS TAB ============
with tab_alerts:
    st.subheader("🚨 Alerts")
    sev_filter = st.multiselect("Filter by severity", ["Action","Watch","Info"], default=["Action","Watch"])
    alerts = work[work["severity"].isin(sev_filter)][
        ["loan_id","customer_name","loan_type","severity","risk_score","reason_codes","emi_gap","emi_gap_pct","days_delay","bounce_flag"]
    ].sort_values(["severity","risk_score"], ascending=[False,False])

    if alerts.empty:
        st.success("No alerts at the selected severity levels.")
    else:
        display = alerts.copy()
        display["emi_gap_pct"] = display["emi_gap_pct"].map(lambda v: f"{v:.0%}")
        st.dataframe(display, use_container_width=True, height=320)
        st.download_button("⬇️ Download Risk Report (CSV)", alerts.to_csv(index=False), "risk_report.csv", "text/csv")

# ============ NOTIFY TAB ============
with tab_notify:
    st.subheader("📣 Send Alerts")

    base = work[work["severity"].isin(["Action","Watch"])]
    def msg_df(df):
        return df[["loan_id","customer_name","loan_type","severity","risk_score","reason_codes","emi_gap","emi_gap_pct","days_delay","bounce_flag"]].copy()

    scope = st.selectbox("Which alerts to send?", ["Action only", "Action + Watch", "All severities"])
    if scope == "Action only":
        to_send = msg_df(work[work["severity"] == "Action"])
    elif scope == "Action + Watch":
        to_send = msg_df(base)
    else:
        to_send = msg_df(work)

    with st.expander("Preview first 5 messages", expanded=True):
        for _, row in to_send.head(5).iterrows():
            st.code(build_alert_message(row), language=None)

    if st.button("Send / Simulate Alerts"):
        if to_send.empty:
            st.info("No alerts to send for current scope.")
        else:
            st.toast(f"Prepared {len(to_send)} alert(s).", icon="✅")
            sent = 0
            if not dry_run and teams_webhook:
                for _, row in to_send.iterrows():
                    ok = send_to_teams(teams_webhook, build_alert_message(row))
                    if ok:
                        sent += 1
                if sent:
                    st.success(f"Posted {sent} alert(s) to Teams.")
                else:
                    st.warning("Attempted to post alerts but none succeeded. Check webhook/permissions.")
            else:
                with st.expander("Simulation output"):
                    for _, row in to_send.iterrows():
                        st.write(f"📧 {default_officer_email}")
                        st.code(build_alert_message(row), language=None)
                st.warning("Dry run ON or no Teams webhook provided → simulation only.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("<hr style='border:0;height:1px;background:#eee;margin:1.5rem 0;'>", unsafe_allow_html=True)
st.caption("Prototype • Hidden/condensed column mapping • Loan type segmentation • Risk Landscape • Alert simulation")
