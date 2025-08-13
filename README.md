# Early Warning Loan Default Agent (Prototype)

A lightweight Streamlit app that monitors loan repayments, detects early warning signals (partials, delays, bounced payments), runs anomaly detection, scores risk, assigns **severity tiers** (Info / Watch / Action), visualizes portfolio risk, and **simulates or posts alerts** to Microsoft Teams.

## ✨ What it does
- Upload CSV or use sample data
- Auto-detect & map common column names
- Feature engineering: **EMI gap**, **days delay**, **bounce flag**
- **Anomaly detection** (IsolationForest) on repayment patterns
- **Reason codes** and **numeric risk score**
- **Severity tiers**: Info / Watch / Action (configurable)
- **Charts**: severity distribution, EMI gap histogram, risk scatter plot
- **Alerts**: download CSV; simulate messages; optional **Teams** posting via Incoming Webhook

## 🗂️ Columns expected (auto-mapped)
- `loan_id`, `customer_name` (optional), `emi_due_date`, `emi_amount`, `amount_paid`, `payment_date` (optional), `bounce_flag` (0/1 or non-empty code)

## 🚀 Quickstart
```bash
git clone https://github.com/<your-username>/early-warning-agent.git
cd early-warning-agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
