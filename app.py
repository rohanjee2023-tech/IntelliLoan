"""
INTELLOAN - AI POWERED LOAN APPROVAL SYSTEM

Run:
    streamlit run app.py

Required files in the same folder:
    best_loan_approval_model.pkl
    loan_preprocessor.pkl

Assessment history is stored in:
    intelliloan_records.db
"""

from pathlib import Path
from datetime import datetime
import sqlite3
import uuid

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IntelliLoan | Credit Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent

MODEL_PATH = BASE / "best_loan_approval_model.pkl"
PREPROCESSOR_PATH = BASE / "loan_preprocessor.pkl"
DATABASE_PATH = BASE / "intelliloan_records.db"


# ============================================================
# 2. PROFESSIONAL DASHBOARD DESIGN
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #F4F7FB;
    color: #18304F;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif;
    color: #18304F;
}

[data-testid="stSidebar"] {
    background-color: #17385F;
}

[data-testid="stSidebar"] * {
    color: white;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #D5E1F0;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E2E9F2;
    border-radius: 15px;
    padding: 18px;
    box-shadow: 0 4px 15px rgba(20, 45, 80, 0.04);
}

div[data-testid="stMetricLabel"] {
    color: #64758B;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #18304F;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
}

[data-testid="stForm"] {
    background: white;
    border: 1px solid #E2E9F2;
    border-radius: 16px;
    padding: 20px;
}

.stFormSubmitButton > button {
    background: #2F6FED;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 700;
    min-height: 45px;
}

.stDownloadButton > button {
    border-radius: 10px;
    font-weight: 700;
}

.hero {
    background: linear-gradient(115deg, #17385F, #24528A, #2F6FED);
    padding: 28px;
    border-radius: 20px;
    color: white;
    margin-bottom: 24px;
}

.hero h1 {
    color: white !important;
    margin-bottom: 8px;
}

.hero p {
    color: #D8E6FB;
    margin-bottom: 0;
}

.eyebrow {
    color: #B9D2FF;
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 800;
}

.section-note {
    color: #6D7C91;
    font-size: 14px;
}

hr {
    border-color: #E2E9F2;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD THE TRAINED MODEL
# ============================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def get_category_options(preprocessor):
    encoder = (
        preprocessor
        .named_transformers_["cat"]
        .named_steps["encoder"]
    )

    columns = preprocessor.transformers_[1][2]

    return {
        column: [str(value) for value in categories]
        for column, categories in zip(
            columns, encoder.categories_
        )
    }


model = None
preprocessor = None
options = None
model_error = None

try:
    model, preprocessor = load_artifacts()
    options = get_category_options(preprocessor)
except Exception as error:
    model_error = str(error)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def calculate_emi(principal, annual_rate, months):
    monthly_rate = annual_rate / 1200

    if monthly_rate == 0:
        return principal / months

    factor = (1 + monthly_rate) ** months

    return (
        principal
        * monthly_rate
        * factor
        / (factor - 1)
    )


def format_inr(value):
    if value is None or not np.isfinite(float(value)):
        return "N/A"

    return "₹{:,.0f}".format(float(value))


def classify_risk(probability, low_threshold, high_threshold):
    if probability >= low_threshold:
        return "LOW"

    if probability >= high_threshold:
        return "MEDIUM"

    return "HIGH"


# ============================================================
# 5. SQLITE DATABASE
# ============================================================

def get_connection():
    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=20,
    )

    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS assessments (
                application_id TEXT PRIMARY KEY,
                assessed_at TEXT,
                applicant_name TEXT,
                age INTEGER,
                gender TEXT,
                marital_status TEXT,
                dependents INTEGER,
                education_level TEXT,
                employment_status TEXT,
                employer_category TEXT,
                applicant_income REAL,
                coapplicant_income REAL,
                total_annual_income REAL,
                credit_score INTEGER,
                existing_loans INTEGER,
                dti_ratio REAL,
                savings REAL,
                collateral_value REAL,
                loan_amount REAL,
                loan_term_months INTEGER,
                loan_purpose TEXT,
                property_area TEXT,
                interest_rate REAL,
                loan_to_income_ratio REAL,
                approval_probability REAL,
                prediction TEXT,
                risk_band TEXT,
                monthly_emi REAL,
                monthly_household_income REAL,
                emi_income_ratio REAL,
                total_repayment REAL,
                affordable INTEGER
            )
        """)


def get_all_records():
    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM assessments
            ORDER BY assessed_at DESC
            """,
            connection,
        )


def save_record(record):
    columns = list(record.keys())

    placeholders = ", ".join(
        ["?"] * len(columns)
    )

    query = (
        f"INSERT INTO assessments "
        f"({', '.join(columns)}) "
        f"VALUES ({placeholders})"
    )

    with get_connection() as connection:
        connection.execute(
            query,
            list(record.values()),
        )


def make_csv(dataframe):
    return dataframe.to_csv(
        index=False
    ).encode("utf-8-sig")


initialize_database()

records = get_all_records()


# ============================================================
# 6. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown("## 🏦 IntelliLoan")

    st.caption("CREDIT INTELLIGENCE PLATFORM")

    st.divider()

    page = st.radio(
        "CORE MODULES",
        [
            "Dashboard",
            "Loan Application",
            "Application Audit Log",
            "Model & Rules",
        ],
    )

    st.divider()

    if model_error:
        st.error("Model not loaded")
        st.caption(model_error)
    else:
        st.success("Model loaded successfully")

    st.caption(
        f"Database: {DATABASE_PATH.name}"
    )

    st.caption(
        "Assessment records are stored locally."
    )


# ============================================================
# 7. DASHBOARD HEADER
# ============================================================

st.markdown(
    """
    <div style="
        background: linear-gradient(115deg, #17385F, #24528A, #2F6FED);
        padding: 28px;
        border-radius: 20px;
        color: white;
        margin-bottom: 24px;
    ">
        <h1 style="color: white;">IntelliLoan Credit Dashboard</h1>
        <p style="color: #D8E6FB;">
            AI-powered loan assessment, credit risk analysis,
            applicant history and financial insights.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# 8. CALCULATE PORTFOLIO STATISTICS
# ============================================================

total_applications = len(records)

if total_applications > 0:

    approved_count = int(
        (records["prediction"] == "Likely APPROVED").sum()
    )

    rejected_count = (
        total_applications - approved_count
    )

    approval_rate = (
        approved_count / total_applications
    ) * 100

    average_credit_score = (
        records["credit_score"].mean()
    )

    average_loan_amount = (
        records["loan_amount"].mean()
    )

    average_probability = (
        records["approval_probability"].mean()
    ) * 100

    affordable_count = int(
        records["affordable"].sum()
    )

else:

    approved_count = 0
    rejected_count = 0
    approval_rate = 0
    average_credit_score = 0
    average_loan_amount = 0
    average_probability = 0
    affordable_count = 0


# ============================================================
# 9. EXECUTIVE DASHBOARD
# ============================================================

if page == "Dashboard":

    st.subheader("Executive Overview")

    st.markdown(
        '<p class="section-note">'
        'Summary of all applications assessed and saved so far.'
        '</p>',
        unsafe_allow_html=True,
    )

    # KPI cards

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Evaluated",
        total_applications,
    )

    c2.metric(
        "Predicted Approval Rate",
        f"{approval_rate:.1f}%",
    )

    c3.metric(
        "Average Credit Score",
        f"{average_credit_score:.0f}"
        if total_applications else "N/A",
    )

    c4.metric(
        "Average Loan Amount",
        format_inr(average_loan_amount)
        if total_applications else "N/A",
    )

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "Predicted Approvals",
        approved_count,
    )

    c6.metric(
        "Predicted Rejections",
        rejected_count,
    )

    c7.metric(
        "Average Approval Probability",
        f"{average_probability:.1f}%"
        if total_applications else "N/A",
    )

    c8.metric(
        "Passed Affordability Check",
        affordable_count,
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Prediction Distribution")

        if total_applications:

            prediction_counts = (
                records["prediction"]
                .value_counts()
            )

            st.bar_chart(prediction_counts)

        else:

            st.info(
                "No assessments yet. "
                "Create your first application."
            )

    with right:

        st.subheader("Risk Distribution")

        if total_applications:

            risk_counts = (
                records["risk_band"]
                .value_counts()
                .reindex(
                    ["LOW", "MEDIUM", "HIGH"],
                    fill_value=0,
                )
            )

            st.bar_chart(risk_counts)

        else:

            st.info(
                "Risk analytics will appear "
                "after your first assessment."
            )

    st.divider()

    st.subheader("Assessment Trend")

    if total_applications:

        trend = records.copy()

        trend["assessed_at"] = pd.to_datetime(
            trend["assessed_at"],
            errors="coerce",
        )

        trend["Date"] = (
            trend["assessed_at"].dt.strftime("%Y-%m-%d")
        )

        daily_counts = (
            trend.groupby("Date")
            .size()
            .rename("Applications")
        )

        st.line_chart(daily_counts)

    else:

        st.info(
            "The assessment trend will populate automatically."
        )

    st.divider()

    st.subheader("Recent Applications")

    if total_applications:

        recent = records.head(10).copy()

        recent["approval_probability"] = (
            recent["approval_probability"] * 100
        ).round(1).astype(str) + "%"

        recent["loan_amount"] = (
            recent["loan_amount"].map(format_inr)
        )

        recent["monthly_emi"] = (
            recent["monthly_emi"].map(format_inr)
        )

        st.dataframe(
            recent[
                [
                    "application_id",
                    "assessed_at",
                    "applicant_name",
                    "loan_amount",
                    "credit_score",
                    "approval_probability",
                    "prediction",
                    "risk_band",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "📥 Download All Records (CSV)",
            data=make_csv(records),
            file_name="intelliloan_all_records.csv",
            mime="text/csv",
        )

    else:

        st.info(
            "Your saved applications will appear here."
        )


# ============================================================
# 10. NEW LOAN APPLICATION
# ============================================================

elif page == "Loan Application":

    st.subheader("New Loan Assessment")

    st.markdown(
        '<p class="section-note">'
        'Enter applicant information and generate a prediction. '
        'Successful assessments are saved automatically.'
        '</p>',
        unsafe_allow_html=True,
    )

    if model_error:

        st.error(
            "Cannot load the model. Check that both model "
            "files exist beside app.py."
        )

        st.stop()

    with st.form("loan_application"):

        st.markdown("### 01. Applicant Profile")

        c1, c2, c3 = st.columns(3)

        with c1:

            applicant_name = st.text_input(
                "Applicant Name / Reference"
            )

            age = st.number_input(
                "Age", 18, 80, 35
            )

            gender = st.selectbox(
                "Gender", options["Gender"]
            )

            marital = st.selectbox(
                "Marital Status",
                options["Marital_Status"],
            )

        with c2:

            dependents = st.number_input(
                "Dependents", 0, 10, 1
            )

            education = st.selectbox(
                "Education Level",
                options["Education_Level"],
            )

            employment = st.selectbox(
                "Employment Status",
                options["Employment_Status"],
            )

            employer = st.selectbox(
                "Employer Category",
                options["Employer_Category"],
            )

        with c3:

            credit = st.number_input(
                "Credit Score", 300, 900, 700
            )

            existing = st.number_input(
                "Existing Loans", 0, 15, 1
            )

            dti = st.number_input(
                "DTI Ratio (0–1)",
                0.0, 1.0, 0.30, 0.01,
            )

        st.divider()

        st.markdown("### 02. Financial Profile")

        f1, f2, f3 = st.columns(3)

        with f1:

            app_inc = st.number_input(
                "Applicant Annual Income (₹)",
                0, 50_000_000, 900_000, 50_000,
            )

            co_inc = st.number_input(
                "Co-applicant Annual Income (₹)",
                0, 50_000_000, 300_000, 50_000,
            )

        with f2:

            savings = st.number_input(
                "Savings (₹)",
                0, 100_000_000, 500_000, 50_000,
            )

            collateral = st.number_input(
                "Collateral Value (₹)",
                0, 200_000_000, 2_000_000, 100_000,
            )

        with f3:

            amount = st.number_input(
                "Requested Loan Amount (₹)",
                10_000, 100_000_000, 800_000, 50_000,
            )

            term = st.selectbox(
                "Loan Tenure",
                [12, 24, 36, 48, 60, 72, 84],
                index=4,
                format_func=lambda x: f"{x} months",
            )

        st.divider()

        st.markdown("### 03. Loan Details")

        l1, l2, l3 = st.columns(3)

        with l1:

            purpose = st.selectbox(
                "Loan Purpose",
                options["Loan_Purpose"],
            )

        with l2:

            area = st.selectbox(
                "Property Area",
                options["Property_Area"],
            )

        with l3:

            rate = st.number_input(
                "Interest Rate (% p.a.)",
                1.0, 40.0, 10.0, 0.25,
            )

        submitted = st.form_submit_button(
            "🔍 Assess and Save Application",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        if not applicant_name.strip():

            st.error(
                "Enter an applicant name or reference."
            )

            st.stop()

        # Build input DataFrame

        row = pd.DataFrame([{
            "Applicant_Income": app_inc,
            "Coapplicant_Income": co_inc,
            "Employment_Status": employment,
            "Age": age,
            "Marital_Status": marital,
            "Dependents": dependents,
            "Credit_Score": credit,
            "Existing_Loans": existing,
            "DTI_Ratio": dti,
            "Savings": savings,
            "Collateral_Value": collateral,
            "Loan_Amount": amount,
            "Loan_Term": term,
            "Loan_Purpose": purpose,
            "Property_Area": area,
            "Education_Level": education,
            "Gender": gender,
            "Employer_Category": employer,
        }])

        # Feature engineering must match training.

        row["Total_Income"] = (
            row["Applicant_Income"]
            + row["Coapplicant_Income"]
        )

        row["Loan_to_Income_Ratio"] = (
            row["Loan_Amount"]
            / row["Total_Income"].replace(0, np.nan)
        )

        try:

            transformed = preprocessor.transform(row)

            classes = list(model.classes_)

            if "Yes" in classes:

                yes_index = classes.index("Yes")

            elif 1 in classes:

                yes_index = classes.index(1)

            else:

                st.error(
                    f"Could not identify the approval class. "
                    f"Model classes: {classes}"
                )

                st.stop()

            probability = float(
                model.predict_proba(
                    transformed
                )[0][yes_index]
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )

            st.stop()

        # Prediction

        prediction = (
            "Likely APPROVED"
            if probability >= 0.5
            else "Likely REJECTED"
        )

        # Session policy thresholds.

        low_cut = st.session_state.get(
            "low_cut", 0.70
        )

        high_cut = st.session_state.get(
            "high_cut", 0.40
        )

        foir = st.session_state.get(
            "foir", 0.40
        )

        if high_cut >= low_cut:

            high_cut = 0.40
            low_cut = 0.70

        risk = classify_risk(
            probability,
            low_cut,
            high_cut,
        )

        total_income = float(
            row["Total_Income"].iloc[0]
        )

        monthly_income = total_income / 12

        monthly_emi = calculate_emi(
            amount,
            rate,
            term,
        )

        emi_ratio = (
            monthly_emi / monthly_income
            if monthly_income > 0
            else float("inf")
        )

        affordable = emi_ratio <= foir

        loan_to_income = (
            float(row["Loan_to_Income_Ratio"].iloc[0])
            if pd.notna(
                row["Loan_to_Income_Ratio"].iloc[0]
            )
            else None
        )

        application_id = (
            "IL-" + uuid.uuid4().hex[:10].upper()
        )

        assessed_at = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

        # Store the complete assessment.

        record = {
            "application_id": application_id,
            "assessed_at": assessed_at,
            "applicant_name": applicant_name.strip(),
            "age": int(age),
            "gender": gender,
            "marital_status": marital,
            "dependents": int(dependents),
            "education_level": education,
            "employment_status": employment,
            "employer_category": employer,
            "applicant_income": float(app_inc),
            "coapplicant_income": float(co_inc),
            "total_annual_income": total_income,
            "credit_score": int(credit),
            "existing_loans": int(existing),
            "dti_ratio": float(dti),
            "savings": float(savings),
            "collateral_value": float(collateral),
            "loan_amount": float(amount),
            "loan_term_months": int(term),
            "loan_purpose": purpose,
            "property_area": area,
            "interest_rate": float(rate),
            "loan_to_income_ratio": loan_to_income,
            "approval_probability": probability,
            "prediction": prediction,
            "risk_band": risk,
            "monthly_emi": monthly_emi,
            "monthly_household_income": monthly_income,
            "emi_income_ratio": (
                emi_ratio
                if np.isfinite(emi_ratio)
                else None
            ),
            "total_repayment": monthly_emi * term,
            "affordable": int(affordable),
        }

        try:

            save_record(record)

        except Exception as error:

            st.error(
                f"Could not save assessment: {error}"
            )

            st.stop()

        # Display results.

        st.success(
            f"Assessment saved successfully. "
            f"Reference: {application_id}"
        )

        st.subheader("Assessment Results")

        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "Prediction",
            prediction,
        )

        r2.metric(
            "Approval Probability",
            f"{probability * 100:.1f}%",
        )

        r3.metric(
            "Risk Band",
            risk,
        )

        r4.metric(
            "Monthly EMI",
            format_inr(monthly_emi),
        )

        st.progress(
            float(np.clip(probability, 0, 1)),
            text=(
                f"Estimated approval probability: "
                f"{probability * 100:.1f}%"
            ),
        )

        st.divider()

        st.subheader("Affordability Analysis")

        a1, a2, a3 = st.columns(3)

        a1.metric(
            "Monthly Household Income",
            format_inr(monthly_income),
        )

        a2.metric(
            "EMI / Income",
            f"{emi_ratio * 100:.1f}%"
            if np.isfinite(emi_ratio)
            else "N/A",
        )

        a3.metric(
            "Total Repayments",
            format_inr(monthly_emi * term),
        )

        if affordable:

            st.success(
                "The EMI passes the configured "
                "income-ratio check."
            )

        else:

            st.warning(
                "The proposed EMI exceeds the configured "
                "income-ratio limit."
            )

        st.caption(
            "This is decision support, not a final lending decision. "
            "The affordability check does not deduct all existing "
            "debt payments and household expenses."
        )

        # Download the individual assessment.

        individual = pd.DataFrame([record])

        st.download_button(
            "📥 Download This Assessment (CSV)",
            data=make_csv(individual),
            file_name=f"{application_id}.csv",
            mime="text/csv",
        )


# ============================================================
# 11. APPLICATION AUDIT LOG
# ============================================================

elif page == "Application Audit Log":

    st.subheader("Application Audit Log")

    st.markdown(
        "Search, filter and export previously saved applications."
    )

    records = get_all_records()

    if records.empty:

        st.info(
            "No saved applications. "
            "Create an assessment first."
        )

    else:

        f1, f2, f3 = st.columns(3)

        with f1:

            search_text = st.text_input(
                "Search applicant or application ID"
            )

        with f2:

            prediction_filter = st.selectbox(
                "Prediction",
                [
                    "All",
                    "Likely APPROVED",
                    "Likely REJECTED",
                ],
            )

        with f3:

            risk_filter = st.selectbox(
                "Risk Band",
                ["All", "LOW", "MEDIUM", "HIGH"],
            )

        filtered = records.copy()

        if search_text.strip():

            query = search_text.strip().lower()

            matches = (
                filtered["applicant_name"]
                .astype(str)
                .str.lower()
                .str.contains(query, na=False)
            )

            matches |= (
                filtered["application_id"]
                .astype(str)
                .str.lower()
                .str.contains(query, na=False)
            )

            filtered = filtered[matches]

        if prediction_filter != "All":

            filtered = filtered[
                filtered["prediction"] == prediction_filter
            ]

        if risk_filter != "All":

            filtered = filtered[
                filtered["risk_band"] == risk_filter
            ]

        st.caption(
            f"Showing {len(filtered)} of {len(records)} records."
        )

        display = filtered.copy()

        display["approval_probability"] = (
            display["approval_probability"] * 100
        ).round(2)

        display["loan_amount"] = (
            display["loan_amount"].map(format_inr)
        )

        display["monthly_emi"] = (
            display["monthly_emi"].map(format_inr)
        )

        display["total_repayment"] = (
            display["total_repayment"].map(format_inr)
        )

        display["affordable"] = (
            display["affordable"]
            .map({1: "Yes", 0: "No"})
        )

        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True,
        )

        st.divider()

        d1, d2 = st.columns(2)

        with d1:

            st.download_button(
                "📥 Download Filtered Records",
                data=make_csv(filtered),
                file_name="intelliloan_filtered_records.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with d2:

            st.download_button(
                "📥 Download ALL Records",
                data=make_csv(records),
                file_name="intelliloan_all_records.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with st.expander("Delete all saved records"):

            st.warning(
                "This permanently deletes all saved assessments. "
                "Download a backup CSV first if needed."
            )

            confirm = st.checkbox(
                "I understand that this action cannot be undone."
            )

            if st.button(
                "Delete All Records",
                disabled=not confirm,
            ):

                with get_connection() as connection:

                    connection.execute(
                        "DELETE FROM assessments"
                    )

                st.success("All records deleted.")

                st.rerun()


# ============================================================
# 12. MODEL INFORMATION AND RULES
# ============================================================

elif page == "Model & Rules":

    st.subheader("Model & System Information")

    if model_error:

        st.error(
            "Model files could not be loaded."
        )

        st.code(model_error)

    else:

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Classifier",
            type(model).__name__,
        )

        m2.metric(
            "Total Saved Assessments",
            len(get_all_records()),
        )

        m3.metric(
            "Number of Model Classes",
            len(model.classes_),
        )

        st.divider()

        st.subheader("Model File Paths")

        st.code(
            f"Application directory:\n{BASE}\n\n"
            f"XGBoost model:\n{MODEL_PATH}\n\n"
            f"Preprocessor:\n{PREPROCESSOR_PATH}\n\n"
            f"Database:\n{DATABASE_PATH}"
        )

        st.subheader("Decision Rules")

        st.write(
            "**Prediction:** Approval probability ≥ 50% "
            "is displayed as Likely APPROVED."
        )

        st.write(
            "**Risk band:** Probability thresholds classify "
            "results as LOW, MEDIUM or HIGH."
        )

        st.write(
            "**Affordability:** Monthly EMI is compared "
            "with monthly household income."
        )

        st.write(
            "**Record storage:** Successful assessments "
            "are saved in a local SQLite database."
        )

        st.divider()

        st.subheader("Important Limitations")

        st.warning(
            "This is a demonstration and decision-support system. "
            "Approval probability is not the same as probability of default. "
            "The affordability check does not deduct all existing debt "
            "payments or household expenses. Validate model performance, "
            "probability calibration, fairness and applicable lending "
            "requirements before real-world use."
        )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#718096; font-size:12px;">
        INTELLOAN | AI-ASSISTED CREDIT ASSESSMENT
        <br>
        Demonstration and decision-support system
    </div>
    """,
    unsafe_allow_html=True,
)