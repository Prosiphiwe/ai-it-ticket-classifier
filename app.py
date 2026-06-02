import matplotlib.pyplot as plt
import streamlit as st
import joblib
import pandas as pd
import os


plt.style.use("seaborn-v0_8-whitegrid")

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: bold;
    text-align: center;
    color: #1f4e79;
}

.sub-title {
    text-align: center;
    color: #6c757d;
    margin-bottom: 20px;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    text-align: center;
}

</style>

<div class="main-title">🧠 AI IT Service Desk</div>
<div class="sub-title">ServiceNow-Style Ticket Dashboard</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================
# PAGE CONFIG (DASHBOARD STYLE)
# =========================
st.set_page_config(
    page_title="AI IT Service Desk",
    page_icon="🧠",
    layout="wide"
)

# =========================
# HEADER
# =========================

st.markdown("---")

# =========================
# LOAD MODELS
# =========================
model_category = joblib.load("model_category.pkl")
model_priority = joblib.load("model_priority.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# =========================
# FIXES
# =========================
def suggest_fix(category):
    fixes = {
        "Network": "Restart router, check VPN, verify cables.",
        "Hardware": "Check physical components or replace faulty hardware.",
        "Software": "Reinstall or update the application.",
        "Access": "Reset password or verify permissions.",
        "Security": "Run antivirus scan or check security policies.",
        "Operating System": "Run system updates or repair OS files.",
        "Email": "Restart Outlook or check mail server.",
        "Performance": "Close background apps or check CPU usage."
    }
    return fixes.get(category, "No suggestion available.")

def confidence_label(conf):
    if conf >= 70:
        return "🟢 High"
    elif conf >= 40:
        return "🟡 Medium"
    else:
        return "🔴 Low"

# =========================
# TICKET SAVER
# =========================
def save_ticket(description, category, priority, cat_conf, pri_conf):
    file = "tickets.csv"

    if not os.path.exists(file):
        df = pd.DataFrame(columns=[
            "Ticket ID", "Description", "Category", "Priority",
            "Category Confidence", "Priority Confidence"
        ])
        df.to_csv(file, index=False)

    df = pd.read_csv(file)

    ticket_id = f"INC-{len(df)+1:03d}"

    new_ticket = {
        "Ticket ID": ticket_id,
        "Description": description,
        "Category": category,
        "Priority": priority,
        "Category Confidence": round(cat_conf, 2),
        "Priority Confidence": round(pri_conf, 2)
    }

    df = pd.concat([df, pd.DataFrame([new_ticket])], ignore_index=True)
    df.to_csv(file, index=False)

    return ticket_id

# =========================
# INPUT SECTION
# =========================
text = st.text_area("Enter IT issue description:")

# =========================
# BUTTON ACTION
# =========================

if st.button("Predict Ticket"):

    if text.strip() == "":
        st.warning("Please enter a ticket description")

    else:
        # Vectorize
        vec = vectorizer.transform([text])

        # Predict
        category = model_category.predict(vec)[0]
        priority = model_priority.predict(vec)[0]

        # Confidence
        category_probs = model_category.predict_proba(vec)[0]
        priority_probs = model_priority.predict_proba(vec)[0]

        cat_probs = category_probs
        pri_probs = priority_probs

        category_conf = max(cat_probs) * 100
        priority_conf = max(pri_probs) * 100

        # Save ticket
        ticket_id = save_ticket(
            text,
            category,
            priority,
            category_conf,
            priority_conf
        )

        # 🔥 SHOW RESULTS (THIS MUST BE INSIDE BUTTON)
        st.success(f"🎯 Category: {category}")
        st.warning(f"⚡ Priority: {priority}")
        st.info(f"🆔 Ticket ID: {ticket_id}")
        st.info(f"🔧 Fix: {suggest_fix(category)}")

st.markdown("---")
st.subheader("📊 Service Desk Analytics Dashboard")

if os.path.exists("tickets.csv"):
    df = pd.read_csv("tickets.csv")

    if len(df) > 0:
    
        # =========================
        # KPI METRICS
        # =========================
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📦 Total Tickets", len(df))

        with col2:
            st.metric("🔴 High Priority", len(df[df["Priority"] == "High"]))

        with col3:
            st.metric("🌐 Network Issues", len(df[df["Category"] == "Network"]))

    
        # =========================
        # CATEGORY CHART
        # =========================
        st.markdown("### 🎯 Ticket Categories")

        fig1, ax1 = plt.subplots(figsize=(6, 4))

        df["Category"].value_counts().plot(
        kind="bar",
        ax=ax1,
        color="#2E86C1"
        )

        ax1.set_title("Ticket Categories")
        ax1.set_ylabel("Number of Tickets")
        ax1.set_xlabel("Category")

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig1)

        # =========================
        # PRIORITY CHART
        # =========================
        st.markdown("### ⚡ Priority Breakdown")

        fig2, ax2 = plt.subplots(figsize=(5, 5))

        df["Priority"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax2,
        colors=["#E74C3C", "#F1C40F", "#2ECC71"]
)

        ax2.set_ylabel("")
        ax2.set_title("Priority Distribution")

        st.pyplot(fig2)

    else:
        st.info("No tickets available yet.")
else:
    st.info("Create tickets first to see analytics.")