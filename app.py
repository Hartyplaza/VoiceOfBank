# VoiceOfBank — Streamlit Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from pathlib import Path

st.set_page_config(
    page_title = "VoiceOfBank",
    page_icon  = "🏦",
    layout     = "wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    font-size: 2.2rem;
    font-weight: 800;
    color: #1e3a5f;
    margin-bottom: 0;
}
.sub-header {
    font-size: 1rem;
    color: #64748b;
    margin-bottom: 2rem;
}
.metric-card {
    background: #f8fafc;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border-left: 4px solid #4f8ef7;
}
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e3a5f;
    margin-bottom: 0.5rem;
}
.positive { color: #16a34a; font-weight: 700; }
.negative { color: #dc2626; font-weight: 700; }
.neutral  { color: #ca8a04; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BANKS      = ["Monzo", "Starling", "Barclays", "HSBC", "NatWest", "Lloyds"]
CHALLENGER = ["Monzo", "Starling"]
BANK_COLORS = {
    "Monzo"   : "#ef4444",
    "Starling": "#3b82f6",
    "Barclays": "#1d4ed8",
    "HSBC"    : "#dc2626",
    "NatWest" : "#7c3aed",
    "Lloyds"  : "#065f46",
}
API_URL = "http://localhost:8000"

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = Path("data/processed")
    df       = pd.read_csv(base / "reviews_bert.csv", parse_dates=["date"])
    lda      = pd.read_csv(base / "topics_lda.csv",   parse_dates=["date"])
    try:
        results = pd.read_csv(base / "classification_results.csv")
    except FileNotFoundError:
        results = None
    return df, lda, results

df, lda_df, results_df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## VoiceOfBank")
    st.markdown("Customer Voice Intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Overview", "Sentiment Analysis", "Complaint Topics", "Live Classifier"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Data**")
    st.markdown(f"Reviews: {len(df):,}")
    st.markdown(f"Banks: {df['bank'].nunique()}")
    st.markdown(f"Date range: {df['date'].min().strftime('%b %Y')} - {df['date'].max().strftime('%b %Y')}")

# ── Page: Overview ────────────────────────────────────────────────────────────
if page == "Overview":
    st.markdown('<p class="main-header">VoiceOfBank</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Customer Voice Intelligence from 30,000 UK Bank App Reviews</p>',
                unsafe_allow_html=True)

    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews", f"{len(df):,}")
    with col2:
        neg_rate = (df["bert_label"] == "Negative").mean() * 100
        st.metric("Overall Negative Rate", f"{neg_rate:.1f}%")
    with col3:
        best_bank = df.groupby("bank")["bert_prob_pos"].mean().idxmax()
        st.metric("Most Positive Bank", best_bank)
    with col4:
        worst_bank = df.groupby("bank")["bert_prob_pos"].mean().idxmin()
        st.metric("Most Negative Bank", worst_bank)

    st.markdown("---")

    # Average rating by bank
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Average Star Rating by Bank</p>', unsafe_allow_html=True)
        avg_rating = df.groupby("bank")["rating"].mean().sort_values(ascending=False).reset_index()
        colors = ["#22c55e" if b in CHALLENGER else "#4f8ef7" for b in avg_rating["bank"]]
        fig = go.Figure(go.Bar(
            x=avg_rating["bank"], y=avg_rating["rating"],
            marker_color=colors,
            text=avg_rating["rating"].round(2),
            textposition="outside",
        ))
        fig.update_layout(
            height=320, margin=dict(t=10,b=40,l=40,r=10),
            yaxis=dict(range=[0, 5.5]),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Negative Review Rate by Bank (%)</p>', unsafe_allow_html=True)
        neg_rate = df.groupby("bank").apply(
            lambda x: (x["bert_label"] == "Negative").mean() * 100
        ).sort_values(ascending=False).reset_index()
        neg_rate.columns = ["bank", "neg_rate"]
        colors2 = ["#22c55e" if b in CHALLENGER else "#4f8ef7" for b in neg_rate["bank"]]
        fig2 = go.Figure(go.Bar(
            x=neg_rate["bank"], y=neg_rate["neg_rate"],
            marker_color=colors2,
            text=neg_rate["neg_rate"].round(1).astype(str) + "%",
            textposition="outside",
        ))
        fig2.update_layout(
            height=320, margin=dict(t=10,b=40,l=40,r=10),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Sentiment drift
    st.markdown('<p class="section-header">Monthly Sentiment Drift (RoBERTa)</p>', unsafe_allow_html=True)
    df["year_month_dt"] = pd.to_datetime(df["year_month"])
    monthly = df.groupby(["year_month_dt","bank"])["bert_prob_pos"].mean().reset_index()

    fig3 = go.Figure()
    for bank in BANKS:
        sub = monthly[monthly["bank"] == bank].sort_values("year_month_dt")
        fig3.add_trace(go.Scatter(
            x=sub["year_month_dt"], y=sub["bert_prob_pos"],
            name=bank, mode="lines+markers",
            line=dict(color=BANK_COLORS[bank], width=2.5 if bank in CHALLENGER else 1.5,
                      dash="solid" if bank in CHALLENGER else "dash"),
            marker=dict(size=4),
        ))
    fig3.add_hline(y=0.5, line_dash="dot", line_color="gray", opacity=0.5)
    fig3.update_layout(
        height=350, margin=dict(t=10,b=40,l=50,r=10),
        xaxis_title="Month", yaxis_title="Mean P(Positive)",
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Page: Sentiment Analysis ──────────────────────────────────────────────────
elif page == "Sentiment Analysis":
    st.markdown('<p class="main-header">Sentiment Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">VADER vs RoBERTa sentiment comparison across all banks</p>',
                unsafe_allow_html=True)

    selected_bank = st.selectbox("Select Bank", ["All Banks"] + BANKS)

    if selected_bank != "All Banks":
        data = df[df["bank"] == selected_bank]
    else:
        data = df

    col1, col2, col3 = st.columns(3)
    with col1:
        pos_pct = (data["bert_label"] == "Positive").mean() * 100
        st.metric("Positive", f"{pos_pct:.1f}%")
    with col2:
        neu_pct = (data["bert_label"] == "Neutral").mean() * 100
        st.metric("Neutral", f"{neu_pct:.1f}%")
    with col3:
        neg_pct = (data["bert_label"] == "Negative").mean() * 100
        st.metric("Negative", f"{neg_pct:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">RoBERTa Sentiment Distribution</p>', unsafe_allow_html=True)
        counts = data["bert_label"].value_counts().reindex(["Positive","Neutral","Negative"]).fillna(0)
        fig = go.Figure(go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.5,
            marker_colors=["#22c55e","#eab308","#ef4444"],
        ))
        fig.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Aspect Sentiment Heatmap</p>', unsafe_allow_html=True)
        aspects = {
            "Customer Service" : "aspect_customer_service",
            "App & Interface"  : "aspect_app_interface",
            "Fees & Charges"   : "aspect_fees_charges",
            "Transfer & Speed" : "aspect_transfer_speed",
            "Security & Fraud" : "aspect_security_fraud",
        }
        heatmap_data = []
        for bank in BANKS:
            row = []
            sub = df[df["bank"] == bank]
            for col_name in aspects.values():
                if col_name in sub.columns:
                    row.append(sub[col_name].mean())
                else:
                    row.append(0)
            heatmap_data.append(row)

        fig2 = go.Figure(go.Heatmap(
            z=heatmap_data,
            x=list(aspects.keys()),
            y=BANKS,
            colorscale="RdYlGn",
            zmid=0,
            zmin=-0.5, zmax=0.5,
            text=[[f"{v:.2f}" for v in row] for row in heatmap_data],
            texttemplate="%{text}",
        ))
        fig2.update_layout(height=300, margin=dict(t=10,b=60,l=80,r=10))
        st.plotly_chart(fig2, use_container_width=True)

    # VADER vs RoBERTa comparison
    st.markdown('<p class="section-header">VADER vs RoBERTa Agreement</p>', unsafe_allow_html=True)
    agree = (data["vader_label"] == data["bert_label"]).mean() * 100
    st.write(f"Overall agreement rate: **{agree:.1f}%**")

    agree_by_sentiment = data.groupby("sentiment").apply(
        lambda x: (x["vader_label"] == x["bert_label"]).mean() * 100
    ).reset_index()
    agree_by_sentiment.columns = ["Sentiment", "Agreement %"]

    fig3 = go.Figure(go.Bar(
        x=agree_by_sentiment["Sentiment"],
        y=agree_by_sentiment["Agreement %"],
        marker_color=["#22c55e","#ef4444","#eab308"],
        text=agree_by_sentiment["Agreement %"].round(1).astype(str) + "%",
        textposition="outside",
    ))
    fig3.update_layout(
        height=280, margin=dict(t=10,b=40,l=50,r=10),
        yaxis=dict(range=[0,110]),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Page: Complaint Topics ────────────────────────────────────────────────────
elif page == "Complaint Topics":
    st.markdown('<p class="main-header">Complaint Topics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">LDA topic modelling on 1-2 star reviews</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Complaints", f"{len(lda_df):,}")
    with col2:
        complaint_rate = len(lda_df) / len(df) * 100
        st.metric("Complaint Rate", f"{complaint_rate:.1f}%")

    st.markdown("---")

    # Topic distribution by bank
    st.markdown('<p class="section-header">Complaint Topics by Bank (%)</p>', unsafe_allow_html=True)
    topic_bank = lda_df.groupby(["bank","lda_topic_label"]).size().unstack(fill_value=0)
    topic_bank_pct = topic_bank.div(topic_bank.sum(axis=1), axis=0).mul(100).round(1)

    fig = px.bar(
        topic_bank_pct.reset_index().melt(id_vars="bank", var_name="Topic", value_name="Pct"),
        x="bank", y="Pct", color="Topic", barmode="group",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        height=400, margin=dict(t=10,b=40,l=50,r=10),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="", yaxis_title="% of Complaints",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Spike analysis
    st.markdown('<p class="section-header">Complaint Spike Analysis — Barclays & Lloyds (Feb-Apr 2026)</p>',
                unsafe_allow_html=True)

    spike_banks = ["Barclays", "Lloyds"]
    before = lda_df[
        (lda_df["bank"].isin(spike_banks)) &
        (lda_df["date"] < "2026-02-01")
    ]
    during = lda_df[
        (lda_df["bank"].isin(spike_banks)) &
        (lda_df["date"] >= "2026-02-01") &
        (lda_df["date"] <= "2026-04-30")
    ]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Complaints Before Spike", len(before))
    with col2:
        st.metric("Complaints During Spike", len(during))

    col1, col2 = st.columns(2)
    with col1:
        before_counts = before["lda_topic_label"].value_counts().head(8)
        fig2 = go.Figure(go.Bar(
            y=before_counts.index[::-1], x=before_counts.values[::-1],
            orientation="h", marker_color="#4f8ef7",
        ))
        fig2.update_layout(
            title="Before Spike", height=280,
            margin=dict(t=30,b=10,l=150,r=10),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        during_counts = during["lda_topic_label"].value_counts().head(8)
        fig3 = go.Figure(go.Bar(
            y=during_counts.index[::-1], x=during_counts.values[::-1],
            orientation="h", marker_color="#ef4444",
        ))
        fig3.update_layout(
            title="During Spike (Feb-Apr 2026)", height=280,
            margin=dict(t=30,b=10,l=150,r=10),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig3, use_container_width=True)

# ── Page: Live Classifier ─────────────────────────────────────────────────────
elif page == "Live Classifier":
    st.markdown('<p class="main-header">Live Review Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter a bank review and get real-time sentiment prediction</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        review_text = st.text_area(
            "Review text",
            placeholder="e.g. The app keeps crashing after the latest update...",
            height=120,
        )
        bank_select = st.selectbox("Bank (optional)", ["Not specified"] + BANKS)
        bank_val    = None if bank_select == "Not specified" else bank_select

        if st.button("Classify Review", type="primary", use_container_width=True):
            if not review_text.strip():
                st.warning("Please enter a review to classify.")
            else:
                with st.spinner("Classifying..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/predict",
                            json={"text": review_text, "bank": bank_val},
                            timeout=30,
                        )
                        if resp.status_code == 200:
                            result = resp.json()
                            st.session_state["last_result"] = result
                        else:
                            st.error(f"API error: {resp.status_code}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to API. Make sure the FastAPI server is running on port 8000.")

    with col2:
        st.markdown("**Example reviews:**")
        examples = [
            "Amazing app, transfers are instant and the design is beautiful",
            "Locked out for 3 days, customer service completely useless",
            "It works fine, nothing special about it",
            "Constant crashes after latest update, absolutely terrible",
        ]
        for ex in examples:
            st.caption(f'"{ex[:60]}..."')

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        st.markdown("---")
        st.markdown("### Prediction Results")

        col1, col2, col3 = st.columns(3)
        label = result["roberta_label"]
        conf  = result["roberta_confidence"] * 100

        color_map = {"Positive": "normal", "Negative": "inverse", "Neutral": "off"}
        with col1:
            st.metric("RoBERTa Prediction", label, f"{conf:.1f}% confidence")
        with col2:
            st.metric("XGBoost Prediction", result["xgb_label"],
                      f"{result['xgb_confidence']*100:.1f}% confidence")
        with col3:
            agree = "Agree" if result["roberta_label"] == result["xgb_label"] else "Disagree"
            st.metric("Model Agreement", agree)

        st.markdown("**RoBERTa Probability Breakdown:**")
        prob_df = pd.DataFrame({
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Probability": [
                result["roberta_prob_positive"],
                result["roberta_prob_neutral"],
                result["roberta_prob_negative"],
            ]
        })
        fig = go.Figure(go.Bar(
            x=prob_df["Sentiment"],
            y=prob_df["Probability"],
            marker_color=["#22c55e","#eab308","#ef4444"],
            text=(prob_df["Probability"] * 100).round(1).astype(str) + "%",
            textposition="outside",
        ))
        fig.update_layout(
            height=280, margin=dict(t=10,b=40,l=50,r=10),
            yaxis=dict(range=[0, 1.15]),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)
