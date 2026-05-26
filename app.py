import streamlit as st
import pandas as pd
import numpy as np
from ml_model import load_artifacts, recommend_country
from ai_explainer import get_country_explanation

# ── Page config ──
st.set_page_config(
    page_title="Uni-Bridge 🌉",
    page_icon="🎓",
    layout="wide"
)

# ── Load ML artifacts ──
@st.cache_resource
def load_ml():
    return load_artifacts(model_dir='.')

model, scaler, le_country, feature_cols, country_df = load_ml()

# ── Header ──
st.title("🌉 Uni-Bridge")
st.subheader("Bridge To Your Dream Universities")
st.markdown("---")

# ── Input Section ──
st.markdown("### 📋 Your Profile")

col1, col2, col3 = st.columns(3)

with col1:
    gpa   = st.slider("GPA", 1.5, 4.0, 3.2, 0.1)
    ielts = st.slider("IELTS Score", 4.5, 9.0, 6.5, 0.5)

with col2:
    budget = st.selectbox(
        "Annual Budget (USD)",
        [8000, 12000, 18000, 25000, 35000, 50000, 70000],
        index=3
    )
    region = st.selectbox(
        "Preferred Region",
        ['Europe', 'Asia', 'North America', 'South America',
         'Oceania', 'Middle East', 'Africa']
    )

with col3:
    degree = st.selectbox("Degree Level", ['Bachelor', 'Master', 'PhD'])
    field  = st.selectbox(
        "Field of Study",
        ['Engineering', 'Business', 'Medicine', 'Arts',
         'Computer Science', 'Law', 'Natural Sciences', 'Social Sciences']
    )

st.markdown("---")

# ── Recommend Button ──
if st.button("🚀 Find My Universities", type="primary", use_container_width=True):

    with st.spinner("Finding best countries for you..."):
        results = recommend_country(
            gpa=gpa,
            ielts=ielts,
            budget_usd=budget,
            preferred_region=region,
            degree_level=degree,
            field_of_study=field,
            model=model,
            scaler=scaler,
            le_country=le_country,
            feature_cols=feature_cols,
            country_df=country_df,
            top_n=5
        )

    if results.empty:
        st.error("❌ No recommendations found. Try changing your preferences.")
    else:
        # ── Results Table ──
        st.markdown("### 🌍 Recommended Countries")
        st.dataframe(results, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🤖 AI-Powered Insights")

        # ── Tabs for top 3 countries ──
        top3 = results.head(3)
        tab_labels = [f"🌍 {row['Country']}" for _, row in top3.iterrows()]
        tabs = st.tabs(tab_labels)

        for tab, (_, row) in zip(tabs, top3.iterrows()):
            with tab:

                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("ML Confidence",   f"{row['ML_Confidence_%']:.1f}%")
                m2.metric("Annual Cost",      f"${row['Annual_Cost_USD']:,}")
                m3.metric("Region Match",     row['Region_Match'])
                m4.metric("Admission Chance", row['Admission_Chance'])

                st.markdown("---")

                with st.spinner(f"Getting AI insights for {row['Country']}..."):
                    explanation = get_country_explanation(
                        country=row['Country'],
                        gpa=gpa,
                        ielts=ielts,
                        budget_usd=budget,
                        field_of_study=field,
                        degree_level=degree,
                        ml_confidence=row['ML_Confidence_%']
                    )

                st.markdown(explanation)
