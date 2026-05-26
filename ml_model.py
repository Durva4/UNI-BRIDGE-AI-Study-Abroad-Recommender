import pandas as pd
import numpy as np
import joblib
import glob
import os
from sklearn.preprocessing import LabelEncoder

# ── Region Map ──
REGION_MAP = {
    'Australia': 'Oceania',      'New Zealand': 'Oceania',
    'Canada': 'North America',   'Mexico': 'North America',
    'Dominican Republic': 'North America', 'Panama': 'North America',
    'Argentina': 'South America','Brazil': 'South America',
    'Colombia': 'South America', 'Ecuador': 'South America',
    'Peru': 'South America',     'Uruguay': 'South America',
    'Germany': 'Europe',         'France': 'Europe',
    'Netherlands': 'Europe',     'Sweden': 'Europe',
    'Norway': 'Europe',          'Denmark': 'Europe',
    'Finland': 'Europe',         'Belgium': 'Europe',
    'Austria': 'Europe',         'Switzerland': 'Europe',
    'Italy': 'Europe',           'Spain': 'Europe',
    'Portugal': 'Europe',        'Greece': 'Europe',
    'Poland': 'Europe',          'Hungary': 'Europe',
    'Romania': 'Europe',         'Bulgaria': 'Europe',
    'Croatia': 'Europe',         'Cyprus': 'Europe',
    'Slovenia': 'Europe',        'Serbia': 'Europe',
    'Iceland': 'Europe',         'Luxembourg': 'Europe',
    'Ireland': 'Europe',         'Ukraine': 'Europe',
    'India': 'Asia',             'Japan': 'Asia',
    'Singapore': 'Asia',         'Malaysia': 'Asia',
    'Indonesia': 'Asia',         'Taiwan': 'Asia',
    'Thailand': 'Asia',          'Bangladesh': 'Asia',
    'Uzbekistan': 'Asia',
    'Saudi Arabia': 'Middle East','Kuwait': 'Middle East',
    'Bahrain': 'Middle East',    'Lebanon': 'Middle East',
    'Israel': 'Middle East',
    'Egypt': 'Africa',           'Nigeria': 'Africa',
    'South Africa': 'Africa',    'Ghana': 'Africa',
    'Morocco': 'Africa',         'Tunisia': 'Africa',
}


def load_artifacts(model_dir='.'):
    """
    Load all saved ML artifacts from disk.
    Returns: model, scaler, le_country, feature_columns, country_df
    """
    # Load model
    model_files = glob.glob(os.path.join(model_dir, 'best_model_*.pkl'))
    if not model_files:
        raise FileNotFoundError("No best_model_*.pkl found! Run notebook Step 8 first.")
    model = joblib.load(model_files[0])

    # Load scaler, encoder, feature columns
    scaler         = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))
    le_country     = joblib.load(os.path.join(model_dir, 'label_encoder_country.pkl'))
    feature_cols   = joblib.load(os.path.join(model_dir, 'feature_columns.pkl'))

    # Load country data
    country_df = pd.read_csv(os.path.join(model_dir, 'final_df.csv'))

    # Add Region and cost columns
    country_df['Region'] = country_df['Country'].map(REGION_MAP).fillna('Other')
    country_df['Annual_Cost_USD'] = country_df['Tuition_USD'] + country_df['Rent_USD'] * 12

    # Add QS Composite
    country_df['QS_Composite'] = (
        country_df['Academic_Reputation'] * 0.4 +
        country_df['Employer_Reputation'] * 0.3 +
        country_df['Research_Quality']    * 0.2 +
        country_df['Employment_Outcomes'] * 0.1
    )

    return model, scaler, le_country, feature_cols, country_df


def recommend_country(
    gpa, ielts, budget_usd,
    preferred_region, degree_level, field_of_study,
    model, scaler, le_country, feature_cols, country_df,
    top_n=5
):
    """
    ML-based country recommendation function.

    Parameters:
        gpa, ielts, budget_usd     : Student profile
        preferred_region           : e.g. 'Europe', 'Asia'
        degree_level               : 'Bachelor', 'Master', 'PhD'
        field_of_study             : e.g. 'Engineering', 'Business'
        model, scaler, le_country  : Loaded ML artifacts
        feature_cols               : List of training feature columns
        country_df                 : Country dataset with Region & costs
        top_n                      : Number of recommendations

    Returns:
        DataFrame with top N recommended countries
    """

    # Check if model needs scaling (LR / KNN)
    needs_scale = isinstance(model, tuple)
    if needs_scale:
        model, _ = model

    model_classes   = model.classes_
    qs_max          = country_df['QS_Composite'].max()
    records         = []

    for _, c_row in country_df.iterrows():
        country = c_row['Country']

        # Skip countries not seen during training
        if country not in le_country.classes_:
            continue

        total_cost   = c_row['Tuition_USD'] + c_row['Rent_USD'] * 12
        region_match = int(preferred_region == c_row['Region'])

        # ── Categorical dict ──
        cat_dict = {
            'Preferred_Region': preferred_region,
            'Degree_Level'    : degree_level,
            'Field_of_Study'  : field_of_study,
        }

        # ── Numeric dict (only student features — no country leakage) ──
        num_dict = {
            'GPA'       : gpa,
            'IELTS'     : ielts,
            'Budget_USD': budget_usd,
        }

        # ── Build feature vector ──
        cat_df = pd.get_dummies(
            pd.DataFrame([cat_dict]),
            columns=['Preferred_Region', 'Degree_Level', 'Field_of_Study'],
            drop_first=False
        )
        num_df = pd.DataFrame([num_dict])
        fv = pd.concat([num_df, cat_df], axis=1)

        # Align with training columns
        fv = fv.reindex(columns=feature_cols, fill_value=0)
        fv = fv.astype(float)

        # Scale if needed
        if hasattr(scaler, 'transform'):
            try:
                fv_input = scaler.transform(fv)
            except Exception:
                fv_input = fv.values
        else:
            fv_input = fv.values

        # ── Predict ──
        proba           = model.predict_proba(fv_input)[0]
        encoded_country = le_country.transform([country])[0]

        if encoded_country not in model_classes:
            continue

        proba_idx = np.where(model_classes == encoded_country)[0][0]
        ml_prob   = float(proba[proba_idx])

        # ── Scoring ──
        affordable        = 1 if budget_usd >= total_cost else 0
        affordability_score = min(budget_usd / (total_cost + 1), 2.0) / 2.0
        qs_score          = min(c_row['QS_Composite'] / qs_max, 1.0)

        final_score = (
            0.25 * ml_prob +
            0.30 * region_match +
            0.25 * affordability_score +
            0.10 * qs_score +
            0.10 * (1 - qs_score)   # diversity bonus
        )

        # ── Admission chance ──
        if ml_prob >= 0.20:
            chance = 'Safe'
        elif ml_prob >= 0.08:
            chance = 'Moderate'
        else:
            chance = 'Ambitious'

        records.append({
            'Country'           : country,
            'Final_Score'       : final_score,
            'ML_Confidence_%'   : round(ml_prob * 100, 2),
            'Region'            : c_row['Region'],
            'region_match_num'  : region_match,
            'Region_Match'      : '✅' if region_match else '—',
            'Annual_Cost_USD'   : int(total_cost),
            'Tuition_USD'       : int(c_row['Tuition_USD']),
            'Rent_USD'          : int(c_row['Rent_USD']),
            'Affordable'        : '✅' if affordable else '⚠️',
            'Admission_Chance'  : chance,
        })

    rec_df = pd.DataFrame(records)

    if rec_df.empty:
        return rec_df

    # Sort: region match first → affordable → final score
    rec_df = rec_df.sort_values(
        ['region_match_num', 'Affordable', 'Final_Score'],
        ascending=[False, False, False]
    ).head(top_n)

    rec_df = rec_df.drop(columns=['region_match_num'])
    rec_df['Final_Score'] = rec_df['Final_Score'].round(4)
    rec_df.index = range(1, len(rec_df) + 1)

    return rec_df
