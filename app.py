import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="SL Used Car Price Ensemble Dashboard", layout="wide")

@st.cache_resource
def load_artifacts():
    bundle = joblib.load("artifacts/models_bundle.joblib")
    with open("artifacts/metrics.json", "r") as f:
        metrics = json.load(f)
    with open("artifacts/ui_options.json", "r") as f:
        ui_options = json.load(f)
    preds = pd.read_csv("artifacts/test_predictions.csv")
    return bundle, metrics, ui_options, preds

bundle, metrics, ui_options, preds_df = load_artifacts()
metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
best_model_name = metrics_df.sort_values("RMSE (LKR M)").iloc[0]["Model"]

st.title("Sri Lankan Used Car Price Prediction — Ensemble Learning Dashboard")
st.caption("CM4371 Machine Learning & Pattern Recognition | Bagging vs. Boosting vs. Stacking vs. RF+ANN Hybrid")

tab1, tab2, tab3, tab4 = st.tabs([
    "Live Price Predictor",
    "Model Comparison & Charts",
    "Why Models Behaved This Way",
    "GenAI Disclosure & About"
])

# --- TAB 1: LIVE PREDICTOR ---
with tab1:
    st.subheader("Estimate Sri Lankan Market Value (LKR Millions)")

    c1, c2, c3 = st.columns(3)
    with c1:
        brands = sorted(ui_options["brand_to_models"].keys())
        default_brand_idx = brands.index("TOYOTA") if "TOYOTA" in brands else 0
        brand = st.selectbox("Brand", brands, index=default_brand_idx)
        model_name = st.selectbox("Model", ui_options["brand_to_models"][brand])
        yom = st.slider("Year of Manufacture (YOM)", 1985, 2025, 2017)

    with c2:
        engine_cc = st.number_input("Engine Capacity (cc)", min_value=600, max_value=5000, value=1500, step=100)
        mileage_km = st.number_input("Mileage (KM)", min_value=0, max_value=400000, value=65000, step=5000)
        gear = st.selectbox("Transmission (Gear)", ui_options["gears"])
        fuel = st.selectbox("Fuel Type", ui_options["fuels"])

    with c3:
        towns = ui_options["towns"]
        default_town_idx = towns.index("Colombo") if "Colombo" in towns else 0
        town = st.selectbox("Town", towns, index=default_town_idx)
        condition = st.selectbox("Condition", ui_options["conditions"])
        leasing = st.selectbox("Leasing Status", ui_options["leasings"])

    st.markdown("**Vehicle Features:**")
    f1, f2, f3, f4 = st.columns(4)
    ac = int(f1.checkbox("Air Condition", value=True))
    ps = int(f2.checkbox("Power Steering", value=True))
    pm = int(f3.checkbox("Power Mirror", value=True))
    pw = int(f4.checkbox("Power Window", value=True))

    input_df = pd.DataFrame([{
        "Brand": brand,
        "Model": model_name,
        "Gear": gear,
        "Fuel Type": fuel,
        "Town": town,
        "Leasing": leasing,
        "Condition": condition,
        "YOM": yom,
        "Engine_CC": float(engine_cc),
        "Mileage_KM": float(mileage_km),
        "AIR CONDITION": ac,
        "POWER STEERING": ps,
        "POWER MIRROR": pm,
        "POWER WINDOW": pw,
    }])

    if st.button("Predict Price Across All 5 Ensemble Models", type="primary"):
        row_preds = {}
        for name, pipe in bundle["models"].items():
            row_preds[name] = round(float(pipe.predict(input_df)[0]), 2)

        X_in = bundle["hybrid_prep"].transform(input_df)
        hyb_val = bundle["hybrid_rf"].predict(X_in)[0] + bundle["hybrid_ann"].predict(X_in)[0]
        row_preds["Hybrid (RF + ANN)"] = round(float(hyb_val), 2)

        st.success(f"Best Model ({best_model_name}): **LKR {row_preds[best_model_name]} Million**")
        cols = st.columns(len(row_preds))
        for i, (m_name, val) in enumerate(row_preds.items()):
            cols[i].metric(label=m_name, value=f"LKR {val}M")

# --- TAB 2: COMPARISON & CHARTS ---
with tab2:
    st.subheader("Test Set Performance (Lower RMSE/MAE & Higher R² is Better)")
    st.dataframe(metrics_df.sort_values("RMSE (LKR M)"), use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_rmse = px.bar(
            metrics_df.sort_values("RMSE (LKR M)"),
            x="RMSE (LKR M)", y="Model", orientation="h",
            title="RMSE Comparison (LKR Millions)", text="RMSE (LKR M)",
            color="RMSE (LKR M)", color_continuous_scale="Blues_r"
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col_b:
        fig_r2 = px.bar(
            metrics_df.sort_values("R2 Score", ascending=False),
            x="R2 Score", y="Model", orientation="h",
            title="R² Score Comparison", text="R2 Score",
            color="R2 Score", color_continuous_scale="Greens"
        )
        fig_r2.update_xaxes(range=[0.85, 0.95])
        st.plotly_chart(fig_r2, use_container_width=True)

    selected_model = st.selectbox(
        "Inspect Actual vs. Predicted Prices:",
        [c for c in preds_df.columns if c != "Actual"]
    )
    fig_scatter = px.scatter(
        preds_df, x="Actual", y=selected_model,
        title=f"Actual vs. Predicted Price (LKR M) — {selected_model}",
        labels={"Actual": "Actual Price (LKR M)", selected_model: "Predicted Price (LKR M)"},
        opacity=0.55
    )
    max_val = float(preds_df["Actual"].max())
    fig_scatter.add_shape(
        type="line", x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(color="red", dash="dash")
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- TAB 3: INTERPRETATION ---
with tab3:
    st.subheader("Why Did the Models Perform This Way?")
    st.markdown("""
    * **Hybrid (RF + ANN) — Best Overall (`RMSE: 1.219M LKR`, `R²: 0.9143`):**
      Random Forest cleanly partitions high-cardinality categorical attributes (Brand, Model, Fuel Type, Town), while the Neural Network (`MLPRegressor`) learns the smooth continuous residuals from Year of Manufacture, Mileage, and Engine CC. This overcomes Random Forest's step-function limitation.
    * **Bagging (Random Forest) — Lowest MAE (`0.576M LKR`, `R²: 0.9112`):**
      Real Sri Lankan classified listings contain noisy outliers (e.g., negotiable pricing and varying vehicle conditions). Bootstrap averaging dampens variance effectively, letting Random Forest outperform standalone boosting models.
    * **Boosting (XGBoost & Gradient Boosting) — Fastest Training (`XGBoost: 1.61s`):**
      Sequential error correction is more sensitive to noisy scraped outliers, resulting in slightly higher RMSE (`1.285M` and `1.307M`). However, XGBoost achieved over 90.4% $R^2$ in just **1.61 seconds**, making it the fastest model to iterate on.
    * **Stacking (RF + XGB + GBR → Ridge) — High Compute Cost (`145.64s`, `R²: 0.9081`):**
      Blending diverse base learners improved over standalone boosting, but 5-fold cross-validation made it nearly **9× slower** than the RF + ANN hybrid without surpassing its accuracy.
    """)

# --- TAB 4: GENAI DISCLOSURE ---
with tab4:
    st.subheader("Generative AI Disclosure & Dataset")
    st.info(
        "In accordance with CM4371 assignment guidelines, Generative AI (Gemini) was used to assist with "
        "Python data-cleaning boilerplate, Streamlit UI layout, and structuring the model comparison pipeline. "
    )
    st.markdown(
        "**Dataset:** Real Sri Lankan used vehicle listings (9,788 records scraped from local marketplaces like "
        "ikman.lk and Riyasewana.lk), converted from LKR Lakhs to LKR Millions."
    )