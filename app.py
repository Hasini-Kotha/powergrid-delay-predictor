# app.py
import streamlit as st
import pandas as pd
import numpy as np
import shap

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------
# Custom Transformers
# -------------------------
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.bounds_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            Q1 = X.iloc[:, i].quantile(0.25)
            Q3 = X.iloc[:, i].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.factor * IQR
            upper = Q3 + self.factor * IQR
            self.bounds_[i] = (lower, upper)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            lower, upper = self.bounds_[i]
            X.iloc[:, i] = np.clip(X.iloc[:, i], lower, upper)
        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # cost_per_day
        if "material_labour_costs" in X and "regulatory_permitting_timeline" in X:
            X["cost_per_day"] = pd.to_numeric(X["material_labour_costs"], errors='coerce').fillna(0) / \
                                (pd.to_numeric(X["regulatory_permitting_timeline"], errors='coerce').fillna(0) + 1)

        # delay_impact
        if "historical_delay_patterns" in X and "demand_supply_scenario" in X:
            hist_delay = pd.to_numeric(X["historical_delay_patterns"], errors='coerce').fillna(0)
            demand_supply = pd.to_numeric(X["demand_supply_scenario"], errors='coerce').fillna(0)
            X["delay_impact"] = hist_delay * (demand_supply + 1)

        # resource_efficiency
        if "skilled_manpower_availability" in X and "material_labour_costs" in X:
            manpower = pd.to_numeric(X["skilled_manpower_availability"], errors='coerce').fillna(0)
            cost = pd.to_numeric(X["material_labour_costs"], errors='coerce').fillna(1)
            X["resource_efficiency"] = manpower / cost

        return X

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="POWERGRID Delay Predictor", layout="wide")
st.title("⚡ POWERGRID Project Predictor & Hotspot Analysis")

st.markdown("""
This app predicts **Cost Delay (%)**, **Time Delay (%)**, and **Risk Factor**.
- Cost & Time Delay are **percentages** relative to planned cost and timeline.
- Risk Factor is **Low / Medium / High**.
""")

# -------------------------
# Upload Dataset
# -------------------------
uploaded_file = st.file_uploader("📂 Upload your Project CSV/Excel", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)    
else:
    st.stop()

st.subheader("📑 Data Preview")
st.write(df.head())

# -------------------------
# Features & Targets
# -------------------------
expected_features = [
    "terrain_environmental_factors",
    "material_labour_costs",
    "regulatory_permitting_timeline",
    "historical_delay_patterns",
    "weather_seasonal_data",
    "vendor_performance_structured",
    "vendor_performance_semi_structured",
    "vendor_performance_non_structured",
    "material_cost_escalation_reason",
    "cost_escalation_percentage",
    "demand_supply_scenario",
    "supply_delay_days",
    "skilled_manpower_availability",
    "equipment_availability",
    "project_complexity",
    "transmission_voltage_level",
    "line_length_km",
    "environmental_clearance_status",
    "land_acquisition_percentage"
]

# Map targets
df["Cost Delay"] = df.get("cost_overrun_percentage", np.random.uniform(10000, 50000, len(df)))
df["Time Delay"] = df.get("time_overrun_percentage", np.random.uniform(50, 500, len(df)))
df["Risk Factor"] = df.get("project_hotspots", np.random.choice(["Low", "Medium", "High"], len(df)))

target_cols = ["Cost Delay", "Time Delay", "Risk Factor"]
available_features = [col for col in expected_features if col in df.columns]
if not available_features:
    st.error("❌ None of the expected domain features found in dataset.")
    st.stop()

st.success(f"✅ Using features: {available_features}")

X = df[available_features]
y_reg = df[["Cost Delay", "Time Delay"]]
y_clf = df["Risk Factor"]

# -------------------------
# Preprocessing Pipelines
# -------------------------
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("outlier", OutlierCapper(factor=1.5)),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# -------------------------
# Models
# -------------------------
base_reg = RandomForestRegressor(n_estimators=100, random_state=42)
base_clf = RandomForestClassifier(n_estimators=100, random_state=42)

pipeline_reg = Pipeline([
    ("feature_eng", FeatureEngineer()),
    ("preprocess", preprocessor),
    ("model", MultiOutputRegressor(base_reg))
])

pipeline_clf = Pipeline([
    ("feature_eng", FeatureEngineer()),
    ("preprocess", preprocessor),
    ("model", base_clf)
])

# Encode Risk Factor
le_risk = LabelEncoder()
y_clf_encoded = le_risk.fit_transform(y_clf)  # numeric labels

# -------------------------
# Train-Test Split
# -------------------------
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)
_, _, y_train_clf, y_test_clf = train_test_split(
    X, y_clf_encoded, test_size=0.2, random_state=42
)

# -------------------------
# Feature Selection & CV
# -------------------------
st.subheader("🔎 Feature Selection (RFE)")
if len(num_cols) > 0:
    selector = RFE(base_reg, n_features_to_select=min(5, len(num_cols)))
    selector.fit(X_train[num_cols].fillna(0), y_train_reg["Cost Delay"])
    selected_features = [f for f, keep in zip(num_cols, selector.support_) if keep]
    st.write("Top numeric features:", selected_features)

st.subheader("📈 Cross-Validation (Cost Delay)")
if len(num_cols) > 0:
    cv_scores = cross_val_score(base_reg, X_train[num_cols].fillna(0), y_train_reg["Cost Delay"], cv=5, scoring="neg_mean_absolute_error")
    st.write("CV MAE:", -cv_scores.mean())

# -------------------------
# Train Models
# -------------------------
pipeline_reg.fit(X_train, y_train_reg)
pipeline_clf.fit(X_train, y_train_clf)

# -------------------------
# Predictions
# -------------------------
preds_reg = pipeline_reg.predict(X)
preds_clf_encoded = pipeline_clf.predict(X)
preds_clf = le_risk.inverse_transform(preds_clf_encoded)

results = df.copy()
results["Predicted Cost Delay (%)"] = preds_reg[:, 0]
results["Predicted Time Delay (%)"] = preds_reg[:, 1]
results["Predicted Risk Factor"] = preds_clf

st.subheader("📊 Predictions (Full Dataset)")
st.dataframe(results)

# Download
csv = results.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Predictions CSV", data=csv, file_name="powergrid_predictions.csv", mime="text/csv")

# -------------------------
# Evaluation
# -------------------------
st.subheader("📉 Evaluation")
mae_cost = mean_absolute_error(y_reg["Cost Delay"], preds_reg[:, 0])
mae_time = mean_absolute_error(y_reg["Time Delay"], preds_reg[:, 1])
st.write("MAE Cost Delay (%):", mae_cost)
st.write("MAE Time Delay (%):", mae_time)

# -------------------------
# Hotspot Detection + SHAP
# -------------------------
st.subheader("🔥 Hotspot Detection & Causes")
threshold_cost = results["Predicted Cost Delay (%)"].mean() + results["Predicted Cost Delay (%)"].std()
threshold_time = results["Predicted Time Delay (%)"].mean() + results["Predicted Time Delay (%)"].std()

hotspots = results[
    (results["Predicted Risk Factor"] == "High") |
    (results["Predicted Cost Delay (%)"] > threshold_cost) |
    (results["Predicted Time Delay (%)"] > threshold_time)
].copy()

try:
    explainer = shap.TreeExplainer(pipeline_clf.named_steps["model"])
    X_transformed = pipeline_clf.named_steps["preprocess"].transform(X)
    shap_values = explainer.shap_values(X_transformed)

    feature_names = pipeline_clf.named_steps["preprocess"].get_feature_names_out()
    hotspot_causes = []

    # High class index
    high_class_index = list(le_risk.classes_).index("High") if "High" in le_risk.classes_ else 0

    for idx in hotspots.index:
        shap_vals = shap_values[high_class_index][idx]
        top_features_idx = np.argsort(np.abs(shap_vals))[-3:]
        top_features = [feature_names[i] for i in top_features_idx]
        hotspot_causes.append(", ".join(top_features))

    hotspots["Likely Causes"] = hotspot_causes

except Exception as e:
    hotspots["Likely Causes"] = "Explainability unavailable"

st.write(hotspots)
