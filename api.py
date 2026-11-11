"""
api.py

A RESTish API, built with Flask, that serves FOUR machine learning artifacts
produced in earlier labs:

    1. Regression       -- SME monthly revenue prediction
    2. Classification   -- SME credit risk category
    3. Clustering       -- Mall customer segment assignment
    4. Association Rule -- Product recommendation from a cart
       Mining (ARM)

The point of this file is to show you that a fitted scikit-learn /
imbalanced-learn Pipeline is a SINGLE object that already knows how to impute
missing values, encode categorical columns, scale numeric columns, and (for
the classifier) apply SMOTE only during training -- never during inference.
Because all of that preprocessing lives inside the Pipeline itself, this file
never manually label-encodes anything, never manually engineers date features,
and never manually reorders columns before calling .predict(). It builds a
DataFrame straight from the incoming JSON and hands it to the pipeline as-is.
This is the main lesson of the lab: "the pipeline IS the deployment artifact."

Run this file directly to start a local development server:

    python api.py
"""

import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

# -----------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing)
# -----------------------------------------------------------------------
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "https://localhost:443", "https://127.0.0.1:443",
    "http://localhost:63342",  # VS Code Live Server extension
    "null",  # allows opening a frontend .html file directly (file://) with no web server
]}})

# -----------------------------------------------------------------------
# The best practice is to load the four persisted artifacts ONCE, at startup,
# not per-request.
# -----------------------------------------------------------------------
MODEL_DIR = "./model"

regression_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "lasso_regressor_for_sme_revenue.joblib")
)

classification_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "svc_classifier_for_sme_credit_risk.joblib")
)

clustering_artifact = joblib.load(
    os.path.join(MODEL_DIR, "kmeans_mall_customer_segmentation.joblib")
)
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]

arm_artifact = joblib.load(
    os.path.join(MODEL_DIR, "apriori_recommendation_rules.joblib")
)
recommendation_lookup = arm_artifact["recommendation_lookup"]
sku_lookup = arm_artifact["sku_lookup"]

print("All the four artifacts have been loaded. The API is ready.")


# -----------------------------------------------------------------------
# Member 1 contribution: lightweight input validation for the regression
# and classification endpoints.
#
# Without this, a missing field doesn't fail cleanly -- it propagates into
# the pipeline's ColumnTransformer and surfaces as an opaque 500 Internal
# Server Error with a scikit-learn stack trace, which is useless to a
# frontend developer or API consumer trying to integrate against this
# service. Validating the payload up front and returning a 400 with the
# exact missing field names is the difference between "the API told me
# what I did wrong" and "the API crashed."
#
# NULLABLE_FIELDS lists fields that are allowed to be explicitly null in
# the JSON body (structural missingness that the pipeline's own imputer is
# designed to handle) -- those still must be PRESENT as a key, just not
# necessarily non-null.
# -----------------------------------------------------------------------
REGRESSION_REQUIRED_FIELDS = [
    "sector", "county", "quarter", "business_age_years", "num_employees",
    "owner_education_level", "monthly_marketing_spend_kes",
    "avg_daily_foot_traffic", "online_presence_score", "loan_amount_kes",
    "interest_rate_pct", "inflation_rate_pct", "competitor_density",
    "customer_satisfaction_score",
]
REGRESSION_NULLABLE_FIELDS = {"avg_daily_foot_traffic"}

CLASSIFICATION_REQUIRED_FIELDS = [
    "sector", "county", "quarter", "loan_purpose", "owner_education_level",
    "business_age_years", "num_employees", "monthly_revenue_kes",
    "loan_amount_kes", "loan_term_months", "interest_rate_pct",
    "collateral_value_kes", "debt_to_income_ratio", "credit_score",
    "num_previous_loans", "missed_payments_count", "online_presence_score",
    "customer_satisfaction_score", "competitor_density",
]
CLASSIFICATION_NULLABLE_FIELDS = {"collateral_value_kes"}


def validate_payload(payload, required_fields, nullable_fields):
    """
    Returns a list of human-readable problems with the payload, or an empty
    list if it is valid. Does not raise -- the caller decides how to turn
    this into an HTTP response.
    """
    if payload is None:
        return ["Request body is missing or is not valid JSON "
                "(did you set Content-Type: application/json?)"]

    problems = []
    missing = [f for f in required_fields if f not in payload]
    if missing:
        problems.append(f"Missing required field(s): {', '.join(missing)}")

    unexpectedly_null = [
        f for f in required_fields
        if f in payload and payload[f] is None and f not in nullable_fields
    ]
    if unexpectedly_null:
        problems.append(
            f"Field(s) must not be null: {', '.join(unexpectedly_null)}"
        )

    return problems


# 1. REGRESSION -- predict monthly revenue (KES) for an SME

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/sme-revenue-regressor/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"sector": "Technology", "county": "Nairobi", "quarter": "Q3",
#             "business_age_years": 4.5, "num_employees": 12,
#             "owner_education_level": "Secondary",
#             "monthly_marketing_spend_kes": 45000,
#             "avg_daily_foot_traffic": null,
#             "online_presence_score": 78.0, "loan_amount_kes": 250000,
#             "interest_rate_pct": 14.5, "inflation_rate_pct": 6.5,
#             "competitor_density": 5, "customer_satisfaction_score": 4.2}'
@app.route("/api/v1/models/sme-revenue-regressor/predictions", methods=["POST"])
def predict_revenue():
    payload = request.get_json(silent=True)

    problems = validate_payload(
        payload, REGRESSION_REQUIRED_FIELDS, REGRESSION_NULLABLE_FIELDS
    )
    if problems:
        return jsonify({"errors": problems}), 400

    input_row = pd.DataFrame([payload])

    predicted_log_revenue = regression_pipeline.predict(input_row)[0]
    import numpy as np
    predicted_revenue_kes = float(np.expm1(predicted_log_revenue))

    return jsonify({
        "predicted_monthly_revenue_kes": round(predicted_revenue_kes, 2),
        "model": "Lasso Regressor",
    })


# 2. CLASSIFICATION -- predict SME credit risk category (Low/Medium/High)

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/sme-credit-risk-classifier/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"sector": "Retail", "county": "Nairobi", "quarter": "Q2",
#             "loan_purpose": "Working Capital", "owner_education_level": "Secondary",
#             "business_age_years": 2.1, "num_employees": 4,
#             "monthly_revenue_kes": 850000, "loan_amount_kes": 150000,
#             "loan_term_months": 12, "interest_rate_pct": 16.5,
#             "collateral_value_kes": null, "debt_to_income_ratio": 0.62,
#             "credit_score": 590, "num_previous_loans": 1,
#             "missed_payments_count": 3, "online_presence_score": 40.0,
#             "customer_satisfaction_score": 3.4, "competitor_density": 6}'
@app.route("/api/v1/models/sme-credit-risk-classifier/predictions", methods=["POST"])
def predict_credit_risk():
    payload = request.get_json(silent=True)

    problems = validate_payload(
        payload, CLASSIFICATION_REQUIRED_FIELDS, CLASSIFICATION_NULLABLE_FIELDS
    )
    if problems:
        return jsonify({"errors": problems}), 400

    input_row = pd.DataFrame([payload])

    predicted_class = classification_pipeline.predict(input_row)[0]
    predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
    class_labels = classification_pipeline.named_steps["model"].classes_

    return jsonify({
        "predicted_credit_risk_category": str(predicted_class),
        "class_probabilities": {
            str(label): round(float(prob), 4)
            for label, prob in zip(class_labels, predicted_probabilities)
        },
        "model": "Support Vector Classifier (tuned, SMOTE-resampled at training only)",
    })


# 3. CLUSTERING -- assign a new mall customer to an existing segment

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/mall-customer-segmenter/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"Age": 27, "Annual Income (k$)": 45, "Spending Score (1-100)": 72}'
@app.route("/api/v1/models/mall-customer-segmenter/predictions", methods=["POST"])
def predict_customer_segment():
    payload = request.get_json()
    input_row = pd.DataFrame(
        [[
            payload["Age"],
            payload["Annual Income (k$)"],
            payload["Spending Score (1-100)"]
        ]],
        columns=clustering_scaler.feature_names_in_
    )

    scaled_input = clustering_scaler.transform(input_row)
    assigned_cluster = int(clustering_kmeans.predict(scaled_input)[0])

    return jsonify({
        "assigned_cluster": assigned_cluster,
        "model": "K-Means (k=5, standardized features)",
    })


# 4. ASSOCIATION RULE MINING -- recommend products for a shopping cart

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/ecommerce-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"cart": ["H06", "H07"]}'
@app.route("/api/v1/models/ecommerce-recommender/predictions", methods=["POST"])
def recommend_products():
    payload = request.get_json()
    cart = set(payload.get("cart", []))
    top_n = int(payload.get("top_n", 3))

    candidates = {}
    for antecedent, consequent_list in recommendation_lookup.items():
        if antecedent.issubset(cart):
            for sku, confidence, lift in consequent_list:
                if sku not in cart:
                    if sku not in candidates or confidence > candidates[sku][0]:
                        candidates[sku] = (confidence, lift)

    ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:top_n]
    recommendations = [
        {
            "sku": str(sku),
            "product_name": sku_lookup.loc[sku, "product_name"],
            "confidence": round(float(confidence), 3),
            "lift": round(float(lift), 2),
        }
        for sku, (confidence, lift) in ranked
    ]

    return jsonify({
        "cart": sorted(cart),
        "recommendations": recommendations,
        "model": "Apriori association rules, cleaned + precomputed lookup",
    })


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)