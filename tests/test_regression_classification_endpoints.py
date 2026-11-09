"""
tests/test_regression_classification_endpoints.py

Member 1 contribution: smoke tests for the two endpoints I own --
the SME revenue regressor and the SME credit risk classifier.

These are integration tests: they expect api.py to already be running
locally (python api.py) on http://127.0.0.1:5000. Run them with:

    python tests/test_regression_classification_endpoints.py
"""

import sys

import requests

BASE_URL = "http://127.0.0.1:5000/api/v1/models"

VALID_REGRESSION_PAYLOAD = {
    "sector": "Technology", "county": "Nairobi", "quarter": "Q3",
    "business_age_years": 4.5, "num_employees": 12,
    "owner_education_level": "Secondary",
    "monthly_marketing_spend_kes": 45000,
    "avg_daily_foot_traffic": None,
    "online_presence_score": 78.0, "loan_amount_kes": 250000,
    "interest_rate_pct": 14.5, "inflation_rate_pct": 6.5,
    "competitor_density": 5, "customer_satisfaction_score": 4.2,
}

VALID_CLASSIFICATION_PAYLOAD = {
    "sector": "Retail", "county": "Nairobi", "quarter": "Q2",
    "loan_purpose": "Working Capital", "owner_education_level": "Secondary",
    "business_age_years": 2.1, "num_employees": 4,
    "monthly_revenue_kes": 850000, "loan_amount_kes": 150000,
    "loan_term_months": 12, "interest_rate_pct": 16.5,
    "collateral_value_kes": None, "debt_to_income_ratio": 0.62,
    "credit_score": 590, "num_previous_loans": 1,
    "missed_payments_count": 3, "online_presence_score": 40.0,
    "customer_satisfaction_score": 3.4, "competitor_density": 6,
}

results = []


def check(name, condition):
    results.append((name, condition))
    print(f"{'PASS' if condition else 'FAIL'} -- {name}")


def test_regression_valid_request():
    response = requests.post(
        f"{BASE_URL}/sme-revenue-regressor/predictions",
        json=VALID_REGRESSION_PAYLOAD,
    )
    check("regression: valid request returns 200", response.status_code == 200)
    body = response.json()
    check(
        "regression: response has a positive predicted revenue",
        "predicted_monthly_revenue_kes" in body
        and body["predicted_monthly_revenue_kes"] > 0,
    )


def test_regression_missing_field_returns_400():
    bad_payload = dict(VALID_REGRESSION_PAYLOAD)
    del bad_payload["sector"]
    response = requests.post(
        f"{BASE_URL}/sme-revenue-regressor/predictions", json=bad_payload
    )
    check(
        "regression: missing field returns 400 (not a 500 crash)",
        response.status_code == 400,
    )
    check(
        "regression: 400 response names the missing field",
        "sector" in str(response.json()),
    )


def test_classification_valid_request():
    response = requests.post(
        f"{BASE_URL}/sme-credit-risk-classifier/predictions",
        json=VALID_CLASSIFICATION_PAYLOAD,
    )
    check("classification: valid request returns 200", response.status_code == 200)
    body = response.json()
    check(
        "classification: response includes a predicted category",
        body.get("predicted_credit_risk_category") in ("Low", "Medium", "High"),
    )
    check(
        "classification: class probabilities sum to ~1.0",
        abs(sum(body.get("class_probabilities", {}).values()) - 1.0) < 0.01,
    )


def test_classification_missing_field_returns_400():
    bad_payload = dict(VALID_CLASSIFICATION_PAYLOAD)
    del bad_payload["credit_score"]
    response = requests.post(
        f"{BASE_URL}/sme-credit-risk-classifier/predictions", json=bad_payload
    )
    check(
        "classification: missing field returns 400 (not a 500 crash)",
        response.status_code == 400,
    )


if __name__ == "__main__":
    try:
        test_regression_valid_request()
        test_regression_missing_field_returns_400()
        test_classification_valid_request()
        test_classification_missing_field_returns_400()
    except requests.exceptions.ConnectionError:
        print(
            "Could not reach the API. Start it first with: python api.py\n"
            "(it must be running on http://127.0.0.1:5000)"
        )
        sys.exit(1)

    failed = [name for name, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed.")
    if failed:
        print("Failed checks:", ", ".join(failed))
        sys.exit(1)