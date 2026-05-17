# Randomised Signature Engine

Fast path signature regression using random projections (Johnson‑Lindenstrauss). The full signature (depth 3, 14 terms) is projected into a low‑dimensional space (e.g., 32 dimensions) before ridge regression. This makes the engine O(d) per step instead of O(d^n) and works well for daily retraining.

- **Window:** 63 days
- **Signature depth:** 3
- **Random projection dimension:** 32
- **Ridge α:** 1.0
- **Output:** top 3 ETFs per universe by predicted return

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
