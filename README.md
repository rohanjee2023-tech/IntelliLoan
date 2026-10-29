# IntelliLoan dashboard

1. Run the last cells of your notebook; they create `best_loan_approval_model.pkl` and `loan_preprocessor.pkl`.
2. Put both files in this folder, next to `app.py`.
3. `pip install -r requirements.txt`
4. `streamlit run app.py`  (opens at http://localhost:8501)

Use the same scikit-learn / xgboost versions as the notebook, otherwise the .pkl files may not load.
