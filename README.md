# Bank Loan Approval Prediction

A beginner-friendly machine-learning project that predicts whether a loan application is likely to be approved (`Y`) or not approved (`N`).

## Project workflow
1. Load and inspect the loan dataset.
2. Clean missing categorical and numerical values using imputers.
3. Convert categorical features using One-Hot Encoding.
4. Train a Random Forest classifier.
5. Evaluate the model using accuracy, classification report, and confusion matrix.
6. Enter applicant details in a Streamlit web app and get a prediction.

## Project structure
```text
Bank_Loan_Approval_Prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── loan_prediction.csv
└── models/
```

## Run in VS Code (Windows)
1. Extract the ZIP file.
2. Open VS Code → **File → Open Folder** → select `Bank_Loan_Approval_Prediction`.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
5. Activate it:
   ```bash
   .venv\Scripts\activate
   ```
   If PowerShell blocks activation, run:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .venv\Scripts\Activate.ps1
   ```
6. Install libraries:
   ```bash
   pip install -r requirements.txt
   ```
7. Train the model and view evaluation:
   ```bash
   python train_model.py
   ```
8. Start the web application:
   ```bash
   streamlit run app.py
   ```
9. A browser tab should open. If it does not, copy the Local URL shown in the terminal (usually `http://localhost:8501`).

## Dataset
The included `data/loan_prediction.csv` is a **synthetic sample dataset** so the project runs immediately. It is not the original Kaggle data and its scores are only for demonstration.

For your assignment's real dataset:
1. Download the **Loan Prediction Problem Dataset** from Kaggle:
   https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
2. Extract the download.
3. Copy its `train_u6lujuX_CVtuZ9i.csv` file into this project's `data` folder.
4. Rename it to `loan_prediction.csv` (replace the demo file).
5. Run `python train_model.py` again.

The training script expects the standard Kaggle columns, including `Loan_Status`. The Kaggle test CSV generally has no `Loan_Status`, so use the training CSV for model training/evaluation.

## Main features used
- Gender, Married, Dependents, Education, Self_Employed
- ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term
- Credit_History, Property_Area

## Important note
This is an educational classification project, not a real credit-scoring system. Loan decisions must not be made using this demo model. The sample data is synthetic, and model predictions can be inaccurate or unfair.
