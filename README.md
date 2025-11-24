# worker-no-show-prediction

Worker No-Show Prediction (DailyGo)

Purpose
-------
Predict which workers are likely to not show up for an assigned gig so we can apply low-cost interventions (reminders, small bonuses, replacements).

Context (how we built it)
-------------------------
At DailyGo we observed three main drivers of no-shows:
  1. Long commute distance (>5 km)
  2. Low pay and missing basic facilities (no food/drinks)
  3. Workers' prior no-show history

This repo implements a simple, production-minded pipeline that:
  - Creates shift-based features (morning/afternoon/night),
  - Flags long commutes and repeat no-show workers,
  - Tests multiple classifiers and picks the best one automatically.

Models tested
-------------
We evaluate four models: Logistic Regression, Decision Tree, Random Forest and Gradient Boosting.  
Metrics printed: Accuracy, Precision, Recall, ROC AUC. Best model (by AUC) is saved to `models/`.

How to run
----------
1. Prepare the data in `data/sample_noshow.csv` (sample included).
2. Install dependencies: `pip install -r requirements.txt`.
3. Quick EDA: `python src/exploration.py`.
4. Train & evaluate models: `python src/train_models.py`.
   - This prints metrics for each model and saves the best model to `models/`.

Notes
-----
- Dataset in `data/` is a small sample; the scripts are written to be readable and easy to extend.
- In production, replace the sample CSV with real data, use cross-validation, and add feature-logging.
