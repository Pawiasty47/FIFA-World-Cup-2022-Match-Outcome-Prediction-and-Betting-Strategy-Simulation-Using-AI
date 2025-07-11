# 🏆 FIFA World Cup 2022 – AI Match Prediction & Betting Simulation

This project uses **Artificial Intelligence** to simulate the **FIFA World Cup 2022** based on historical football match data and evaluates betting outcomes using real bookmaker odds.

## 📌 Project Overview

The goal of this project is to:

* Train AI models to learn how national football teams perform based on historical data.
* Simulate the 2022 FIFA World Cup tournament, including both **group stage** and **knockout rounds**.
* Compare predicted match outcomes with real results.
* Simulate a bettor placing 10 PLN on each match to determine potential profit or loss.
* Handle missing data gracefully (e.g., matches not covered by bookmakers).
* Provide analysis using two different machine learning algorithms: **Random Forest** and **XGBoost**.

---

## 🧠 AI Models

Two classifiers were used and compared:

* `RandomForestClassifier` (sklearn)
* `XGBClassifier` (XGBoost)

Each model was trained using team identity and tournament type (encoded with `OneHotEncoder`) as input features, and match result (home win / away win / draw) as target.

---

## 📊 Data Sources

* `results.csv`: Full historical international match results.
* `shootouts.csv`: Penalty shootout outcomes for modeling penalty performance.
* `mecze.csv`: Real bookmaker odds for selected World Cup 2022 matches.
* `results2.csv`: Actual match results from World Cup 2022 (for comparison with predictions).

---

## 💡 Features Implemented

* **Match prediction** using historical data
* **Penalty shootout resolution model** for draws in knockout rounds
* **Tournament simulation**, including group stage and elimination brackets
* **Betting simulation** for each match:

  * If prediction was correct → profit = (odds × 10 PLN − 10 PLN)
  * If prediction was wrong → loss = −10 PLN
* **Graceful handling of mismatched or missing matches** (e.g., if a simulated match doesn't exist in the betting odds or real result dataset)

---

## 📈 Example Output

* Simulated group matches and standings
* Predicted outcomes of knockout rounds
* Total profit/loss of a bettor following AI predictions
* Comparison of prediction accuracy across models

---

## 🛠 How to Run

1. Clone this repository.
2. Ensure required dependencies are installed (see below).
3. Place the following CSV files in the same directory:

   * `results.csv`
   * `shootouts.csv`
   * `mecze.csv`
   * `results2.csv`
4. Run the main Python script:

   ```bash
   python a.py
   ```

---

## 📦 Dependencies

* `pandas`
* `numpy`
* `scikit-learn`
* `xgboost`
* `random`
* `collections`

Install required packages using:

```bash
pip install pandas numpy scikit-learn xgboost
```

---

## 🧪 Potential Improvements

* Add more features (e.g., ELO ratings, goal differences, player statistics)
* Enhance the betting simulation with dynamic strategies (e.g., martingale)
* Visualize the tournament tree or prediction accuracy using `matplotlib` or `plotly`
* Create a web interface for simulation

---
