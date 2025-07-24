# 🏏 Football Match Outcome Prediction Using Logistic Regression

This project predicts the outcome of football matches (Home Win, Away Win, or Draw) using historical match data and a Logistic Regression model. It involves feature engineering from past match records to derive team performance statistics.

---

## 📂 Dataset

The model expects a CSV file named `results.csv` with historical match data containing at least the following columns:

* `date`: Match date
* `home_team`: Home team name
* `away_team`: Away team name
* `home_score`: Home team score
* `away_score`: Away team score
* `neutral`: Boolean flag indicating if the match was played at a neutral venue

---

## 🔍 Project Overview

### 1. **Data Loading**

Loads the dataset and prints its shape.

```python
df = pd.read_csv('results.csv')
```

### 2. **Data Preprocessing**

* Converts the `date` column into datetime format.
* Sorts matches by date.
* Creates a new column `outcome` to classify match results:

  * `HomeWin`: Home team scored more
  * `AwayWin`: Away team scored more
  * `Draw`: Both teams scored equally

### 3. **Feature Engineering: Team Form Stats**

The script computes rolling averages for the **last 5 matches** before each game for both the home and away teams:

* Average goals scored
* Average goals conceded
* Average points obtained (Win = 3, Draw = 1, Loss = 0)

Features created:

* `home_team_avg_goals_scored_last_5`
* `home_team_avg_goals_conceded_last_5`
* `home_team_avg_points_last_5`
* `away_team_avg_goals_scored_last_5`
* `away_team_avg_goals_conceded_last_5`
* `away_team_avg_points_last_5`

### 4. **Cleaning the Data**

Replaces any infinite values and drops rows with missing values:

```python
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
```

### 5. **Preparing Data for Modeling**

The following columns are used as features:

```python
features = [
    'home_team_avg_goals_scored_last_5',
    'home_team_avg_goals_conceded_last_5',
    'away_team_avg_goals_scored_last_5',
    'away_team_avg_goals_conceded_last_5',
    'home_team_avg_points_last_5',
    'away_team_avg_points_last_5',
    'neutral'
]
```

* The target variable is `outcome`, label-encoded.
* Features are scaled using `StandardScaler`.

### 6. **Model Training**

A **Logistic Regression** classifier is trained:

```python
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)
```

### 7. **Evaluation**

* Accuracy, Confusion Matrix, and Classification Report are printed:

```python
accuracy_score()
confusion_matrix()
classification_report()
```

### 8. **(Optional) Example Prediction**

An optional block is provided to test the model on a random test sample:

* Shows features, true vs. predicted outcome, and class probabilities.

---

## 📊 Model Summary

* Algorithm: Logistic Regression
* Features: Recent 5-match performance statistics
* Classes: HomeWin, AwayWin, Draw
* Evaluation Metrics: Accuracy, Confusion Matrix

---

## 🛠️ Future Improvements

* Integrate Elo ratings or betting odds
* Use advanced models like XGBoost or Neural Networks
* Add rolling averages for additional stats (shots, xG, possession)

---

## ✨ Credits

Developed by \[Your Name] as a demonstration of predictive modeling in sports analytics.
