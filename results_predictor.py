import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler



# ---- Loading the dataset ----
df = pd.read_csv('results.csv')
print("Shape of the dataset:",df.shape)
# print(df.head())  


# ---- Data preprocessing and feature engineering ----

# Converting date column to date time object
df['date'] = pd.to_datetime(df['date'])

df = df.sort_values('date')

# For each match, determining if the match was a 'Home Win', 'Away Win' or a 'Draw' 
# creating a new column df['outcomes'] to store these categorical values
def get_match_outcome(row):
    if row['home_score'] > row['away_score']:
        return 'HomeWin'
    elif row['home_score'] < row['away_score']:
        return 'AwayWin'
    else:
        return 'Draw' 

df['outcome'] = df.apply(get_match_outcome,axis=1)
# print(df.head())

# Calculating Aggregated Features

df['home_team_avg_goals_scored_last_5']=0.0
df['away_team_avg_goals_scored_last_5']=0.0
df['home_team_avg_goals_conceded_last_5']=0.0
df['away_team_avg_goals_conceded_last_5']=0.0
df['home_team_avg_points_last_5']=0.0
df['away_team_avg_points_last_5']=0.0

def calculate_points(home_score,away_score,is_home_team):
    if is_home_team:
        if home_score>away_score: return 3
        elif home_score == away_score: return 1
        else: return 0
    else:
        if away_score>home_score: return 3
        elif away_score == home_score: return 1
        else: return 0


print("\nCalculating historical team statistics (this might take a moment)...")

for i, row in df.iterrows():
    print(i)
    current_date = row['date']
    home_team = row['home_team']
    away_team = row['away_team']
    
    home_team_past_matches = df[(df['date']<current_date)& ((df['home_team']==home_team)| (df['away_team']==home_team))].tail(5)

    away_team_past_matches = df[(df['date']<current_date)& ((df['home_team']==away_team)| (df['away_team']==away_team))].tail(5)

    home_goals_scored = 0
    home_goals_conceded = 0
    home_points = 0

    for _, match in home_team_past_matches.iterrows():
        if match['home_team']==home_team:
            home_goals_scored += match['home_score']
            home_goals_conceded += match['away_score']
            home_points += calculate_points(match['home_score'], match['away_score'], True)
        else:
            home_goals_scored += match['away_score']
            home_goals_conceded += match['home_score']
            home_points += calculate_points(match['home_score'],match['away_score'], False)
        
    if len(home_team_past_matches)>0:
        df.at[i, 'home_team_avg_goals_scored_last_5'] = home_goals_scored/len(home_team_past_matches)
        df.at[i, 'home_team_avg_goals_conceded_last_5'] = home_goals_conceded/len(home_team_past_matches)
        df.at[i, 'home_team_avg_point_last_5'] = home_points/len(home_team_past_matches)

    away_goals_scored = 0
    away_goals_conceded = 0
    away_points = 0

    for _,match in away_team_past_matches.iterrows():
        if match['home_team'] == away_team:
            away_goals_scored += match['home_score']
            away_goals_conceded += match['away_score']
            away_points += calculate_points(match['home_score'], match['away_score'], True)
        else:
            away_goals_scored += match['away_score']
            away_goals_conceded += match['home_score']
            away_points += calculate_points(match['home_score'], match['away_score'], False)
        
    if len(away_team_past_matches) > 0:
        df.at[i, 'away_team_avg_goals_scored_last_5'] = away_goals_scored / len(away_team_past_matches)
        df.at[i, 'away_team_avg_goals_conceded_last_5'] = away_goals_conceded / len(away_team_past_matches)
        df.at[i, 'away_team_avg_points_last_5'] = away_points / len(away_team_past_matches)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Shape after feature engineering and dropping NaNs: {df.shape}")


# Features to use for the model

features = [
    'home_team_avg_goals_scored_last_5',
    'home_team_avg_goals_conceded_last_5',
    'away_team_avg_goals_scored_last_5',
    'away_team_avg_goals_conceded_last_5',
    'home_team_avg_points_last_5',
    'away_team_avg_points_last_5',
    'neutral' # Boolean flag
]

X = df[features]
y = df['outcome']

X['neutral'] = X['neutral'].astype(int)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\n Outcome encoding mapping: {list(le.classes_)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=X.index)

X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

model = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
model.fit(X_train, y_train)
print('Logistic Regression Model Trained Successfully')

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

accuracy = accuracy_score(y_test,y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, index=le.classes_, columns=le.classes_)
print(conf_matrix_df)

# --- Example Prediction (Optional) ---
# print("\n--- Example Prediction ---")
# sample_index = X_test.sample(1, random_state=1).index[0]
# sample_data = X_test.loc[sample_index].values.reshape(1, -1)
# true_outcome_encoded = y_test[y_test.index == sample_index].iloc[0]
# true_outcome = le.inverse_transform([true_outcome_encoded])[0]

# predicted_outcome_encoded = model.predict(sample_data)[0]
# predicted_outcome = le.inverse_transform([predicted_outcome_encoded])[0]
# predicted_probabilities = model.predict_proba(sample_data)[0]

# print(f"Sample Match (Index {sample_index}):")
# print(f"  Features: {X_test.loc[sample_index].to_dict()}")
# print(f"  True Outcome: {true_outcome}")
# print(f"  Predicted Outcome: {predicted_outcome}")
# print("  Predicted Probabilities (AwayWin, Draw, HomeWin):")
# for i, class_name in enumerate(le.classes_):
#     print(f"    {class_name}: {predicted_probabilities[i]:.4f}")
