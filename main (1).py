import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


# Wczytanie danych
df = pd.read_csv("results.csv")
penalties_df = pd.read_csv("shootouts.csv")
odds_df = pd.read_csv("mecze.csv")
true_results_df = pd.read_csv("results2.csv")

sns.set(style="whitegrid")

# Tworzenie kolumny 'result'
def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'home_win'
    elif row['home_score'] < row['away_score']:
        return 'away_win'
    else:
        return 'draw'
df['result'] = df.apply(get_result, axis=1)

# Kursy bukmacherskie
odds_lookup = {}
for _, row in odds_df.iterrows():
    key = tuple(sorted([row["Team1"], row["Team2"]]))
    odds_lookup[key] = {
        "Team1": row["Team1"],
        "Team2": row["Team2"],
        "Odds1": row["Odds1"],
        "Draw": row["Draw"],
        "Odds2": row["Odds2"]
    }

true_results = {}
for _, row in true_results_df.iterrows():
    key = tuple(sorted([row["home_team"], row["away_team"]]))
    if row["home_score"] > row["away_score"]:
        outcome = "home_win"
    elif row["home_score"] < row["away_score"]:
        outcome = "away_win"
    else:
        outcome = "draw"
    true_results[key] = outcome

STAKE = 10
profit_progress = []
predictions = []
actual_results = []
profit_rf = [0]
profit_xgb = [0]

profit_progress_rf = []
predictions_rf = []
actual_results_rf = []

profit_progress_xgb = []
predictions_xgb = []
actual_results_xgb = []


def bet_on_predicted_result_with_tracking(team1, team2, predicted_result, profit_tracker,
                                             progress_log, predictions_log, actual_log):
    key = tuple(sorted([team1, team2]))
    if key not in odds_lookup or key not in true_results:
        print(f"⚠️ Brak danych dla meczu: {team1} vs {team2}")
        return

    actual_result = true_results[key]
    odds_data = odds_lookup[key]
    if predicted_result == "home_win":
        odds = odds_data["Odds1"] if team1 == odds_data["Team1"] else odds_data["Odds2"]
    elif predicted_result == "away_win":
        odds = odds_data["Odds2"] if team2 == odds_data["Team2"] else odds_data["Odds1"]
    else:
        odds = odds_data["Draw"]

    if predicted_result == actual_result:
        profit = STAKE * odds
        print(f"✅ Trafiony zakład ({predicted_result}) – zysk: {round(profit, 2)} zł")
    else:
        profit = -STAKE
        print(f"❌ Nietrafiony zakład ({predicted_result}, faktycznie: {actual_result}) – strata: {STAKE} zł")

    profit_tracker[0] += profit
    progress_log.append(profit_tracker[0])
    predictions_log.append(predicted_result)
    actual_log.append(actual_result)

# Przygotowanie danych
features = ['home_team', 'away_team', 'tournament']
target = 'result'
X = df[features]
y = df[target]

categorical_features = ['home_team', 'away_team', 'tournament']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Modele
rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced'))
])
xgb_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1,
                                  use_label_encoder=False, eval_metric='mlogloss'))
])

# Label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3)

rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

accuracy_rf = accuracy_score(y_test, rf_model.predict(X_test))
accuracy_xgb = accuracy_score(y_test, xgb_model.predict(X_test))

print("📊 Dokładność modelu (Random Forest):", accuracy_rf)
print("📊 Dokładność modelu (XGBoost):", accuracy_xgb)

# Model karnych
penalties_df['key'] = penalties_df.apply(lambda x: tuple(sorted([x['home_team'], x['away_team']])), axis=1)
df['key'] = df.apply(lambda x: tuple(sorted([x['home_team'], x['away_team']])), axis=1)
df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year
merged = penalties_df.merge(df[['key', 'tournament', 'year']], on='key', how='left').dropna()
merged['label'] = (merged['winner'] == merged['home_team']).astype(int)

X_pen = merged[['home_team', 'away_team', 'tournament', 'year']]
y_pen = merged['label']
preprocessor_pen = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['home_team', 'away_team', 'tournament'])
], remainder='passthrough')

pipeline_pen = Pipeline([
    ('prep', preprocessor_pen),
    ('clf', RandomForestClassifier(n_estimators=100))
])
pipeline_pen.fit(X_pen, y_pen)

def predict_penalty_winner(t1, t2):
    df_input = pd.DataFrame([{'home_team': t1, 'away_team': t2, 'tournament': 'FIFA World Cup', 'year': 2022}])
    return t1 if pipeline_pen.predict_proba(df_input)[0][1] > 0.5 else t2

def predict_match_model(home, away, tournament, model):
    df_input = pd.DataFrame([{
        'home_team': home, 'away_team': away, 'tournament': tournament
    }])
    pred = model.predict(df_input)[0]
    return label_encoder.inverse_transform([pred])[0]

# Symulacja grup
def simulate_group(group_teams, model, profit_tracker, progress_log, predictions_log, actual_log, tournament="FIFA World Cup"):
    points = {team: 0 for team in group_teams}
    for i in range(len(group_teams)):
        for j in range(i+1, len(group_teams)):
            t1, t2 = group_teams[i], group_teams[j]
            result = predict_match_model(t1, t2, tournament, model)
            print(f"{t1} vs {t2} ➜ {result}")
            bet_on_predicted_result_with_tracking(t1, t2, result, profit_tracker, progress_log, predictions_log, actual_log)
            if result == "home_win":
                points[t1] += 3
            elif result == "away_win":
                points[t2] += 3
            else:
                points[t1] += 1
                points[t2] += 1
    sorted_teams = sorted(points.items(), key=lambda x: x[1], reverse=True)
    print("📊 Tabela końcowa:")
    for team, pts in sorted_teams:
        print(f"{team}: {pts} pkt")
    return None, points, [team for team, _ in sorted_teams[:2]]

# Faza pucharowa
def knockout_stage(group_winners, model, profit_tracker, progress_log, predictions_log, actual_log, tournament="FIFA World Cup"):
    round_16 = [
        (group_winners["A"][0], group_winners["B"][1]),
        (group_winners["C"][0], group_winners["D"][1]),
        (group_winners["E"][0], group_winners["F"][1]),
        (group_winners["G"][0], group_winners["H"][1]),
        (group_winners["B"][0], group_winners["A"][1]),
        (group_winners["D"][0], group_winners["C"][1]),
        (group_winners["F"][0], group_winners["E"][1]),
        (group_winners["H"][0], group_winners["G"][1]),
    ]

    def play_round(pairs, name):
        print(f"\n🏟️ {name}")
        winners = []
        for t1, t2 in pairs:
            result = predict_match_model(t1, t2, tournament, model)
            print(f"{t1} vs {t2} ➜ {result}")
            bet_on_predicted_result_with_tracking(t1, t2, result, profit_tracker, progress_log, predictions_log, actual_log)
            if result == "home_win":
                winners.append(t1)
            elif result == "away_win":
                winners.append(t2)
            else:
                winner = predict_penalty_winner(t1, t2)
                print(f"⚽ Rzuty karne ➜ {winner}")
                winners.append(winner)
        return winners

    qf = list(zip(*[iter(play_round(round_16, "1/8 finału"))]*2))
    sf = list(zip(*[iter(play_round(qf, "Ćwierćfinały"))]*2))
    final = list(zip(*[iter(play_round(sf, "Półfinały"))]*2))
    champion = play_round(final, "🏆 FINAŁ")[0]
    print(f"\n👑 MISTRZ ŚWIATA: {champion}")

# Definicje grup
group_definitions = {
    "A": ["Netherlands", "Senegal", "Ecuador", "Qatar"],
    "B": ["England", "Iran", "United States", "Wales"],
    "C": ["Argentina", "Poland", "Mexico", "Saudi Arabia"],
    "D": ["France", "Australia", "Tunisia", "Denmark"],
    "E": ["Spain", "Costa Rica", "Germany", "Japan"],
    "F": ["Belgium", "Canada", "Morocco", "Croatia"],
    "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
    "H": ["Portugal", "Ghana", "Uruguay", "South Korea"]
}

# Symulacja Random Forest
print("\n================= RANDOM FOREST =================")
group_winners_rf = {}
for group, teams in group_definitions.items():
    _, _, top_two = simulate_group(teams, rf_model, profit_rf,
                                      profit_progress_rf, predictions_rf, actual_results_rf)
    group_winners_rf[group] = top_two
knockout_stage(group_winners_rf, rf_model, profit_rf,
                  profit_progress_rf, predictions_rf, actual_results_rf)

# 🔁 Symulacja XGB
print("\n================= XGBOOST =================")
group_winners_xgb = {}
for group, teams in group_definitions.items():
    _, _, top_two = simulate_group(teams, xgb_model, profit_xgb,
                                      profit_progress_xgb, predictions_xgb, actual_results_xgb)
    group_winners_xgb[group] = top_two
knockout_stage(group_winners_xgb, xgb_model, profit_xgb,
                  profit_progress_xgb, predictions_xgb, actual_results_xgb)

# ✅ Wykresy z podziałem na modele

def plot_profit_curve(profit_progress, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(profit_progress, marker='o', linestyle='-', label=f'{model_name}', color='green')
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"Symulacja zysków/strat z zakładów – {model_name}")
    plt.xlabel("Numer zakładu")
    plt.ylabel("Skumulowany zysk (zł)")
    plt.legend()
    plt.grid(True)
    plt.show()

print(f"\n💰 Zysk Random Forest: {round(profit_rf[0], 2)} zł")
print(f"💰 Zysk XGBoost:       {round(profit_xgb[0], 2)} zł")
# 🟢 Wykresy wyników
plot_profit_curve(profit_progress_rf, "Random Forest")

plot_profit_curve(profit_progress_xgb, "XGBoost")

# 🔍 Porównanie dokładności modeli – wykres
def plot_model_accuracies(acc_rf, acc_xgb):
    accuracy_data = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost'],
        'Accuracy': [acc_rf, acc_xgb]
    })

    plt.figure(figsize=(6, 4))
    sns.barplot(data=accuracy_data, x='Model', y='Accuracy', palette='viridis')
    plt.title('Porównanie dokładności modeli')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracy_data['Accuracy']):
        plt.text(i, v + 0.01, f"{v:.2%}", ha='center', va='bottom')
    plt.ylabel("Dokładność")
    plt.tight_layout()
    plt.show()

plot_model_accuracies(accuracy_rf, accuracy_xgb)
def plot_confusion_matrices(y_true, y_pred_rf, y_pred_xgb, label_encoder):
    labels = label_encoder.classes_
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm_rf = confusion_matrix(y_true, y_pred_rf, labels=range(len(labels)))
    cm_xgb = confusion_matrix(y_true, y_pred_xgb, labels=range(len(labels)))

    sns.heatmap(cm_rf, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=axes[0], cmap="Blues")
    axes[0].set_title("Random Forest")
    axes[0].set_xlabel("Predykcja")
    axes[0].set_ylabel("Rzeczywiste")

    sns.heatmap(cm_xgb, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=axes[1], cmap="Greens")
    axes[1].set_title("XGBoost")
    axes[1].set_xlabel("Predykcja")
    axes[1].set_ylabel("Rzeczywiste")

    plt.tight_layout()
    plt.show()

# Predykcje
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

from sklearn.metrics import classification_report

# Raport tekstowy
print("\n📄 Raport klasyfikacji – Random Forest")
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

print("\n📄 Raport klasyfikacji – XGBoost")
print(classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_))


# Wykres
plot_confusion_matrices(y_test, y_pred_rf, y_pred_xgb, label_encoder)
def penalty_prediction_accuracy(penalties_df, model, label_encoder):
    correct = 0
    total = 0

    for _, row in penalties_df.iterrows():
        pred = predict_match_model(row['home_team'], row['away_team'], 'FIFA World Cup', model)
        if pred == 'home_win' and row['winner'] == row['home_team']:
            correct += 1
        elif pred == 'away_win' and row['winner'] == row['away_team']:
            correct += 1
        elif pred == 'draw':  # remis przed karnymi
            correct += 0.5  # częściowy sukces?
        total += 1
    return correct / total if total else None

acc_rf_penalty = penalty_prediction_accuracy(penalties_df, rf_model, label_encoder)
acc_xgb_penalty = penalty_prediction_accuracy(penalties_df, xgb_model, label_encoder)

# Wykres
plt.figure(figsize=(6, 4))
sns.barplot(x=["Random Forest", "XGBoost"], y=[acc_rf_penalty, acc_xgb_penalty], palette='coolwarm')
plt.title("Skuteczność modeli na meczach z rzutami karnymi")
plt.ylim(0, 1)
for i, v in enumerate([acc_rf_penalty, acc_xgb_penalty]):
    plt.text(i, v + 0.01, f"{v:.2%}", ha='center', va='bottom')
plt.ylabel("Skuteczność trafienia zwycięzcy meczu z karnymi")
plt.tight_layout()
plt.show()

