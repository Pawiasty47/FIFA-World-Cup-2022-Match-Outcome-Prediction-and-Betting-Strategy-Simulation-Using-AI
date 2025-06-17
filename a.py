import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import random
from collections import defaultdict

# 1. Wczytaj dane
df = pd.read_csv("results.csv")
penalties_df = pd.read_csv("shootouts.csv")
odds_df = pd.read_csv("mecze.csv")
true_results_df = pd.read_csv("results2.csv")

# 2. Utwórz etykietę: kto wygrał
def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'home_win'
    elif row['home_score'] < row['away_score']:
        return 'away_win'
    else:
        return 'draw'

df['result'] = df.apply(get_result, axis=1)


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

total_profit = 0
STAKE = 10

# 3. Kolumny, które nas interesują
features = ['home_team', 'away_team', 'tournament']
target = 'result'

X = df[features]
y = df[target]

# Wczytaj dane o karnych
penalty_stats = defaultdict(lambda: {'wins': 0, 'total': 0})

for _, row in penalties_df.iterrows():
    ht = row['home_team']
    at = row['away_team']
    winner = row['winner']

    if winner == ht:
        penalty_stats[ht]['wins'] += 1
    elif winner == at:
        penalty_stats[at]['wins'] += 1

    penalty_stats[ht]['total'] += 1
    penalty_stats[at]['total'] += 1

true_results = {}
for _, row in true_results_df.iterrows():
    t1 = row["home_team"]
    t2 = row["away_team"]
    h_score = row["home_score"]
    a_score = row["away_score"]

    if h_score > a_score:
        outcome = "home_win"
    elif h_score < a_score:
        outcome = "away_win"
    else:
        outcome = "draw"

    key = tuple(sorted([t1, t2]))
    true_results[key] = outcome

def bet_on_predicted_result(team1, team2, predicted_result):
    global total_profit

    key = tuple(sorted([team1, team2]))

    if key not in odds_lookup:
        print(f"⚠️ Brak kursów dla meczu: {team1} vs {team2} – pomijam zakład.")
        return

    if key not in true_results:
        print(f"⚠️ Brak faktycznego wyniku dla meczu: {team1} vs {team2} – pomijam zakład.")
        return

    actual_result = true_results[key]
    odds_data = odds_lookup[key]

    # Dopasowanie kursu do przewidywanego wyniku
    if predicted_result == "home_win":
        odds = odds_data["Odds1"] if team1 == odds_data["Team1"] else odds_data["Odds2"]
    elif predicted_result == "away_win":
        odds = odds_data["Odds2"] if team2 == odds_data["Team2"] else odds_data["Odds1"]
    else:
        odds = odds_data["Draw"]

    # Sprawdzenie trafności
    if predicted_result == actual_result:
        profit = STAKE * odds - STAKE
        print(f"✅ Trafiony zakład ({predicted_result}) – zysk: {round(profit, 2)} zł")
    else:
        profit = -STAKE
        print(f"❌ Nietrafiony zakład ({predicted_result}, faktycznie: {actual_result}) – strata: {STAKE} zł")

    total_profit += profit



# 4. OneHotEncoding dla drużyn i turnieju
categorical_features = ['home_team', 'away_team', 'tournament']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 5. Pipeline: preprocessing + model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    ))
])
# Preprocessing – te same kolumny co w modelu głównym
pen_preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['home_team', 'away_team'])
    ])

penalty_model = Pipeline(steps=[
    ('preprocessor', pen_preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    ))
])

# Trening modelu rzutów karnych


# 6. Podział i trening
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# 7. Ocena
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("📊 Dokładność modelu (OneHot + RF):", accuracy)

def get_penalty_score(team):
    stats = penalty_stats.get(team, {'wins': 0, 'total': 0})
    if stats['total'] == 0:
        return 0.5  # jeśli brak danych – neutralna skuteczność
    return stats['wins'] / stats['total']
def predict_penalty_winner(team1, team2):
    score1 = get_penalty_score(team1)
    score2 = get_penalty_score(team2)

    if score1 > score2:
        return team1
    elif score2 > score1:
        return team2
    else:
        return random.choice([team1, team2])  # jeśli remis w skuteczności

# 8. Predykcja nowego meczu
# 🧠 Ulepszona predykcja z opcją neutralnego boiska
def predict_match(home, away, tournament, neutral=0):
    df_input = pd.DataFrame([{
        'home_team': home,
        'away_team': away,
        'tournament': tournament
    }])
    return model.predict(df_input)[0]

# 🧩 Symulacja grupy — zwraca miejsca 1. i 2.
def simulate_group(group_teams, tournament="FIFA World Cup"):
    points = {team: 0 for team in group_teams}
    matches = []

    for i in range(len(group_teams)):
        for j in range(i+1, len(group_teams)):
            team1 = group_teams[i]
            team2 = group_teams[j]

            result = predict_match(team1, team2, tournament)
            print(f"{team1} vs {team2} ➜ {result}")
            bet_on_predicted_result(team1, team2, result)
            if result == "home_win":
                points[team1] += 3
            elif result == "away_win":
                points[team2] += 3
            else:
                points[team1] += 1
                points[team2] += 1

            matches.append((team1, team2, result))

    print("\n📊 TABELA KOŃCOWA GRUPY:")
    sorted_teams = sorted(points.items(), key=lambda x: x[1], reverse=True)
    for team, pts in sorted_teams:
        print(f"{team}: {pts} pkt")

    return matches, points, [team for team, _ in sorted_teams[:2]]  # zwraca 2 najlepsze drużyny

# 📦 Wczytaj grupy i zapisz zwycięzców
group_winners = {}
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

for group_name, teams in group_definitions.items():
    _, _, top_two = simulate_group(teams)
    group_winners[group_name] = top_two

# 🔁 Faza pucharowa
def knockout_stage(group_winners, predict_func,tournament="FIFA World Cup"):
    round_16_pairs = [
        (group_winners["A"][0], group_winners["B"][1]),
        (group_winners["C"][0], group_winners["D"][1]),
        (group_winners["E"][0], group_winners["F"][1]),
        (group_winners["G"][0], group_winners["H"][1]),
        (group_winners["B"][0], group_winners["A"][1]),
        (group_winners["D"][0], group_winners["C"][1]),
        (group_winners["F"][0], group_winners["E"][1]),
        (group_winners["H"][0], group_winners["G"][1]),
    ]

    def play_round(pairs, round_name):
        print(f"\n🏟️ {round_name}")
        winners = []

        for t1, t2 in pairs:
            result = predict_match(t1, t2, tournament)
            print(f"{t1} vs {t2} ➜ {result}")
            bet_on_predicted_result(t1, t2, result)
            if result == "home_win":
                winner = t1
            elif result == "away_win":
                winner = t2
            else:
                # 🔮 Rzuty karne – użyj modelu penalty_model
                winner = predict_penalty_winner(t1, t2)
                print(
                    f"⚽ Rzuty karne – {t1} ({get_penalty_score(t1):.2f}) vs {t2} ({get_penalty_score(t2):.2f}) ➜ przewidywany zwycięzca: {winner}")

                if winner not in [t1, t2]:
                    print(f"❗ Model wskazał nieistniejącego uczestnika ({winner}) – losujemy między {t1}, {t2}")
                    winner = random.choice([t1, t2])
                print(f"⚽ Rzuty karne – przewidywany zwycięzca: {winner}")

            print(f"✅ Awansuje: {winner}")
            winners.append(winner)

        return winners

    qf_pairs = list(zip(*[iter(play_round(round_16_pairs, "1/8 finału"))]*2))
    sf_pairs = list(zip(*[iter(play_round(qf_pairs, "Ćwierćfinały"))]*2))
    final_pair = list(zip(*[iter(play_round(sf_pairs, "Półfinały"))]*2))
    winner = play_round(final_pair, "🏆 FINAŁ")[0]
    print(f"\n👑 MISTRZ ŚWIATA: {winner}")

# ✅ Uruchom fazę pucharową
knockout_stage(group_winners, predict_match)
print(f"\n💰 ŁĄCZNY ZYSK/STRATA ze wszystkich zakładów: {round(total_profit, 2)} zł")
