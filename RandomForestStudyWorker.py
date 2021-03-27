from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd
import optuna

df = pd.read_pickle('./processed-data.pkl')

X = df.drop(['outcome'], axis=1).values
y = df['outcome'].values

def objective(trial):
    n_est = trial.suggest_int('n_estimators', 4, 150)
    d = trial.suggest_int('max_depth', 25, 80)

    clf = RandomForestClassifier(n_estimators = n_est, max_depth = d, random_state = 78, class_weight={
        0: 20,
        1: 1,
        2: 1,
        3: 5
    })

    score = cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits=3, shuffle=True), scoring='accuracy')
    return score.mean()

if __name__ == "__main__":
    study = optuna.load_study(study_name='RandomForestStudy', storage='postgresql://optuna:optuna@localhost:5432/covid-studies')
    study.optimize(objective, show_progress_bar=True)