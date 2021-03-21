from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import pandas as pd
import optuna

df = pd.read_pickle('./processed-data.pkl')

X = df.drop(['outcome'], axis=1).values
y = df['outcome'].values

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 400)
    max_depth = int(trial.suggest_float('max_depth', 1, 50, log=True))

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    score = model_selection.cross_val_score(rf, X, y, cv=3, scoring='accuracy')
    return score.mean()

study = optuna.load_study(study_name='RandomForestCrossValStudy2', storage='postgresql://optuna:optuna@localhost:5432/covid-studies')
study.optimize(objective, show_progress_bar=True)