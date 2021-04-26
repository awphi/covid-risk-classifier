import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd
import optuna

df = pd.read_pickle('./processed-data.pkl')

X = df.drop(['outcome'], axis=1).values
y = df['outcome'].values

def objective(trial):
    param = {
        "silent": 1,
        "eval_metric": "mlogloss",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear"]),
        "reg_lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "reg_alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        "objective": "multi:softprob",
        'verbosity': 0,
        'use_label_encoder': False
    }

    if param["booster"] == "gbtree":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["learning_rate"] = trial.suggest_loguniform("learning_rate", 1e-8, 1.0)
        param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)

    clf = xgb.XGBClassifier(**param)

    score = cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits=3, shuffle=True), scoring='accuracy')
    return score.mean()

if __name__ == "__main__":
    study = optuna.load_study(study_name='XGBStudy', storage='postgresql://optuna:optuna@localhost:5432/covid-studies')
    study.optimize(objective, show_progress_bar=True)