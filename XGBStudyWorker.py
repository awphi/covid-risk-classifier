from sklearn import model_selection
import pandas as pd
import optuna
import xgboost as xgb

df = pd.read_pickle('./processed-data.pkl')

X = df.drop(['outcome'], axis=1).values
y = df['outcome'].values

def objective(trial):
    clf = xgb.XGBClassifier(        
        verbosity = 0,
        objective = "multi:softprob",
        use_label_encoder = False,
        n_estimators = trial.suggest_int("n_estimators", 20, 200),
        #"booster" = trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        reg_lambda = trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        reg_alpha = trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        subsample = 0.8,#trial.suggest_float("subsample", 1e-8, 1.0, log=True),
        colsample_bytree = 0.8,
        max_depth =  trial.suggest_int("max_depth", 1, 9),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 20),
        learning_rate = trial.suggest_float("learning_rate", 1e-8, 1.0, log=True),
        gamma = trial.suggest_float("gamma", 1e-8, 1.0, log=True),
    )

    score = model_selection.cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    return score.mean()


if __name__ == "__main__":
    study = optuna.load_study(study_name='XGBStudy', storage='postgresql://optuna:optuna@localhost:5432/covid-studies')
    study.optimize(objective, show_progress_bar=True)