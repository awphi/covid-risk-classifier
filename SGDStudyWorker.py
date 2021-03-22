from sklearn import model_selection
import pandas as pd
import optuna
from sklearn.linear_model import SGDClassifier

df = pd.read_pickle('./processed-data.pkl')

X = df.drop(['outcome'], axis=1).values
y = df['outcome'].values

def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-8, 1.0, log=True)
    
    clf = SGDClassifier(
        loss='hinge', 
        max_iter=1000, 
        alpha=alpha
    )

    score = model_selection.cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    return score.mean()


if __name__ == "__main__":
    study = optuna.load_study(study_name='SGDTestStudy', storage='postgresql://optuna:optuna@localhost:5432/covid-studies')
    study.optimize(objective, show_progress_bar=True)