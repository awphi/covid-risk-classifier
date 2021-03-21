from sklearn import model_selection
import pandas as pd
import optuna
from sklearn.linear_model import SGDClassifier

df = pd.read_pickle('./processed-data.pkl')

X = df.drop(['outcome'], axis=1).values
y = df['outcome'].values

def objective(trial):
    clf = SGDClassifier(loss='hinge')

    score = model_selection.cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    return score.mean()


if __name__ == "__main__":
    study = optuna.load_study(study_name='XGBStudy', storage='postgresql://optuna:optuna@localhost:5432/covid-studies')
    study.optimize(objective, show_progress_bar=True)