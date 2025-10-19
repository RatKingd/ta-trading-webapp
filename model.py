import pandas as pd, numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

FEATURES = ['ret1','ret5','sma_ratio','rsi']

def build_dataset(series: dict[str, pd.DataFrame], threshold=0.01) -> pd.DataFrame:
    rows = []
    for ticker, df in series.items():
        f = df.copy()
        f['ticker'] = ticker
        f['future_ret'] = f['Adj Close'].pct_change().shift(-1)
        f['label'] = (f['future_ret'] > threshold).astype(int)
        f = f.dropna()
        if not f.empty:
            rows.append(f)
    return pd.concat(rows).sort_index()

def train_ensemble(df_all: pd.DataFrame, n_splits=4):
    X = df_all[FEATURES].values
    y = df_all['label'].values
    tscv = TimeSeriesSplit(n_splits=n_splits)
    models, scores = [], []
    for tr, te in tscv.split(X):
        m = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              n_estimators=250, max_depth=4, subsample=0.9, colsample_bytree=0.9)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        models.append(m)
        scores.append((accuracy_score(y[te], pred), f1_score(y[te], pred)))
    return models, scores

def ensemble_predict(models, X):
    return np.mean([m.predict_proba(X)[:,1] for m in models], axis=0)
