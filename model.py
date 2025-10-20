import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

def build_dataset(price_df: pd.DataFrame, price_col="Adj Close") -> pd.DataFrame:
    """
    בונה DataFrame ללמידה: תכונות מהאינדיקטורים ותגית יעד – עלייה ביום הבא (1/0).
    """
    df = price_df.copy()
    df = df.dropna(subset=[price_col]).reset_index(drop=True)

    # יעד: האם המחיר מחר גבוה מהיום
    df["target"] = (df[price_col].shift(-1) > df[price_col]).astype(int)

    feats = ["rsi", "sma20", "sma50", "ret1"]
    df = df.dropna(subset=feats + ["target"]).reset_index(drop=True)
    return df[feats + ["target"]]

def train_ensemble(df: pd.DataFrame, n_splits=4) -> tuple[RandomForestClassifier, float]:
    """
    מאמן RandomForest פשוט + ציון Cross-Validation ממוצע (F1 macro).
    """
    X = df.drop(columns=["target"])
    y = df["target"]
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    cv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")
    clf.fit(X, y)
    return clf, float(scores.mean())
