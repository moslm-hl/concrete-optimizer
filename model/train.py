from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


FEATURES = [
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age",
]


def charger_dataset(csv_path: str | Path) -> pd.DataFrame:
    # Charge le dataset Kaggle depuis un fichier CSV
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    rename_map = {
        "cement": "Cement",
        "blast_furnace_slag": "Blast Furnace Slag",
        "blast furnace slag": "Blast Furnace Slag",
        "fly_ash": "Fly Ash",
        "fly ash": "Fly Ash",
        "water": "Water",
        "superplasticizer": "Superplasticizer",
        "coarse_aggregate": "Coarse Aggregate",
        "coarse aggregate": "Coarse Aggregate",
        "fine_aggregate": "Fine Aggregate",
        "fine aggregate": "Fine Aggregate",
        "age": "Age",
        "concrete_compressive_strength": "Concrete compressive strength",
        "concrete compressive strength": "Concrete compressive strength",
    }
    df = df.rename(columns={c: rename_map.get(c.strip().lower(), c) for c in df.columns})
    return df


def detecter_colonne_cible(df: pd.DataFrame) -> str:
    # Détecte automatiquement la colonne cible (résistance) dans le dataset
    candidats = [
        "concrete_compressive_strength",
        "Concrete compressive strength",
        "Concrete compressive strength(MPa, megapascals)",
        "Compressive strength",
        "Strength",
    ]
    for c in candidats:
        if c in df.columns:
            return c
    raise ValueError("Colonne cible introuvable dans le CSV")


def entrainer_modele(df: pd.DataFrame, random_state: int = 42) -> tuple[object, dict]:
    # Entraîne un RandomForestRegressor et retourne le modèle + métriques
    y_col = detecter_colonne_cible(df)
    x = df[FEATURES].astype(float)
    y = df[y_col].astype(float)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=random_state)
    model = RandomForestRegressor(n_estimators=300, random_state=random_state)
    model.fit(x_tr, y_tr)
    pred = model.predict(x_te)
    metrics = {
        "mae": float(mean_absolute_error(y_te, pred)),
        "rmse": float(mean_squared_error(y_te, pred) ** 0.5),
        "r2": float(r2_score(y_te, pred)),
    }
    return model, {"metrics": metrics, "y_test": y_te.to_numpy(), "y_pred": pred}


def sauvegarder_artifacts(out_dir: Path, model: object, info: dict) -> None:
    # Sauvegarde le modèle, métriques, prédictions et importance des variables
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.pkl")
    (out_dir / "metrics.json").write_text(json.dumps(info["metrics"], indent=2), encoding="utf-8")
    eval_df = pd.DataFrame({"y_true": info["y_test"], "y_pred": info["y_pred"]})
    eval_df.to_csv(out_dir / "eval_predictions.csv", index=False)
    imp = getattr(model, "feature_importances_", np.zeros(len(FEATURES)))
    pd.DataFrame({"feature": FEATURES, "importance": imp}).to_csv(out_dir / "feature_importance.csv", index=False)


def main() -> None:
    # Point d'entrée CLI pour entraîner et sauvegarder le modèle
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "data" / "concrete_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {csv_path}")
    df = charger_dataset(csv_path)
    model, info = entrainer_modele(df)
    sauvegarder_artifacts(Path(__file__).resolve().parent, model, info)
    print("Entraînement terminé. Modèle sauvegardé dans model/model.pkl")


if __name__ == "__main__":
    main()
