from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd


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


def charger_modele(model_path: str | os.PathLike) -> object:
    # Charge un modèle scikit-learn sauvegardé avec joblib
    return joblib.load(model_path)


def predire_fc28(modele: object, entree: dict) -> float:
    # Prédit fc28 (MPa) à partir d'un dictionnaire d'entrée
    norm = {str(k).strip().lower(): v for k, v in entree.items()}
    values = {}
    for k in FEATURES:
        values[k] = float(entree.get(k, norm.get(k.lower())))
    x = pd.DataFrame([values])
    y = modele.predict(x)[0]
    return float(y)


def chemin_modele_defaut() -> Path:
    # Retourne le chemin par défaut du modèle (model/model.pkl)
    return Path(__file__).resolve().parent / "model.pkl"

