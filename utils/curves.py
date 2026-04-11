from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from model.predict import charger_modele, chemin_modele_defaut, predire_fc28
from utils.formulas import calculer_module_young


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


RANGES = {
    "Cement": (100.0, 700.0),
    "Blast Furnace Slag": (0.0, 300.0),
    "Fly Ash": (0.0, 250.0),
    "Water": (120.0, 240.0),
    "Superplasticizer": (0.0, 30.0),
    "Coarse Aggregate": (800.0, 1300.0),
    "Fine Aggregate": (500.0, 1100.0),
    "Age": (1.0, 365.0),
}


DEFAULTS = {
    "Cement": 300.0,
    "Blast Furnace Slag": 0.0,
    "Fly Ash": 0.0,
    "Water": 180.0,
    "Superplasticizer": 5.0,
    "Coarse Aggregate": 1000.0,
    "Fine Aggregate": 800.0,
    "Age": 28.0,
}


@st.cache_resource
def _charger_modele_cached() -> object:
    p = chemin_modele_defaut()
    if not p.exists():
        raise FileNotFoundError(
            "Le modèle n'existe pas encore. Lance d'abord: python -m model.train"
        )
    return charger_modele(p)


def page_courbes() -> None:
    st.header("Courbes")
    st.write(
        "Choisis un ingrédient à faire varier. L'application calcule la courbe de fc28 "
        "et la courbe du module de Young associée, en gardant les autres paramètres fixes."
    )

    with st.form("courbes_form"):
        col1, col2 = st.columns(2)

        values: dict[str, float] = {}
        for i, k in enumerate(FEATURES):
            lo, hi = RANGES[k]
            default = DEFAULTS[k]
            step = 1.0
            if k in {"Superplasticizer"}:
                step = 0.5
            if k in {"Age"}:
                step = 1.0

            with col1 if i < 4 else col2:
                values[k] = float(
                    st.number_input(
                        k + " (kg/m³)" if k != "Age" else "Âge (jours)",
                        min_value=float(lo),
                        max_value=float(hi),
                        value=float(default),
                        step=float(step),
                    )
                )

        ingredient = st.selectbox(
            "Ingrédient à faire varier",
            FEATURES,
            index=0,
        )

        submitted = st.form_submit_button("Générer les courbes", type="primary")

    if not submitted:
        return

    try:
        model = _charger_modele_cached()
    except Exception as e:
        st.error(str(e))
        return

    lo, hi = RANGES[ingredient]
    xs = np.linspace(lo, hi, 50)

    fc_vals: list[float] = []
    e_vals: list[float] = []

    for x in xs:
        entree = dict(values)
        entree[ingredient] = float(x)
        fc = predire_fc28(model, entree)
        fc_vals.append(float(fc))
        e_vals.append(float(calculer_module_young(fc)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(xs, fc_vals, color="tab:blue")
    ax1.set_title("Résistance à 28 jours (fc28) en fonction de " + ingredient)
    ax1.set_xlabel(ingredient + (" (jours)" if ingredient == "Age" else " (kg/m³)"))
    ax1.set_ylabel("fc28 (MPa)")
    ax1.grid(True, alpha=0.25)

    ax2.plot(xs, e_vals, color="tab:green")
    ax2.set_title("Module de Young (E) en fonction de " + ingredient)
    ax2.set_xlabel(ingredient + (" (jours)" if ingredient == "Age" else " (kg/m³)"))
    ax2.set_ylabel("E (MPa)")
    ax2.grid(True, alpha=0.25)

    st.pyplot(fig)
