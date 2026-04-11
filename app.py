from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from model.predict import charger_modele, chemin_modele_defaut, predire_fc28
from utils.formulas import (
    calculer_module_young,
    calculer_rapport_ec,
    calculer_resistance_traction,
    classifier_beton,
    optimiser_formulation,
    recommander_addition,
    recommander_adjuvant,
    recommander_ciment,
    seuil_ec_pour_type,
    target_realiste,
)
from utils.validator import valider_ingredients


APP_TITLE = "Optimiseur de Bétons Hydrauliques Spéciaux"
AUTHOR = "Réalisé par : Moslm HLIMI — 1AGC2 — ENIT — 2025/2026"


CONCRETE_TYPES = [
    {"Type": "Béton Ordinaire (BO)", "fc28 (MPa)": "20–40", "Usage": "Bâtiments courants"},
    {"Type": "Béton Hautes Performances (BHP)", "fc28 (MPa)": "60–100", "Usage": "Ponts, tunnels, tours"},
    {"Type": "Béton Ultra-Hautes Performances (BUHP)", "fc28 (MPa)": "> 150", "Usage": "Ouvrages exceptionnels"},
    {"Type": "Béton Autoplaçant (BAP)", "fc28 (MPa)": "30–80", "Usage": "Zones très ferraillées"},
    {"Type": "Béton Fibré", "fc28 (MPa)": "40–80", "Usage": "Dalles industrielles, tunnels"},
    {"Type": "Béton Léger", "fc28 (MPa)": "15–40", "Usage": "Isolation thermique"},
    {"Type": "Béton Lourd", "fc28 (MPa)": ">> 40", "Usage": "Protection nucléaire"},
    {"Type": "Béton Projeté (Gunite)", "fc28 (MPa)": "30–60", "Usage": "Soutènement, stabilisation"},
    {"Type": "Béton Étanche", "fc28 (MPa)": "25–50", "Usage": "Réservoirs, barrages"},
]

REAL_WORLD_EXAMPLES = [
    {"Ouvrage": "Viaduc de Millau (France, 2004)", "Type": "BHP", "fc": "80 MPa", "Note": "Pylônes 245 m"},
    {"Ouvrage": "Burj Khalifa (Dubaï, 2010)", "Type": "BHP", "fc": "80 MPa", "Note": "828 m"},
    {"Ouvrage": "Passerelle de Sherbrooke (Canada, 1997)", "Type": "BUHP", "fc": "> 200 MPa", "Note": "Dalle 30 mm, -70% poids"},
    {"Ouvrage": "Tunnel sous la Manche (FR/UK)", "Type": "Projeté + Étanche", "fc": "60 MPa", "Note": "Soutènement"},
    {"Ouvrage": "Barrage Sidi Salem (Tunisie)", "Type": "Étanche", "fc": "—", "Note": "Milieu sulfaté"},
    {"Ouvrage": "Barrage Sidi El Barrak (Tunisie)", "Type": "Spécial", "fc": "—", "Note": "Eaux riches en sulfates"},
    {"Ouvrage": "One World Trade Center (NY, USA)", "Type": "BHP", "fc": "96 MPa", "Note": "Poteaux bas"},
    {"Ouvrage": "Pont de Normandie (France)", "Type": "BHP", "fc": "60 MPa", "Note": "Portée 856 m"},
]


def _charger_artifact_csv(path: Path) -> pd.DataFrame | None:
    # Charge un CSV s'il existe, sinon retourne None
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def _charger_modele_cached() -> object:
    # Charge le modèle ML (cache Streamlit)
    p = chemin_modele_defaut()
    if not p.exists():
        raise FileNotFoundError(
            "Le modèle n'existe pas encore. Lance d'abord: python -m model.train"
        )
    return charger_modele(p)


def _page_accueil() -> None:
    # Page 1: Accueil
    st.title(APP_TITLE)
    st.subheader("Outil de prédiction et d'optimisation basé sur l'Intelligence Artificielle")
    st.write(
        "Cette application prédit la résistance à la compression à 28 jours (fc28) "
        "à partir d'une formulation de béton (dosages en kg/m³ et âge en jours), "
        "et fournit des recommandations inspirées des Bétons Hydrauliques Spéciaux."
    )
    st.write(AUTHOR)

    st.markdown("## Types de bétons spéciaux (résumé)")
    st.dataframe(pd.DataFrame(CONCRETE_TYPES), use_container_width=True)


def _inputs_formulation() -> dict:
    # Construit le formulaire d'entrée des ingrédients
    col1, col2 = st.columns(2)
    with col1:
        cement = st.number_input("Ciment (kg/m³)", min_value=0.0, value=300.0, step=5.0)
        slag = st.number_input("Laitier de haut-fourneau (kg/m³)", min_value=0.0, value=0.0, step=5.0)
        flyash = st.number_input("Cendres volantes (kg/m³)", min_value=0.0, value=0.0, step=5.0)
        water = st.number_input("Eau (kg/m³)", min_value=0.0, value=180.0, step=1.0)
    with col2:
        sp = st.number_input("Superplastifiant (kg/m³)", min_value=0.0, value=5.0, step=0.5)
        coarse = st.number_input("Granulats grossiers (kg/m³)", min_value=0.0, value=1000.0, step=10.0)
        fine = st.number_input("Sable / granulats fins (kg/m³)", min_value=0.0, value=800.0, step=10.0)
        age = st.number_input("Âge (jours)", min_value=1.0, value=28.0, step=1.0)

    ciment_type = st.selectbox(
        "Type de ciment (NF EN 197-1)",
        ["CEM I", "CEM II", "CEM III", "CEM IV", "CEM V"],
        index=0,
    )
    addition = st.selectbox(
        "Addition minérale",
        ["Fumée de Silice", "Cendres Volantes", "Laitier", "Aucun"],
        index=3,
    )
    expo = st.selectbox(
        "Classe d'exposition (EN 206)",
        ["XC", "XS", "XD", "XF", "XA"],
        index=0,
    )

    return {
        "Cement": cement,
        "Blast Furnace Slag": slag,
        "Fly Ash": flyash,
        "Water": water,
        "Superplasticizer": sp,
        "Coarse Aggregate": coarse,
        "Fine Aggregate": fine,
        "Age": age,
        "CementType": ciment_type,
        "MineralAddition": addition,
        "ExposureClass": expo,
    }


def _page_prediction() -> None:
    # Page 2: Prédiction
    st.header("Prédiction")
    st.write("Renseigne une formulation puis clique sur le bouton pour prédire fc28.")

    entree = _inputs_formulation()
    if st.button("Prédire la Résistance fc28", type="primary", use_container_width=True):
        try:
            valider_ingredients(entree)
            model = _charger_modele_cached()
            fc = predire_fc28(model, entree)
        except Exception as e:
            st.error(str(e))
            return

        cat = classifier_beton(fc)
        e_young = calculer_module_young(fc)
        ft = calculer_resistance_traction(fc)
        ec = calculer_rapport_ec(entree["Water"], entree["Cement"])

        st.markdown("### Résultat")
        st.metric("fc28 prédit (MPa)", f"{fc:.2f}")
        st.info(f"Catégorie: {cat}")

        st.markdown("### Indicateurs calculés")
        c1, c2, c3 = st.columns(3)
        c1.metric("Module de Young E (MPa)", f"{e_young:.0f}")
        c2.metric("Résistance à la traction ft (MPa)", f"{ft:.2f}")
        c3.metric("Rapport E/C", f"{ec:.3f}")

        seuil = seuil_ec_pour_type(cat)
        if ec > seuil:
            st.warning(f"E/C élevé pour cette catégorie (seuil recommandé ≈ {seuil:.2f}).")

        ciment_rec = recommander_ciment(entree["ExposureClass"])
        st.markdown("### Recommandations")
        st.write(f"- Ciment recommandé selon exposition: **{ciment_rec}**")
        st.write(f"- Addition recommandée (type visé): **{recommander_addition(cat)}**")
        st.write(f"- Adjuvant recommandé (type visé): **{recommander_adjuvant(cat)}**")


def _optimiser_depuis_target(target_fc: float, type_cible: str) -> tuple[dict, list[str]]:
    # Génère une formulation proposée à partir d'une cible (heuristique)
    warns: list[str] = []
    if not target_realiste(target_fc, type_cible):
        warns.append("Objectif potentiellement irréaliste pour le type sélectionné.")
    propo = optimiser_formulation(target_fc)
    if "BHP" in type_cible or "BUHP" in type_cible:
        propo["Superplasticizer"] = max(float(propo.get("Superplasticizer", 0.0)), 8.0)
    return propo, warns


def _page_optimisation() -> None:
    # Page 3: Optimisation
    st.header("Optimisation")
    st.write("Définis un objectif de résistance et laisse l'application suggérer une formulation.")

    type_cible = st.selectbox(
        "Type de béton visé",
        [
            "Béton Ordinaire (BO)",
            "Béton Standard",
            "BHP — Béton Hautes Performances",
            "BUHP — Béton Ultra-Hautes Performances",
        ],
        index=1,
    )
    target_fc = st.number_input("fc28 cible (MPa)", min_value=5.0, max_value=250.0, value=60.0, step=1.0)

    st.markdown("### Formulation de départ")
    _ = _inputs_formulation()

    if st.button("Optimiser la formulation", type="primary", use_container_width=True):
        try:
            model = _charger_modele_cached()
            optim, warns = _optimiser_depuis_target(float(target_fc), type_cible)
            valider_ingredients(optim)
            fc_pred = predire_fc28(model, optim)
        except Exception as e:
            st.error(str(e))
            return

        st.markdown("### Résultat d'optimisation")
        st.metric("fc28 prédit pour la formulation proposée (MPa)", f"{fc_pred:.2f}")

        if warns:
            for w in warns:
                st.warning(w)

        st.markdown("### Formulation proposée (kg/m³)")
        st.dataframe(
            pd.DataFrame([{k: optim[k] for k in [
                "Cement",
                "Blast Furnace Slag",
                "Fly Ash",
                "Water",
                "Superplasticizer",
                "Coarse Aggregate",
                "Fine Aggregate",
                "Age",
            ]}]),
            use_container_width=True,
        )

        ec = calculer_rapport_ec(optim["Water"], optim["Cement"])
        st.write(f"- Recommandation E/C (type visé): **≤ {seuil_ec_pour_type(type_cible):.2f}**")
        st.write(f"- E/C proposé: **{ec:.3f}**")
        st.write(f"- Addition recommandée: **{recommander_addition(type_cible)}**")
        st.write(f"- Adjuvant recommandé: **{recommander_adjuvant(type_cible)}**")


def _page_visualisation() -> None:
    # Page 4: Visualisation
    st.header("Visualisation")

    model_dir = Path(__file__).resolve().parent / "model"
    imp_df = _charger_artifact_csv(model_dir / "feature_importance.csv")
    eval_df = _charger_artifact_csv(model_dir / "eval_predictions.csv")

    if imp_df is None or eval_df is None:
        st.warning("Artifacts de visualisation indisponibles. Lance: python -m model.train")
        return

    st.markdown("### Importance des variables (Random Forest)")
    imp_df = imp_df.sort_values("importance", ascending=True)
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.barh(imp_df["feature"], imp_df["importance"]) 
    ax1.set_xlabel("Importance")
    ax1.set_ylabel("Variable")
    st.pyplot(fig1)

    st.markdown("### Prédit vs Réel")
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.scatter(eval_df["y_true"], eval_df["y_pred"], alpha=0.6)
    lims = [
        float(min(eval_df["y_true"].min(), eval_df["y_pred"].min())),
        float(max(eval_df["y_true"].max(), eval_df["y_pred"].max())),
    ]
    ax2.plot(lims, lims, "r--")
    ax2.set_xlabel("fc28 réel (MPa)")
    ax2.set_ylabel("fc28 prédit (MPa)")
    st.pyplot(fig2)

    st.markdown("### Plages de résistance par type")
    type_df = pd.DataFrame(CONCRETE_TYPES)
    st.dataframe(type_df, use_container_width=True)

    st.markdown("### Exemples d'ouvrages (rapport)")
    st.dataframe(pd.DataFrame(REAL_WORLD_EXAMPLES), use_container_width=True)


def _page_references() -> None:
    # Page 5: Références & Normes
    st.header("Références & Normes")
    st.markdown("### Normes")
    st.write("- NF EN 197-1")
    st.write("- EN 206")
    st.write("- EN 934-2")
    st.write("- NF EN 1992-1-1")

    st.markdown("### Références")
    st.write("- Mehta & Monteiro (2014)")
    st.write("- De Larrard (1999)")
    st.write("- Richard & Cheyrezy (1995)")
    st.write("- AFGC (2013)")
    st.write("- ACI 211 (2002)")
    st.write("- ACI 237 (2007)")


def main() -> None:
    # Point d'entrée Streamlit
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Aller à",
        ["Accueil", "Prédiction", "Optimisation", "Visualisation", "Références & Normes"],
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.write(AUTHOR)

    if page == "Accueil":
        _page_accueil()
    elif page == "Prédiction":
        _page_prediction()
    elif page == "Optimisation":
        _page_optimisation()
    elif page == "Visualisation":
        _page_visualisation()
    else:
        _page_references()


if __name__ == "__main__":
    main()
