import math


def classifier_beton(fc: float) -> str:
    # Classifie le béton selon sa résistance fc28 (MPa)
    if fc < 20:
        return "⚠️ Résistance insuffisante"
    if fc < 40:
        return "🟡 Béton Ordinaire (BO)"
    if fc < 60:
        return "🟠 Béton Standard"
    if fc < 100:
        return "🔵 BHP — Béton Hautes Performances"
    if fc < 150:
        return "🟣 Très Hautes Performances"
    return "🔴 BUHP — Béton Ultra-Hautes Performances"


def calculer_module_young(fc: float) -> float:
    # Calcule le module d'élasticité E (MPa) selon NF EN 1992-1-1
    fc = max(0.0, float(fc))
    return 9500.0 * ((fc + 8.0) ** (1.0 / 3.0))


def calculer_resistance_traction(fc: float) -> float:
    # Calcule la résistance à la traction ft (MPa) selon la règle ft ≈ fc / 10
    return float(fc) / 10.0


def calculer_rapport_ec(water: float, cement: float) -> float:
    # Calcule le rapport eau/ciment (E/C) à partir des dosages (kg/m³)
    cement = max(1e-9, float(cement))
    return float(water) / cement


def seuil_ec_pour_type(type_beton: str) -> float:
    # Donne un seuil simple du rapport E/C recommandé selon le type de béton
    t = (type_beton or "").lower()
    if "buhp" in t or "ultra" in t:
        return 0.25
    if "bhp" in t or "hautes" in t:
        return 0.40
    return 0.60


def recommander_adjuvant(type_beton: str) -> str:
    # Retourne l'adjuvant recommandé selon le type de béton ciblé
    t = (type_beton or "").lower()
    if "bap" in t or "autop" in t:
        return "Agents de viscosité + Superplastifiant (EN 934-2)"
    if "buhp" in t or "ultra" in t or "bhp" in t:
        return "Superplastifiant (HRWR) — EN 934-2 T3"
    return "Plastifiant — EN 934-2 T2"


def recommander_ciment(classe_exposition: str) -> str:
    # Retourne le type de ciment recommandé selon la classe d'exposition EN 206
    c = (classe_exposition or "").upper().strip()
    if c.startswith("XS"):
        return "CEM III/A"
    if c.startswith("XD"):
        return "CEM II/A-S"
    if c.startswith("XA"):
        return "CEM IV/B"
    if c.startswith("XF"):
        return "CEM II/A-S"
    return "CEM I"


def recommander_addition(type_beton: str) -> str:
    # Retourne l'ajout minéral recommandé selon le type de béton
    t = (type_beton or "").lower()
    if "buhp" in t or "ultra" in t:
        return "Fumée de Silice"
    if "bhp" in t or "hautes" in t:
        return "Fumée de Silice + (Cendres Volantes ou Laitier)"
    if "etanche" in t or "hydraul" in t:
        return "Laitier de Haut-Fourneau"
    return "Cendres Volantes"


def _bornes_par_type(type_beton: str) -> tuple[float, float]:
    # Donne des bornes approximatives fc28 (MPa) par type de béton
    t = (type_beton or "").lower()
    if "ordinaire" in t or "bo" in t:
        return 20.0, 40.0
    if "standard" in t:
        return 40.0, 60.0
    if "bhp" in t:
        return 60.0, 100.0
    if "très" in t or "tres" in t:
        return 100.0, 150.0
    if "buhp" in t:
        return 150.0, 250.0
    return 20.0, 100.0


def target_realiste(target_fc: float, type_beton: str) -> bool:
    # Vérifie si une résistance cible est réaliste au regard du type sélectionné
    lo, hi = _bornes_par_type(type_beton)
    return lo <= float(target_fc) <= hi


def _estimer_ec_pour_fc(target_fc: float) -> float:
    # Estime un E/C cible (heuristique) à partir de la résistance visée
    fc = float(target_fc)
    if fc >= 150:
        return 0.23
    if fc >= 100:
        return 0.30
    if fc >= 60:
        return 0.38
    if fc >= 40:
        return 0.45
    return 0.55


def optimiser_formulation(target_fc: float) -> dict:
    # Suggère une formulation simple (kg/m³) pour approcher une résistance cible
    ec = _estimer_ec_pour_fc(target_fc)
    cement = 320.0 + max(0.0, (float(target_fc) - 40.0)) * 3.0
    cement = min(max(cement, 250.0), 650.0)
    water = max(120.0, min(220.0, ec * cement))
    sp = 5.0 if target_fc < 60 else 8.0
    slag = 0.0 if target_fc < 60 else 60.0
    flyash = 30.0 if target_fc < 60 else 0.0
    return {
        "Cement": round(cement, 1),
        "Blast Furnace Slag": round(slag, 1),
        "Fly Ash": round(flyash, 1),
        "Water": round(water, 1),
        "Superplasticizer": round(sp, 1),
        "Coarse Aggregate": 1000.0,
        "Fine Aggregate": 800.0,
        "Age": 28.0,
    }

