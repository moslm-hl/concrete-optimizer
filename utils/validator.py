def valider_positif(nom: str, valeur: float, mini: float = 0.0) -> None:
    # Valide qu'une valeur numérique est >= mini, sinon lève une ValueError
    if valeur is None:
        raise ValueError(f"{nom} est requis")
    if float(valeur) < float(mini):
        raise ValueError(f"{nom} doit être >= {mini}")


def valider_ingredients(data: dict) -> None:
    # Valide les ingrédients principaux attendus par le modèle
    champs = [
        ("Cement", 0.0),
        ("Blast Furnace Slag", 0.0),
        ("Fly Ash", 0.0),
        ("Water", 0.0),
        ("Superplasticizer", 0.0),
        ("Coarse Aggregate", 0.0),
        ("Fine Aggregate", 0.0),
        ("Age", 1.0),
    ]
    for nom, mini in champs:
        valider_positif(nom, data.get(nom), mini)

