Optimiseur de Betons Hydrauliques Speciaux

Auteur Moslm HLIMI
Classe 1AGC2
Ecole Ecole Nationale des Ingenieurs de Tunis
Annee universitaire 2025 2026

Ce projet est une application web en Python qui aide a estimer la resistance a la compression du beton fc28
L utilisateur saisit les dosages des constituants du beton en kg par metre cube et saisit aussi lage en jours
L application utilise un modele de machine learning entraine sur le dataset Kaggle Concrete Compressive Strength
Le modele predit fc28 en MPa puis l application calcule aussi des indicateurs simples utilises en genie civil comme le module de Young et la resistance en traction
L interface est faite avec Streamlit et propose plusieurs pages en francais pour la prediction l optimisation et la visualisation
Les graphiques affichent l importance des variables et la comparaison entre les valeurs reelles et predites

Installation
Installer Python 64 bit puis creer un environnement virtuel venv dans le dossier du projet
Activer lenvironnement puis installer les dependances depuis le fichier requirements txt
Les dependances principales sont streamlit scikit learn pandas numpy matplotlib et joblib

Execution
Verifier que le fichier concrete data csv est present dans le dossier data
Entrainer le modele en lancant le module model train
Cela genere le fichier model pkl ainsi que des fichiers de metriques et de visualisation dans le dossier model
Lancer ensuite l application Streamlit avec la commande streamlit run app py
Ouvrir le lien local affiche dans le terminal pour utiliser l interface
Pour une nouvelle session on peut relancer uniquement l application si le modele est deja present

Bibliotheques utilisees
streamlit scikit learn pandas numpy matplotlib joblib
