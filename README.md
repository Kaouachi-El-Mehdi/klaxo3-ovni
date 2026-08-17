# Klaxo-3 — réception des relevés

Analyse du fichier de signalements d'OVNI transmis par la sonde Klaxo-3
(projet IPSSI, Machine Learning Avancé, partie 1).

## Lancer l'analyse

```bash
pip install -r requirements.txt
python analyse.py
```

Le script télécharge `releves_klaxo3.csv` s'il est absent, puis exécute
toutes les phases d'une traite jusqu'aux derniers chiffres. Détails,
décisions et résultats commentés : voir [RAPPORT.md](RAPPORT.md).
