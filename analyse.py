"""
Bureau d'Analyse Terrestre - reception des releves de la sonde Klaxo-3.

Se relance d'une traite : telechargement (si absent) -> chargement brut ->
typage -> etiquette canular -> premier modele -> fuite de donnees -> modele
final -> comparaison a la baseline. Tous les chiffres imprimes ici sont ceux
qui sont recopies dans RAPPORT.md.
"""

import csv
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

DATA_URL = (
    "https://raw.githubusercontent.com/planetsig/ufo-reports/master/"
    "csv-data/ufo-complete-geocoded-time-standardized.csv"
)
DATA_PATH = Path(__file__).resolve().parent / "releves_klaxo3.csv"

COLUMNS = [
    "datetime", "city", "state", "country", "shape", "duration_seconds",
    "duration_hours_min", "comments", "date_posted", "latitude", "longitude",
]

RANDOM_STATE = 42


def section(titre):
    print()
    print("=" * 70)
    print(titre)
    print("=" * 70)


def ensure_data():
    if not DATA_PATH.exists():
        print(f"Telechargement de {DATA_PATH.name} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH


# ---------------------------------------------------------------------------
# Phase 1 : ouvrir la caisse
# ---------------------------------------------------------------------------
def phase1_ouvrir_la_caisse(path):
    section("PHASE 1 : ouvrir la caisse")

    with open(path, encoding="utf-8", errors="replace") as f:
        total_lignes = sum(1 for _ in f)

    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        lignes = list(csv.reader(f))

    lignes_ok = [r for r in lignes if len(r) == len(COLUMNS)]
    lignes_a_part = [r for r in lignes if len(r) != len(COLUMNS)]

    assert total_lignes == len(lignes_ok) + len(lignes_a_part)

    print(f"Lignes dans le fichier          : {total_lignes}")
    print(f"Lignes chargees (11 champs)     : {len(lignes_ok)}")
    print(f"Lignes mises a part (!= 11)     : {len(lignes_a_part)}")

    nb_champs = sorted({len(r) for r in lignes_a_part})
    print(f"Nombre de champs rencontres parmi les lignes mises a part : {nb_champs}")
    print("Exemple de ligne problematique (12 champs au lieu de 11) :")
    print(lignes_a_part[0])

    df = pd.DataFrame(lignes_ok, columns=COLUMNS)
    return df, lignes_a_part


# ---------------------------------------------------------------------------
# Phase 2 : rien n'est du bon type
# ---------------------------------------------------------------------------
def phase2_typage(df):
    section("PHASE 2 : rien n'est du bon type")
    df = df.copy()

    anomalies = {}

    # latitude / longitude -> numerique
    lat_num = pd.to_numeric(df["latitude"], errors="coerce")
    lon_num = pd.to_numeric(df["longitude"], errors="coerce")
    anomalies["latitude"] = df.loc[lat_num.isna(), "latitude"].tolist()
    anomalies["longitude"] = df.loc[lon_num.isna(), "longitude"].tolist()
    df["latitude"] = lat_num
    df["longitude"] = lon_num

    # duration_seconds -> numerique
    dur_num = pd.to_numeric(df["duration_seconds"], errors="coerce")
    anomalies["duration_seconds"] = df.loc[dur_num.isna(), "duration_seconds"].tolist()
    df["duration_seconds"] = dur_num

    # datetime -> date
    dt = pd.to_datetime(df["datetime"], format="%m/%d/%Y %H:%M", errors="coerce")
    anomalies["datetime"] = df.loc[dt.isna(), "datetime"].tolist()
    df["datetime_parsed"] = dt

    # date_posted -> date
    dp = pd.to_datetime(df["date_posted"], format="%m/%d/%Y", errors="coerce")
    anomalies["date_posted"] = df.loc[dp.isna(), "date_posted"].tolist()
    df["date_posted_parsed"] = dp

    # champs vides (categoriels) : pas une conversion, mais une anomalie a part entiere
    vides = {
        "shape": int((df["shape"] == "").sum()),
        "country": int((df["country"] == "").sum()),
        "state": int((df["state"] == "").sum()),
    }

    print("Echecs de conversion par champ :")
    print(f"  latitude          : {len(anomalies['latitude'])} / {len(df)}"
          f"  -> exemple : {anomalies['latitude'][:3]}")
    print(f"  longitude         : {len(anomalies['longitude'])} / {len(df)}")
    print(f"  duration_seconds  : {len(anomalies['duration_seconds'])} / {len(df)}"
          f"  -> valeurs : {anomalies['duration_seconds']}")
    print(f"  datetime          : {len(anomalies['datetime'])} / {len(df)}"
          f"  -> exemple : {anomalies['datetime'][:3]} (toutes finissent par 24:00)")
    print(f"  date_posted       : {len(anomalies['date_posted'])} / {len(df)}")
    print()
    print("Champs categoriels laisses vides par le temoin :")
    for champ, n in vides.items():
        print(f"  {champ:<9} : {n} lignes vides ({n / len(df):.2%})")

    print()
    print("Diagnostic des 4 anomalies retenues :")
    print("  1. latitude ('33q.200088', 1 valeur)      -> service de transmission "
          "(geocodage), une seule valeur suffit a rendre toute la colonne texte.")
    print("  2. duration_seconds (5 valeurs, ex. \"2`\") -> temoin : formats de duree "
          "libres (fractions, mots) tapes directement dans un champ numerique.")
    print("  3. datetime (1220 valeurs en '...24:00')  -> service de transmission : "
          "artefact du script de standardisation des heures (24:00 n'existe pas).")
    print("  4. shape vide (2922 lignes)                -> temoin : n'a pas su/voulu "
          "decrire la forme observee.")

    return df, anomalies, vides


# ---------------------------------------------------------------------------
# Phase 3 : le Conseil veut trier les canulars
# ---------------------------------------------------------------------------
def phase3_canular(df):
    section("PHASE 3 : le Conseil veut trier les canulars")
    df = df.copy()

    df["is_hoax"] = df["comments"].str.contains("hoax", case=False, na=False).astype(int)

    n = int(df["is_hoax"].sum())
    prop = df["is_hoax"].mean()
    print("Regle : un releve est marque canular si son champ 'comments' contient "
          "le mot 'hoax' (insensible a la casse).")
    print(f"Releves marques canulars : {n} / {len(df)} ({prop:.2%})")

    incertain = df["comments"].str.contains(r"hoax\?\?|possible hoax", case=False,
                                              na=False, regex=True).sum()
    autres_mots = df["comments"].str.contains(
        r"\bfake\b|\bprank\b|made up|\bjoke\b|not real", case=False, na=False, regex=True
    )
    rates = int((autres_mots & ~df["is_hoax"].astype(bool)).sum())

    print(f"Limite (fausse alerte) : {incertain} des {n} 'hoax' trouves sont en realite "
          "des '((HOAX??))' ou 'possible hoax' ajoutes par un employe NUFORC qui n'etait "
          "pas sur - la regle les compte comme certains.")
    print(f"Limite (rate) : {rates} commentaires disent 'fake', 'prank', 'made up' ou "
          "'joke' sans jamais ecrire le mot 'hoax' - la regle ne les attrape pas.")

    return df


# ---------------------------------------------------------------------------
# Construction des features (partagee par les phases 4, 5 et 6)
# ---------------------------------------------------------------------------
def construire_features(df, inclure_comments):
    df = df.copy()

    dt_fixed = df["datetime"].str.replace(" 24:00", " 00:00", regex=False)
    dt_parsed = pd.to_datetime(dt_fixed, format="%m/%d/%Y %H:%M", errors="coerce")
    dt_parsed = dt_parsed + pd.to_timedelta(
        df["datetime"].str.endswith(" 24:00").fillna(False).astype(int), unit="D"
    )

    num = pd.DataFrame({
        "hour": dt_parsed.dt.hour,
        "weekday": dt_parsed.dt.dayofweek,
        "month": dt_parsed.dt.month,
        "duration_seconds": df["duration_seconds"],
        "latitude": df["latitude"],
        "longitude": df["longitude"],
    })
    num = num.fillna(num.median(numeric_only=True))

    cat = pd.DataFrame({
        "shape": df["shape"].replace("", "inconnu"),
        "country": df["country"].replace("", "inconnu"),
        "state": df["state"].replace("", "inconnu"),
    })

    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=20)
    cat_enc = encoder.fit_transform(cat)

    blocs = [csr_matrix(num.values), cat_enc]

    if inclure_comments:
        vect = TfidfVectorizer(max_features=500, stop_words="english")
        texte = df["comments"].fillna("")
        blocs.append(vect.fit_transform(texte))

    X = hstack(blocs).tocsr()
    return X


def entrainer_evaluer(X, y, titre):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    rappel = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    exactitude = accuracy_score(y_test, y_pred)

    print(f"[{titre}] evalue sur {len(y_test)} releves jamais vus a l'entrainement")
    print(f"  Rappel (canulars attrapes / 100 reels)  : {rappel * 100:.1f}")
    print(f"  Precision (vrais / 100 signales)        : {precision * 100:.1f}")
    print(f"  Exactitude (accuracy)                   : {exactitude * 100:.1f}")

    return {"rappel": rappel, "precision": precision, "exactitude": exactitude,
            "y_test": y_test, "y_pred": y_pred}


# ---------------------------------------------------------------------------
# Phase 4 : le premier verdict
# ---------------------------------------------------------------------------
def phase4_premier_verdict(df):
    section("PHASE 4 : le premier verdict")
    y = df["is_hoax"].values
    X = construire_features(df, inclure_comments=True)
    resultat = entrainer_evaluer(X, y, "Phase 4 - avec comments (texte)")
    return resultat


# ---------------------------------------------------------------------------
# Phase 5 : le Conseil ne vous croit pas (fuite de donnees)
# ---------------------------------------------------------------------------
def phase5_fuite(df, resultat_phase4):
    section("PHASE 5 : le Conseil ne vous croit pas")

    table = [
        ("datetime", "temoin", "au moment de l'observation", "non"),
        ("city / state / country", "temoin (localisation donnee au signalement)",
         "au moment de l'observation", "non"),
        ("shape", "temoin", "au moment de l'observation", "non"),
        ("duration_seconds / duration_hours_min", "temoin",
         "au moment de l'observation", "non"),
        ("latitude / longitude", "service de transmission (geocodage automatique)",
         "juste apres l'observation", "non"),
        ("comments", "temoin ET employe NUFORC (notes '((HOAX??))' ajoutees dans le "
         "meme champ)", "observation + relecture, des semaines plus tard",
         "OUI - c'est la source de notre etiquette elle-meme"),
        ("date_posted", "employe (mise en ligne apres traitement du dossier)",
         "des semaines plus tard", "oui, en principe (le dossier est deja traite)"),
    ]

    print("Colonne | Qui ecrit | Quand | Savait deja si canular ?")
    for ligne in table:
        print("  - " + " | ".join(ligne))

    print()
    print("Colonnes qui sortent du modele : comments (et tout ce qui en derive) et "
          "date_posted, car repondent 'oui' a la derniere question.")

    y = df["is_hoax"].values
    X_clean = construire_features(df, inclure_comments=False)
    resultat_propre = entrainer_evaluer(X_clean, y, "Phase 5 - sans comments/date_posted")

    print()
    print("Comparaison phase 4 (avant) vs phase 5 (apres) :")
    print(f"  Rappel    : {resultat_phase4['rappel']*100:.1f}  ->  "
          f"{resultat_propre['rappel']*100:.1f}")
    print(f"  Precision : {resultat_phase4['precision']*100:.1f}  ->  "
          f"{resultat_propre['precision']*100:.1f}")
    print()
    print("Explication : en phase 4, le vectoriseur TF-IDF apprenait sur le mot "
          "'hoax' lui-meme, present dans les notes ajoutees par un employe qui "
          "avait deja tranche. Le modele ne detectait pas des canulars, il relisait "
          "notre propre etiquette dans le texte. Une fois ce texte retire, il doit "
          "vraiment generaliser a partir de la forme, la duree et la localisation, "
          "et ses scores retombent a un niveau honnete.")

    return resultat_propre


# ---------------------------------------------------------------------------
# Phase 6 : le modele le plus bete du Bureau
# ---------------------------------------------------------------------------
def phase6_stagiaire(df, resultat_propre):
    section("PHASE 6 : le modele le plus bete du Bureau")

    y_test = resultat_propre["y_test"]
    y_pred_modele = resultat_propre["y_pred"]

    y_pred_stagiaire = np.zeros_like(y_test)

    exactitude_stagiaire = accuracy_score(y_test, y_pred_stagiaire)
    exactitude_modele = accuracy_score(y_test, y_pred_modele)
    rappel_stagiaire = recall_score(y_test, y_pred_stagiaire, zero_division=0)

    print(f"Exactitude du stagiaire (toujours 'pas un canular') : "
          f"{exactitude_stagiaire*100:.2f} %")
    print(f"Exactitude de notre modele (phase 5)                : "
          f"{exactitude_modele*100:.2f} %")
    print(f"Rappel du stagiaire (canulars attrapes)              : "
          f"{rappel_stagiaire*100:.1f}")
    print(f"Rappel de notre modele (phase 5)                     : "
          f"{resultat_propre['rappel']*100:.1f}")

    print()
    print("Mesure presentee au Conseil : le RAPPEL, pas l'exactitude.")
    print("Les canulars sont environ 0,9% des releves : repondre toujours 'non' "
          "obtient donc une excellente exactitude sans jamais attraper un seul "
          "canular (rappel = 0). L'exactitude ne prouve rien ici, elle recompense "
          "la classe majoritaire ; seul le rappel (et la precision) montre qu'on "
          "attrape reellement des canulars.")


# ---------------------------------------------------------------------------
def main():
    path = ensure_data()

    df_brut, lignes_a_part = phase1_ouvrir_la_caisse(path)
    df_type, anomalies, vides = phase2_typage(df_brut)
    df_label = phase3_canular(df_type)
    resultat_phase4 = phase4_premier_verdict(df_label)
    resultat_phase5 = phase5_fuite(df_label, resultat_phase4)
    phase6_stagiaire(df_label, resultat_phase5)

    section("FIN")
    print("Toutes les phases se sont executees sans intervention manuelle.")


if __name__ == "__main__":
    main()
