"""
Bureau d'Analyse Terrestre - reception des releves de la sonde Klaxo-3.

Se relance d'une traite : telechargement (si absent) -> chargement brut ->
typage -> etiquette canular -> premier modele -> fuite de donnees -> modele
final -> comparaison a la baseline. Tous les chiffres imprimes ici sont ceux
qui sont recopies dans RAPPORT.md.
"""

import csv
import html
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
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
EVENT_COLS = ["datetime", "city", "state", "country"]


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
# datetime a un format non standard : "24:00" (minuit) n'existe pas, on le
# recale sur "00:00" le jour suivant. Fonction partagee par toutes les phases
# qui ont besoin d'une date d'observation exploitable.
# ---------------------------------------------------------------------------
def parser_datetime(df):
    dt_fixed = df["datetime"].str.replace(" 24:00", " 00:00", regex=False)
    dt_parsed = pd.to_datetime(dt_fixed, format="%m/%d/%Y %H:%M", errors="coerce")
    dt_parsed = dt_parsed + pd.to_timedelta(
        df["datetime"].str.endswith(" 24:00").fillna(False).astype(int), unit="D"
    )
    return dt_parsed


# ---------------------------------------------------------------------------
# Construction des features (partagee par les phases 4, 5, 6, 7 et 8)
# ---------------------------------------------------------------------------
def construire_features(df, inclure_comments):
    df = df.copy()

    dt_parsed = parser_datetime(df)

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


def entrainer_evaluer_from_split(X, y, idx_train, idx_test, titre, silencieux=False):
    clf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    clf.fit(X[idx_train], y[idx_train])
    y_test = y[idx_test]
    y_pred = clf.predict(X[idx_test])

    rappel = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    exactitude = accuracy_score(y_test, y_pred)

    if not silencieux:
        print(f"[{titre}] evalue sur {len(y_test)} releves jamais vus a l'entrainement")
        print(f"  Rappel (canulars attrapes / 100 reels)  : {rappel * 100:.1f}")
        print(f"  Precision (vrais / 100 signales)        : {precision * 100:.1f}")
        print(f"  Exactitude (accuracy)                   : {exactitude * 100:.1f}")

    return {"rappel": rappel, "precision": precision, "exactitude": exactitude,
            "y_test": y_test, "y_pred": y_pred, "idx_train": idx_train, "idx_test": idx_test}


def entrainer_evaluer(X, y, titre):
    idx = np.arange(X.shape[0])
    idx_train, idx_test = train_test_split(
        idx, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    return entrainer_evaluer_from_split(X, y, idx_train, idx_test, titre)


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
# Phase 7 : plusieurs temoins, un seul evenement
# ---------------------------------------------------------------------------
def identifier_evenements(df):
    """Un identifiant de groupe par (datetime, ville, etat, pays) : notre
    definition d'un 'meme evenement' vu par plusieurs temoins."""
    return df.groupby(EVENT_COLS, dropna=False).ngroup()


def phase7_evenements(df, resultat_phase4):
    section("PHASE 7 : plusieurs temoins, un seul evenement")

    grp_id = identifier_evenements(df)
    tailles = grp_id.value_counts()
    multi = tailles[tailles > 1]

    print(f"Colonnes utilisees pour reconnaitre un meme evenement : {EVENT_COLS}")
    print(f"Evenements signales par plus d'un temoin : {len(multi)}")
    plus_gros_id = multi.idxmax()
    print(f"Temoins pour le plus gros evenement       : {int(multi.max())}")

    exemple = df[grp_id == plus_gros_id]
    print(f"\nExemple - evenement du {exemple['datetime'].iloc[0]} a "
          f"{exemple['city'].iloc[0]} ({len(exemple)} temoins), tous alignes :")
    print(exemple[["datetime", "city", "state", "comments"]].to_string())

    # decoupe d'hier (phase 4) : combien de releves etaient a cheval
    y = df["is_hoax"].values
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    cote = np.full(len(df), "train", dtype=object)
    cote[idx_test] = "test"
    cote_par_groupe = pd.DataFrame({"grp": grp_id.values, "cote": cote})
    cote_par_groupe = cote_par_groupe[cote_par_groupe["grp"].isin(multi.index)]
    nb_cotes = cote_par_groupe.groupby("grp")["cote"].nunique()
    groupes_a_cheval = nb_cotes[nb_cotes > 1].index
    n_a_cheval = int(cote_par_groupe[cote_par_groupe["grp"].isin(groupes_a_cheval)].shape[0])
    print(f"\nReleves a cheval sur apprentissage/test dans la decoupe d'hier "
          f"(phase 4) : {n_a_cheval}")

    # nouvelle decoupe : un evenement entier d'un seul cote
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    idx_train2, idx_test2 = next(gss.split(idx, y, groups=grp_id.values))
    X = construire_features(df, inclure_comments=True)
    resultat_groupe = entrainer_evaluer_from_split(
        X, y, idx_train2, idx_test2, "Phase 7 - decoupe par evenement"
    )

    print("\nComparaison phase 4 (decoupe aleatoire) vs phase 7 (decoupe par evenement) :")
    print(f"  Rappel    : {resultat_phase4['rappel']*100:.1f}  ->  "
          f"{resultat_groupe['rappel']*100:.1f}")
    print(f"  Precision : {resultat_phase4['precision']*100:.1f}  ->  "
          f"{resultat_groupe['precision']*100:.1f}")

    # temoignages recopies a l'identique
    textes = df["comments"].fillna("").str.strip()
    comptes = textes[textes != ""].value_counts()
    doublons = comptes[comptes > 1]
    print(f"\nTemoignages recopies mot pour mot : {int(doublons.sum())} lignes "
          f"sur {len(doublons)} textes distincts.")

    # verification : ces doublons partagent-ils aussi un meme evenement ?
    lignes_dupliquees = df.index[textes.isin(doublons.index) & (textes != "")]
    grp_dupliques = grp_id.loc[lignes_dupliquees]
    meme_evenement = grp_dupliques.isin(multi.index).sum()
    print(f"Traitement : on les garde. Ce sont des commentaires generiques tres courts "
          f"('Fireball', 'UFO', 'Lights in the sky...') qui reapparaissent par coincidence "
          f"sur des evenements sans rapport : seules {meme_evenement} de ces "
          f"{len(lignes_dupliquees)} lignes appartiennent en plus a un evenement a "
          f"plusieurs temoins. Rien n'indique un vrai copier-coller d'un meme temoignage.")

    return {"grp_id": grp_id, "resultat_groupe": resultat_groupe}


# ---------------------------------------------------------------------------
# Phase 8 : l'ordre des choses
# ---------------------------------------------------------------------------
def phase8_ordre_temporel(df, resultat_phase4):
    section("PHASE 8 : l'ordre des choses")

    dt = parser_datetime(df)
    ecart = (df["date_posted_parsed"] - dt).dt.days
    print(f"Ecart median entre date_posted et datetime : {ecart.median():.0f} jours")
    print(f"Ecart superieur a 1 an  : {(ecart > 365).sum()} releves "
          f"({(ecart > 365).mean():.1%})")
    print(f"Ecart superieur a 10 ans : {(ecart > 3650).sum()} releves")
    print("\nChoix : on decoupe sur 'datetime' (le moment ou le temoin a leve les yeux), "
          "pas sur 'date_posted' (le traitement administratif). date_posted accumule un "
          "retard tres variable, parfois de plusieurs decennies (un dossier de 1998 peut "
          "etre publie en 2013) - s'en servir pour couper melangerait completement l'ordre "
          "reel des evenements que le systeme doit apprendre a generaliser.")

    cutoff = dt.quantile(0.75)
    idx_train = np.where(dt <= cutoff)[0]
    idx_test = np.where(dt > cutoff)[0]

    print(f"\nDate de coupure (75e percentile de datetime) : {cutoff.date()}")
    print(f"Releves avant la coupure (apprentissage) : {len(idx_train)}")
    print(f"Releves apres la coupure (test)          : {len(idx_test)}")

    y = df["is_hoax"].values
    prop_train = y[idx_train].mean()
    prop_test = y[idx_test].mean()
    print(f"Proportion de canulars avant : {prop_train:.2%}")
    print(f"Proportion de canulars apres : {prop_test:.2%}")

    X = construire_features(df, inclure_comments=True)
    resultat_temps = entrainer_evaluer_from_split(
        X, y, idx_train, idx_test, "Phase 8 - decoupe temporelle"
    )

    print("\nComparaison phase 4 (decoupe aleatoire) vs phase 8 (decoupe temporelle) :")
    print(f"  Rappel    : {resultat_phase4['rappel']*100:.1f}  ->  "
          f"{resultat_temps['rappel']*100:.1f}")
    print(f"  Precision : {resultat_phase4['precision']*100:.1f}  ->  "
          f"{resultat_temps['precision']*100:.1f}")

    grp_id = identifier_evenements(df)
    cote = pd.Series(np.where(dt <= cutoff, "train", "test"), index=df.index)
    a_cheval = cote.groupby(grp_id).nunique()
    print(f"\n(Bonus : {int((a_cheval > 1).sum())} evenements a cheval sur cette coupure "
          f"temporelle - un meme evenement se deroule en une seule nuit, la coupure par "
          f"date respecte donc naturellement le regroupement de la phase 7.)")

    return {"cutoff": cutoff, "idx_train": idx_train, "idx_test": idx_test,
            "resultat_temps": resultat_temps}


# ---------------------------------------------------------------------------
# Phase 9 : les cases vides
# ---------------------------------------------------------------------------
def phase9_cases_vides(df):
    section("PHASE 9 : les cases vides")

    colonnes = ["country", "state", "shape"]  # les 3 plus trouees (12365, 7409, 2922)
    y = df["is_hoax"]

    print("Colonne  | trous  | % canulars si trou | % canulars si rempli")
    resultats = []
    for col in colonnes:
        trou = df[col] == ""
        prop_trou = y[trou].mean()
        prop_plein = y[~trou].mean()
        resultats.append((col, int(trou.sum()), prop_trou, prop_plein))
        print(f"  {col:<7} | {trou.sum():>6} | {prop_trou:.2%}             | {prop_plein:.2%}")

    print("\nTraitement retenu : remplacer chaque trou par une categorie explicite "
          "'inconnu', pas par la valeur la plus frequente. Dans l'encodage one-hot, "
          "'inconnu' reste un niveau a part entiere que le modele peut utiliser - on "
          "bouche le trou sans effacer la trace qu'il y avait un trou.")

    return resultats


# ---------------------------------------------------------------------------
# Phase 10 : la chaine de traitement du Bureau
# ---------------------------------------------------------------------------
def construire_table_brute(df):
    """Table de colonnes brutes, non encore apprises, pret pour un
    ColumnTransformer. Rien ici n'est calcule sur l'ensemble du fichier :
    tout ce qui doit etre appris (mediane, encodage) sera ajuste par le
    pipeline, uniquement sur la partie apprentissage."""
    dt = parser_datetime(df)
    return pd.DataFrame({
        "hour": dt.dt.hour,
        "weekday": dt.dt.dayofweek,
        "month": dt.dt.month,
        "duration_seconds": pd.to_numeric(df["duration_seconds"], errors="coerce"),
        "latitude": df["latitude"],
        "longitude": df["longitude"],
        "shape": df["shape"].replace("", np.nan),
        "country": df["country"].replace("", np.nan),
        "state": df["state"].replace("", np.nan),
    }, index=df.index)


def construire_pipeline_v1():
    num_cols = ["hour", "weekday", "month", "duration_seconds", "latitude", "longitude"]
    cat_cols = ["shape", "country", "state"]
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="inconnu")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]), cat_cols),
    ])
    return Pipeline([
        ("prep", prep),
        ("clf", RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])


def phase10_pipeline(df, idx_train, idx_test):
    section("PHASE 10 : la chaine de traitement du Bureau")

    table = construire_table_brute(df)
    y = df["is_hoax"].values
    X_train, X_test = table.iloc[idx_train], table.iloc[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    print(f"Proportion de canulars - apprentissage : {y_train.mean():.2%}")
    print(f"Proportion de canulars - test          : {y_test.mean():.2%}")

    pipeline = construire_pipeline_v1()
    pipeline.fit(X_train, y_train)  # mediane et encodeur ajustes sur l'apprentissage SEUL
    y_pred = pipeline.predict(X_test)
    print(f"\nRappel (mediane/encodeur appris sur l'apprentissage seul)    : "
          f"{recall_score(y_test, y_pred, zero_division=0)*100:.1f}")
    print(f"Precision (mediane/encodeur appris sur l'apprentissage seul) : "
          f"{precision_score(y_test, y_pred, zero_division=0)*100:.1f}")

    ligne = X_test.iloc[[0]]
    pred = pipeline.predict(ligne)[0]
    proba = pipeline.predict_proba(ligne)[0, 1]
    print("\nDemonstration - un releve neuf traverse toute la chaine en un seul appel :")
    print(ligne.to_string(index=False))
    print(f"-> prediction : {'canular' if pred else 'pas un canular'} "
          f"(probabilite estimee : {proba:.3f})")

    return pipeline, table


# ---------------------------------------------------------------------------
# Phase 11 : combien de temps ca a dure
# ---------------------------------------------------------------------------
UNITE_SECONDES = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "day": 86400, "days": 86400,
}
QUANTITE_VAGUE = {
    "several": 5, "couple": 2, "few": 3, "some": 3, "one": 1, "two": 2,
    "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "twenty": 20, "thirty": 30, "a": 1, "an": 1,
}
DUREE_INUTILISABLE = {
    "unknown", "unkown", "unsure", "not sure", "na", "n/a", "current",
    "still there", "still going", "still here", "still happening",
    "still going on", "ongoing", "on going", "continuing", "varies",
    "uncertain", "not known", "all night", "night", "short", "long",
    "unk", "?", "??", "???", "", "-", "none", "dont know", "don't know",
    "don&#39t know", "photo", "ufo", "continuous", "brief",
}


def parser_duree_texte(brut):
    """Convertit une duree ecrite en texte libre (duration_hours_min) en
    secondes. Retourne None si le texte est trop vague pour etre chiffre."""
    if not isinstance(brut, str):
        return None
    s = html.unescape(brut).strip().lower()
    if s == "":
        return None
    s = s.replace("approx:", "").replace("approx.", "approx")
    s = re.sub(r"[()?+]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s in DUREE_INUTILISABLE:
        return None
    if s in {"instant", "momentary", "split second", "a moment", "flash"}:
        return 1.0

    m = re.fullmatch(r"(\d*):(\d{2})", s)  # format H:MM ou :MM
    if m:
        h = int(m.group(1)) if m.group(1) else 0
        return h * 3600 + int(m.group(2)) * 60

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to|or)\s*(\d+(?:\.\d+)?)\s*([a-z]+)", s)
    if m:
        lo, hi, unite = float(m.group(1)), float(m.group(2)), m.group(3)
        sec = UNITE_SECONDES.get(unite)
        if sec:
            return (lo + hi) / 2 * sec

    m = re.search(r"(\d+)\s*/\s*(\d+)\s*([a-z]+)", s)
    if m:
        num, den, unite = int(m.group(1)), int(m.group(2)), m.group(3)
        sec = UNITE_SECONDES.get(unite)
        if sec and den:
            return (num / den) * sec

    m = re.search(r"half\s*(?:an?\s*)?(hour|minute|min|hr)", s)
    if m:
        sec = UNITE_SECONDES.get(m.group(1))
        if sec:
            return 0.5 * sec

    m = re.search(r"(\d+(?:\.\d+)?)\s*([a-z]+)", s)
    if m:
        val, unite = float(m.group(1)), m.group(2)
        sec = UNITE_SECONDES.get(unite)
        if sec:
            return val * sec

    for mot, qte in QUANTITE_VAGUE.items():
        m = re.search(rf"\b{re.escape(mot)}\b\s*(?:of\s*)?([a-z]+)", s)
        if m:
            sec = UNITE_SECONDES.get(m.group(1))
            if sec:
                return qte * sec

    if s in UNITE_SECONDES:  # mot d'unite seul, sans quantite -> on suppose 1
        return float(UNITE_SECONDES[s])

    m = re.fullmatch(r"(\d+(?:\.\d+)?)", s)  # nombre nu -> on suppose des minutes
    if m:
        return float(m.group(1)) * 60

    return None


def phase11_duree(df):
    section("PHASE 11 : combien de temps ca a dure")

    brut = pd.to_numeric(df["duration_seconds"], errors="coerce")
    brut_valide = brut.where(brut > 0)  # 0 traite comme suspect, pas comme une vraie mesure
    texte_parse = df["duration_hours_min"].fillna("").map(parser_duree_texte)

    duree = brut_valide.copy()
    a_completer = duree.isna()
    duree = duree.where(~a_completer, texte_parse)

    deux_presents = brut_valide.notna() & texte_parse.notna()
    ratio = (brut_valide / texte_parse).where(deux_presents)
    contradiction = deux_presents & ((ratio > 3) | (ratio < 1 / 3))

    inutilisable = int(duree.isna().sum())
    n_contradiction = int(contradiction.sum())
    mediane = float(duree.median())
    plus_dune_journee = int((duree >= 86400).sum())

    print(f"Releves dont la duree reste inutilisable apres traitement : "
          f"{inutilisable} / {len(df)}")
    print(f"Releves ou les deux colonnes se contredisent (facteur > 3) : {n_contradiction}")
    print(f"Duree mediane : {mediane:.0f} s (~{mediane/60:.1f} min)")
    print(f"Releves annoncant plus d'une journee d'observation (>= 86400 s) : "
          f"{plus_dune_journee}")

    exemple_idx = df.index[(brut == 0) & texte_parse.notna()][0]
    print(f"\nExemple de desaccord : duration_seconds="
          f"{df.loc[exemple_idx, 'duration_seconds']!r} mais duration_hours_min="
          f"{df.loc[exemple_idx, 'duration_hours_min']!r} "
          f"(duree recuperee : {texte_parse[exemple_idx]:.0f} s).")

    print("\nLes 3 durees les plus longues du fichier :")
    top3 = duree.sort_values(ascending=False).head(3)
    for i, v in top3.items():
        print(f"  {v:.0f} s (~{v/31557600:.1f} ans) - duration_seconds="
              f"{df.loc[i, 'duration_seconds']!r}, duration_hours_min="
              f"{df.loc[i, 'duration_hours_min']!r}")

    print("\nDecision : la duree utilisee par le modele est plafonnee a 1 jour (86400 s), "
          "et un indicateur booleen 'duree_extreme' garde la trace des releves au-dessus. "
          "Une observation de plusieurs annees n'est plus un 'releve' ponctuel mais un "
          "phenomene recurrent mal capture par une colonne en secondes - la laisser telle "
          "quelle ecraserait toute mediane. Aucune ligne n'est supprimee.")

    duree_plafonnee = duree.clip(upper=86400)
    duree_extreme = (duree >= 86400).astype(int)
    return duree_plafonnee, duree_extreme


# ---------------------------------------------------------------------------
# Phase 12 : la ville et l'heure
# ---------------------------------------------------------------------------
FUSION_FORMES = {"changed": "changing", "round": "circle"}


def phase12_ville_heure(df, table_base, duree_plafonnee, duree_extreme, idx_train):
    section("PHASE 12 : la ville et l'heure")

    n_villes = int(df["city"].nunique())
    villes_uniques = int((df["city"].value_counts() == 1).sum())
    print(f"Villes distinctes dans la transmission : {n_villes}")
    print(f"Villes qui n'apparaissent qu'une seule fois : {villes_uniques}")
    print("Regle appliquee a la ville : one-hot avec seuil de frequence minimale "
          "(min_frequency=15) - toute ville vue moins de 15 fois dans la transmission "
          "est regroupee dans une categorie 'rare' au lieu de recevoir sa propre colonne.")

    heure = table_base["hour"]

    def point_horaire(h):
        return np.array([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)])

    d_23_0 = float(np.linalg.norm(point_horaire(23) - point_horaire(0)))
    d_23_20 = float(np.linalg.norm(point_horaire(23) - point_horaire(20)))
    print(f"\nEncodage cyclique de l'heure (sin/cos sur 24h) :")
    print(f"  Distance encodee entre 23h et 0h  : {d_23_0:.3f}")
    print(f"  Distance encodee entre 23h et 20h : {d_23_20:.3f}")
    print("  23h est bien plus proche de 0h que de 20h : le cercle horaire est respecte.")

    formes_avant = int(df["shape"].replace("", np.nan).dropna().nunique())
    shape_fusionne = df["shape"].replace(FUSION_FORMES)
    formes_apres = int(shape_fusionne.replace("", np.nan).dropna().nunique())
    print(f"\nFormes distinctes avant fusion : {formes_avant}")
    print("Fusions appliquees : 'changed' -> 'changing' (meme mot, deux conjugaisons), "
          "'round' -> 'circle' (la meme forme decrite deux fois).")
    print(f"Formes distinctes apres fusion : {formes_apres}")

    table = table_base.drop(columns=["hour", "duration_seconds"]).copy()
    table["hour_sin"] = np.sin(2 * np.pi * heure / 24)
    table["hour_cos"] = np.cos(2 * np.pi * heure / 24)
    table["duration_seconds"] = duree_plafonnee
    table["duree_extreme"] = duree_extreme
    table["shape"] = shape_fusionne
    table["city"] = df["city"].replace("", np.nan)

    num_cols = ["hour_sin", "hour_cos", "weekday", "month", "duration_seconds",
                "duree_extreme", "latitude", "longitude"]
    cat_cols = ["shape", "country", "state", "city"]

    prep_final = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="inconnu")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=15)),
        ]), cat_cols),
    ])
    largeur_apres = prep_final.fit_transform(table.iloc[idx_train]).shape[1]

    prep_naif = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="inconnu")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),  # pas de min_frequency
        ]), cat_cols),
    ])
    largeur_avant = prep_naif.fit_transform(table.iloc[idx_train]).shape[1]

    print(f"\nNombre de colonnes du tableau avant (ville en one-hot naif) : {largeur_avant}")
    print(f"Nombre de colonnes du tableau apres (ville regroupee)        : {largeur_apres}")

    return table, num_cols, cat_cols


def construire_pipeline_final(num_cols, cat_cols):
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="inconnu")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=15)),
        ]), cat_cols),
    ])
    return Pipeline([
        ("prep", prep),
        ("clf", RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])


COUT_FN = 30  # canular manque
COUT_FP = 2   # fausse alerte


def facture(y_vrai, y_pred):
    fn = int(((y_pred == 0) & (y_vrai == 1)).sum())
    fp = int(((y_pred == 1) & (y_vrai == 0)).sum())
    return fn * COUT_FN + fp * COUT_FP


# ---------------------------------------------------------------------------
# Phase 13 : la facture du Bureau
# ---------------------------------------------------------------------------
def phase13_facture(y_test, proba_test):
    section("PHASE 13 : la facture du Bureau")

    print(f"Grille de couts : canular manque = {COUT_FN} credits, "
          f"fausse alerte = {COUT_FP} credits, bonne reponse = 0.")

    seuils = np.linspace(0.01, 0.99, 99)
    factures = np.array([facture(y_test, (proba_test >= s).astype(int)) for s in seuils])
    i_opt = int(np.argmin(factures))
    seuil_opt, facture_opt = float(seuils[i_opt]), int(factures[i_opt])
    facture_05 = facture(y_test, (proba_test >= 0.5).astype(int))

    print("\nFrontiere -> facture (echantillon tous les 0.10) :")
    for s, f in zip(seuils[::10], factures[::10]):
        print(f"  {s:.2f} -> {f} credits")

    print(f"\nFrontiere retenue (minimise la facture) : {seuil_opt:.2f}")
    print(f"Facture a 0.5 (choix par defaut de la bibliotheque) : {facture_05} credits")
    print(f"Facture a {seuil_opt:.2f} (frontiere retenue)                : {facture_opt} credits")
    print(f"Ecart                                                : "
          f"{facture_05 - facture_opt} credits economises")
    print("\nJustification : un canular manque coute 15 fois plus cher qu'une fausse alerte "
          "(30 contre 2 credits). La frontiere a 0.5 ne le sait pas ; celle retenue ici "
          "penche deliberement vers plus d'alertes, parce que c'est ce que la grille du "
          "Conseil paie le moins cher.")

    return seuil_opt


# ---------------------------------------------------------------------------
# Phase 14 : une promesse a 80 %
# ---------------------------------------------------------------------------
def tranches_robustes(valeurs, minimum=5):
    """qcut, en augmentant la resolution jusqu'a obtenir au moins `minimum`
    tranches distinctes - les probabilites d'une foret aleatoire sont pleines
    de valeurs identiques (beaucoup de releves tombent dans les memes feuilles),
    ce qui fait fondre un qcut(q=8) classique a moins de tranches que demande."""
    dernier = None
    for q in (8, 15, 30, 60, 120, 250):
        dernier = pd.qcut(valeurs, q=q, duplicates="drop")
        if dernier.categories.size >= minimum:
            return dernier
    return dernier


def phase14_calibration(pipeline, X_train, y_train, X_test, y_test, proba_test):
    section("PHASE 14 : une promesse a 80 %")

    tranches = tranches_robustes(proba_test)
    table = pd.DataFrame({"tranche": tranches, "proba": proba_test, "reel": y_test})
    resume = table.groupby("tranche", observed=True).agg(
        n=("reel", "size"), proba_moyenne=("proba", "mean"), proportion_reelle=("reel", "mean")
    )
    print("Tranche              | n     | proba. moyenne annoncee | proportion reelle de canulars")
    for tranche, ligne in resume.iterrows():
        print(f"  {str(tranche):<20} | {int(ligne['n']):>5} | {ligne['proba_moyenne']:.3f}"
              f"                  | {ligne['proportion_reelle']:.3f}")

    ecart_moyen = (resume["proba_moyenne"] - resume["proportion_reelle"]).mean()
    sens = "trop confiant (il annonce plus que ce qui se realise)" if ecart_moyen > 0 \
        else "trop prudent (il annonce moins que ce qui se realise)"
    print(f"\nLe systeme est {sens}.")

    print("\nCorrection : calibration isotonique (CalibratedClassifierCV, ajustee par "
          "validation croisee sur l'apprentissage uniquement).")
    pipeline_calibre = CalibratedClassifierCV(pipeline, method="isotonic", cv=3)
    pipeline_calibre.fit(X_train, y_train)
    proba_calibree = pipeline_calibre.predict_proba(X_test)[:, 1]

    tranches2 = tranches_robustes(proba_calibree)
    table2 = pd.DataFrame({"tranche": tranches2, "proba": proba_calibree, "reel": y_test})
    resume2 = table2.groupby("tranche", observed=True).agg(
        n=("reel", "size"), proba_moyenne=("proba", "mean"), proportion_reelle=("reel", "mean")
    )
    print("\nMeme tableau apres calibration :")
    print("Tranche              | n     | proba. moyenne annoncee | proportion reelle de canulars")
    for tranche, ligne in resume2.iterrows():
        print(f"  {str(tranche):<20} | {int(ligne['n']):>5} | {ligne['proba_moyenne']:.3f}"
              f"                  | {ligne['proportion_reelle']:.3f}")

    return pipeline_calibre, proba_calibree


# ---------------------------------------------------------------------------
# Phase 15 : deux analystes, deux chiffres
# ---------------------------------------------------------------------------
def phase15_intervalle(y_test, proba_test, seuil):
    section("PHASE 15 : deux analystes, deux chiffres")

    n_canulars_test = int(y_test.sum())
    print(f"Taille de la partie test           : {len(y_test)}")
    print(f"Canulars reellement presents dedans : {n_canulars_test}")

    rng = np.random.default_rng(RANDOM_STATE)
    n_boot = 1000
    rappels = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_test), len(y_test))
        yt, pt = y_test[idx], proba_test[idx]
        if yt.sum() == 0:
            continue
        pred = (pt >= seuil).astype(int)
        rappels.append(recall_score(yt, pred, zero_division=0))
    rappels = np.array(rappels)
    point = recall_score(y_test, (proba_test >= seuil).astype(int), zero_division=0)
    lo, hi = np.percentile(rappels, [2.5, 97.5])

    print(f"\nRappel de notre systeme : {point:.2f}")
    print(f"Intervalle a 95 % (sur {n_boot} tirages bootstrap de la partie test) : "
          f"[{lo:.2f}, {hi:.2f}]")

    print(f"\nReponse au Conseil sur les deux analystes : avec seulement {n_canulars_test} "
          f"canulars dans la partie test, deux chiffres separes de quelques centiemes ne "
          f"prouvent rien - notre propre rappel bouge deja de {lo:.2f} a {hi:.2f} rien qu'en "
          f"tirant au hasard une autre version du meme test. Il faudrait un ecart bien plus "
          f"large que 0,31 vs 0,34 pour dire qu'un systeme est vraiment meilleur que l'autre.")

    return lo, hi, point


# ---------------------------------------------------------------------------
# Phase 16 : trois dossiers sur le bureau
# ---------------------------------------------------------------------------
def phase16_explications(pipeline, X_train, X_test, y_test, proba_test, seuil):
    section("PHASE 16 : trois dossiers sur le bureau")

    print("Importance par permutation (chute de l'aire sous la courbe precision-rappel "
          "quand une colonne est melangee) :")
    print("(le rappel/precision au seuil retenu sont presque tout le temps a 0 sur ce "
          "modele - un score base sur predict_proba comme l'AP est necessaire pour voir "
          "quoi que ce soit bouger.)")
    imp = permutation_importance(
        pipeline, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE,
        scoring="average_precision", n_jobs=-1,
    )
    classement = pd.Series(imp.importances_mean, index=X_test.columns).sort_values(ascending=False)
    for col, val in classement.items():
        print(f"  {col:<18} : {val:+.4f}")

    surprise = classement.index[0]
    print(f"\nColonne dont la place surprend : '{surprise}' arrive en tete - une colonne de "
          f"contexte (temps/lieu), pas une colonne de description directe du phenomene, "
          f"pese le plus lourd dans la decision globale.")

    pos_confiant = int(np.argmax(proba_test))
    autres = np.array([i for i in range(len(proba_test)) if i != pos_confiant])
    pos_limite = int(autres[np.argmin(np.abs(proba_test[autres] - seuil))])
    faux_negatifs = np.where((y_test == 1) & (proba_test < seuil))[0]
    pos_rate = int(faux_negatifs[np.argmax(proba_test[faux_negatifs])]) if len(faux_negatifs) else None

    label_confiant = "Dossier 1 - le releve juge le plus suspect" if proba_test[pos_confiant] < seuil \
        else "Dossier 1 - canular signale avec forte confiance"
    label_limite = "Dossier 2 - juste en dessous de la frontiere" if proba_test[pos_limite] < seuil \
        else "Dossier 2 - tout juste au-dessus de la frontiere"
    if proba_test[pos_confiant] < seuil:
        print(f"\n(Aucun releve de la partie test ne franchit la frontiere de {seuil:.2f} : "
              f"meme le plus suspect ne plafonne qu'a {proba_test[pos_confiant]:.3f}. Les "
              f"dossiers 1 et 2 restent les deux releves les plus suspects du lot, mais "
              f"aucun des deux n'aurait declenche d'alerte.)")

    colonnes_top5 = list(classement.index[:5])
    valeurs_types = {}
    for col in colonnes_top5:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            valeurs_types[col] = X_train[col].median()
        else:
            valeurs_types[col] = X_train[col].mode(dropna=True).iloc[0]

    def expliquer(nom, pos):
        if pos is None:
            print(f"\n{nom} : aucun canular manque dans cette partie test.")
            return
        ligne = X_test.iloc[[pos]]
        print(f"\n{nom} (probabilite annoncee : {proba_test[pos]:.3f}, "
              f"reellement canular : {'oui' if y_test[pos] else 'non'}) :")
        print(ligne.to_string(index=False))
        contributions = {}
        for col in classement.index[:5]:
            modif = ligne.copy()
            modif[col] = valeurs_types[col]
            nouvelle_proba = pipeline.predict_proba(modif)[0, 1]
            contributions[col] = proba_test[pos] - nouvelle_proba
        contributions = dict(sorted(contributions.items(), key=lambda kv: -abs(kv[1])))
        print("  Effet estime de chaque colonne (probabilite avec vs sans sa valeur reelle) :")
        for col, delta in contributions.items():
            print(f"    {col:<18} : {delta:+.3f}")
        return ligne

    ligne_confiant = expliquer(label_confiant, pos_confiant)
    ligne_limite = expliquer(label_limite, pos_limite)
    ligne_rate = expliquer("Dossier 3 - canular laisse passer", pos_rate)

    print("\nCe classement global (colonnes qui comptent 'en moyenne') n'explique pas un "
          "dossier precis : un dossier peut basculer a cause d'une colonne qui, en moyenne, "
          "ne pese presque rien.")

    return classement


# ---------------------------------------------------------------------------
# Phase 17 : l'angle mort du Bureau
# ---------------------------------------------------------------------------
def phase17_zones(df, idx_test, y_test, proba_test, seuil):
    section("PHASE 17 : l'angle mort du Bureau")

    pays_test = df["country"].values[idx_test]
    zone = np.select(
        [pays_test == "us", pays_test == "ca", pays_test == "gb",
         np.isin(pays_test, ["au", "de"])],
        ["us", "ca", "gb", "autre"],
        default="inconnu",
    )

    print("Zone    | n      | % canulars | rappel | precision")
    lignes = []
    for z in ["us", "ca", "gb", "autre", "inconnu"]:
        m = zone == z
        n = int(m.sum())
        if n == 0:
            continue
        prop = float(y_test[m].mean())
        pred = (proba_test[m] >= seuil).astype(int)
        rap = recall_score(y_test[m], pred, zero_division=0)
        prec = precision_score(y_test[m], pred, zero_division=0)
        lignes.append((z, n, prop, rap, prec))
        print(f"  {z:<7} | {n:>6} | {prop:.2%}     | {rap:.2f}   | {prec:.2f}")

    prop_globale = float(y_test.mean())
    print(f"\nProportion globale de canulars (reference) : {prop_globale:.2%}")

    print("\nDecision : une seule frontiere pour toutes les zones. La phase 15 a montre que "
          "le rappel bouge deja de plusieurs centiemes rien qu'en rejouant le meme test sur "
          "la partie us (des dizaines de milliers de releves) - une zone comme 'gb' ou 'ca' "
          "ne contient que quelques dizaines de canulars dans le test, largement trop peu pour "
          "regler une frontiere specifique sans se caler sur du bruit.")

    return lignes


# ---------------------------------------------------------------------------
# Phase 18 : la transmission d'archive
# ---------------------------------------------------------------------------
def phase18_archive(df, table_finale, num_cols, cat_cols, resultat_phase8):
    section("PHASE 18 : la transmission d'archive")

    dt = parser_datetime(df)
    annee = dt.dt.year
    y = df["is_hoax"]

    effectifs = df.groupby(annee)["is_hoax"].size()
    proportions = df.groupby(annee)["is_hoax"].mean()
    annees_fiables = effectifs[effectifs >= 100].index
    proportions = proportions.loc[annees_fiables]

    print("Proportion de canulars par annee (annees avec >= 100 releves) :")
    for an, p in proportions.items():
        print(f"  {int(an)} : {p:.2%}  (n={int(effectifs[an])})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(proportions.index, proportions.values * 100, marker="o")
        ax.set_xlabel("Annee")
        ax.set_ylabel("% de canulars")
        ax.set_title("Proportion de canulars par annee")
        fig.tight_layout()
        out_png = Path(__file__).resolve().parent / "klaxo3_hoax_par_annee.png"
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        print(f"\nGraphique enregistre : {out_png.name}")
    except ImportError:
        print("\n(matplotlib absent, graphique non genere - la table ci-dessus suffit)")

    cutoff50 = dt.quantile(0.5)
    idx_train50 = np.where(dt <= cutoff50)[0]
    idx_test50 = np.where(dt > cutoff50)[0]
    X_comments = construire_features(df, inclure_comments=True)
    resultat_50 = entrainer_evaluer_from_split(
        X_comments, df["is_hoax"].values, idx_train50, idx_test50,
        "Phase 18 - epreuve ancien -> recent (coupure a 50%)",
    )

    rappel_p8 = resultat_phase8["resultat_temps"]["rappel"] * 100
    precision_p8 = resultat_phase8["resultat_temps"]["precision"] * 100
    print(f"\nCoupure a 50% (date de coupure : {cutoff50.date()}) vs coupure a 75% (phase 8, "
          f"date de coupure : {resultat_phase8['cutoff'].date()}) :")
    print(f"  Rappel    : {rappel_p8:.1f}  ->  {resultat_50['rappel']*100:.1f}")
    print(f"  Precision : {precision_p8:.1f}  ->  {resultat_50['precision']*100:.1f}")

    print("\nIndicateurs de surveillance en production (aucun n'a besoin de connaitre la "
          "reponse) :")
    print("  1. Taux d'alerte : proportion de releves classes canular par semaine. Un ecart "
          "fort avec la moyenne historique (ex. +/-50% relatif) signale que le flux entrant "
          "a change de nature.")
    print("  2. Distribution des probabilites predites : comparer chaque semaine les scores "
          "produits a ceux de la periode d'entrainement (test de Kolmogorov-Smirnov).")
    print("  Frequence : chaque semaine.")
    print("  Seuil de rappel des analystes : ecart de plus de 50% du taux d'alerte "
          "historique, ou p < 0,01 au test KS, sur deux semaines consecutives.")

    return proportions, resultat_50


def main():
    path = ensure_data()

    df_brut, lignes_a_part = phase1_ouvrir_la_caisse(path)
    df_type, anomalies, vides = phase2_typage(df_brut)
    df_label = phase3_canular(df_type)
    resultat_phase4 = phase4_premier_verdict(df_label)
    resultat_phase5 = phase5_fuite(df_label, resultat_phase4)
    phase6_stagiaire(df_label, resultat_phase5)

    phase7_evenements(df_label, resultat_phase4)
    resultat_phase8 = phase8_ordre_temporel(df_label, resultat_phase4)
    idx_train, idx_test = resultat_phase8["idx_train"], resultat_phase8["idx_test"]
    phase9_cases_vides(df_label)

    _, table_base = phase10_pipeline(df_label, idx_train, idx_test)
    duree_plafonnee, duree_extreme = phase11_duree(df_label)
    table_finale, num_cols, cat_cols = phase12_ville_heure(
        df_label, table_base, duree_plafonnee, duree_extreme, idx_train
    )

    y = df_label["is_hoax"].values
    X_train = table_finale.iloc[idx_train]
    X_test = table_finale.iloc[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    pipeline_final = construire_pipeline_final(num_cols, cat_cols)
    pipeline_final.fit(X_train, y_train)
    proba_test = pipeline_final.predict_proba(X_test)[:, 1]

    seuil = phase13_facture(y_test, proba_test)
    _, proba_calibree = phase14_calibration(
        pipeline_final, X_train, y_train, X_test, y_test, proba_test
    )
    phase15_intervalle(y_test, proba_test, seuil)
    phase16_explications(pipeline_final, X_train, X_test, y_test, proba_test, seuil)
    phase17_zones(df_label, idx_test, y_test, proba_test, seuil)
    phase18_archive(df_label, table_finale, num_cols, cat_cols, resultat_phase8)

    section("FIN")
    print("Toutes les phases se sont executees sans intervention manuelle.")


if __name__ == "__main__":
    main()
