# RAPPORT.md — Bureau d'Analyse Terrestre

Ce rapport suit `analyse.py` phase par phase. Tous les chiffres ci-dessous sont
réaffichés par le script quand on le relance d'une traite.

## Phase 1 : ouvrir la caisse

- Lignes dans le fichier : **88 875**
- Lignes chargées normalement (11 champs) : **88 679**
- Lignes mises à part (nombre de champs ≠ 11) : **196**

88 679 + 196 = 88 875, ça tombe juste : aucune ligne n'a disparu, on a juste
mis les 196 de côté le temps de comprendre ce qui clochait.

Elles ont toutes **12 champs** au lieu de 11. Exemple :

```
10/1/2006 12:00,,,,,0,,,"((EDITORIAL COMMENT ABOUT THE UFO PHENOMEN))  ufo+alien+reptiles",10/30/2006,0,0
```

Il y a un champ vide en trop entre `duration_seconds` et le commentaire
(`duration_hours_min` et `comments` sont vides, puis le texte arrive en 9ᵉ
position au lieu de la 8ᵉ, ce qui décale `date_posted`, `latitude` et
`longitude` d'une case). Aucune de ces 196 lignes n'a de `shape` renseignée
non plus — ça sent plus le bug du service qui a compilé le CSV qu'une erreur
du témoin. On les garde à part sans les recaler à la main : les décaler
aurait été un choix arbitraire, et elles ne pèsent que 0,22% du total.

## Phase 2 : rien n'est du bon type

Anomalies rencontrées lors de la conversion des 88 679 lignes propres :

| Champ | Type visé | Échecs | Exemple de valeur fautive |
|---|---|---|---|
| `latitude` | nombre | **1** | `33q.200088` |
| `longitude` | nombre | 0 | — |
| `duration_seconds` | nombre | **5** | `2\``, `8\``, `0.5\``, deux valeurs vides |
| `datetime` | date | **1220** | `10/10/2005 24:00` (toujours `24:00`) |
| `date_posted` | date | 0 | — |

Quatre anomalies, quatre origines différentes :

1. **`latitude` — 1 valeur (`33q.200088`)** : un `q` glissé au milieu d'un
   nombre. Une seule ligne sur 88 679, mais ça suffit à faire lire toute la
   colonne comme du texte si on ne force pas la conversion. Ça vient du
   **service de transmission** (le géocodage qui a produit le CSV).
2. **`duration_seconds` — 5 valeurs (ex. `` `2` ``, `` `8` ``)** : un
   guillemet simple mal encodé en backtick. À côté, `duration_hours_min` dit
   des choses comme `"each a few seconds"` ou `"1/2 segundo"` — ce sont des
   durées écrites librement par le **témoin**, juste mal réencodées ensuite.
3. **`datetime` — 1220 valeurs finissant par `24:00`** : ça n'existe pas
   (minuit c'est `00:00` le jour suivant). Le motif revient sur plein de
   dates différentes, donc c'est un bug du script de standardisation —
   **service de transmission**.
4. **`shape` vide — 2922 lignes (3,30%)** : le témoin n'a pas su ou pas voulu
   décrire la forme — **témoin**.

Au passage, mêmes causes probables : `country` vide sur 12 365 lignes
(13,94%) et `state` vide sur 7409 lignes (8,35%) — localisation trop vague
pour que le géocodage s'en sorte.

Rien n'est supprimé ici : les valeurs qui résistent à la conversion
deviennent `NaN` / `NaT`, la ligne reste.

## Phase 3 : le Conseil veut trier les canulars

**Règle (une phrase) :** un relevé est marqué canular si son champ `comments`
contient le mot « hoax » (recherche insensible à la casse).

- Relevés marqués canulars : **802 / 88 679 (0,90 %)**

**Faux positifs :** 674 des 802 « hoax » viennent en fait d'une note
`((HOAX??))` ou `((NUFORC Note: Possible hoax?? PD))` ajoutée par un employé
NUFORC qui n'était pas sûr. La règle les compte comme certains alors que le
point d'interrogation dit l'inverse.

**Ratés :** 65 commentaires disent « fake », « prank », « made up » ou
« joke » sans jamais écrire « hoax ». La règle passe complètement à côté.

## Phase 4 : le premier verdict

Modèle : `RandomForestClassifier` (200 arbres, `class_weight="balanced"`),
entraîné sur 75% des relevés et évalué sur les 25% restants (**22 170
relevés jamais vus à l'entraînement**, tirage stratifié pour garder ~0,9% de
canulars des deux côtés).

Features utilisées dans cette phase : heure / jour de semaine / mois extraits
de `datetime`, `duration_seconds`, `latitude`, `longitude`, `shape`,
`country`, `state` **et un TF-IDF (500 mots) du champ `comments`**.

- Sur 100 canulars réellement présents dans le test, le système en attrape :
  **94,0**
- Sur 100 relevés que le système signale, le nombre qui en sont vraiment :
  **100,0**

Ces deux nombres viennent uniquement du test (les 22 170 relevés jamais
montrés au modèle avant).

C'est presque trop beau pour être vrai. La phase suivante explique pourquoi.

## Phase 5 : le Conseil ne vous croit pas

Tableau des colonnes utilisées par le modèle de la phase 4 :

| Colonne | Qui écrit cette information | À quel moment | Savait déjà si canular ? |
|---|---|---|---|
| `datetime` | témoin | au moment de l'observation | non |
| `city` / `state` / `country` | témoin (localisation donnée dans le signalement) | au moment de l'observation | non |
| `shape` | témoin | au moment de l'observation | non |
| `duration_seconds` / `duration_hours_min` | témoin | au moment de l'observation | non |
| `latitude` / `longitude` | service de transmission (géocodage automatique à partir de la localisation du témoin) | juste après l'observation | non |
| `comments` | **témoin ET employé NUFORC** — les notes `((HOAX??))` sont écrites *dans le même champ* que le témoignage | témoignage le jour même, note ajoutée des semaines plus tard lors de la relecture | **OUI — c'est littéralement la source de notre étiquette** |
| `date_posted` | employé (date de mise en ligne, après traitement du dossier) | des semaines plus tard | oui, en principe : le dossier est déjà traité quand il est publié |

`comments` et `date_posted` répondent « oui » à la quatrième case. Elles
sortent du modèle. On relance à l'identique sur les colonnes restantes.

**Comparaison des deux nombres de la phase 4, avant / après :**

| | Rappel (canulars attrapés / 100) | Précision (signalés qui sont vrais / 100) |
|---|---|---|
| Avant (avec `comments`) | 94,0 | 100,0 |
| Après (sans `comments` ni `date_posted`) | **0,0** (0/201 canulars du test) | **0,0** (0/3 alertes du test) |

**Pourquoi ça s'effondre :** notre étiquette « canular » c'est littéralement
« le mot *hoax* est dans `comments` ». Donner `comments` au modèle revient à
lui donner la réponse écrite noir sur blanc — le TF-IDF a juste appris à
repérer le mot « hoax », pas un vrai canular. Une fois ce champ retiré, la
forme, la durée et le pays n'apportent quasiment rien (taux de canular entre
0,8% et 2,7% selon le pays, entre 1,1% et 2,2% selon la forme — à peine
différent du taux de base à 0,9%). Le score de la phase 4 n'était pas un peu
gonflé, il venait entièrement de la fuite.

*(La chute est totale, pas juste une baisse — ça confirme que l'étiquette de
la phase 3 vient d'un texte écrit après coup par quelqu'un qui connaissait
déjà la réponse.)*

## Phase 6 : le modèle le plus bête du Bureau

Sur le même ensemble de test (22 170 relevés) :

| | Exactitude (accuracy) | Rappel |
|---|---|---|
| Stagiaire (toujours « pas un canular ») | **99,09 %** | 0,0 |
| Notre modèle (phase 5, sans fuite) | **99,08 %** | 0,0 |

**Ce qu'on présente au Conseil : le rappel (et la précision), pas
l'exactitude.**

Les canulars ne pèsent que 0,9% des relevés, donc répondre « non » à chaque
fois donne déjà une exactitude excellente sans jamais en attraper un seul —
rappel nul. L'exactitude récompense juste la classe majoritaire ici, elle ne
fait quasi aucune différence entre un système qui ne fait rien et un système
qui essaie vraiment. Le rappel et la précision, eux, disent ce qui compte
vraiment : combien de canulars sont attrapés, et combien des alertes sont
justes.

*Constat honnête : une fois la fuite retirée, notre modèle ne fait pas mieux
que le stagiaire. Ça ne veut pas dire que le canular est indétectable, juste
que ces 10 colonnes structurées ne suffisent pas — il faudrait plus de
signal (un vrai texte de témoignage sans les notes d'employé), ou une autre
source pour l'étiquette.*
