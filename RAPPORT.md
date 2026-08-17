# RAPPORT.md — Bureau d'Analyse Terrestre

Ce rapport suit `analyse.py` phase par phase. Tous les chiffres ci-dessous sont
réaffichés par le script quand on le relance d'une traite.

## Phase 1 : ouvrir la caisse

- Lignes dans le fichier : **88 875**
- Lignes chargées normalement (11 champs) : **88 679**
- Lignes mises à part (nombre de champs ≠ 11) : **196**

88 679 + 196 = 88 875, les trois nombres sont cohérents : aucune ligne n'a
disparu, on l'a juste rangée dans un tas à part le temps de comprendre ce
qu'elle contenait.

Les 196 lignes mises de côté ont toutes **12 champs** au lieu de 11. Exemple
concret :

```
10/1/2006 12:00,,,,,0,,,"((EDITORIAL COMMENT ABOUT THE UFO PHENOMEN))  ufo+alien+reptiles",10/30/2006,0,0
```

Ce qui cloche : entre `duration_seconds` et le commentaire, il y a un champ
vide en trop (`duration_hours_min` **et** `comments` sont vides, puis le texte
qu'on attendait dans `comments` arrive en 9ᵉ position au lieu de la 8ᵉ, ce qui
décale `date_posted`, `latitude` et `longitude` d'une case). Ces 196 lignes
n'ont jamais eu de `shape` renseignée, on soupçonne un bug du service qui a
compilé le CSV plutôt qu'une erreur du témoin. On les a chargées telles
quelles (en mémoire, dans une liste à part) mais on ne les fait pas entrer
dans le DataFrame de travail des phases suivantes : les décaler à la main
aurait été un choix arbitraire de plus, alors qu'elles pèsent 0,22% du total.

## Phase 2 : rien n'est du bon type

Anomalies rencontrées lors de la conversion des 88 679 lignes propres :

| Champ | Type visé | Échecs | Exemple de valeur fautive |
|---|---|---|---|
| `latitude` | nombre | **1** | `33q.200088` |
| `longitude` | nombre | 0 | — |
| `duration_seconds` | nombre | **5** | `2\``, `8\``, `0.5\``, deux valeurs vides |
| `datetime` | date | **1220** | `10/10/2005 24:00` (toujours `24:00`) |
| `date_posted` | date | 0 | — |

Quatre anomalies de nature différente, avec leur origine :

1. **`latitude` — 1 valeur (`33q.200088`)** : une lettre `q` insérée au milieu
   d'un nombre. Cette seule ligne sur 88 679 suffit à faire lire toute la
   colonne comme du texte si on ne force pas la conversion — origine :
   **service de transmission** (le pipeline de géocodage qui a produit le
   CSV).
2. **`duration_seconds` — 5 valeurs (ex. `` `2` ``, `` `8` ``)** : un
   guillemet simple mal encodé en un backtick. En regardant
   `duration_hours_min` en face (`"each a few seconds"`, `"1/3200"`,
   `"1/2 segundo"`), on voit que ce sont des durées écrites librement par
   le témoin puis mal réencodées — origine : **témoin**.
3. **`datetime` — 1220 valeurs finissant par `24:00`** : une heure `24:00`
   n'existe pas (minuit s'écrit `00:00` le jour suivant). Le motif est
   systématique et touche des dates très variées : c'est la trace d'un bug du
   script de « time-standardization » qui a produit ce fichier — origine :
   **service de transmission**.
4. **`shape` vide — 2922 lignes (3,30%)** : le témoin n'a pas su ou pas voulu
   décrire la forme de ce qu'il a vu — origine : **témoin**.

Bonus notés au passage (mêmes causes probables) : `country` vide sur 12 365
lignes (13,94%) et `state` vide sur 7409 lignes (8,35%) — le témoin n'a pas
précisé une localisation assez fine pour que le géocodage la résolve.

Aucune ligne n'a été supprimée à cette étape : les valeurs qui résistent à la
conversion deviennent `NaN` / `NaT` dans leur propre colonne, la ligne reste.

## Phase 3 : le Conseil veut trier les canulars

**Règle (une phrase) :** un relevé est marqué canular si son champ `comments`
contient le mot « hoax » (recherche insensible à la casse).

- Relevés marqués canulars : **802 / 88 679 (0,90 %)**

**Ce que la règle attrape à tort :** 674 des 802 « hoax » trouvés viennent en
réalité d'une note `((HOAX??))` ou `((NUFORC Note: Possible hoax?? PD))`
ajoutée par un employé NUFORC qui n'était *pas sûr* — la règle les compte
comme des canulars certains alors que le point d'interrogation dit le
contraire.

**Ce que la règle rate :** 65 commentaires contiennent « fake », « prank »,
« made up » ou « joke » sans jamais écrire le mot « hoax » — ces cas
échappent totalement à la règle.

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

Ces deux nombres sont calculés uniquement sur l'ensemble de test (les 22 170
relevés mis de côté avant l'entraînement, jamais montrés au modèle).

Ce score est presque trop beau. C'est justement le sujet de la phase
suivante.

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

**Pourquoi l'écart, en trois lignes :** notre étiquette de canular est
littéralement « le mot *hoax* apparaît dans `comments` ». Donner `comments`
au modèle, c'est lui donner l'étiquette elle-même écrite en toutes lettres —
le TF-IDF a simplement appris à repérer le mot « hoax », pas à repérer un
canular. Une fois ce texte retiré, on a vérifié que la forme, la durée et le
pays ne portent quasiment aucun signal réel sur cette étiquette (taux de
canular entre 0,8% et 2,7% selon le pays, entre 1,1% et 2,2% selon la forme —
à peine différent du taux de base à 0,9%) : le score de la phase 4
n'était donc pas « un peu gonflé », il était **entièrement** dû à la fuite.

*(Nos deux résultats ne sont pas identiques — la chute est même totale, ce
qui confirme que la phase 3 tire son étiquette d'un texte écrit après coup
par un employé qui connaissait déjà la réponse.)*

## Phase 6 : le modèle le plus bête du Bureau

Sur le même ensemble de test (22 170 relevés) :

| | Exactitude (accuracy) | Rappel |
|---|---|---|
| Stagiaire (toujours « pas un canular ») | **99,09 %** | 0,0 |
| Notre modèle (phase 5, sans fuite) | **99,08 %** | 0,0 |

**Mesure présentée au Conseil : le rappel (et la précision), pas
l'exactitude.**

Les canulars ne représentent que 0,9% des relevés. Répondre systématiquement
« non » obtient donc une exactitude excellente sans jamais attraper un seul
canular : son rappel est nul. L'exactitude ne fait ici que récompenser la
classe majoritaire, elle est quasiment identique pour un système qui ne fait
rien et pour un système qui essaie vraiment. Le rappel et la précision, eux,
mesurent directement ce que le Conseil demande : combien de canulars sont
attrapés, et combien des alertes sont fondées. C'est sur ces deux nombres,
pas sur l'exactitude, qu'un système doit être jugé ici.

*Constat honnête à ce stade : une fois la fuite retirée, notre modèle ne fait
pas mieux que le stagiaire. Cela ne veut pas dire que le canular est
indétectable, mais que les 10 colonnes structurées de cette transmission ne
suffisent pas à le détecter — il faudrait soit plus de signal (un vrai texte
de témoignage nettoyé de toute note d'employé), soit une autre source de
vérité pour l'étiquette.*
