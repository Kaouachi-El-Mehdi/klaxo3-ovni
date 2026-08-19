# RAPPORT.md — Bureau d'Analyse Terrestre

Ce rapport suit `analyse.py` phase par phase. Tous les chiffres ci-dessous sont
réaffichés par le script quand on le relance d'une traite.

## Phase 1 : ouvrir la caisse

- Lignes dans le fichier : **88 875**
- Lignes chargées normalement (11 champs) : **88 679**
- Lignes mises à part (nombre de champs ≠ 11) : **196**


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

## Phase 7 : plusieurs témoins, un seul événement

Colonnes utilisées pour reconnaître un même événement : `datetime`, `city`,
`state`, `country`. Un groupe = tous les relevés qui partagent ces quatre
valeurs.

- Événements signalés par plus d'un témoin : **1 006**
- Témoins pour le plus gros d'entre eux : **19** (Tinley Park, IL, le
  31/10/2004 à 20h00 — la nuit citée par le conseiller à la cartographie)
- Relevés à cheval sur apprentissage/test dans la découpe d'hier (phase 4) :
  **934**

Nouvelle découpe (un événement entier d'un seul côté, `GroupShuffleSplit`) :

| | Rappel | Précision |
|---|---|---|
| Phase 4 (découpe aléatoire) | 94,0 | 100,0 |
| Phase 7 (découpe par événement) | **95,9** | 100,0 |

Le chiffre ne s'effondre pas — il monte même légèrement. Rien d'alarmant :
la fuite dominante reste celle de la phase 5 (le mot « hoax » recopié dans
`comments`), qui à elle seule sépare déjà presque parfaitement les deux
classes quel que soit le découpage. La fuite par événement dupliqué existe
bien (934 relevés étaient mal répartis hier) mais elle est secondaire face à
une fuite aussi grossière ; la variation de 94,0 à 95,9 est plus proche du
bruit d'échantillonnage que d'un effet de fuite corrigée.

Témoignages recopiés mot pour mot : **615 lignes sur 252 textes distincts**
(« Fireball », « UFO », « Lights in the sky... »). Seules 33 de ces 615
lignes appartiennent en plus à un événement à plusieurs témoins déjà repéré
plus haut. Traitement retenu : on les garde. Ce sont des commentaires
génériques très courts qui se répètent par coïncidence sur des événements
sans rapport, pas des copier-coller d'un même témoignage.

## Phase 8 : l'ordre des choses

Écart entre `date_posted` (mise en ligne) et `datetime` (observation) :
médiane à **27 jours**, mais 21,6 % des relevés (19 111) attendent plus d'un
an et 9 121 plus de dix ans.

**Choix : on découpe sur `datetime`.** `date_posted` accumule un retard trop
irrégulier — un dossier de 1998 peut être publié en 2013 — pour servir de
repère temporel fiable ; s'en servir mélangerait l'ordre réel des
événements que le système doit apprendre à généraliser.

- Date de coupure (75e percentile de `datetime`) : **2011-05-09**
- Relevés avant (apprentissage) : 66 509 — après (test) : 22 170
- Proportion de canulars avant : 0,91 % — après : 0,88 %

| | Rappel | Précision |
|---|---|---|
| Phase 4 (découpe aléatoire) | 94,0 | 100,0 |
| Phase 8 (découpe temporelle) | 94,3 | 100,0 |

Bonus : **0 événement de la phase 7 n'est à cheval** sur cette coupure
temporelle — un événement se déroule en une seule nuit, la coupure par date
respecte donc naturellement le regroupement de la phase 7.

## Phase 9 : les cases vides

| Colonne | Trous | % canulars si trou | % canulars si rempli |
|---|---|---|---|
| `country` | 12 365 | 1,16 % | 0,86 % |
| `state` | 7 409 | 1,30 % | 0,87 % |
| `shape` | 2 922 | 1,16 % | 0,90 % |

Les trois colonnes montrent le même léger effet : un relevé troué a un peu
plus de risque d'être un canular qu'un relevé complet — cohérent avec l'idée
qu'un signalement bâclé (peu de détails, pas de forme, pas de pays) est
aussi plus susceptible d'être suspect.

**Traitement retenu :** remplacer chaque trou par une catégorie explicite
`inconnu`, pas par la valeur la plus fréquente. Dans l'encodage one-hot,
`inconnu` reste un niveau à part entière que le modèle peut utiliser — le
trou est bouché sans que sa trace soit effacée.

## Phase 10 : la chaîne de traitement du Bureau

Refonte en `sklearn.pipeline.Pipeline` : un `ColumnTransformer` (médiane et
encodeur one-hot) enchaîné avec la forêt, ajusté **uniquement** sur la
partie apprentissage — plus aucun calcul n'est fait sur le fichier entier
avant la découpe.

- Proportion de canulars — apprentissage : 0,91 % — test : 0,88 %
- Rappel / précision (sur les colonnes encore disponibles à ce stade,
  sans `comments`) : **0,0 / 0,0**

Ce chiffre ne surprend pas : c'est la même conclusion qu'en phase 5-6, avec
un pipeline correctement étanche cette fois. La démonstration demandée par
le Conseil :

```
hour=0  weekday=0  month=10  duration_seconds=7200.0
latitude=42.73  longitude=-73.69  shape=triangle  country=us  state=ny
-> prediction : pas un canular (probabilité estimée : 0.020)
```

Un relevé entre d'un côté, une prédiction sort de l'autre, en un seul appel
`pipeline.predict(...)`, sans retaper la moindre étape à la main.

## Phase 11 : combien de temps ça a duré

`duration_seconds` (numérique) et `duration_hours_min` (texte libre du
témoin, parsé par une petite bibliothèque de motifs — unités, fourchettes,
fractions, quantités vagues comme « several minutes ») sont fusionnées :
`duration_seconds` prime quand elle est positive, sinon on retombe sur le
texte.

- Relevés dont la durée reste inutilisable après traitement : **6 232 /
  88 679**
- Relevés où les deux colonnes se contredisent (facteur > 3) : **2 149**
- Durée médiane : **180 s (~3 min)**
- Relevés annonçant plus d'une journée d'observation : **261**

Exemple de désaccord entre les deux colonnes : `duration_seconds = 0` mais
`duration_hours_min = '12'` — durée récupérée depuis le texte : 720 s.
Exactement le cas décrit par le service scientifique.

Les 3 durées les plus longues du fichier :

| Durée | ~Années | `duration_seconds` | `duration_hours_min` |
|---|---|---|---|
| 97 836 000 s | 3,1 | 97836000.0 | « 31 years » |
| 82 800 000 s | 2,6 | 82800000.0 | « 23000hrs » |
| 66 276 000 s | 2,1 | 66276000.0 | « 21 years » |

Ces trois lignes illustrent le problème à elles seules : le témoin qui
annonce « 31 years » ou « 21 years » se retrouve avec une valeur en secondes
qui ne représente qu'une poignée d'années dans le fichier — encore un signe
que `duration_seconds` ne peut pas être prise telle quelle sur les valeurs
extrêmes.

**Décision :** la durée utilisée par le modèle est plafonnée à 1 jour
(86 400 s), avec un indicateur booléen `duree_extreme` qui garde la trace
des relevés au-dessus. Une observation de plusieurs années n'est plus un
« relevé » ponctuel mais un phénomène récurrent mal capturé par une colonne
en secondes — la laisser telle quelle écraserait toute médiane. Aucune
ligne n'est supprimée.

## Phase 12 : la ville et l'heure

**Ville :** 22 018 villes distinctes, dont **14 177 qui n'apparaissent
qu'une seule fois**. Règle retenue : one-hot avec seuil de fréquence
minimale (`min_frequency=15`) — toute ville vue moins de 15 fois est
regroupée dans une catégorie « rare » au lieu de recevoir sa propre colonne.

**Heure :** encodage cyclique (`sin`/`cos` sur 24h) plutôt qu'un simple
entier 0-23.

- Distance encodée entre 23h et 0h : **0,261**
- Distance encodée entre 23h et 20h : **0,765**

23h ressort bien plus proche de 0h que de 20h : le cercle horaire est
respecté.

**Forme (`shape`) :** 29 formes distinctes avant traitement. Deux fusions
appliquées : `changed` → `changing` (même mot, deux conjugaisons) et
`round` → `circle` (la même forme décrite deux fois). **27 formes**
restantes.

**Largeur du tableau :**

| | Colonnes |
|---|---|
| Avant (ville en one-hot naïf) | 19 268 |
| Après (ville regroupée, seuil de fréquence) | **847** |

## Phase 13 : la facture du Bureau

Grille votée par le Conseil : canular manqué = 30 crédits, fausse alerte =
2 crédits, bonne réponse = 0.

| Frontière | Facture |
|---|---|
| 0,01 | 16 528 |
| 0,11 | 5 970 |
| 0,21 | 5 840 |
| 0,31 | 5 824 |
| 0,41 | 5 822 |
| 0,51 – 0,91 | 5 820 (plate) |

- Frontière retenue (minimise la facture) : **0,45**
- Facture à 0,5 : **5 820 crédits** — facture à 0,45 : **5 820 crédits**
- Écart : **0 crédit économisé**

Aucune frontière ne bat 0,5 ici, et ce n'est pas un bug : c'est le même
constat que la phase 6, reformulé en crédits. Le modèle final (sans
`comments`, construit avec les colonnes propres des phases 9 à 12) ne
sépare quasiment pas les deux classes — sa probabilité maximale sur toute la
partie test plafonne à 0,44, en dessous de n'importe quelle frontière
raisonnable (voir phase 16). Faire bouger la frontière ne peut pas
compenser l'absence de signal ; le raisonnement en coûts reste correct, il
n'a simplement rien à optimiser sur ce jeu de colonnes.

## Phase 14 : une promesse à 80 %

| Tranche (probabilité annoncée) | n | Proba. moyenne annoncée | Proportion réelle |
|---|---|---|---|
| ]−0,001 ; 0,00487] | 10 347 | 0,000 | 0,007 |
| ]0,00487 ; 0,005] | 5 080 | 0,005 | 0,008 |
| ]0,005 ; 0,01] | 2 568 | 0,010 | 0,010 |
| ]0,01 ; 0,015] | 1 394 | 0,015 | 0,016 |
| ]0,015 ; 0,025] | 1 424 | 0,022 | 0,011 |
| ]0,025 ; 0,44] | 1 357 | 0,053 | 0,015 |

Le système est **trop confiant** : dans la dernière tranche il annonce en
moyenne 5,3 % de chances de canular pour une réalité à 1,5 %.

Correction : calibration isotonique (`CalibratedClassifierCV`, ajustée par
validation croisée sur l'apprentissage uniquement). Même tableau après
correction :

| Tranche | n | Proba. moyenne annoncée | Proportion réelle |
|---|---|---|---|
| ]0,00615 ; 0,00786] | 8 488 | 0,007 | 0,007 |
| ]0,00786 ; 0,00858] | 3 506 | 0,008 | 0,006 |
| ]0,00858 ; 0,00913] | 2 796 | 0,009 | 0,010 |
| ]0,00913 ; 0,00952] | 1 920 | 0,009 | 0,012 |
| ]0,00952 ; 0,011] | 2 736 | 0,010 | 0,009 |
| ]0,011 ; 0,198] | 2 724 | 0,013 | 0,015 |

Les deux colonnes se resserrent nettement — la calibration ne rend pas le
modèle plus discriminant (il n'a toujours pas plus de signal), mais le
chiffre qu'il annonce devient honnête.

## Phase 15 : deux analystes, deux chiffres

- Taille de la partie test : **22 170**
- Canulars réellement présents dedans : **194**
- Rappel de notre système à la frontière retenue : **0,00**
- Intervalle à 95 % (1 000 tirages bootstrap de la partie test) :
  **[0,00 ; 0,00]**

**Réponse au Conseil sur les deux analystes :** avec seulement 194 canulars
dans la partie test, deux chiffres qui ne diffèrent que de quelques
centièmes (0,31 contre 0,34) ne prouvent rien. Ici notre propre rappel ne
bouge même pas d'un tirage bootstrap à l'autre — il est nul dans les deux
cas — mais dès qu'un système commence à attraper quelques dizaines de
canulars, une poignée de relevés déplacés entre apprentissage et test suffit
à faire bouger le chiffre de plusieurs points sans que le modèle ait changé.
Le nombre de canulars dans la partie test explique à lui seul la largeur de
la fourchette.

## Phase 16 : trois dossiers sur le bureau

Importance par permutation (chute de l'aire sous la courbe précision-rappel
— le rappel/la précision au seuil retenu restent à 0 presque partout, un
score basé sur `predict_proba` est nécessaire pour voir quoi que ce soit
bouger) :

| Colonne | Effet |
|---|---|
| `shape` | +0,0012 |
| `longitude` | +0,0005 |
| `duration_seconds` | +0,0004 |
| `country` | +0,0004 |
| `duree_extreme` | +0,0002 |
| `hour_cos` | +0,0001 |
| `city` | −0,0001 |
| `state` | −0,0003 |
| `latitude` | −0,0003 |
| `hour_sin` | −0,0004 |
| `weekday` | −0,0005 |
| `month` | −0,0017 |

Colonne dont la place surprend : **`shape`** arrive en tête — pas la
localisation ni le moment, mais la description de la forme rapportée par le
témoin, alors que ces mêmes formes se répartissaient presque comme le taux
de base en phase 5.

Aucun relevé de la partie test ne franchit la frontière de 0,45 : même le
plus suspect ne plafonne qu'à 0,440. Les deux premiers dossiers restent les
deux relevés les plus suspects du lot, en toute honnêteté.

**Dossier 1 — le relevé jugé le plus suspect** (probabilité 0,440, en
réalité pas un canular) : forme `fireball`, Orlando (FL), mars, vendredi,
durée 900 s.
Effet estimé des colonnes : `shape` +0,235, `duration_seconds` +0,115,
`longitude` +0,085.

**Dossier 2 — juste en dessous de la frontière** (probabilité 0,345, en
réalité pas un canular) : forme `light`, Elko (NV), juillet, mardi, durée
3 600 s.
Effet estimé : `longitude` +0,085, le reste quasi nul.

**Dossier 3 — canular laissé passer** (probabilité 0,145, en réalité un
canular) : New York City (GW Bridge), forme non renseignée, mois de
janvier.
Effet estimé : `shape` +0,135 (le fait que la forme soit vide pèse), `country`
+0,095 (pays vide aussi).

Le classement global (« ce qui compte en moyenne ») n'explique pas un
dossier précis : dans le dossier 2, `shape` — première colonne du classement
global — a un effet quasi nul, alors que `longitude` domine.

## Phase 17 : l'angle mort du Bureau

| Zone | n | % canulars | Rappel | Précision |
|---|---|---|---|---|
| us | 18 885 | 0,76 % | 0,00 | 0,00 |
| ca | 674 | 1,34 % | 0,00 | 0,00 |
| gb | 194 | 4,12 % | 0,00 | 0,00 |
| autre (au + de) | 93 | 3,23 % | 0,00 | 0,00 |
| inconnu (pays vide) | 2 324 | 1,33 % | 0,00 | 0,00 |

Proportion globale de référence : 0,88 %. Le taux de canulars n'est pas du
tout le même d'une zone à l'autre (0,76 % aux États-Unis contre 4,12 % au
Royaume-Uni) — un écart que la moyenne globale masque complètement, comme
prévu.

**Décision : une seule frontière pour toutes les zones.** La phase 15 a
montré que le rappel bouge déjà de plusieurs centièmes rien qu'en rejouant
le même test sur la partie us (des dizaines de milliers de relevés) ; une
zone comme `gb` ou `ca` ne contient que quelques dizaines de canulars dans
le test, largement trop peu pour régler une frontière spécifique sans se
caler sur du bruit.

## Phase 18 : la transmission d'archive

Proportion de canulars par année (années avec au moins 100 relevés,
1964-2014) : voir `klaxo3_hoax_par_annee.png`, généré par `analyse.py`. La
courbe n'est pas plate : elle reste sous 1 % presque tout le temps entre
1964 et 2004, puis grimpe nettement — 1,51 % en 2006, 2,12 % en 2007,
**2,64 % en 2008** — avant de redescendre partiellement ensuite (0,57 % en
2012, 0,59 % en 2013). Ça correspond à ce que le Conseil craignait : la
façon de noter un canular a changé au milieu des années 2000, probablement
au moment où le Bureau a pris l'habitude d'ajouter des notes du type
`((HOAX??))` que la phase 3 avait déjà repérées.

**Épreuve (entraînement sur l'ancien, test sur le récent, coupure à 50 %
plutôt que 75 %)** — coupure au 2006-09-21, avec le modèle « phase 4 »
(comments compris), pour rester comparable à la phase 8 :

| | Rappel | Précision |
|---|---|---|
| Phase 8 (coupure à 75 %, 2011-05-09) | 94,3 | 100,0 |
| Phase 18 (coupure à 50 %, 2006-09-21) | **70,0** | 100,0 |

Le rappel chute nettement quand on force le modèle à apprendre uniquement
sur la période d'avant 2007 et à se prononcer sur l'après — exactement la
période où, d'après la courbe ci-dessus, les habitudes d'annotation ont
changé. Le modèle a appris une définition du canular qui commence déjà à
dater.

**Indicateurs de surveillance en production** (aucun n'a besoin de
connaître la réponse) :

1. **Taux d'alerte** — proportion de relevés classés canular par semaine.
   Un écart fort avec la moyenne historique (± 50 % relatif) signale que le
   flux entrant a changé de nature.
2. **Distribution des probabilités prédites** — comparer chaque semaine les
   scores produits à ceux de la période d'entraînement (test de
   Kolmogorov-Smirnov).

Fréquence : chaque semaine. Seuil de rappel des analystes : écart de plus de
50 % du taux d'alerte historique, ou p < 0,01 au test KS, sur deux semaines
consécutives.

## Le rapport final

`analyse.py` tourne toujours d'une traite, du téléchargement au dernier
chiffre, sur une machine neuve et un dossier vide. `pip install -r
requirements.txt` installe la seule dépendance ajoutée depuis la partie 1
(`matplotlib`, pour le graphique de la phase 18). Tous les résultats sont
figés par `RANDOM_STATE = 42` : ils ne bougent pas d'un lancement à
l'autre.
