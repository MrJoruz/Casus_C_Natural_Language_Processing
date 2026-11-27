# Plan van Aanpak – Casus Natural Language Processing

## Aanleiding en doel

In vier lessen bouwen we zelf een kleine NLP-pijplijn: tokenisatie met byte-pair encoding, bag-of-words classificatie, n-gram tekstgeneratie en CBOW-embeddings. Elke les levert een command-line script op dat voortbouwt op één gedeelde module `nlp.py`, plus een notebook dat het script toepast op kankergerelateerde teksten.

Doelen:

- Begrijpen hoe tekst wordt omgezet in getallen (tokens, vectoren, embeddings) en welke keuzes daarbij het resultaat bepalen.
- De kernalgoritmen zelf implementeren met alleen de Python-standaardbibliotheek (geen wrappers om externe modules); in de notebooks mag alles.
- Resultaten kritisch interpreteren en vergelijken met bestaande implementaties (`keras_hub`, `sklearn`, `nltk`, TensorFlow Projector).

Uitgangspunt is dat de lessen op elkaar voortbouwen: de `.tok`-bestanden uit les 1 zijn de invoer voor les 3 en 4, en de tokenizer wordt hergebruikt in les 2.

## Op te leveren producten

Alles wordt bijgehouden in één gezamenlijke Git-repository die na afloop via BrightSpace wordt ingeleverd.

| Les | Script | Bestanden in / uit | Notebook (vrije tools) |
| --- | --- | --- | --- |
| 1 | `tokenize` | `.txt` → `.enc`; `.txt` + `.enc` → `.tok`; `.tok` + `.enc` → `.txt` | `les1.ipynb` – Wikipedia NL/EN, 4 tokenbestanden, vergelijking `keras_hub` |
| 2 | `bagofwords` | één of meer `.txt` → `.bow` (multi-hot, frequentie, TF-IDF) | `les2.ipynb` – PubMed-corpora, Naive Bayes, vergelijking `CountVectorizer`/`TfidfVectorizer` |
| 3 | `ngram` | één of meer `.tok` → gegenereerde `.tok` | `les3.ipynb` – Bulkley I+II, 1-/2-/3-grams, vergelijking `nltk.lm.MLE` |
| 4 | `embed` | `.tok` (+ `.enc`) → `.emb` | `les4.ipynb` – gecombineerd corpus, PCA-plot, buren, TF Projector |
| alle | `nlp.py` | gedeelde klassen en functies | – |

Elk script toont bondige hulp bij `-h` / `--help` en importeert de algemene functionaliteit uit `nlp.py`.

## Architectuur en ontwerpkeuzes

Eén module `nlp.py` bevat alle herbruikbare logica; de scripts zijn dunne CLI-lagen met `argparse` eromheen. Zo kunnen de notebooks dezelfde code direct importeren.

### Repositorystructuur

```
nlp-casus/
├── nlp.py            # gedeelde module
├── tokenize          # les 1 (uitvoerbaar, #!/usr/bin/env python3)
├── bagofwords        # les 2
├── ngram             # les 3
├── embed             # les 4
├── les1.ipynb … les4.ipynb
├── data/             # ruwe teksten (.txt), klein genoeg om mee te leveren
├── output/           # .enc, .tok, .bow, .emb (evt. in .gitignore)
├── tests/            # unittests voor nlp.py
└── README.md
```

Let op: een script met de naam `tokenize` kan botsen met de standaardmodule `tokenize` van Python als het met `.py`-extensie in dezelfde map staat. Daarom zonder extensie en met shebang, of de logica volledig in `nlp.py`.

### Inhoud van nlp.py

| Onderdeel | Verantwoordelijkheid | Gebruikt in les |
| --- | --- | --- |
| `pretokenize(text, lowercase, keep_punct)` | tekst splitsen in woorden en interpunctie, optioneel lowercasing | 1, 2 |
| `BPETokenizer` | `fit(text, max_tokens, min_freq)`, `encode(text)`, `decode(tokens)`, `save`/`load` (.enc) | 1–4 |
| `read_tokens` / `write_tokens` | lezen en schrijven van .tok | 1, 3, 4 |
| `BagOfWords` | vocabulaire, `multi_hot`, `frequency`, `tfidf` over meerdere corpora | 2 |
| `NGramModel` | tellen van (n−1)-context → volgend token, `sample(length)`, back-off | 3 |
| `cbow_dataset(tokens, n)` | contextvensters als multi-hot invoer + middelste token als label | 4 |
| `read_embedding` / `write_embedding` | lezen en schrijven van .emb | 4 |

### Bestandsformaten

Alle formaten zijn tekstgebaseerd, zodat ze leesbaar en met Git te vergelijken zijn.

| Extensie | Formaat | Inhoud |
| --- | --- | --- |
| `.enc` | JSON | instellingen (lowercase, interpunctie), basisalfabet, geordende lijst merges, mapping token-id → tekst |
| `.tok` | platte tekst | token-id's gescheiden door spaties, één regel per document of alinea |
| `.bow` | TSV | kopregel met token-id's, daarna één rij getallen per document (eventueel sparse: `id:waarde`) |
| `.emb` | TSV | per regel: token-tekst, gevolgd door de vectorwaarden (direct bruikbaar voor de TF Projector) |

### Keuzes bij tokenisatie

- Pre-tokenisatie op woordgrenzen; merges gebeuren nooit over spaties of interpunctie heen.
- Spatie wordt als prefix aan het woord gekoppeld (zoals GPT: " kanker"), zodat decoderen verliesvrij is.
- Hoofdletters: standaard behouden, met optie `--lowercase`.
- Stopcriteria: `--vocab-size` (maximaal aantal tokens) en `--min-freq` (minimale paarfrequentie); welke het eerst bereikt wordt, telt.

## Aanpak per les

### Les 1 – Tokenisatie (BPE)

Opdracht:

1. Pre-tokenizer schrijven en BPE-training implementeren: paren tellen, vaakste paar samenvoegen, herhalen tot `--vocab-size` of `--min-freq` bereikt is.
2. `encode` (merges in volgorde toepassen) en `decode` implementeren; onbekende tekens afvangen met een `<unk>`-token of byte-fallback.
3. CLI met subcommando's `train`, `encode`, `decode` en de combinatie `train --encode`.
4. Round-trip testen: `decode(encode(tekst)) == tekst`.

Toepassing (`les1.ipynb`):

1. Wikipedia-pagina's Kanker (NL) en Cancer (EN) ophalen met `requests` + BeautifulSoup en opschonen (referenties, navigatie).
2. Per taal een encoding trainen met hetzelfde aantal tokens; beide encodings op beide teksten toepassen (4 `.tok`-bestanden).
3. Aantal tokens per combinatie in een tabel; verwachting: een tekst kost de minste tokens met de encoding van de eigen taal. Ook het effect van de vocabulairegrootte bekijken.
4. Optioneel: vergelijken met de `byte_pair_tokenizer` van `keras_hub`.

### Les 2 – Bag-of-words classificatie

Opdracht:

1. Corpus inlezen met één document per regel; één BPE-encoding trainen over alle opgegeven bestanden samen.
2. Encodings implementeren: multi-hot, telling/frequentie, en TF-IDF volgens de gegeven formule (tf = nₜ,d / n\_d, idf = ln(N / Nₜ)), met N en Nₜ over alle bestanden samen.
3. Per invoerbestand een eigen `.bow` wegschrijven, met één gedeelde vocabulaire.

Toepassing (`les2.ipynb`):

1. Uit `/commons/data/NCBI/PubMed` titels + abstracts halen; publicaties zonder abstract overslaan (`KeyError`). Indelen via keywords/MeSH of zoektermen in de titel: kanker vs. niet-kanker (eventueel meer klassen).
2. Grootte: bv. 1000–3000 abstracts per klasse, ongeveer in balans.
3. `.bow`-bestanden genereren voor alle drie encodings; labelen, husselen, train/test-split (bv. 80/20, vaste seed).
4. `MultinomialNB` (en eventueel `BernoulliNB` voor multi-hot, of `LogisticRegression`) trainen; accuracy en confusion matrix per encoding.
5. Analyse: welke tokens wegen het zwaarst? Komen de zoektermen voor, en wat gebeurt er met de accuracy als we ze verwijderen?
6. Optioneel: vergelijken met `CountVectorizer` en `TfidfVectorizer`.

### Les 3 – N-gram tekstgeneratie

Opdracht:

1. Uit één of meer `.tok`-bestanden alle n-grams tellen: dictionary van (n−1)-context → teller van volgende tokens.
2. Sampling naar kansverdeling, startend vanaf een willekeurige context uit de trainingsdata (`--seed` voor reproduceerbaarheid).
3. Onbekende context oplossen met back-off: terugvallen op een kortere context, en uiteindelijk op unigramfrequenties.
4. Resultaat van `--length` tokens wegschrijven naar `.tok`.

Toepassing (`les3.ipynb`):

1. Bulkley vol. I en II van Project Gutenberg downloaden, Gutenberg-header en -footer strippen, samenvoegen.
2. Tokeniseren met verschillende vocabulairegroottes (bv. 500, 2000, 8000 tokens).
3. Tekst genereren met n = 1, 2, 3 (en eventueel hoger) en terug decoderen.
4. Beoordelen: leesbaarheid neemt toe met n, maar bij grote n en grote vocabulaire gaat het model letterlijk kopiëren (overfitting). Dit kwantificeren met het aandeel gegenereerde n-grams dat ook in de brontekst staat.
5. Optioneel: vergelijken met `nltk.lm.MLE.generate`.

### Les 4 – Embeddings (CBOW)

Opdracht:

1. Uit de tokenreeks vensters maken van 2n+1 tokens: de n linker en n rechter buren als multi-hot invoer, het middelste token als label.
2. `MLPClassifier` trainen met één verborgen laag: `hidden_layer_sizes=(N,)` en `activation='identity'`, zodat de verborgen laag een lineaire projectie is. Parameters `--window` en `--hidden` via de CLI.
3. Embedding per token = de kolom bij dat token in `coefs_[1]` (verborgen → uitvoer); eventueel ook `coefs_[0]` vergelijken. Let op: sklearn noemt het attribuut `coefs_`, niet `coef_`.
4. Met de `.enc` de token-id's terugvertalen naar tekst en wegschrijven als `.emb`.

Toepassing (`les4.ipynb`):

1. Engelse Wikipedia + kanker-abstracts (les 2) + Bulkley (les 3) samenvoegen tot één `.txt`, tokeniseren, embedding trainen.
2. Experimenteren met venstergrootte, aantal hidden neurons en vocabulairegrootte; rekentijd in de gaten houden (eventueel subsample of `max_iter` beperken).
3. PCA naar 2D en scatterplot met labels voor de vaakste tokens.
4. Per token de dichtstbijzijnde buur zoeken (cosinusafstand) en interpreteren: ontstaan er zinvolle paren (bv. tumor/tumour, verschillende orgaannamen)?
5. Optioneel: `.emb` exporteren naar de TensorFlow Embedding Projector (vectors.tsv + metadata.tsv).

## Planning en mijlpalen

Aanname: één week per les plus een halve week afronding. Pas de weken aan op het rooster en de inleverdatum op BrightSpace.

| Week | Focus | Activiteiten | Mijlpaal (af aan eind van de week) |
| --- | --- | --- | --- |
| 1 | Les 1 | repo opzetten, `nlp.py`-skelet, BPE train/encode/decode, Wikipedia ophalen | `tokenize` werkt round-trip; `les1.ipynb` met 4 `.tok`-bestanden en tokentelling |
| 2 | Les 2 | PubMed-extractie (vroeg starten i.v.m. grootte), `BagOfWords`, classificatie | `bagofwords` met 3 encodings; `les2.ipynb` met accuracy per encoding |
| 3 | Les 3 | `NGramModel` + back-off, Gutenberg-teksten, experimenten met n en vocab | `ngram` genereert tekst; `les3.ipynb` met voorbeelden en analyse |
| 4 | Les 4 | CBOW-dataset, `MLPClassifier`, embedding extraheren, PCA en buren | `embed` schrijft `.emb`; `les4.ipynb` met plot en nearest neighbours |
| 5 (½) | Afronding | optionele vergelijkingen, refactoren `nlp.py`, README, notebooks opnieuw draaien | eindversie getagd en ingeleverd via BrightSpace |

Kritisch pad: de BPE-tokenizer uit les 1. Les 2, 3 en 4 hangen ervan af, dus die moet als eerste stabiel en snel genoeg zijn. Het verzamelen van data (PubMed, Gutenberg) kan parallel aan het programmeren.

## Werkwijze en taakverdeling

- **Git-workflow:** `main` bevat altijd werkende code; per les een feature-branch, merge na review door een teamgenoot.
- **Verdeling per les:** één persoon schrijft het script en de `nlp.py`-onderdelen, de ander de dataverzameling en het notebook. Wissel per les, zodat iedereen alle onderdelen kent.
- **Afspraken over `nlp.py`:** interfaces (functienamen, parameters, bestandsformaten) worden aan het begin van elke les samen vastgelegd; docstrings en type hints verplicht.
- **Overleg:** kort moment aan het begin van elke lesdag (wat is af, wat blokkeert), en een issue-lijst in de repository voor open taken.

Nog in te vullen: teamleden en wie welke les trekt.

## Risico's en maatregelen

| Risico | Gevolg | Maatregel |
| --- | --- | --- |
| BPE-training in pure Python is traag op grote teksten | les 2–4 lopen vast op rekentijd | woordfrequenties tellen en merges per uniek woord toepassen i.p.v. op de hele tekst; paartellingen incrementeel bijwerken |
| PubMed-pickles zijn groot en onregelmatig gestructureerd | geheugenproblemen, `KeyError`s | één of twee bestanden gebruiken, defensief uitlezen met `try/except`, subset opslaan als `.txt` |
| Classificatie is triviaal door de zoektermen | accuracy van \~100% zegt weinig | experiment met de zoektermen uit de tekst verwijderd |
| N-gram model kopieert de brontekst letterlijk | lijkt "geloofwaardig" maar is geen generatie | overlap met de bron meten en rapporteren |
| `MLPClassifier` met grote vocabulaire is traag of convergeert niet | geen bruikbare embedding | kleinere vocabulaire, subsample, `max_iter` en `early_stopping` instellen |
| Naamconflict van script `tokenize` met de Python-standaardmodule | onverklaarbare importfouten | script zonder `.py`-extensie, logica in `nlp.py` |
| Uitval of ongelijke werkverdeling in het team | achterstand op het kritisch pad | wekelijkse mijlpalen, rotatie van taken, vroeg signaleren |

## Kwaliteitsbewaking (definition of done)

Een les is af als:

- [ ] het script alle gevraagde functionaliteit heeft en alleen de standaardbibliotheek gebruikt (plus `sklearn` waar de opdracht dat voorschrijft);
- [ ] `script -h` bondige, correcte hulp toont;
- [ ] de algemene logica in `nlp.py` staat, met docstrings en unittests (bv. BPE round-trip, TF-IDF op een klein handmatig narekenbaar voorbeeld);
- [ ] het notebook van begin tot eind draait op een schone kernel en de onderzoeksvragen beantwoordt met een korte, kritische interpretatie;
- [ ] de code gemerged is naar `main` na review door een teamgenoot.

Voor de eindinlevering: README met installatie- en gebruiksinstructies, alle notebooks opnieuw uitgevoerd, en repository-export geüpload naar BrightSpace.
