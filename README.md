# DataLit-InsideAirbnb

Inside Airbnb

TODO:
Forschungsfrage formulieren
Data Preprocessing
Feature Engineering
Model Selection
Explainability Methods
Visualisation of results


$\textbf{Forschungsfrage:}$

- Preis pro Nacht pro Person
- Chance of reservation in gewissem Zeitintervall
- Einfluss verschiedener Faktoren auf Preisgestaltung
- Eingrenzung auf wenige Städte
- Unterschiede bzgl. Wichtigkeit verschiedener Features im Städtevergleich / Jahreszeitvergleich




Data Preprocessing:
Auswahl der Tabellen für Vorhersage (Links für Tabellen zu Berlin; Data Documentation)
listings.csv.gz 
calendar.csv.gz
reviews.csv.gz
Neighbourhoods.csv
Neighbourhoods.geojson
Auswahl der Features in einzelnen Tabellen für Vorhersage
Continuous data
Categorical data → one hot encoding?
NLP data → LLM sentiment feature embedding?
Image links → downloading images? Embedding?
Dates 
Umwandlung der listings.csv in richtiges csv format
data is raw
buggy zeilenumbrüche in Beschreibungen
umwandeln, sodass eine Zeile einem listing eintrag entspricht







Feature Engineering
verschiedene Features sind in Natural Language
listings.csv
name
description
Neighbourhood_overview
Host_about
Bathrooms_text
reviews.csv
comments
general ideas:
listings.csv
individual feature embedding for each categories in listings.csv or
aggregation of individual NLP texts for one summary embedding of the text
reviews.csv
aggregation of all reviews for specific listing and sentiment analysis?


Model Selection / Statistics
PCA of features
Linear Regression Baseline
which features to use for baseline model (all including NLP embeddings?)
regularisation? cross-validation?
Random Forest Regression
XGBoost
Kernel Methods
AutoML in tabular data for ensembling different models → more robust predictions?
Maybe kleines NN testen wenn viel Zeit
Clustering w.r.t feature attribution
Cross validation to check models
Rand index to check distinctness off clusters
Classification to check clustering
? Finding appropriate test-statistics


Caution: some models like XGBoost can handle categorical data well (no need for one-hot encoding) some like standard linear regression need more preprocessing)


Explainability Methods
Feature Attribution
which factors contribute most to the price prediction?
methods for extracting feature attribution in tabular data?
are methods intrinsically interpretable (like sparse linear models)?
feature importance extraction in XGBoost, Kernel Methods, RandomForests?
SHAP value for XGBoost
Integrated Gradient for NNs or other gradient-based methods
Random Forest is explainable by nature
For Black-Box NLP maybe GEEX algorithm (more complicated)
measuring feature importance by removing individual features at inference time and measuring accuracy decrease
removal strategy: NaN, zero, mean, median?


Limitations
Models
which models make which assumptions → are they met in our scenario
Soundness of evaluation methods
Do test statistics make sense
Discuss interpretability of SHAP, GEEX, and integrated gradients
Think about biases in the data
Reliableness of NLP analysis of text
Stability of general prizing
Correct for inflation (might be distinct for some regions)
Correct for differences in currencies















Auf https://insideairbnb.com/get-the-data/ finden sich aktuelle Daten zu Übernachtungsangeboten in verschiedenen Städten/Ländern
Price per Night prediction
Chance of reservation prediction
Eingrenzung auf gewisse Stadt/Städte und Vergleich der gewichtikeit gewisser Features über Städte hinweg
Großes Pro: Daten sind einfach verfügbar, müssen nur heruntergleaden werden

Welche Features beeinflussen den Preis?
Wie unterscheidet sich der Einfluss der Features in verschiedenen Regionen, Ländern, …?
Wie unterscheidet sich der Einfluss der Features in verschiedenen Jahreszeiten?
Währungsverhältnisse? Woher kommt der Tourist und wo bucht er?
…







