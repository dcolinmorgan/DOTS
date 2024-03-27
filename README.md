# Current Events Scraper & Featurizer

Using OpenSearch and Google News APIs, this tool pulls news stories and extracts features from the text. The features are then stored in a CSV file.

Can gather stories from multiple sources and languages. GNews maxes out at ~3000 stories per day, OpenSearch has no limit.  OpenSearch uses scroll and slice to pull a large number of stories .

Clone current version & run [dots_feat.py](https://github.com/dcolinmorgan/dots/blob/main/dots/dots_feat.py)
--------------------------------------------------
requirements :
spacy,
bs4,
scikit-learn,
transformers,
torch,
opensearch-py,
requests,
xml,
nltk,
numpy,
pandas
 
### the example below will pull 10 OS stories and return 5 features each in additon to location and date to a file

```python
    git clone https://github.com/dcolinmorgan/dots
    python dots/dots_feat.py -n 10 -f 5  
```

>"'Gaza Strip', '16-01-2024', ","['neighborhoods', 'rebels', 'widespread famine', 'egypt', 'disease']" <br>
>"'Miseno, Campania, Italy', '16-01-2024', ","['disasters', 'mount vesuvius', 'ancient cataclysm', 'costruzione', 'beach']"<br>
>"'Clarendon, Clarendon, Jamaica', '16-01-2024', ","['new bowen', 'fight', 'whatsapp', 'st catherine', 'jamaica']"<br>
>"'Philadelphia, Pennsylvania, United States', '16-01-2024', ","['meteorologists', 'snow shovels', 'snowstorm', 'accuweather alerts', 'accuweather meteorologists']"<br>
>"'New Bedford, Massachusetts, United States', '16-01-2024', ","['massachusetts law', 'saturday', 'ariel dorsey', 'traffic', 'united states']"<br>
>"'Corofin, Clare, Ireland', '16-01-2024', ","['emergency services', 'breathing', 'rescue service', 'firefighters', 'afternoon']"<br>
>"'United States', '16-01-2024', ","['preparedness', 'earthquake', 'quake', 'morning', 'disaster']"<br>
>"'Syria', '16-01-2024', ","['neighboring countries', 'early recovery', 'cholera', 'symptom', 'mohamad katoub']"<br>
>"'Iceland', '16-01-2024', ","['lava flows', 'evacuation', 'eruptions', 'jóhannesson', 'lúðvík pétursson']"<br>


here is an example produced every day via `gh_actions` parsing GNews stories and extracting features:
 [Table](DOTS/output/lobstr3_dots_feats.csv)
