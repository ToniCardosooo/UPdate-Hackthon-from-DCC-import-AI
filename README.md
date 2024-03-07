# UPDate Abuse Hackathon
## Team: from DCC import AI


### Overview

This project was developed for the 2024 UPDate Hackathon powered by NuCC and AbuseTotal, during the 2023/2024 academic year, at the Faculty of Sciences of the University of Porto, by the students André Sousa, António Cardoso, Bárbara Santos e Paulo Silva and you may find its [original repository on Github](https://github.com/ToniCardosooo/UPdate-Hackthon-from-DCC-import-AI).


### Setup

To run this build, you must have a [Twitch API Key](https://api-docs.igdb.com/#getting-started) and an [OpenAI Key](https://platform.openai.com/api-keys). Additionally, at the end of this file, you may find the Python and library versions used.


### The Problem

We were tasked with creating an AI service/agent that leverages LLMs to process, refine and/or enrich this data to produce intelligence on a daily basis. For this task, we decided to create a platform that fetches Twitch data about old and recent games and produces a possible future scoring and review for recently released games based on old similar ones.

### The Pipeline

To do this, we started by 

### Files

- `main.py` holds the main code to access our solution. Simply run `python main.py` to get started.
- `model.ipynb` holds the parameter tuning for the machine learning model used, along with a quick PCA analysis.
- `llm.py` holds functions to access the llm and the embeddings used for generation.
- `xgb.py` holds a function to generate a xgb model with the correct PCA function (stored in `pca.pkl`) and generate the model used in the main file.


### Versions

| Module | Version |
|:--------|:--------|
| Python | 3.10.13 |
| xgboost | 2.0.3 |
| matplotlib | 3.8.3 |
| scikit-learn | 1.4.1.post1 |
| Jinja2 | 3.1.2 |
| numpy | 1.25.1 |
| openai | 1.13.3 |
| pandas | 2.0.3 |
| scipy | 1.12.0 |
