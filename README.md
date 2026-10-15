# Impact de la taille des chunks et de l'overlap sur les performances d'un système RAG


Étude contrôlée de l'effet de la **taille des chunks** (32 → 1 024 tokens) et du **taux d'overlap** (0 → 50 %) sur un système RAG, à trois niveaux : couverture des réponses, qualité de la recherche et qualité des réponses générées par un LLM. Le coût est également mesuré.

## Résultats clés
- **Les petits chunks gagnent.** À budget de contexte fixe (1 024 tokens), le Hit@budget passe de **93,8 %** avec des chunks de 64 tokens à **39,4 %** avec des chunks de 1 024 tokens. La cause est la dilution sémantique des embeddings denses.
- **25 % d'overlap** apporte +6,8 pts de recherche et +6,9 pts de F1 à 64 tokens, pour +33 % d'index. Son effet sur la couverture est prédit par un modèle probabiliste simple (corrélation de 0,99).
- **BM25 reste stable jusqu'à 256 tokens**, alors que la recherche dense chute : la dilution est propre au retrieval dense.
- **La génération suit la recherche** (Spearman = 1,0). Le F1 atteint 66,4, contre 15,1 sans RAG et 80,8 avec l'oracle. Les hallucinations passent de 84 % à environ 2 % grâce à un prompt strict avec abstention.

## Contenu
| Fichier | Description |
|---|---|
| `RAG_chunking_overlap_Islem_Alimi.ipynb` | Notebook unique : introduction, état de l'art, données, méthodologie, code, protocole, résultats, analyse critique, références |

## Configuration expérimentale
- **Données :** [SQuAD v1.1, split validation](https://huggingface.co/datasets/rajpurkar/squad). Les 48 articles Wikipédia sont reconstitués et la position exacte des réponses est connue.
- **Retriever :** `BAAI/bge-small-en-v1.5`, avec BM25 comme baseline.
- **Générateur :** `Qwen/Qwen2.5-1.5B-Instruct`, avec un décodage glouton.
- **Configurations :** 34 configurations de chunking (stratégies `fixed` et `sentence`), comparées à k fixe et à budget de tokens fixe.
- **Statistiques :** intervalles de confiance à 95 % et tests appariés par bootstrap.

## Reproduire
Ouvrir le notebook dans Google Colab avec un GPU T4, puis lancer « Exécuter tout » (≈ 30–40 min). Tous les paramètres se trouvent dans `CONFIG`, avec la graine 42. Les résultats (CSV et figures) sont écrits dans `results/`.
