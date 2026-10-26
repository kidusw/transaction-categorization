"""Types the skops model format may load (MLflow 3 saves sklearn models with skops).

skops refuses to load any type that isn't allow-listed, so a tampered model file can't
run arbitrary code the way a pickle can. Shared by training (when saving) and the API
(when loading an exported model without MLflow).
"""

SKOPS_TRUSTED_TYPES = [
    "txcat.models.baseline.RuleBaseline",
    "txcat.models.embeddings.EmbeddingEncoder",
    "txcat.models.features.direction_features",
    "txcat.models.features._direction_names",
    "lightgbm.sklearn.LGBMClassifier",
    "lightgbm.basic.Booster",
    "collections.OrderedDict",
]
