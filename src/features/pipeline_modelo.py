"""
Pipeline de Modelagem (3 estágios) — compatível com SHAP.

Estágios:
  1. Preparo     -> usa a função preparar_dados do módulo preprocessamento.py
                   (modo fit=False, recebe params do treino)
  2. Imputação   -> SimpleImputer com estratégia 'mediana' (camada de segurança
                   para eventuais NaN remanescentes / novos dados)
  3. Modelo      -> Estimador scikit-learn-compatível (árvore, forest, boosting)

O objeto final é um Pipeline sklearn, que:
  - Permite .fit(X_train, y_train)
  - Permite .predict(X) e .predict_proba(X)
  - Expõe .named_steps["modelo"].feature_importances_  (SHAP-friendly)
  - Pode ser passado para shap.TreeExplainer (quando modelo for àrvore)
"""
from __future__ import annotations

import os
import pickle
from copy import deepcopy
from typing import Any, Dict

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer

# Ajusta import para funcionar tanto via CLI quanto em notebooks
if __package__ in (None, ""):
    import sys
    _SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    from features.preprocessamento import preparar_dados, carregar_params
else:
    from .preprocessamento import preparar_dados, carregar_params


# ----------------------------------------------------------------
# Wrapper do nosso preparador como transformador sklearn.
# Dessa forma, ele fica dentro do Pipeline (3 estágios).
# ----------------------------------------------------------------
class PreparoTransformador:
    """1º estágio do pipeline: wrapper de preparar_dados() como scikit-learn transformer."""

    def __init__(self, params: Dict[str, Any]):
        self.params = deepcopy(params)
        self.feature_names_in_ = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        """Apenas anota os nomes das colunas de entrada. Os parâmetros já foram aprendidos."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        """Aplica preparar_dados no modo fit=False (sem aprender nada)."""
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X_df = X.copy()
        # preparar_dados(fit=False) exige DataFrame com colunas originais.
        X_tr = preparar_dados(X_df, params=self.params, fit=False)
        self.feature_names_out_ = list(X_tr.columns)
        return X_tr.values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_ or [])


# ----------------------------------------------------------------
# Construtor do pipeline completo.
# ----------------------------------------------------------------
def montar_pipeline(modelo: Any, params_prep: Dict[str, Any]) -> ImbPipeline:
    """Cria pipeline com os 3 estágios.

    Parâmetros
    ----------
    modelo : estimador sklearn-compt (deve implementar fit/predict/predict_proba)
    params_prep : dict vindo de preparar_dados(..., fit=True)

    Retorna
    -------
    Pipeline com as etapas nomeadas:
      ('preparo',  PreparoTransformador(params_prep)),
      ('imputacao', SimpleImputer(strategy='median')),
      ('modelo', modelo)
    """
    pipe = ImbPipeline(steps=[
        ("preparo",   PreparoTransformador(params_prep)),
        ("imputacao", SimpleImputer(strategy="median", copy=True, add_indicator=False)),
        ("modelo",    deepcopy(modelo)),
    ])
    return pipe


def obter_nomes_features(pipe: ImbPipeline) -> list:
    """Obtém os nomes das features APÓS os estágios 'preparo' e 'imputacao'.
    Útil para SHAP.
    """
    prep = pipe.named_steps.get("preparo")
    if hasattr(prep, "feature_names_out_") and prep.feature_names_out_:
        return list(prep.feature_names_out_)
    if hasattr(prep, "feature_names_in_"):
        return list(prep.feature_names_in_)
    raise ValueError("Pipeline ainda não foi fitado; rode pipe.fit(X, y) antes.")


def obter_feature_importances(pipe: ImbPipeline) -> pd.DataFrame:
    """Retorna DataFrame com feature_importances_ do modelo do pipeline.
    Se modelo não tiver feature_importances_, levanta AttributeError.
    """
    modelo = pipe.named_steps["modelo"]
    names  = obter_nomes_features(pipe)
    imps   = getattr(modelo, "feature_importances_", None)
    if imps is None:
        raise AttributeError("Este modelo não expõe feature_importances_.")
    df = pd.DataFrame({"feature": names, "importance": imps})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def salvar_pipeline(pipe: ImbPipeline, caminho: str) -> None:
    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
    with open(caminho, "wb") as f:
        pickle.dump(pipe, f)


def carregar_pipeline(caminho: str) -> ImbPipeline:
    with open(caminho, "rb") as f:
        return pickle.load(f)
