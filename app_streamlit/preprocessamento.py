"""
Módulo de Preparação dos Dados — Reutilizável no Streamlit.

Função principal:
    preparar_dados(df, params=None, fit=True, target_col="inadimplente_2anos",
                   test_size=0.25, seed=42)

O pipeline:
  0. Split estratificado 75-25 (executado APENAS quando fit=True).
  1. Flag renda ausente + imputação por mediana (aprende a mediana SÓ no treino).
  2. Flag dependentes ausentes.
  3. Trata colunas de atrasos: top-capping em 20 + flag única para códigos 96/98.
  4. Top-capping em 50 000 na renda_mensal.
  5. Features novas:
      5.1 renda_por_dependente   = renda / (dependentes + 1)
      5.2 sobra_caixa            = renda_mensal * (1 - razao_divida)

Retornos:
    - Quando fit=True:  (X_train, X_test, y_train, y_test, params)
    - Quando fit=False: (X_transformed)       --- p/ Streamlit em produção.

params é um dict serializável (pickle) com TODOS os valores aprendidos no treino,
para evitar data leakage na hora de aplicar no teste / em produção.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED_DEFAULT = 42
TEST_SIZE_DEFAULT = 0.25

# ----- Constantes do pipeline (definidas pelo negócio) -----
TOP_CAP_ATRASOS = 20
TOP_CAP_RENDA = 50_000
CODIGOS_SISTEMA_ATRASOS = (96, 98)
COLUNAS_ATRASO = [
    "atrasos_30_59_dias",
    "atrasos_60_89_dias",
    "atrasos_90_mais_dias",
]
COLUNAS_ORIGINAIS_OBRIGATORIAS = [
    "idade",
    "renda_mensal",
    "dependentes",
    "uso_limite_rotativo",
    "razao_divida",
    "linhas_credito_abertas",
    "financiamentos_imobiliarios",
] + COLUNAS_ATRASO


# ----------------------------------------------------------------
# 1. Pipeline de preparação como função única e idempotente
# ----------------------------------------------------------------
def preparar_dados(
    df: pd.DataFrame,
    params: Union[Dict[str, Any], None] = None,
    fit: bool = True,
    target_col: str = "inadimplente_2anos",
    test_size: float = TEST_SIZE_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]], pd.DataFrame]:
    """Pipeline único de preparação de dados (treino e inferência).

    Parâmetros
    ----------
    df : DataFrame com as colunas originais. Opcionalmente com target.
    params : Dict gerado nesta função quando fit=True. Passe este mesmo dict
             para aplicar parâmetros do treino em dados novos (fit=False).
    fit : bool - True durante preparação do projeto; False p/ produção.
    target_col : nome da coluna alvo.
    test_size : fração do holdout.
    seed : semente global de reprodutibilidade.

    Retorna
    -------
    fit=True  -> (X_train, X_test, y_train, y_test, params)
    fit=False -> X_transformado  (DataFrame, sem o target)
    """
    # --- Checks iniciais ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` deve ser um pandas.DataFrame.")
    if fit and target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada no df (modo fit=True).")

    faltantes = [c for c in COLUNAS_ORIGINAIS_OBRIGATORIAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Colunas obrigatórias faltando: {faltantes}")

    df = df.copy()

    # ------------------------------------------------------------------
    # 0. Split estratificado — SÓ quando fit=True
    # ------------------------------------------------------------------
    if fit:
        y_full = df[target_col].copy()
        X_full = df.drop(columns=[target_col])
        X_train, X_test, y_train, y_test = train_test_split(
            X_full,
            y_full,
            test_size=test_size,
            random_state=seed,
            stratify=y_full,
        )
        params = _fit_params(X_train, y_train)
        X_train_p = _transform(X_train, params)
        X_test_p  = _transform(X_test,  params)
        return X_train_p, X_test_p, y_train, y_test, params
    else:
        if params is None:
            raise ValueError("params é obrigatório no modo fit=False (inferência/produção).")
        # Sem target
        X = df.drop(columns=[target_col], errors="ignore")
        return _transform(X, params)


# ----------------------------------------------------------------
# 2. Sub-rotinas internas (fit + transform separados = sem leakage)
# ----------------------------------------------------------------
def _fit_params(X: pd.DataFrame, y: pd.Series | None = None) -> Dict[str, Any]:
    """Aprende os parâmetros estritamente a partir da base de treino."""
    params: Dict[str, Any] = {}
    params["mediana_renda_mensal"] = float(np.nanmedian(X["renda_mensal"]))
    params["top_cap_atrasos"]      = TOP_CAP_ATRASOS
    params["top_cap_renda"]        = TOP_CAP_RENDA
    params["codigos_sistema"]      = list(CODIGOS_SISTEMA_ATRASOS)
    params["colunas_atraso"]       = list(COLUNAS_ATRASO)
    return params


def _transform(X_raw: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Aplica as transformações usando params (nunca aprende nada novo)."""
    X = X_raw.copy()

    # 1 e 2 --- Renda: flag + imputação
    X["renda_ausente"] = X["renda_mensal"].isnull().astype(int)
    X["renda_mensal"]  = X["renda_mensal"].fillna(params["mediana_renda_mensal"])

    # 3 --- Dependentes: flag ausente
    X["dependentes_ausentes"] = X["dependentes"].isnull().astype(int)
    # Preenche dependentes com 0 como fallback (clientes sem dependentes declarados).
    # Não usamos mediana aqui — decisão de negócio de que missing = nenhum dependente
    # informado, mas deixamos a flag para não perder informação.
    X["dependentes"] = X["dependentes"].fillna(0).astype(float)

    # 4 --- Colunas de atrasos: top-capping em 20 + flag única de código de sistema
    cod_sis = tuple(params["codigos_sistema"])
    cols_a  = params["colunas_atraso"]

    # Detecta se QUALQUER das 3 colunas tem 96/98 (investigação já provou:
    # sempre as 3 ao mesmo tempo. A flag única é suficiente.)
    mask_sistema = np.zeros(len(X), dtype=bool)
    for c in cols_a:
        mask_sistema = mask_sistema | X[c].isin(cod_sis).values

    X["cod_sistema_atrasos"] = mask_sistema.astype(int)

    # Substitui os códigos 96/98 por np.nan momentaneamente antes do top-cap
    for c in cols_a:
        X[c] = X[c].where(~X[c].isin(cod_sis), other=params["top_cap_atrasos"])

    # Top-cap fixo em 20 nas colunas de atraso
    cap = params["top_cap_atrasos"]
    for c in cols_a:
        X[c] = X[c].clip(upper=cap).astype(int)

    # 5 --- Top-cap de renda em R$ 50.000
    X["renda_mensal"] = X["renda_mensal"].clip(upper=params["top_cap_renda"])

    # 6.1 --- Renda por dependente = renda / (dependentes + 1)  (+1 evita div/0)
    X["renda_por_dependente"] = X["renda_mensal"] / (X["dependentes"] + 1.0)

    # 6.2 --- Sobra caixa = renda * (1 - razao_divida)
    # Evita valores absurdamente grandes por causa de razao_divida > 1. A própria
    # base de dados tem razao_divida truncada em 2,0. Não precisamos clipar aqui
    # mas garantimos numpy casting para float.
    X["sobra_caixa"] = X["renda_mensal"].astype(float) * (1.0 - X["razao_divida"].astype(float))

    # Ordem das colunas: primeiro features originais (sem alvo), depois flags, depois novas
    ordem_cols = (
        ["idade", "renda_mensal", "dependentes", "uso_limite_rotativo", "razao_divida",
         "linhas_credito_abertas", "financiamentos_imobiliarios",
         "atrasos_30_59_dias", "atrasos_60_89_dias", "atrasos_90_mais_dias",
         "renda_ausente", "dependentes_ausentes", "cod_sistema_atrasos",
         "renda_por_dependente", "sobra_caixa"]
    )
    # Mantém todas as colunas extras, mas na ordem desejada primeiro
    extras = [c for c in X.columns if c not in ordem_cols]
    X = X[ordem_cols + extras]
    return X


# ----------------------------------------------------------------
# 3. Helpers de salvamento / carregamento de params (p/ Streamlit)
# ----------------------------------------------------------------
def salvar_params(params: Dict[str, Any], caminho: str) -> None:
    """Serializa params em arquivo pickle."""
    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
    with open(caminho, "wb") as f:
        pickle.dump(params, f)


def carregar_params(caminho: str) -> Dict[str, Any]:
    """Carrega params salvos em pickle."""
    with open(caminho, "rb") as f:
        return pickle.load(f)


def _fmt_num(v):
    try:
        if pd.isna(v):
            return "NA"
        if isinstance(v, (int, np.integer)):
            return f"{int(v)}"
        if abs(v) >= 1000:
            return f"{v:,.2f}"
        return f"{v:.3f}"
    except Exception:
        return str(v)


# ----------------------------------------------------------------
# 4. Ponto de entrada CLI: executa o pipeline completo e salva artefatos.
# ----------------------------------------------------------------
if __name__ == "__main__":
    import sys

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATA_RAW = os.path.join(BASE_DIR, "data", "raw", "credito_tratado.csv")
    DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(DATA_PROC, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"[INFO] Carregando {DATA_RAW} ...")
    df_raw = pd.read_csv(DATA_RAW)

    print(f"[INFO] Split 75-25 estratificado + preparando dados ...")
    X_train, X_test, y_train, y_test, params = preparar_dados(
        df_raw,
        fit=True,
        test_size=0.25,
        seed=SEED_DEFAULT,
    )

    # --- Verifica distribuição do alvo pós split ---
    print("\n[INFO] Distribuição do alvo após split estratificado:")
    dist = pd.DataFrame({
        "Conjunto": ["Treino", "Teste", "Base completa"],
        "Tamanho":  [len(y_train), len(y_test), len(df_raw)],
        "N Inadimplentes": [int(y_train.sum()), int(y_test.sum()), int(df_raw["inadimplente_2anos"].sum())],
        "% Inadimplentes": [
            f"{y_train.mean()*100:.2f}",
            f"{y_test.mean()*100:.2f}",
            f"{df_raw['inadimplente_2anos'].mean()*100:.2f}",
        ],
    })
    print(dist.to_string(index=False))

    # --- Salva artefatos ---
    xtrain_path = os.path.join(DATA_PROC, "X_train.csv")
    xtest_path  = os.path.join(DATA_PROC, "X_test.csv")
    ytrain_path = os.path.join(DATA_PROC, "y_train.csv")
    ytest_path  = os.path.join(DATA_PROC, "y_test.csv")
    params_path = os.path.join(MODELS_DIR, "preprocessamento_params.pkl")

    X_train.to_csv(xtrain_path, index=False)
    X_test.to_csv(xtest_path,  index=False)
    y_train.to_frame(name="inadimplente_2anos").to_csv(ytrain_path, index=False)
    y_test.to_frame(name="inadimplente_2anos").to_csv(ytest_path, index=False)
    salvar_params(params, params_path)

    print(f"\n[INFO] Artefatos salvos:")
    print(f"  X_train              : {xtrain_path}")
    print(f"  X_test               : {xtest_path}")
    print(f"  y_train              : {ytrain_path}")
    print(f"  y_test               : {ytest_path}")
    print(f"  Parâmetros (pickle)  : {params_path}")

    # --- Resumo final da base TREINO ---
    print("\n[INFO] Resumo da base X_train após tratamentos:")
    print(f"  Linhas : {len(X_train):,}")
    print(f"  Colunas: {len(X_train.columns)}")
    tipos = pd.DataFrame({
        "Coluna": X_train.columns,
        "Tipo Pandas": [str(X_train[c].dtype) for c in X_train.columns],
        "Nulos": [int(X_train[c].isnull().sum()) for c in X_train.columns],
        "Min": [_fmt_num(X_train[c].min()) for c in X_train.columns],
        "Max": [_fmt_num(X_train[c].max()) for c in X_train.columns],
        "Média": [_fmt_num(X_train[c].mean()) for c in X_train.columns],
    })
    print(tipos.to_string(index=False))

    print("\n[INFO] Parâmetros fitados no treino (salvos em pickle p/ uso em produção):")
    for k, v in params.items():
        print(f"  · {k:30s} = {v}")

    sys.exit(0)
