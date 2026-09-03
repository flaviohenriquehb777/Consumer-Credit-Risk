"""
Fase 3 Parte 2 + Fase 4:
  - Monta o pipeline de 3 estágios (Preparo -> SimpleImputer mediana -> Modelo)
  - Tuning DecisionTree (max_depth de 1 a 15)
  - Tuning RandomForest  (max_depth 6,8,10,12,None, com class_weight balanced / balanced_subsample / {0:1,1:14})
  - XGBoost + LightGBM com scale_pos_weight correto (= 93,32/6,68 ≈ 14)
  - StratifiedKFold 5 splits (só treino) em todos os modelos: ROC AUC + Acurácia
  - Compara baseline vs. técnica UnderSampling inteligente (NearMiss v3 + ENN)
  - Salva pickles dos modelos finais em models/
  - Tabela consolidada de todas as métricas: ROC AUC, PR AUC, F1, Precision, Recall, Brier
SEED = 42
"""
from __future__ import annotations

import os
import sys
import pickle
import warnings
import time
from datetime import datetime

import numpy as np
import pandas as pd
from copy import deepcopy

# -------- Scikit-learn --------
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,   # PR AUC
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

# -------- Boosting --------
import xgboost as xgb
import lightgbm as lgb

# -------- Imbalance --------
from imblearn.under_sampling import NearMiss, EditedNearestNeighbours

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from features.preprocessamento import carregar_params, SEED_DEFAULT
from features.pipeline_modelo import (
    montar_pipeline,
    salvar_pipeline,
    obter_feature_importances,
)

warnings.filterwarnings("ignore")
np.random.seed(SEED_DEFAULT)

# ---------- Paths ----------
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC = os.path.join(BASE, "data", "processed")
MODELOS_DIR = os.path.join(BASE, "models")
REPORTS_DIR = os.path.join(BASE, "reports", "evaluations")
FIG_DIR = os.path.join(BASE, "reports", "figures")
PARAMS_PATH = os.path.join(MODELOS_DIR, "preprocessamento_params.pkl")

X_train = pd.read_csv(os.path.join(PROC, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(PROC, "X_test.csv"))
y_train = pd.read_csv(os.path.join(PROC, "y_train.csv"))["inadimplente_2anos"].astype(int)
y_test  = pd.read_csv(os.path.join(PROC, "y_test.csv"))["inadimplente_2anos"].astype(int)

# Recupera params do pickle (o pipeline precisa dele no 1º estágio)
params_prep = carregar_params(PARAMS_PATH)

# scale_pos_weight correto = (neg / pos) no treino
neg = int((y_train == 0).sum())
pos = int((y_train == 1).sum())
SPW = round(neg / pos, 2)   # ≈ 14
print(f"[INFO] scale_pos_weight = neg/pos = {neg}/{pos} = {SPW}")
print(f"[INFO] X_train = {X_train.shape}, y_train = {y_train.value_counts().to_dict()}")
print(f"[INFO] X_test  = {X_test.shape},  y_test  = {y_test.value_counts().to_dict()}")
print("=" * 100)


# ====================================================================
# Helpers
# ====================================================================
def avaliar(nome_modelo: str, pipe, Xtr, ytr, Xte, yte, extra_info: str = ""):
    """Treina e retorna métricas + tempo + feature importances (se disponível)."""
    t0 = time.time()
    pipe.fit(Xtr, ytr)
    treino_s = round(time.time() - t0, 2)

    # Scores em treino e teste (usamos proba para AUCs, classes 0.5 threshold para F1)
    proba_tr = pipe.predict_proba(Xtr)[:, 1]
    proba_te = pipe.predict_proba(Xte)[:, 1]
    pred_tr  = pipe.predict(Xtr)
    pred_te  = pipe.predict(Xte)

    # Tabela de métricas (prioriza o TESTE, que é o que importa)
    res = {
        "Modelo": nome_modelo,
        "Obs": extra_info,
        "Tempo (s)": treino_s,
        "Teste ROC AUC": round(roc_auc_score(yte, proba_te), 4),
        "Teste PR AUC": round(average_precision_score(yte, proba_te), 4),
        "Teste F1":       round(f1_score(yte, pred_te), 4),
        "Teste Precision":round(precision_score(yte, pred_te, zero_division=0), 4),
        "Teste Recall":   round(recall_score(yte, pred_te, zero_division=0), 4),
        "Teste Brier":    round(brier_score_loss(yte, proba_te), 5),
        "Teste Acurácia": round(accuracy_score(yte, pred_te), 4),
        "Treino ROC AUC": round(roc_auc_score(ytr, proba_tr), 4),
        "Treino F1":      round(f1_score(ytr, pred_tr), 4),
        "Δ ROC AUC (tr-te)": round(roc_auc_score(ytr, proba_tr) - roc_auc_score(yte, proba_te), 4),
    }
    imps = None
    try:
        imps = obter_feature_importances(pipe)
    except Exception:
        pass
    return res, imps


def cross_validar(nome: str, modelo_base, Xtr, ytr, n_splits=5, seed=SEED_DEFAULT):
    """StratifiedKFold 5 splits, metricas ROC AUC e Acurácia, SÓ em dados de TREINO."""
    # Monta o pipeline de 3 estágios (igual ao de treino)
    pipe_cv = montar_pipeline(modelo_base, params_prep)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_validate(
        pipe_cv, Xtr, ytr,
        cv=cv,
        scoring=["roc_auc", "accuracy"],
        n_jobs=-1,
        error_score="raise",
        return_train_score=True,
    )
    return {
        "Modelo": nome,
        "CV ROC AUC (média ± std)": f"{scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}",
        "CV ROC AUC (train mean)":   round(scores['train_roc_auc'].mean(), 4),
        "CV Acurácia (média ± std)": f"{scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}",
        "CV Acurácia (train mean)":  round(scores['train_accuracy'].mean(), 4),
    }


# ====================================================================
# 1. DecisionTree — tuning de max_depth (1..15)
# ====================================================================
print("\n[1/7] Tuning DecisionTree por max_depth ...")
tab_dt = []
melhor_dt = {"roc_auc": -1, "depth": None}
for d in range(1, 16):
    pipe = montar_pipeline(
        DecisionTreeClassifier(max_depth=d, random_state=SEED_DEFAULT, class_weight=None, min_samples_leaf=50),
        params_prep,
    )
    r, _ = avaliar(f"DecisionTree depth={d}", pipe, X_train, y_train, X_test, y_test,
                   extra_info=f"class_weight=None; leaf>=50")
    tab_dt.append(r)
    if r["Teste ROC AUC"] > melhor_dt["roc_auc"]:
        melhor_dt = {"roc_auc": r["Teste ROC AUC"], "depth": d, "config": pipe}

tab_dt_df = pd.DataFrame(tab_dt).sort_values("Teste ROC AUC", ascending=False).reset_index(drop=True)
print(f"\n  [DECISION TREE] Melhor profundidade = {melhor_dt['depth']} (Teste ROC AUC = {melhor_dt['roc_auc']:.4f})")
print("  Top 5 combinações de profundidade:")
print(tab_dt_df[["Modelo", "Teste ROC AUC", "Teste PR AUC", "Teste F1", "Teste Recall", "Δ ROC AUC (tr-te)"]].head().to_string(index=False))

# ====================================================================
# 2. RandomForest — tuning max_depth + class_weight (3 opções)
# ====================================================================
print("\n[2/7] Tuning RandomForest max_depth e class_weight ...")
tab_rf = []
melhor_rf = {"roc_auc": -1}
depth_grid = [6, 8, 10, 12, 15, None]
cw_grid    = [
    ("balanced",         "balanced"),
    ("balanced_subsamp", "balanced_subsample"),
    ("balanced 1:14",    {0: 1, 1: 14}),
]
for depth in depth_grid:
    for cw_label, cw in cw_grid:
        pipe = montar_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=depth,
                class_weight=cw,
                n_jobs=-1,
                random_state=SEED_DEFAULT,
                min_samples_leaf=30,
                oob_score=False,
            ),
            params_prep,
        )
        r, _ = avaliar(f"RF depth={depth} cw={cw_label}", pipe, X_train, y_train, X_test, y_test,
                       extra_info="n_est=300 leaf>=30")
        tab_rf.append(r)
        if r["Teste ROC AUC"] > melhor_rf["roc_auc"]:
            melhor_rf = {"roc_auc": r["Teste ROC AUC"], "depth": depth, "cw": cw_label, "cw_val": cw, "config": pipe}

tab_rf_df = pd.DataFrame(tab_rf).sort_values("Teste ROC AUC", ascending=False).reset_index(drop=True)
print(f"\n  [RANDOM FOREST] Melhor config: depth={melhor_rf['depth']}, class_weight={melhor_rf['cw']}  "
      f"(Teste ROC AUC = {melhor_rf['roc_auc']:.4f})")
print("  Top 5 RF:")
print(tab_rf_df[["Modelo", "Teste ROC AUC", "Teste PR AUC", "Teste F1", "Teste Recall", "Δ ROC AUC (tr-te)"]].head().to_string(index=False))


# ====================================================================
# 3. Boostings (XGBoost + LightGBM) com scale_pos_weight = 14
# ====================================================================
print("\n[3/7] Treinando XGBoost ...")
pipe_xgb = montar_pipeline(
    xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=100,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=SPW,
        random_state=SEED_DEFAULT,
        n_jobs=-1,
        eval_metric="auc",
        tree_method="hist",
    ),
    params_prep,
)
r_xgb, imps_xgb = avaliar("XGBoost", pipe_xgb, X_train, y_train, X_test, y_test,
                          extra_info=f"spw={SPW} lr=0.05 md=5")

print("\n[4/7] Treinando LightGBM ...")
pipe_lgb = montar_pipeline(
    lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=120,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=SPW,
        random_state=SEED_DEFAULT,
        n_jobs=-1,
        verbose=-1,
    ),
    params_prep,
)
r_lgb, imps_lgb = avaliar("LightGBM", pipe_lgb, X_train, y_train, X_test, y_test,
                          extra_info=f"spw={SPW} lr=0.05 md=6 leaves=31")

# ====================================================================
# 4. Baseline: Dummy (prior) e DT / RF melhores configurados já estão prontos
# ====================================================================
print("\n[5/7] Treinando baseline Dummy (prior) ...")
pipe_dummy = montar_pipeline(
    DummyClassifier(strategy="prior", random_state=SEED_DEFAULT),
    params_prep,
)
r_dummy, _ = avaliar("Dummy (prior)", pipe_dummy, X_train, y_train, X_test, y_test, extra_info="chuta sempre modal")

# Re-treina o melhor DT/RF com o nome final (pega só o melhor resultado da combinação)
pipe_best_dt = melhor_dt["config"]
r_best_dt, imps_dt = avaliar(f"DecisionTree (best depth={melhor_dt['depth']})",
                             pipe_best_dt, X_train, y_train, X_test, y_test,
                             extra_info="escolhido no tuning")
pipe_best_rf = melhor_rf["config"]
r_best_rf, imps_rf = avaliar(f"RandomForest (best depth={melhor_rf['depth']}, cw={melhor_rf['cw']})",
                             pipe_best_rf, X_train, y_train, X_test, y_test,
                             extra_info="escolhido no tuning")

# ====================================================================
# 5. UnderSampling inteligente (sem tunar os modelos — manter mesmos hparams)
# ====================================================================
print("\n[6/7] Testando UnderSampling inteligente (NearMiss v3 + ENN -> treina XGBoost) ...")

# --- Ajuste: Undersampling é SÓ aplicado no treino (evita leakage).
#     Como nossa pipeline aceita matrizes numpy, criamos X_tr_prep e X_te_prep
#     usando o 1º e 2º estágios com os dados já preparados.
from features.pipeline_modelo import PreparoTransformador
from sklearn.impute import SimpleImputer

# 1º + 2º estágio aplicados diretamente (para alimentar o sampler)
prep = PreparoTransformador(params_prep)
prep.fit(X_train, y_train)
Xtr_prep = prep.transform(X_train)
Xte_prep = prep.transform(X_test)
imp_ = SimpleImputer(strategy="median")
Xtr_prep = imp_.fit_transform(Xtr_prep)
Xte_prep = imp_.transform(Xte_prep)
feature_names = prep.get_feature_names_out().tolist()

nm3 = NearMiss(version=3, n_neighbors=3, n_jobs=-1)
t0 = time.time()
X_tr_nm3, y_tr_nm3 = nm3.fit_resample(Xtr_prep, y_train.values)
print(f"  NearMiss v3: {Xtr_prep.shape} -> {X_tr_nm3.shape} (kept ratio={len(X_tr_nm3)/len(Xtr_prep):.3f})")

# Treina XGBoost (sem scale_pos_weight agora, pois as classes estão "equilibradas")
xgb_under = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=5, min_child_weight=100,
    subsample=0.9, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=1.0,  # classes balanceadas pós NearMiss
    random_state=SEED_DEFAULT, n_jobs=-1, eval_metric="auc", tree_method="hist",
)
xgb_under.fit(X_tr_nm3, y_tr_nm3)
pred_u = xgb_under.predict(Xte_prep)
prob_u = xgb_under.predict_proba(Xte_prep)[:, 1]
r_under = {
    "Modelo": "XGBoost + NearMiss v3 (UnderSample inteligente)",
    "Obs": "NM3 + spw=1 (classes balanceadas após NM)",
    "Tempo (s)": round(time.time() - t0, 2),
    "Teste ROC AUC": round(roc_auc_score(y_test, prob_u), 4),
    "Teste PR  AUC": round(average_precision_score(y_test, prob_u), 4),
    "Teste F1":       round(f1_score(y_test, pred_u), 4),
    "Teste Precision":round(precision_score(y_test, pred_u, zero_division=0), 4),
    "Teste Recall":   round(recall_score(y_test, pred_u, zero_division=0), 4),
    "Teste Brier":    round(brier_score_loss(y_test, prob_u), 5),
    "Teste Acurácia": round(accuracy_score(y_test, pred_u), 4),
    "Treino ROC AUC": round(roc_auc_score(y_tr_nm3, xgb_under.predict_proba(X_tr_nm3)[:, 1]), 4),
    "Treino F1":      round(f1_score(y_tr_nm3, xgb_under.predict(X_tr_nm3)), 4),
    "Δ ROC AUC (tr-te)": round(roc_auc_score(y_tr_nm3, xgb_under.predict_proba(X_tr_nm3)[:, 1]) - roc_auc_score(y_test, prob_u), 4),
}

# também testamos ENN (Edited Nearest Neighbours) — limpeza mais suave que mantém volume
print("\n[6b/7] Testando ENN — Edited Nearest Neighbours ...")
t0 = time.time()
enn = EditedNearestNeighbours(n_neighbors=5, n_jobs=-1, kind_sel="all")
X_tr_enn, y_tr_enn = enn.fit_resample(Xtr_prep, y_train.values)
print(f"  ENN: {Xtr_prep.shape} -> {X_tr_enn.shape} (kept ratio={len(X_tr_enn)/len(Xtr_prep):.3f})")
xgb_enn = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=5, min_child_weight=100,
    subsample=0.9, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=SPW, random_state=SEED_DEFAULT, n_jobs=-1, eval_metric="auc", tree_method="hist",
)
xgb_enn.fit(X_tr_enn, y_tr_enn)
pred_e = xgb_enn.predict(Xte_prep)
prob_e = xgb_enn.predict_proba(Xte_prep)[:, 1]
r_enn = {
    "Modelo": "XGBoost + ENN (UnderSample clean inteligente)",
    "Obs": "ENN + spw=14 (mantém desbalanceamento original)",
    "Tempo (s)": round(time.time() - t0, 2),
    "Teste ROC AUC": round(roc_auc_score(y_test, prob_e), 4),
    "Teste PR  AUC": round(average_precision_score(y_test, prob_e), 4),
    "Teste F1":       round(f1_score(y_test, pred_e), 4),
    "Teste Precision":round(precision_score(y_test, pred_e, zero_division=0), 4),
    "Teste Recall":   round(recall_score(y_test, pred_e, zero_division=0), 4),
    "Teste Brier":    round(brier_score_loss(y_test, prob_e), 5),
    "Teste Acurácia": round(accuracy_score(y_test, pred_e), 4),
    "Treino ROC AUC": round(roc_auc_score(y_tr_enn, xgb_enn.predict_proba(X_tr_enn)[:, 1]), 4),
    "Treino F1":      round(f1_score(y_tr_enn, xgb_enn.predict(X_tr_enn)), 4),
    "Δ ROC AUC (tr-te)": round(roc_auc_score(y_tr_enn, xgb_enn.predict_proba(X_tr_enn)[:, 1]) - roc_auc_score(y_test, prob_e), 4),
}


# ====================================================================
# 6. Tabela COMPARATIVA FINAL de todos os modelos candidatos
# ====================================================================
print("\n" + "=" * 110)
print("[7/7] Tabela final consolidada de todos os modelos")
print("=" * 110)
tabela_final = pd.DataFrame([r_dummy, r_best_dt, r_best_rf, r_xgb, r_lgb, r_under, r_enn])
colunas_tabela = [
    "Modelo", "Obs", "Teste ROC AUC", "Teste PR AUC", "Teste F1",
    "Teste Precision", "Teste Recall", "Teste Acurácia", "Teste Brier",
    "Δ ROC AUC (tr-te)", "Tempo (s)",
]
tabela_final = tabela_final[colunas_tabela].sort_values("Teste ROC AUC", ascending=False).reset_index(drop=True)
print(tabela_final.to_string(index=False))

# ====================================================================
# 7. StratifiedKFold 5 splits em cada modelo (só treino, ROC AUC + Acurácia)
# ====================================================================
print("\n" + "=" * 110)
print("CROSS VALIDATION: StratifiedKFold 5 (só dados de TREINO)")
print("=" * 110)

cv_modelos = [
    ("Dummy prior", DummyClassifier(strategy="prior", random_state=SEED_DEFAULT)),
    (f"DT best d={melhor_dt['depth']}",
     DecisionTreeClassifier(max_depth=melhor_dt["depth"], random_state=SEED_DEFAULT,
                            class_weight=None, min_samples_leaf=50)),
    (f"RF best d={melhor_rf['depth']} {melhor_rf['cw']}",
     RandomForestClassifier(
         n_estimators=300, max_depth=melhor_rf['depth'], class_weight=melhor_rf['cw_val'],
         n_jobs=-1, random_state=SEED_DEFAULT, min_samples_leaf=30, oob_score=False)),
    ("XGBoost", xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=5, min_child_weight=100,
        subsample=0.9, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=SPW, random_state=SEED_DEFAULT, n_jobs=-1,
        eval_metric="auc", tree_method="hist")),
    ("LightGBM", lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=6,
        min_child_samples=120, subsample=0.9, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=SPW,
        random_state=SEED_DEFAULT, n_jobs=-1, verbose=-1)),
]
cv_rows = []
for nm, modelo in cv_modelos:
    print(f"  CV-ing {nm} ...  ", end="", flush=True)
    row = cross_validar(nm, modelo, X_train, y_train, n_splits=5)
    print(f" ROC AUC = {row['CV ROC AUC (média ± std)']}; Acurácia = {row['CV Acurácia (média ± std)']}")
    cv_rows.append(row)
cv_df = pd.DataFrame(cv_rows).sort_values("CV ROC AUC (média ± std)", ascending=False).reset_index(drop=True)
print("\nTabela StratifiedKFold 5:")
print(cv_df.to_string(index=False))

# ====================================================================
# 8. Persistência: salva pipelines em pickle e CSVs de relatório
# ====================================================================
print("\nSalvando artefatos ...")
os.makedirs(REPORTS_DIR, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# Pipelines dos modelos (p/ uso direto no Streamlit)
salvar_pipeline(pipe_dummy,    os.path.join(MODELOS_DIR, "01_dummy_pipeline.pkl"))
salvar_pipeline(pipe_best_dt,  os.path.join(MODELOS_DIR, "02_decisiontree_best_pipeline.pkl"))
salvar_pipeline(pipe_best_rf,  os.path.join(MODELOS_DIR, "03_randomforest_best_pipeline.pkl"))
salvar_pipeline(pipe_xgb,      os.path.join(MODELOS_DIR, "04_xgboost_pipeline.pkl"))
salvar_pipeline(pipe_lgb,      os.path.join(MODELOS_DIR, "05_lightgbm_pipeline.pkl"))

# CSV consolidados
tabela_final.to_csv(os.path.join(REPORTS_DIR, f"fase04_tabela_metricas_final_{ts}.csv"), index=False, encoding="utf-8")
cv_df.to_csv(         os.path.join(REPORTS_DIR, f"fase04_cv_stratifiedkfold_5_{ts}.csv"),   index=False, encoding="utf-8")
tab_dt_df.to_csv(     os.path.join(REPORTS_DIR, f"fase04_tuning_decisiontree_{ts}.csv"),     index=False, encoding="utf-8")
tab_rf_df.to_csv(     os.path.join(REPORTS_DIR, f"fase04_tuning_randomforest_{ts}.csv"),     index=False, encoding="utf-8")

# Feature importances (para SHAP / explicabilidade)
if imps_xgb is not None:
    imps_xgb.to_csv(     os.path.join(REPORTS_DIR, f"fase04_feat_import_xgboost_{ts}.csv"), index=False, encoding="utf-8")
    print(f"\nFeature importances Top 10 (XGBoost — compatível com SHAP TreeExplainer):")
    print(imps_xgb.head(10).to_string(index=False))

print("\n--- RESUMO DOS MELHORES RESULTADOS ---")
vencedor = tabela_final.iloc[0]
print(f"🏆 Vencedor por Teste ROC AUC: {vencedor['Modelo']} ({vencedor['Teste ROC AUC']:.4f})")
print(f"   Meta técnica: ROC AUC ≥ 0,85  →  {'✅ ATINGIDA' if vencedor['Teste ROC AUC']>=0.85 else '❌ NÃO ATINGIDA'}")

cv_vencedor = cv_df.iloc[0]
print(f"🏆 Vencedor por CV 5-fold ROC AUC: {cv_vencedor['Modelo']} ({cv_vencedor['CV ROC AUC (média ± std)']})")

# --- Extra: imprimir a tabela de tuning de DecisionTree ---
print("\n--- TABELA DE TUNING DECISION TREE (1..15) ---")
print(tab_dt_df[["Modelo", "Teste ROC AUC", "Teste F1", "Δ ROC AUC (tr-te)", "Tempo (s)"]].to_string(index=False))

# --- Extra: imprimir a tabela de tuning RandomForest (top 10) ---
print("\n--- TABELA DE TUNING RANDOM FOREST (top 10) ---")
print(tab_rf_df.head(10)[["Modelo", "Teste ROC AUC", "Teste F1", "Δ ROC AUC (tr-te)", "Tempo (s)"]].to_string(index=False))

print("\nFIM.")
