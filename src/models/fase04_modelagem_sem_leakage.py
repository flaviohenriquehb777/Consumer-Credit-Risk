"""
Fase 4 Modelagem — CORREÇÃO (padrão sênior / SEM LEAKAGE):

   TODO O TUNING E SELEÇÃO É FEITO SÓ COM DADOS DE TREINO.
   O HOLDOUT (TESTE) SÓ É USADO UMA ÚNICA VEZ, NO PASSO FINAL (batismo).

Pipeline:
   0. Carrega X_train/y_train e separa o TESTE (X_test/y_test) — este é guardado a sete chaves,
      só é aberto na ÚLTIMA etapa.
   1. Tuning DecisionTree max_depth=1..15 → StratifiedKFold 5 no TREINO
      → medir CV OOF: (a) média ROC AUC, (b) média FN×10 + FP×1, (c) média acurácia
   2. Tuning RandomForest grid depth × class_weight → StratifiedKFold 5 no TREINO
   3. Tuning XGBoost → grid pequeno (max_depth, lr, min_child_weight) → StratifiedKFold 5 no TREINO
   4. Tuning LightGBM → grid pequeno → StratifiedKFold 5 no TREINO
   5. Escolhe "melhores hparams POR MODELO" (vencedor de cada família) com base na média de
      ROC AUC do CV OOF.
   6. Refit CADA modelo final em 100% do TREINO usando esses hparams.
   7. ÚLTIMO PASSO: uma única avaliação no HOLDOUT (teste) com TODAS as métricas + custo 10:1,
      inclusive threshold ótimo calculado a partir das predições OOF do treino (nunca do teste).
   8. Salva todos os artefatos: pickles finais e CSVs de tuning (só métricas CV).

SEED = 42
"""
from __future__ import annotations

import os
import sys
import pickle
import time
import warnings
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from features.preprocessamento import carregar_params, SEED_DEFAULT
from features.pipeline_modelo import montar_pipeline, salvar_pipeline, obter_nomes_features

warnings.filterwarnings("ignore")
np.random.seed(SEED_DEFAULT)

# --- Paths ---
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC    = os.path.join(BASE, "data", "processed")
MODELOS = os.path.join(BASE, "models")
REP     = os.path.join(BASE, "reports", "evaluations")
FIG     = os.path.join(BASE, "reports", "figures")
PARAMS  = carregar_params(os.path.join(MODELOS, "preprocessamento_params.pkl"))
os.makedirs(MODELOS, exist_ok=True)
os.makedirs(REP, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

X_train = pd.read_csv(os.path.join(PROC, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROC, "y_train.csv"))["inadimplente_2anos"].astype(int).values
X_test  = pd.read_csv(os.path.join(PROC, "X_test.csv"))
y_test  = pd.read_csv(os.path.join(PROC, "y_test.csv"))["inadimplente_2anos"].astype(int).values

NEG, POS = int((y_train == 0).sum()), int((y_train == 1).sum())
SPW = round(NEG / POS, 2)  # ≈ 13.96
CUSTO_FN, CUSTO_FP = 10, 1
TS = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"[INFO] TREINO: {X_train.shape[0]:,} (POS={POS:,}, NEG={NEG:,})")
print(f"[INFO] HOLDOUT (teste) = {X_test.shape[0]:,}  —  SÓ USADO NO ÚLTIMO PASSO.")
print(f"[INFO] scale_pos_weight (treino) = {SPW}. Custo FN={CUSTO_FN}, FP={CUSTO_FP}.")
print("=" * 120)


# ======================================================================
# Helpers
# ======================================================================
def calcular_custo(y_true, y_pred):
    _tn, _fp, _fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(CUSTO_FN * _fn + CUSTO_FP * _fp), int(_fn), int(_fp)


def cv_oof_scores(nome, modelo_base, grid_linha: dict):
    """5-fold OOF no TREINO, devolve média + std das métricas por fold."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED_DEFAULT)
    folds_roc, folds_f1, folds_acc, folds_custo = [], [], [], []
    oof_proba = np.zeros(len(y_train), dtype=float)
    oof_pred  = np.zeros(len(y_train), dtype=int)

    for tr_idx, va_idx in skf.split(X_train, y_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        pipe = montar_pipeline(deepcopy(modelo_base), PARAMS)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:, 1]
        pred  = pipe.predict(X_va)
        c, _fn, _fp = calcular_custo(y_va, pred)
        folds_roc.append(roc_auc_score(y_va, proba))
        folds_f1.append(f1_score(y_va, pred, zero_division=0))
        folds_acc.append(accuracy_score(y_va, pred))
        folds_custo.append(c)
        oof_proba[va_idx] = proba
        oof_pred[va_idx]  = pred

    return {
        "Modelo": nome,
        **grid_linha,
        "CV ROC AUC (média)": round(float(np.mean(folds_roc)), 5),
        "CV ROC AUC (std)":   round(float(np.std(folds_roc)),  5),
        "CV F1 (média)":      round(float(np.mean(folds_f1)),   5),
        "CV F1 (std)":        round(float(np.std(folds_f1)),    5),
        "CV Acurácia (média)": round(float(np.mean(folds_acc)),5),
        "CV Custo Total OOF (média por fold)": round(float(np.mean(folds_custo)), 2),
        "CV Custo Total OOF (5 folds somado)": int(np.sum(folds_custo)),
        "OOF ROC AUC (sobre as predições agregadas)": round(float(roc_auc_score(y_train, oof_proba)), 5),
        "OOF F1 (agregado)":  round(float(f1_score(y_train, oof_pred, zero_division=0)), 5),
        "OOF Accuracy (agregado)": round(float(accuracy_score(y_train, oof_pred)), 5),
        "OOF proba": oof_proba,  # guardamos para o threshold ótimo
        "OOF pred":  oof_pred,
    }


# ======================================================================
# 1. DecisionTree tuning
# ======================================================================
print("\n[1/4] CV 5-fold DecisionTree — depth 1..15 (TREINO APENAS)")
rows_dt = []
for d in range(1, 16):
    linha = {"max_depth": d, "class_weight": None, "min_samples_leaf": 50}
    modelo = DecisionTreeClassifier(
        max_depth=d, random_state=SEED_DEFAULT,
        class_weight=None, min_samples_leaf=50)
    rows_dt.append(cv_oof_scores(f"DecisionTree d={d}", modelo, linha))

tab_dt = pd.DataFrame([{k:v for k,v in r.items() if k not in ("OOF proba","OOF pred")} for r in rows_dt])
tab_dt = tab_dt.sort_values("CV ROC AUC (média)", ascending=False).reset_index(drop=True)
best_dt_row = rows_dt[int(np.argmax([r["CV ROC AUC (média)"] for r in rows_dt]))]
BEST_DT_DEPTH = int(best_dt_row["max_depth"])
print(f"  🏆 Melhor DecisionTree por média CV ROC AUC: depth={BEST_DT_DEPTH} "
      f"(média={best_dt_row['CV ROC AUC (média)']:.5f}  std={best_dt_row['CV ROC AUC (std)']:.5f})")
print("  Top 5 DT:")
cols_dt = ["max_depth","CV ROC AUC (média)","CV ROC AUC (std)","CV Custo Total OOF (5 folds somado)","OOF ROC AUC (sobre as predições agregadas)"]
print(tab_dt[cols_dt].head().to_string(index=False))

# ======================================================================
# 2. RandomForest tuning
# ======================================================================
print("\n[2/4] CV 5-fold RandomForest — grid 6 depths × 3 class_weights")
depths = [6, 8, 10, 12, 15, None]
cw_grid = [
    ("balanced", "balanced"),
    ("balanced_subs", "balanced_subsample"),
    ("balanced_1_14", {0:1, 1:14}),
]
rows_rf = []
for d in depths:
    for cw_label, cw_val in cw_grid:
        linha = {"max_depth": str(d), "class_weight": cw_label,
                 "n_estimators": 300, "min_samples_leaf": 30}
        modelo = RandomForestClassifier(
            n_estimators=300, max_depth=d, class_weight=cw_val,
            n_jobs=-1, random_state=SEED_DEFAULT, min_samples_leaf=30, oob_score=False)
        rows_rf.append(cv_oof_scores(f"RF d={d} cw={cw_label}", modelo, linha))

tab_rf = pd.DataFrame([{k:v for k,v in r.items() if k not in ("OOF proba","OOF pred")} for r in rows_rf])
tab_rf = tab_rf.sort_values("CV ROC AUC (média)", ascending=False).reset_index(drop=True)
best_rf_row = rows_rf[int(np.argmax([r["CV ROC AUC (média)"] for r in rows_rf]))]
BEST_RF_DEPTH_S = best_rf_row["max_depth"]
BEST_RF_DEPTH = (None if BEST_RF_DEPTH_S.strip() == "None" else int(BEST_RF_DEPTH_S))
BEST_RF_CW_LABEL = best_rf_row["class_weight"]
BEST_RF_CW = [v for (l, v) in cw_grid if l == BEST_RF_CW_LABEL][0]
print(f"  🏆 Melhor RandomForest: depth={BEST_RF_DEPTH_S}, class_weight={BEST_RF_CW_LABEL} "
      f"(média CV ROC AUC = {best_rf_row['CV ROC AUC (média)']:.5f})")
print("  Top 5 RF:")
cols_rf = ["max_depth","class_weight","CV ROC AUC (média)","CV ROC AUC (std)","CV Custo Total OOF (5 folds somado)"]
print(tab_rf[cols_rf].head().to_string(index=False))

# ======================================================================
# 3. XGBoost tuning (grid pequeno: depth × lr × min_child_weight)
# ======================================================================
print("\n[3/4] CV 5-fold XGBoost — grid depth(4,5,6) × lr(0.03,0.05,0.07) × mcw(80,100)")
xgb_grid = []
for md in (4, 5, 6):
    for lr in (0.03, 0.05, 0.07):
        for mcw in (80, 100):
            xgb_grid.append((md, lr, mcw))
rows_xgb = []
for (md, lr, mcw) in xgb_grid:
    linha = {"max_depth": md, "learning_rate": lr, "min_child_weight": mcw, "scale_pos_weight": SPW,
             "n_estimators": 500, "subsample": 0.9, "colsample": 0.85}
    modelo = xgb.XGBClassifier(
        n_estimators=500, max_depth=md, learning_rate=lr, min_child_weight=mcw,
        subsample=0.9, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=SPW, random_state=SEED_DEFAULT, n_jobs=-1,
        eval_metric="auc", tree_method="hist")
    rows_xgb.append(cv_oof_scores(f"XGB md={md} lr={lr} mcw={mcw}", modelo, linha))

tab_xgb = pd.DataFrame([{k:v for k,v in r.items() if k not in ("OOF proba","OOF pred")} for r in rows_xgb])
tab_xgb = tab_xgb.sort_values("CV ROC AUC (média)", ascending=False).reset_index(drop=True)
best_xgb_row = rows_xgb[int(np.argmax([r["CV ROC AUC (média)"] for r in rows_xgb]))]
BEST_XGB = dict(max_depth=int(best_xgb_row["max_depth"]),
                learning_rate=float(best_xgb_row["learning_rate"]),
                min_child_weight=int(best_xgb_row["min_child_weight"]))
print(f"  🏆 Melhor XGBoost: max_depth={BEST_XGB['max_depth']}  lr={BEST_XGB['learning_rate']}  "
      f"min_child_weight={BEST_XGB['min_child_weight']}  (média CV ROC AUC={best_xgb_row['CV ROC AUC (média)']:.5f})")
print("  Top 5 XGB:")
cols_xgb = ["max_depth","learning_rate","min_child_weight","CV ROC AUC (média)","CV ROC AUC (std)"]
print(tab_xgb[cols_xgb].head().to_string(index=False))

# ======================================================================
# 4. LightGBM tuning (grid pequeno)
# ======================================================================
print("\n[4/4] CV 5-fold LightGBM — grid md(5,6,7) × lr(0.03,0.05,0.07) × mcs(100,150)")
lgb_grid = []
for md in (5, 6, 7):
    for lr in (0.03, 0.05, 0.07):
        for mcs in (100, 150):
            lgb_grid.append((md, lr, mcs))
rows_lgb = []
for (md, lr, mcs) in lgb_grid:
    linha = {"max_depth": md, "learning_rate": lr, "min_child_samples": mcs,
             "scale_pos_weight": SPW, "num_leaves": 2**md - 1, "n_estimators": 500}
    modelo = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=lr, max_depth=md, num_leaves=2**md - 1,
        min_child_samples=mcs, subsample=0.9, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=SPW,
        random_state=SEED_DEFAULT, n_jobs=-1, verbose=-1)
    rows_lgb.append(cv_oof_scores(f"LGB md={md} lr={lr} mcs={mcs}", modelo, linha))

tab_lgb = pd.DataFrame([{k:v for k,v in r.items() if k not in ("OOF proba","OOF pred")} for r in rows_lgb])
tab_lgb = tab_lgb.sort_values("CV ROC AUC (média)", ascending=False).reset_index(drop=True)
best_lgb_row = rows_lgb[int(np.argmax([r["CV ROC AUC (média)"] for r in rows_lgb]))]
BEST_LGB = dict(max_depth=int(best_lgb_row["max_depth"]),
                learning_rate=float(best_lgb_row["learning_rate"]),
                min_child_samples=int(best_lgb_row["min_child_samples"]),
                num_leaves=2**int(best_lgb_row["max_depth"]) - 1)
print(f"  🏆 Melhor LightGBM: max_depth={BEST_LGB['max_depth']}  lr={BEST_LGB['learning_rate']}  "
      f"min_child_samples={BEST_LGB['min_child_samples']}  (média CV ROC AUC={best_lgb_row['CV ROC AUC (média)']:.5f})")
print("  Top 5 LGB:")
cols_lgb = ["max_depth","learning_rate","min_child_samples","CV ROC AUC (média)","CV ROC AUC (std)"]
print(tab_lgb[cols_lgb].head().to_string(index=False))

# ======================================================================
# 5. Tabela comparativa final CV (TREINO APENAS)
# ======================================================================
print("\n" + "=" * 120)
print("RANKING FINAL — CV 5-fold NO TREINO (melhores hparams por família)")
print("=" * 120)

def modelo_final_xgb():
    return xgb.XGBClassifier(n_estimators=500, max_depth=BEST_XGB["max_depth"],
                             learning_rate=BEST_XGB["learning_rate"],
                             min_child_weight=BEST_XGB["min_child_weight"],
                             subsample=0.9, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
                             scale_pos_weight=SPW, random_state=SEED_DEFAULT, n_jobs=-1,
                             eval_metric="auc", tree_method="hist")
def modelo_final_lgb():
    return lgb.LGBMClassifier(n_estimators=500, max_depth=BEST_LGB["max_depth"],
                              learning_rate=BEST_LGB["learning_rate"],
                              num_leaves=BEST_LGB["num_leaves"],
                              min_child_samples=BEST_LGB["min_child_samples"],
                              subsample=0.9, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
                              scale_pos_weight=SPW, random_state=SEED_DEFAULT, n_jobs=-1, verbose=-1)
def modelo_final_rf():
    return RandomForestClassifier(n_estimators=300, max_depth=BEST_RF_DEPTH,
                                  class_weight=BEST_RF_CW, n_jobs=-1,
                                  random_state=SEED_DEFAULT, min_samples_leaf=30, oob_score=False)
def modelo_final_dt():
    return DecisionTreeClassifier(max_depth=BEST_DT_DEPTH, random_state=SEED_DEFAULT,
                                  class_weight=None, min_samples_leaf=50)
def modelo_final_dummy():
    return DummyClassifier(strategy="prior", random_state=SEED_DEFAULT)

cv_familia = pd.DataFrame([
    {k:v for k,v in best_dt_row.items() if k not in ("OOF proba","OOF pred")},
    {k:v for k,v in best_rf_row.items() if k not in ("OOF proba","OOF pred")},
    {k:v for k,v in best_xgb_row.items() if k not in ("OOF proba","OOF pred")},
    {k:v for k,v in best_lgb_row.items() if k not in ("OOF proba","OOF pred")},
])
# Dummy entra também
cv_dummy = cv_oof_scores("Dummy prior", modelo_final_dummy(), {"strategy":"prior"})
cv_familia = pd.concat([pd.DataFrame([{k:v for k,v in cv_dummy.items() if k not in ("OOF proba","OOF pred")}]),
                        cv_familia], ignore_index=True)
cv_familia = cv_familia.sort_values("CV ROC AUC (média)", ascending=False).reset_index(drop=True)
print(cv_familia.to_string(index=False))

# ======================================================================
# 6. Refit final em 100% TREINO com os hparams vencedores do CV.
# ======================================================================
print("\nRefit final 100% TREINO (melhores hparams de cada modelo)...")
def fit_final(nome, constr):
    pipe = montar_pipeline(constr(), PARAMS)
    pipe.fit(X_train, y_train)
    return pipe

pipe_dummy = fit_final("Dummy", modelo_final_dummy)
pipe_dt    = fit_final("DT",    modelo_final_dt)
pipe_rf    = fit_final("RF",    modelo_final_rf)
pipe_xgb   = fit_final("XGB",   modelo_final_xgb)
pipe_lgb   = fit_final("LGB",   modelo_final_lgb)
pipes = {"Dummy (prior)": pipe_dummy,
         f"DecisionTree best (d={BEST_DT_DEPTH})": pipe_dt,
         f"RandomForest best (d={BEST_RF_DEPTH_S}, cw={BEST_RF_CW_LABEL})": pipe_rf,
         f"XGBoost best ({BEST_XGB})": pipe_xgb,
         f"LightGBM best ({BEST_LGB})": pipe_lgb}

# --- Threshold ótimo calculado SÓ a partir do OOB do treino (NUNCA do teste) ---
print("\nComputando threshold ótimo (custo 10:1) a partir do OOF treino, por modelo.")

def melhor_threshold_oof(oof_proba, y_true):
    ths = np.linspace(0.01, 0.99, 9801)
    mc, mt = 1e18, None
    for th in ths:
        pred = (oof_proba >= th).astype(int)
        c, *_ = calcular_custo(y_true, pred)
        if c < mc:
            mc = c
            mt = th
    return round(float(mt), 3), int(mc)

# Para os vencedores do CV com OOF proba:
th_dict, custo_oof_dict = {}, {}
for (nome, best_row) in [
    ("DecisionTree best", best_dt_row),
    ("RandomForest best", best_rf_row),
    ("XGBoost best", best_xgb_row),
    ("LightGBM best", best_lgb_row),
]:
    th, c_oof = melhor_threshold_oof(best_row["OOF proba"], y_train)
    th_dict[nome] = th
    custo_oof_dict[nome] = c_oof
    print(f"  · {nome}  →  threshold ótimo (treino OOF) = {th}  |  custo OOF somado = {c_oof}")

# ======================================================================
# 7. UMA ÚNICA AVALIAÇÃO NO HOLDOUT (TESTE)  —  BATISMO.
# ======================================================================
print("\n" + "=" * 120)
print("🥇🥈🥉  BATISMO  DO  HOLDOUT  (única vez que o teste é lido para métricas)  🥇🥈🥉")
print("=" * 120)

holdout_rows = []
for nome, pipe in pipes.items():
    proba = pipe.predict_proba(X_test)[:, 1]
    pred_05  = (proba >= 0.5).astype(int)
    # Threshold ótimo (do treino OOF): só para os modelos que temos OOF
    nome_curto = ([k for k in th_dict if nome.startswith(k.split()[0])] + [None])[0]
    th_otimo = th_dict.get(nome_curto, 0.5)
    pred_ot  = (proba >= th_otimo).astype(int)

    # Cálculo baseline política atual para referência (uma única vez)
    politica_atual = np.zeros(len(y_test), dtype=int)
    c_pol, fn_pol, fp_pol = calcular_custo(y_test, politica_atual)

    c_05, fn_05, fp_05 = calcular_custo(y_test, pred_05)
    c_ot, fn_ot, fp_ot = calcular_custo(y_test, pred_ot)

    holdout_rows.append({
        "Modelo": nome,
        "Th ótimo (vindo do OOF treino)": th_otimo,
        "ROC AUC teste":        round(roc_auc_score(y_test, proba), 5),
        "PR AUC teste":         round(average_precision_score(y_test, proba), 5),
        "F1 teste (th=0.5)":    round(f1_score(y_test, pred_05, zero_division=0), 5),
        "Acurácia teste (0.5)": round(accuracy_score(y_test, pred_05), 5),
        "Brier teste":          round(brier_score_loss(y_test, proba), 5),
        "Custo FN*10+FP th=0.5": c_05,
        "FN th=0.5": fn_05, "FP th=0.5": fp_05,
        "Economia vs Pol. Atual (%, th=0.5)": round((1 - c_05 / c_pol) * 100, 2),
        "Custo FN*10+FP th ÓTIMO": c_ot,
        "FN th ÓTIMO": fn_ot, "FP th ÓTIMO": fp_ot,
        "Economia vs Pol. Atual (%, th ÓTIMO)": round((1 - c_ot / c_pol) * 100, 2),
    })

tab_holdout = pd.DataFrame(holdout_rows).sort_values("ROC AUC teste", ascending=False).reset_index(drop=True)
print(f"Política ATUAL (aprova todos) no holdout: CUSTO TOTAL = {c_pol}  (FN={fn_pol}  FP={fp_pol})")
print()
print(tab_holdout.to_string(index=False))

vencedor = tab_holdout.iloc[0]
print(f"\n🏆 VENCEDOR no HOLDOUT (ROC AUC): {vencedor['Modelo']} — ROC AUC = {vencedor['ROC AUC teste']:.5f}")
print(f"   Meta técnica ROC AUC ≥ 0,85:  {'✅ ATINGIDA' if vencedor['ROC AUC teste']>=0.85 else '❌ NÃO ATINGIDA'}")
print(f"   Economia de custo 10:1 vs política atual (threshold ótimo do treino): "
      f"{vencedor['Economia vs Pol. Atual (%, th ÓTIMO)']:.2f}%")

# ======================================================================
# 8. Persistência: CSV tuning (apenas métricas CV, NÃO contém holdout tuning)
#    + pickles dos modelos finais (refit 100% treino)
# ======================================================================
print("\nPersistindo artefatos...")
tab_dt.to_csv(   os.path.join(REP, f"fase04_cvtune_decisiontree_{TS}.csv"), index=False, encoding="utf-8")
tab_rf.to_csv(   os.path.join(REP, f"fase04_cvtune_randomforest_{TS}.csv"), index=False, encoding="utf-8")
tab_xgb.to_csv(  os.path.join(REP, f"fase04_cvtune_xgboost_{TS}.csv"),        index=False, encoding="utf-8")
tab_lgb.to_csv(  os.path.join(REP, f"fase04_cvtune_lightgbm_{TS}.csv"),       index=False, encoding="utf-8")
cv_familia.to_csv(os.path.join(REP, f"fase04_cv_final_ranking_{TS}.csv"),     index=False, encoding="utf-8")
tab_holdout.to_csv(os.path.join(REP, f"fase04_holdout_avaliacao_unica_{TS}.csv"), index=False, encoding="utf-8")

salvar_pipeline(pipe_dummy, os.path.join(MODELOS, "01_dummy_pipeline.pkl"))
salvar_pipeline(pipe_dt,    os.path.join(MODELOS, "02_decisiontree_best_pipeline.pkl"))
salvar_pipeline(pipe_rf,    os.path.join(MODELOS, "03_randomforest_best_pipeline.pkl"))
salvar_pipeline(pipe_xgb,   os.path.join(MODELOS, "04_xgboost_pipeline.pkl"))
salvar_pipeline(pipe_lgb,   os.path.join(MODELOS, "05_lightgbm_pipeline.pkl"))

# --- Threshold ótimo por modelo para a Fase 5 (será aplicado no Streamlit) ---
with open(os.path.join(MODELOS, "thresholds_otimos_custo_10_1.pkl"), "wb") as f:
    pickle.dump(th_dict, f)
print(f"  Thresholds ótimos (custo 10:1, calculados do OOF treino): {th_dict}")
print("  Pickle salvo em models/thresholds_otimos_custo_10_1.pkl")

print("\nFIM — todo o tuning foi feito SÓ em CV no TREINO; HOLDOUT usado UMA vez.")
