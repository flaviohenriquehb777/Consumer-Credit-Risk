"""
Fase 5b — Decisão do limiar: CUSTO EM PESSOAS (além de R$).
Gera tabelas que respondem, em DETALHE:
  • Quantas pessoas o modelo negou (FP e TP)? E aprovou?
  • % de PESSOAS QUE NÃO DARIAM CALOTE QUE FORAM NEGADAS (FP/Total verdadeiros adimplentes)
  • % de PESSOAS QUE DARIAM CALOTE QUE FORAM NEGADAS  (TP/Total verdadeiros inadimplentes = Recall)
  • Custo R$ de cada categoria (FN, FP) e soma total.

Tudo aprendido SÓ no TREINO OOF.
Depois validação cega UMA VEZ no HOLDOUT (mesmos limiares do treino OOF).
"""
from __future__ import annotations
import os, sys, pickle, warnings
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if SRC not in sys.path: sys.path.insert(0, SRC)
from features.preprocessamento import SEED_DEFAULT, carregar_params
from features.pipeline_modelo import montar_pipeline
warnings.filterwarnings("ignore"); np.random.seed(SEED_DEFAULT)

BASE    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC    = os.path.join(BASE, "data", "processed"); MODELOS = os.path.join(BASE, "models")
REP     = os.path.join(BASE, "reports", "evaluations")
PARAMS  = carregar_params(os.path.join(MODELOS, "preprocessamento_params.pkl"))
CUSTO_FN, CUSTO_FP = 5_000, 500
with open(os.path.join(MODELOS, "thresholds_otimos_custo_REAIS_FN5000_FP500.pkl"),"rb") as f:
    TH = pickle.load(f)
LIMIAR_DT, LIMIAR_XGB = TH["DecisionTree best"], TH["XGBoost best"]
print(f"Limiares vindos do pickle do OOF treino: DT={LIMIAR_DT}  XGB={LIMIAR_XGB}  "
      f"(FN R${CUSTO_FN}, FP R${CUSTO_FP})")

X_train = pd.read_csv(os.path.join(PROC, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROC, "y_train.csv"))["inadimplente_2anos"].astype(int).values
X_test  = pd.read_csv(os.path.join(PROC, "X_test.csv"))
y_test  = pd.read_csv(os.path.join(PROC, "y_test.csv"))["inadimplente_2anos"].astype(int).values
NEG_TRAIN, POS_TRAIN = int((y_train==0).sum()), int((y_train==1).sum())
NEG_TEST,  POS_TEST  = int((y_test ==0).sum()), int((y_test ==1).sum())
SPW = round(NEG_TRAIN / POS_TRAIN, 2)

# 1. OOF 5-fold no TREINO (DT e XGB)
BEST_DT_DEPTH = 7
BEST_XGB_ = dict(max_depth=4, learning_rate=0.03, min_child_weight=80,
                 n_estimators=500, subsample=0.9, colsample_bytree=0.85,
                 reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=SPW,
                 random_state=SEED_DEFAULT, n_jobs=-1, eval_metric="auc", tree_method="hist")
def modelo_dt(): return DecisionTreeClassifier(max_depth=BEST_DT_DEPTH, random_state=SEED_DEFAULT, class_weight=None, min_samples_leaf=50)
def modelo_xgb(): return xgb.XGBClassifier(**BEST_XGB_)

def oof_proba(modelo_base):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED_DEFAULT)
    oof = np.zeros(len(y_train), dtype=float)
    for tr_idx, va_idx in skf.split(X_train, y_train):
        pipe = montar_pipeline(deepcopy(modelo_base), PARAMS)
        pipe.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof[va_idx] = pipe.predict_proba(X_train.iloc[va_idx])[:,1]
    return oof
OOF_DT  = oof_proba(modelo_dt())
OOF_XGB = oof_proba(modelo_xgb())

# 2. Função que gera a tabela "custo em pessoas"
def tabela_pessoas(nome_cenario: str, y_true: np.ndarray, proba: np.ndarray, limiar: float,
                   neg_total: int, pos_total: int, nome_modelo: str):
    pred = (proba >= limiar).astype(int)
    tn, fp, fn, tp = [int(x) for x in confusion_matrix(y_true, pred, labels=[0,1]).ravel()]
    n_total = len(y_true)
    # Categorias "pessoas":
    # Verdadeiros Adimplentes    (TN + FP) = pessoas que na realidade pagariam
    # Verdadeiros Inadimplentes  (TP + FN) = pessoas que na realidade calotariam
    # Dentre os verdadeiros adimplentes:  quantas % foram NEGADAS (FP)?
    # Dentre os verdadeiros inadimplentes:quantas % foram NEGADAS (TP)?
    negados_total               = fp + tp
    aprovados_total             = tn + fn
    negados_que_nao_dariam_calote = fp     # Falsos positivos = negados adimplentes
    negados_que_dariam_calote     = tp     # Verdadeiros positivos = negados inadimplentes
    return {
        "Cenário (base)": nome_cenario,
        "Modelo": nome_modelo,
        "Limiar aplicado": f"{limiar:.2f}",
        "Total de clientes avaliados (N)": f"{n_total:,}",
        # --- Pessoas por realidade real ---
        "Adimplentes de verdade (TOTAL = TN+FP)":  f"{neg_total:,}",
        "Inadimplentes de verdade (TOTAL = TP+FN)": f"{pos_total:,}",
        # --- Pessoas negadas (o que o modelo negou crédito) ---
        "Negou crédito (Total)": f"{negados_total:,}",
        "% Base total negada":   f"{100*negados_total/n_total:.2f}%",
        "Pessoas NÃO dariam calote MAS foram NEGADAS (FP)": f"{negados_que_nao_dariam_calote:,}",
        "% de TODOS os adimplentes que foram NEGADOS (FP/verdadeiros adimplentes)": f"{100*negados_que_nao_dariam_calote/neg_total:.2f}%",
        "Pessoas QUE dariam calote E foram NEGADAS (TP — recall)": f"{negados_que_dariam_calote:,}",
        "% de TODOS os inadimplentes que foram NEGADOS (TP/verdadeiros inadimplentes)": f"{100*negados_que_dariam_calote/pos_total:.2f}%",
        # --- Pessoas aprovadas ---
        "Aprovou crédito (Total)": f"{aprovados_total:,}",
        "% Base total aprovada": f"{100*aprovados_total/n_total:.2f}%",
        "Aprovados que dariam calote (FN — risco vivo)": f"{fn:,}",
        "% Inadimplência real da carteira aprovada (FN/aprovados)": f"{100*fn/(aprovados_total if aprovados_total else np.nan):.2f}%",
        # --- Custo R$ por categoria ---
        "Custo FN (perda principal R$)": f"R$ {int(CUSTO_FN*fn):,.2f}",
        "Custo FP (perda margem R$)":    f"R$ {int(CUSTO_FP*fp):,.2f}",
        "Custo TOTAL (FN×5.000 + FP×500 R$)": f"R$ {int(CUSTO_FN*fn + CUSTO_FP*fp):,.2f}",
        "Custo por cliente (R$)": f"R$ {round((CUSTO_FN*fn + CUSTO_FP*fp)/n_total,2):,.2f}",
    }

# 3. TREINO OOF — 4 linhas: DT, XGB, Política Atual (aprova todos = limiar > max(proba)), Nega todos (limiar 0)
linhas = []
linhas.append(tabela_pessoas("TREINO OOF (n=112.500)", y_train, OOF_DT,  LIMIAR_DT,  NEG_TRAIN, POS_TRAIN, f"DecisionTree best (d={BEST_DT_DEPTH})"))
linhas.append(tabela_pessoas("TREINO OOF (n=112.500)", y_train, OOF_XGB, LIMIAR_XGB, NEG_TRAIN, POS_TRAIN, "XGBoost best (md=4 lr=0.03 mcw=80)"))
# Política atual "aprova todos" → limiar > max(prob) = sempre pred 0 → todos aprovados, FP=0, TP=0, FN=POS_TRAIN, TN=NEG_TRAIN
linhas.append(tabela_pessoas("TREINO OOF (n=112.500)", y_train, OOF_XGB, 999.0, NEG_TRAIN, POS_TRAIN, "🔥 Política ATUAL (APROVA TODOS)"))
# Política "nega todos" → limiar 0 → sempre pred 1 → FN=0, TN=0, TP=POS_TRAIN, FP=NEG_TRAIN
linhas.append(tabela_pessoas("TREINO OOF (n=112.500)", y_train, OOF_XGB, 0.0,   NEG_TRAIN, POS_TRAIN, "❌ Política: NEGAR TODOS"))

# 4. HOLDOUT (validação cega UMA VEZ — usa os mesmos limiares do OOF treino)
with open(os.path.join(MODELOS, "02_decisiontree_best_pipeline.pkl"),"rb") as f: pipe_dt  = pickle.load(f)
with open(os.path.join(MODELOS, "04_xgboost_pipeline.pkl"),"rb") as f:          pipe_xgb = pickle.load(f)
PROBA_TEST_DT  = pipe_dt.predict_proba(X_test)[:,1]
PROBA_TEST_XGB = pipe_xgb.predict_proba(X_test)[:,1]
linhas.append(tabela_pessoas("HOLDOUT (n=37.500) — validação cega", y_test, PROBA_TEST_DT,  LIMIAR_DT,  NEG_TEST, POS_TEST, f"DecisionTree best (d={BEST_DT_DEPTH})"))
linhas.append(tabela_pessoas("HOLDOUT (n=37.500) — validação cega", y_test, PROBA_TEST_XGB, LIMIAR_XGB, NEG_TEST, POS_TEST, "XGBoost best (md=4 lr=0.03 mcw=80)"))
linhas.append(tabela_pessoas("HOLDOUT (n=37.500) — validação cega", y_test, PROBA_TEST_XGB, 999.0, NEG_TEST, POS_TEST, "🔥 Política ATUAL (APROVA TODOS)"))
linhas.append(tabela_pessoas("HOLDOUT (n=37.500) — validação cega", y_test, PROBA_TEST_XGB, 0.0,   NEG_TEST, POS_TEST, "❌ Política: NEGAR TODOS"))

TAB = pd.DataFrame(linhas)
print("\n" + "=" * 230)
print("TABELA: Preço da decisão EM PESSOAS + custo R$ por categoria — 4 cenários")
print("=" * 230)
# Pretty print (quebra em 2 blocos: colunas pessoas, depois colunas custo R$)
cols_pessoas_1 = ["Cenário (base)","Modelo","Limiar aplicado","Total de clientes avaliados (N)",
                  "Adimplentes de verdade (TOTAL = TN+FP)","Inadimplentes de verdade (TOTAL = TP+FN)",
                  "Negou crédito (Total)","% Base total negada",
                  "Pessoas NÃO dariam calote MAS foram NEGADAS (FP)",
                  "% de TODOS os adimplentes que foram NEGADOS (FP/verdadeiros adimplentes)",
                  "Pessoas QUE dariam calote E foram NEGADAS (TP — recall)",
                  "% de TODOS os inadimplentes que foram NEGADOS (TP/verdadeiros inadimplentes)"]
cols_pessoas_2 = ["Aprovou crédito (Total)","% Base total aprovada",
                  "Aprovados que dariam calote (FN — risco vivo)",
                  "% Inadimplência real da carteira aprovada (FN/aprovados)"]
cols_custo     = ["Custo FN (perda principal R$)","Custo FP (perda margem R$)",
                  "Custo TOTAL (FN×5.000 + FP×500 R$)","Custo por cliente (R$)"]

print("\nBLOCO 1 — Visão em PESSOAS (negação)")
print("-"*230)
print(TAB[cols_pessoas_1].to_string(index=False))
print("\nBLOCO 2 — Visão em PESSOAS (aprovação)")
print("-"*230)
print(TAB[cols_pessoas_2].to_string(index=False))
print("\nBLOCO 3 — Custo R$ por categoria (FN e FP)")
print("-"*230)
print(TAB[cols_custo].to_string(index=False))

# Salvando CSV completo
TAB.to_csv(os.path.join(REP, "fase05b_tabela_preco_da_decisao_em_pessoas.csv"),
           index=False, encoding="utf-8", sep=";")
print(f"\nArquivo salvo: reports/evaluations/fase05b_tabela_preco_da_decisao_em_pessoas.csv")

# --- Resumo final das duas %s pedidas pelo usuário (TREINO OOF e HOLDOUT) ---
def print_resumo(cenario_filtro):
    print(f"\n--- Resumo ({cenario_filtro}) das duas %s que você pediu ---")
    sub = TAB[TAB["Cenário (base)"].str.contains(cenario_filtro)]
    for _, r in sub.iterrows():
        print(f"  ▸ {r['Modelo']:<40s}  "
              f"% adimplentes NEGADOS (FP/tot adimpl.) = {r['% de TODOS os adimplentes que foram NEGADOS (FP/verdadeiros adimplentes)']:<8s} | "
              f"% inadimplentes NEGADOS (recall) = {r['% de TODOS os inadimplentes que foram NEGADOS (TP/verdadeiros inadimplentes)']:<8s} | "
              f"Custo FN R$ = {r['Custo FN (perda principal R$)']:<22s} | "
              f"Custo FP R$ = {r['Custo FP (perda margem R$)']:<22s} | "
              f"Total R$ = {r['Custo TOTAL (FN×5.000 + FP×500 R$)']}")
print_resumo("TREINO OOF")
print_resumo("HOLDOUT")
