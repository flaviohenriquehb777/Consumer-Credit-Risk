"""Calcula custo esperado 10:1 (10*FN + 1*FP) para todos os modelos no HOLDOUT."""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, SRC)

PROC = os.path.join(SRC, "..", "data", "processed")
PROC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
MODELOS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))

X_test = pd.read_csv(os.path.join(PROC, "X_test.csv"))
y_test = pd.read_csv(os.path.join(PROC, "y_test.csv"))["inadimplente_2anos"].astype(int).values

PIPES = {
    "Dummy (prior)":            "01_dummy_pipeline.pkl",
    "DecisionTree (d=8)":       "02_decisiontree_best_pipeline.pkl",
    "RandomForest (best)":      "03_randomforest_best_pipeline.pkl",
    "XGBoost":                  "04_xgboost_pipeline.pkl",
    "LightGBM":                 "05_lightgbm_pipeline.pkl",
}

PESO_FN = 10
PESO_FP = 1

# ---------- Política atual (aprova todos) ----------
aprov_todos = np.zeros(len(y_test), dtype=int)  # sempre classe 0 (aprova)
tn, fp, fn, tp = confusion_matrix(y_test, aprov_todos, labels=[0,1]).ravel()
custo_atual = PESO_FN * fn + PESO_FP * fp
custo_por_cliente_atual = custo_atual / len(y_test)

# ---------- Política "negar todos" ----------
negar_todos = np.ones(len(y_test), dtype=int)
tn2, fp2, fn2, tp2 = confusion_matrix(y_test, negar_todos, labels=[0,1]).ravel()
custo_negar = PESO_FN * fn2 + PESO_FP * fp2

TABELA = []
for nome, arq in PIPES.items():
    with open(os.path.join(MODELOS, arq), "rb") as f:
        pipe = pickle.load(f)
    proba = pipe.predict_proba(X_test)[:, 1]

    # ---- A) Threshold DEFAULT (0,5) ----
    pred_05 = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_05, labels=[0,1]).ravel()
    custo_05 = PESO_FN * fn + PESO_FP * fp
    cpc_05   = custo_05 / len(y_test)
    acuracia = (tp + tn) / len(y_test)
    recall   = tp / max(1, (tp + fn))
    prec     = tp / max(1, (tp + fp))

    # ---- B) Threshold OTIMO (grid search minimiza custo 10:1 no HOLDOUT) ----
    ths = np.linspace(0.01, 0.99, 9801)
    melhor_custo = float("inf")
    melhor_th = None
    melhor_fn = melhor_fp = -1
    for th in ths:
        pred = (proba >= th).astype(int)
        _tn, _fp, _fn, _tp = confusion_matrix(y_test, pred, labels=[0,1], normalize=None).ravel()
        c = PESO_FN * _fn + PESO_FP * _fp
        if c < melhor_custo:
            melhor_custo = c
            melhor_th = th
            melhor_fn, melhor_fp = _fn, _fp

    TABELA.append({
        "Modelo": nome,
        "Th=0.5 — Custo Total":  int(custo_05),
        "Th=0.5 — Custo/cliente": round(cpc_05, 3),
        "Th=0.5 — N_FN":          int(fn),
        "Th=0.5 — N_FP":          int(fp),
        "Th=0.5 — Acurácia":      round(acuracia, 4),
        "Th=0.5 — Recall (pos)":  round(recall, 4),
        "Th=0.5 — Precision":     round(prec, 4),
        "Th=0.5 — vs Pol. Atual (economia %)": round((1 - custo_05/custo_atual)*100, 2),
        "Th OTIMO p/ custo 10:1": round(melhor_th, 3),
        "Th OTIMO — Custo Total": int(melhor_custo),
        "Th OTIMO — Custo/cliente": round(melhor_custo / len(y_test), 3),
        "Th OTIMO — N_FN": int(melhor_fn),
        "Th OTIMO — N_FP": int(melhor_fp),
        "Th OTIMO — vs Pol. Atual (economia %)": round((1 - melhor_custo/custo_atual)*100, 2),
    })

# ---------- Row: Política Atual (aprova todos = regras manuais baseline do CRISP) ----------
linha_atual = {
    "Modelo": "🔥 Política ATUAL (aprova TODOS — equivalente ao Dummy)",
    "Th=0.5 — Custo Total":  int(custo_atual),
    "Th=0.5 — Custo/cliente": round(custo_por_cliente_atual, 3),
    "Th=0.5 — N_FN":          int(fn),
    "Th=0.5 — N_FP":          int(fp),
    "Th=0.5 — Acurácia":      round((int(tn)+int(tp))/len(y_test), 4),
    "Th=0.5 — Recall (pos)":  round(int(tp)/max(1,(int(tp)+int(fn))), 4) if (int(tp)+int(fn)) else 0,
    "Th=0.5 — Precision":     round(int(tp)/max(1,(int(tp)+int(fp))), 4) if (int(tp)+int(fp)) else 0,
    "Th=0.5 — vs Pol. Atual (economia %)": 0.0,
    "Th OTIMO p/ custo 10:1": "—",
    "Th OTIMO — Custo Total": int(custo_atual),
    "Th OTIMO — Custo/cliente": round(custo_por_cliente_atual, 3),
    "Th OTIMO — N_FN": int(fn),
    "Th OTIMO — N_FP": int(fp),
    "Th OTIMO — vs Pol. Atual (economia %)": 0.0,
}
# Corrige valores reais da linha "política atual" (usou a última confusão matrix; sobrescreve)
tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_test, aprov_todos, labels=[0,1]).ravel()
linha_atual.update({
    "Th=0.5 — N_FN": int(fn_a), "Th=0.5 — N_FP": int(fp_a),
    "Th OTIMO — N_FN": int(fn_a), "Th OTIMO — N_FP": int(fp_a),
    "Th=0.5 — Custo Total": int(PESO_FN*fn_a + PESO_FP*fp_a),
    "Th=0.5 — Custo/cliente": round((PESO_FN*fn_a + PESO_FP*fp_a)/len(y_test), 3),
    "Th OTIMO — Custo Total": int(PESO_FN*fn_a + PESO_FP*fp_a),
    "Th OTIMO — Custo/cliente": round((PESO_FN*fn_a + PESO_FP*fp_a)/len(y_test), 3),
})
TABELA.insert(0, linha_atual)

# ---------- Row: Política "negar todos" ----------
linha_negar = {
    "Modelo": "Política extrema: NEGAR TODOS",
    "Th=0.5 — Custo Total":  int(custo_negar),
    "Th=0.5 — Custo/cliente": round(custo_negar/len(y_test), 3),
    "Th=0.5 — N_FN":          int(fn2),
    "Th=0.5 — N_FP":          int(fp2),
    "Th=0.5 — Acurácia":      round((tp2+tn2)/len(y_test), 4),
    "Th=0.5 — Recall (pos)":  round(tp2/max(1,(tp2+fn2)), 4),
    "Th=0.5 — Precision":     round(tp2/max(1,(tp2+fp2)), 4),
    "Th=0.5 — vs Pol. Atual (economia %)": round((1-custo_negar/custo_atual)*100, 2),
    "Th OTIMO p/ custo 10:1": "—",
    "Th OTIMO — Custo Total": int(custo_negar),
    "Th OTIMO — Custo/cliente": round(custo_negar/len(y_test), 3),
    "Th OTIMO — N_FN": int(fn2),
    "Th OTIMO — N_FP": int(fp2),
    "Th OTIMO — vs Pol. Atual (economia %)": round((1-custo_negar/custo_atual)*100, 2),
}
TABELA.insert(1, linha_negar)

TAB = pd.DataFrame(TABELA)
print(f"Tamanho HOLDOUT = {len(y_test)} clientes.")
print(f"PESO_FN = {PESO_FN}, PESO_FP = {PESO_FP}.")
print(f"Política 'aprova todos' — CUSTO TOTAL = {int(custo_atual)} —  "
      f"FN={int(fn_a)} ×10 + FP={int(fp_a)} ×1")
print(f"Política 'negar  todos' — CUSTO TOTAL = {int(custo_negar)} —  "
      f"FN={int(fn2)} ×10 + FP={int(fp2)} ×1")
print()
print("=" * 170)
print("TABELA A: Todos os modelos. Threshold 0,5 (padrão)")
print("=" * 170)
cols_a = ["Modelo","Th=0.5 — Custo Total","Th=0.5 — Custo/cliente",
          "Th=0.5 — N_FN","Th=0.5 — N_FP","Th=0.5 — Recall (pos)",
          "Th=0.5 — Precision","Th=0.5 — vs Pol. Atual (economia %)"]
print(TAB[cols_a].sort_values("Th=0.5 — Custo Total", ascending=True).to_string(index=False))

print()
print("=" * 170)
print("TABELA B: Todos os modelos com THRESHOLD OTIMIZADO para custo 10:1")
print("=" * 170)
cols_b = ["Modelo", "Th OTIMO p/ custo 10:1",
          "Th OTIMO — Custo Total","Th OTIMO — Custo/cliente",
          "Th OTIMO — N_FN","Th OTIMO — N_FP",
          "Th OTIMO — vs Pol. Atual (economia %)"]
print(TAB[cols_b].sort_values("Th OTIMO — Custo Total", ascending=True).to_string(index=False))

# Ranking final
vencedor_A = TAB.sort_values("Th=0.5 — Custo Total").iloc[0]["Modelo"]
vencedor_B_row = TAB[TAB["Th OTIMO — Custo Total"] == TAB.loc[TAB["Modelo"].str.startswith("🔥")==False, "Th OTIMO — Custo Total"].min()].iloc[0]
print("\n🏆 Ranking — CUSTO MÍNIMO (threshold 0.5 padrão):         ", vencedor_A)
print("🏆 Ranking — CUSTO MÍNIMO (threshold OTIMIZADO p/ 10:1):   ",
      vencedor_B_row["Modelo"],
      f"   (threshold = {vencedor_B_row['Th OTIMO p/ custo 10:1']}, "
      f"economia vs política atual = {vencedor_B_row['Th OTIMO — vs Pol. Atual (economia %)']}%).")
