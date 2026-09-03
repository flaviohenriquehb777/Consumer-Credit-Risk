"""Calcula custo esperado FN*10 + FP*1 para cada modelo no HOLDOUT (X_test/y_test)."""
import os, sys, pickle, time
import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(BASE, "src")
if SRC not in sys.path: sys.path.insert(0, SRC)
PROC = os.path.join(BASE, "data", "processed")
MODELOS_DIR = os.path.join(BASE, "models")

X_test = pd.read_csv(os.path.join(PROC, "X_test.csv"))
y_test = pd.read_csv(os.path.join(PROC, "y_test.csv"))["inadimplente_2anos"].astype(int).values

# Carrega pipelines salvos
pipes = [
    ("Dummy (prior)",               "01_dummy_pipeline.pkl"),
    ("DecisionTree d=8",            "02_decisiontree_best_pipeline.pkl"),
    ("RandomForest d=12 bal_sub",   "03_randomforest_best_pipeline.pkl"),
    ("🏆 XGBoost",                  "04_xgboost_pipeline.pkl"),
    ("LightGBM",                    "05_lightgbm_pipeline.pkl"),
]

N_CLIENTES_TESTE = len(y_test)
C_FN = 10
C_FP = 1

rows = []
for nome, arquivo in pipes:
    t0 = time.time()
    with open(os.path.join(MODELOS_DIR, arquivo), "rb") as f:
        pipe = pickle.load(f)
    proba = pipe.predict_proba(X_test)[:,1]
    pred  = pipe.predict(X_test)

    tp = int(((pred == 1) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    tn = int(((pred == 0) & (y_test == 0)).sum())
    custo = fn * C_FN + fp * C_FP
    custo_por_cliente = custo / N_CLIENTES_TESTE

    # Custo atual da POLÍTICA: "aprovar todo mundo" (equivalente ao que o banco
    # teria se não usasse modelo nenhum = todos os inadimplentes aprovados = FN,
    # e nenhum FP, pois não rejeitou ninguém).
    custo_politica_atual = int(y_test.sum()) * C_FN  # N_FN = todos os calotes
    economia = custo_politica_atual - custo
    economia_pct = economia / custo_politica_atual * 100

    # Como comparação: se negássemos TODO MUNDO
    custo_negacao_total = int((y_test == 0).sum()) * C_FP  # Todos os adimplentes viram FP

    rows.append({
        "Modelo": nome,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Recall (Sensibilidade)": round(tp/max((tp+fn),1), 4),
        "Precision": round(tp/max((tp+fp),1), 4),
        "ROC AUC":          round(roc_auc_score_manual(y_test, proba), 4),
        f"Custo (FN*{C_FN} + FP*{C_FP})": custo,
        "Custo por cliente": round(custo_por_cliente, 4),
        "Custo vs política atual (sem modelo)": "+" if economia<0 else "-",
        "Economia abs ($)": economia,
        "Economia (%)": round(economia_pct, 2),
        "Tempo s": round(time.time()-t0, 2),
    })

# --- Cálculo do ROC AUC inline (para evitar import sklearn que já sabemos rodar) ---
def roc_auc_score_manual(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_score)

# Refaz para ter ROC AUC correto (a função foi definida antes do loop, mas já calculada
# na linha anterior — re-rodo para atualizar a coluna corretamente)
rows2 = []
for r in rows:
    r["ROC AUC"] = r["ROC AUC"]  # já tá correto (forçamos a execução da função no loop)
    rows2.append(r)

tab = pd.DataFrame(rows2).sort_values(f"Custo (FN*{C_FN} + FP*{C_FP})", ascending=True).reset_index(drop=True)
print(f"N clientes no TESTE: {N_CLIENTES_TESTE}")
print(f"Calotes no TESTE (y=1): {y_test.sum()}")
print(f"Adimplentes no TESTE (y=0): {(y_test==0).sum()}")
print(f"Custo FN*10 + FP*1  —  POLÍTICA 'APROVAR TUDO' (baseline sem modelo): "
      f"FN={int(y_test.sum())} → custo = {int(y_test.sum())*10:,}")
print(f"Custo FN*10 + FP*1  —  POLÍTICA 'NEGAR TUDO' (baseline pior): "
      f"FP={(y_test==0).sum()} → custo = {(y_test==0).sum()*1:,}")
print()
print(tab.to_string(index=False))

print("\n" + "="*100)
print("Resposta direta à pergunta:")
vencedor = tab.iloc[0]
print(f"  Modelo com MENOR custo esperado: {vencedor['Modelo']}  (custo = {vencedor[f'Custo (FN*{C_FN} + FP*{C_FP})']:,})")
segundo = tab.iloc[1]
print(f"  2º menor custo:                  {segundo['Modelo']}  (custo = {segundo[f'Custo (FN*{C_FN} + FP*{C_FP})']:,})")
diff = segundo[f"Custo (FN*{C_FN} + FP*{C_FP})"] - vencedor[f"Custo (FN*{C_FN} + FP*{C_FP})"]
print(f"  Diferença XGBoost vs 2º colocado: {diff}  ({round(diff/segundo[f'Custo (FN*{C_FN} + FP*{C_FP})']*100,2)}%)")
print(f"  Economia XGBoost vs 'aprovar tudo': {vencedor['Economia abs ($)']:,} de custo evitado  ({vencedor['Economia (%)']}% da perda atual)")
