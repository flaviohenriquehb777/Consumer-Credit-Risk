"""
Fase 5 — Decisão do limiar de probabilidade (threshold) com base em CUSTO REAL em R$.
Decisão tomada SÓ com predições OOF do TREINO (holdout é validação cega final).

Custos (passados pelo usuário):
    CUSTO_FN = R$ 5.000  ← aprovamos quem deu calote (perdemos principal)
    CUSTO_FP = R$   500  ← negamos quem pagaria  (perdemos margem do cliente)

Passo a passo:
    1. Carrega os pickles dos modelos oficiais (refit 100% treino).
    2. Refaz OOF 5-fold no TREINO (DT e XGBoost), para obter as predições OOF
       (fora de fold) em todos os 112.500 clientes do treino.
    3. Varre limiares de 0,01 em 0,01 (0,01 → 0,99) e calcula custo total R$.
    4. Escolhe threshold ótimo por modelo.
    5. TABELA FINAL COMPARATIVA: menor custo possível / threshold / % vs política atual.
    6. VALIDAÇÃO CEGA UMA VEZ NO HOLDOUT (não altera nada, só confirma o valor real).
    7. Persiste: CSVs da curva, pickles thresholds (parâmetro para o Streamlit), gráficos.
"""
from __future__ import annotations

import os, sys, pickle, warnings
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if SRC not in sys.path: sys.path.insert(0, SRC)
from features.preprocessamento import SEED_DEFAULT, carregar_params
from features.pipeline_modelo import montar_pipeline

warnings.filterwarnings("ignore")
np.random.seed(SEED_DEFAULT)
sns.set_theme(style="whitegrid", palette="viridis")

BASE    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC    = os.path.join(BASE, "data", "processed")
MODELOS = os.path.join(BASE, "models")
FIG     = os.path.join(BASE, "reports", "figures")
REP     = os.path.join(BASE, "reports", "evaluations")
os.makedirs(FIG, exist_ok=True); os.makedirs(REP, exist_ok=True)

PARAMS  = carregar_params(os.path.join(MODELOS, "preprocessamento_params.pkl"))
CUSTO_FN, CUSTO_FP = 5_000, 500
RAZAO_FN_FP = CUSTO_FN / CUSTO_FP  # = 10,0
TS = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"Custos (em R$): FN={CUSTO_FN:,.2f}   FP={CUSTO_FP:,.2f}   Razão FN/FP={RAZAO_FN_FP:.1f}:1")
print("Thresholds serão aprendidos SÓ no TREINO (OOF 5-fold). Holdout só é lido no PASSO 6.")
print("=" * 140)

X_train = pd.read_csv(os.path.join(PROC, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROC, "y_train.csv"))["inadimplente_2anos"].astype(int).values
X_test  = pd.read_csv(os.path.join(PROC, "X_test.csv"))
y_test  = pd.read_csv(os.path.join(PROC, "y_test.csv"))["inadimplente_2anos"].astype(int).values
NEG, POS = int((y_train == 0).sum()), int((y_train == 1).sum())
SPW = round(NEG / POS, 2)

print(f"TREINO = {len(y_train):,} (NEG={NEG:,} POS={POS:,})")
print(f"HOLDOUT = {len(y_test):,} (FECHADO até o passo 6)")

# ============================================================
# 1. Definição dos modelos (mesmos hparams da rodada oficial SEM LEAKAGE)
# ============================================================
BEST_DT_DEPTH = 7
BEST_XGB = dict(max_depth=4, learning_rate=0.03, min_child_weight=80,
                n_estimators=500, subsample=0.9, colsample_bytree=0.85,
                reg_alpha=0.1, reg_lambda=1.0,
                scale_pos_weight=SPW, random_state=SEED_DEFAULT,
                n_jobs=-1, eval_metric="auc", tree_method="hist")

def modelo_dt():
    return DecisionTreeClassifier(max_depth=BEST_DT_DEPTH, random_state=SEED_DEFAULT,
                                  class_weight=None, min_samples_leaf=50)
def modelo_xgb():
    return xgb.XGBClassifier(**BEST_XGB)

MODELOS_OBJ = {
    f"DecisionTree (d={BEST_DT_DEPTH})": modelo_dt(),
    f"XGBoost best ({BEST_XGB['max_depth']}, lr={BEST_XGB['learning_rate']}, mcw={BEST_XGB['min_child_weight']})": modelo_xgb(),
}

# ============================================================
# 2. Função CUSTO e OOF 5-fold no TREINO (NÃO lê holdout)
# ============================================================
def calcular_custo_rs(y_true: np.ndarray, proba: np.ndarray, limiar: float):
    """Retorna (custo_RS, n_FN, n_FP, n_TP, n_TN, taxa_aprovacao, inadimplencia_da_carteira_aprovada)."""
    pred = (proba >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    custo = CUSTO_FN * int(fn) + CUSTO_FP * int(fp)
    n_aprovados = int(tn + fn)  # preditos 0 = aprovados
    inadimplencia_carteira_aprovada = (int(fn) / n_aprovados) if n_aprovados > 0 else np.nan
    return dict(
        limiar=round(float(limiar), 2),
        custo_RS=int(custo),
        custo_por_cliente_RS=round(float(custo / len(y_true)), 2),
        n_FN=int(fn), n_FP=int(fp), n_TP=int(tp), n_TN=int(tn),
        taxa_aprovacao=round(float(n_aprovados / len(y_true)), 4),
        inadimplencia_carteira_aprovada_pct=round(100 * float(inadimplencia_carteira_aprovada), 2),
        perda_principal_RS=int(CUSTO_FN * int(fn)),
        perda_margem_RS=int(CUSTO_FP * int(fp)),
    )

def oof_treino(nome_modelo, modelo_base):
    """5-fold OOF, devolve o array de probabilidades OOF dos 112.500."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED_DEFAULT)
    oof = np.zeros(len(y_train), dtype=float)
    for tr_idx, va_idx in skf.split(X_train, y_train):
        pipe = montar_pipeline(deepcopy(modelo_base), PARAMS)
        pipe.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof[va_idx] = pipe.predict_proba(X_train.iloc[va_idx])[:, 1]
    return oof

print("\n[PASSO 2] OOF 5-fold no TREINO para DecisionTree + XGBoost...")
OOF = {}
for nome, mc in MODELOS_OBJ.items():
    oof_proba = oof_treino(nome, mc)
    OOF[nome] = oof_proba
    print(f"  ✓ {nome}  OOF shape = {oof_proba.shape}  min={oof_proba.min():.3f}  max={oof_proba.max():.3f}")

# ============================================================
# 3. Curva de custo × limiar (0,01 → 0,99, passo 0,01) no TREINO OOF
# ============================================================
print("\n[PASSO 3] Varredura 0,01→0,99 em custo R$ (SÓ TREINO OOF)...")
LIMIARES = np.round(np.arange(0.01, 1.00, 0.01), 2)
CURVAS = {}
for nome, oof_proba in OOF.items():
    rows = []
    for th in LIMIARES:
        rows.append(calcular_custo_rs(y_train, oof_proba, th))
    df = pd.DataFrame(rows).sort_values("limiar").reset_index(drop=True)
    CURVAS[nome] = df
    melhor_linha = df.loc[df["custo_RS"].idxmin()]
    print(f"  {nome}:")
    print(f"      limiar ótimo (OOF treino) = {melhor_linha['limiar']:.2f}")
    print(f"      menor custo R$            = R$ {melhor_linha['custo_RS']:,.2f}")
    print(f"      FN / FP                   = {int(melhor_linha['n_FN']):,} / {int(melhor_linha['n_FP']):,}")
    print(f"      taxa aprovação            = {100*melhor_linha['taxa_aprovacao']:.2f}%  |  inadimpl. carteira aprovada = {melhor_linha['inadimplencia_carteira_aprovada_pct']}%")

# Política ATUAL (aprova todos = aprova 100%): limiar artificial 1,01 (apenas ninguém negado)
linha_atual_treino = calcular_custo_rs(y_train, OOF[list(OOF.keys())[0]], 1.01)  # todos aprovados (FN=POS, FP=0)
# Mas "aprova todos" deve prever 0 para todos → limiar > max(proba) → pred = sempre 0
# Calculamos exato:
pred_aprova_todos = np.zeros(len(y_train), dtype=int)
TN_at, FP_at, FN_at, TP_at = confusion_matrix(y_train, pred_aprova_todos, labels=[0,1]).ravel()
custo_atual_treino_rs = int(CUSTO_FN * FN_at + CUSTO_FP * FP_at)

# Política "nega todos"
pred_nega_todos = np.ones(len(y_train), dtype=int)
TN_nt, FP_nt, FN_nt, TP_nt = confusion_matrix(y_train, pred_nega_todos, labels=[0,1]).ravel()
custo_negar_todos = int(CUSTO_FN * FN_nt + CUSTO_FP * FP_nt)

print(f"\n🔥 Política ATUAL ('aprova todos' — baseline CRISP) no TREINO:")
print(f"   FN={int(FN_at):,}  FP={int(FP_at):,}  CUSTO R$ = R$ {custo_atual_treino_rs:,.2f}  |  custo/cliente R$ {round(custo_atual_treino_rs/len(y_train),2)}")
print(f"❌ Política extrema 'nega todos' no TREINO:")
print(f"   FN={int(FN_nt):,}  FP={int(FP_nt):,}  CUSTO R$ = R$ {custo_negar_todos:,.2f}")

# ============================================================
# 4. TABELA COMPARATIVA FINAL — SÓ TREINO OOF
# ============================================================
print("\n" + "=" * 140)
print("📋📊📈  TABELA COMPARATIVA (TREINO OOF, 112.500 clientes) — DT vs XGB vs Política Atual")
print("=" * 140)
tabela = []
for nome, df in CURVAS.items():
    m = df.loc[df["custo_RS"].idxmin()]
    economia = (1 - m["custo_RS"] / custo_atual_treino_rs) * 100
    tabela.append({
        "Modelo": nome,
        "Limiar Ótimo (aprendido no OOF treino)": f"{m['limiar']:.2f}",
        "Custo Total Ótimo (R$)": f"R$ {m['custo_RS']:,.2f}",
        "Custo por Cliente (R$)": f"R$ {m['custo_por_cliente_RS']:,.2f}",
        "N_FN (perda principal)": f"{int(m['n_FN']):,}",
        "N_FP (perda margem)": f"{int(m['n_FP']):,}",
        "Perda Principal (R$)": f"R$ {int(m['perda_principal_RS']):,.2f}",
        "Perda Margem (R$)": f"R$ {int(m['perda_margem_RS']):,.2f}",
        "Taxa de Aprovação": f"{100*m['taxa_aprovacao']:.2f}%",
        "Inadimplência da Carteira Aprovada": f"{m['inadimplencia_carteira_aprovada_pct']:.2f}%",
        "Economia vs Política Atual (R$)": f"R$ {custo_atual_treino_rs - int(m['custo_RS']):,.2f}",
        "Economia vs Política Atual (%)": f"{economia:.2f}%",
    })
tabela.append({
    "Modelo": "🔥 Política ATUAL (aprova TODOS)",
    "Limiar Ótimo (aprendido no OOF treino)": "—",
    "Custo Total Ótimo (R$)": f"R$ {custo_atual_treino_rs:,.2f}",
    "Custo por Cliente (R$)": f"R$ {round(custo_atual_treino_rs/len(y_train),2):,.2f}",
    "N_FN (perda principal)": f"{int(FN_at):,}",
    "N_FP (perda margem)": f"{int(FP_at):,}",
    "Perda Principal (R$)": f"R$ {int(CUSTO_FN*FN_at):,.2f}",
    "Perda Margem (R$)": f"R$ {int(CUSTO_FP*FP_at):,.2f}",
    "Taxa de Aprovação": "100,00%",
    "Inadimplência da Carteira Aprovada": f"{100*FN_at/(TN_at+FN_at):.2f}%",
    "Economia vs Política Atual (R$)": "R$ 0,00",
    "Economia vs Política Atual (%)": "0,00%",
})
tabela.append({
    "Modelo": "❌ Política: NEGAR TODOS",
    "Limiar Ótimo (aprendido no OOF treino)": "0 (sempre negar)",
    "Custo Total Ótimo (R$)": f"R$ {custo_negar_todos:,.2f}",
    "Custo por Cliente (R$)": f"R$ {round(custo_negar_todos/len(y_train),2):,.2f}",
    "N_FN (perda principal)": f"{int(FN_nt):,}",
    "N_FP (perda margem)": f"{int(FP_nt):,}",
    "Perda Principal (R$)": f"R$ {int(CUSTO_FN*FN_nt):,.2f}",
    "Perda Margem (R$)": f"R$ {int(CUSTO_FP*FP_nt):,.2f}",
    "Taxa de Aprovação": "0,00%",
    "Inadimplência da Carteira Aprovada": "—",
    "Economia vs Política Atual (R$)": f"R$ {custo_atual_treino_rs - custo_negar_todos:,.2f}",
    "Economia vs Política Atual (%)": f"{(1-custo_negar_todos/custo_atual_treino_rs)*100:.2f}%",
})
df_tabela = pd.DataFrame(tabela)
print(df_tabela.to_string(index=False))

# ============================================================
# 5. Gráficos de curva CUSTO R$ × LIMIAR (TREINO OOF)
# ============================================================
print("\n[PASSO 5] Salvando gráficos...")

# 5a. Gráfico sobreposto DT vs XGB, com linha do custo atual (aplica todos)
fig, ax = plt.subplots(figsize=(13, 6))
for nome, df in CURVAS.items():
    ax.plot(df["limiar"], df["custo_RS"], label=nome, linewidth=2.2)
    idx = df["custo_RS"].idxmin()
    ax.scatter([df.loc[idx,"limiar"]], [df.loc[idx,"custo_RS"]], s=200, zorder=5,
               marker="*", edgecolor="black", linewidth=0.5, label=f"mín {nome} = {df.loc[idx,'limiar']:.2f} (R${df.loc[idx,'custo_RS']:,.0f})")
ax.axhline(custo_atual_treino_rs, ls="--", color="crimson", alpha=0.8, linewidth=1.6,
           label=f"🔥 Política Atual (aprova TODOS) = R${custo_atual_treino_rs:,.0f}")
ax.axhline(custo_negar_todos, ls=":", color="gray", alpha=0.6, linewidth=1.2,
           label=f"❌ Nega TODOS = R${custo_negar_todos:,.0f}")
ax.set_xlabel("Limiar de probabilidade (1% em 1%)")
ax.set_ylabel("Custo Total Esperado (R$)  —  FN×5.000 + FP×500")
ax.set_title("Decisão do limiar: Custo R$ × Limiar  (SÓ OOF TREINO — n=112.500)")
ax.legend(fontsize=8, loc="upper right", ncol=1)
ax.set_xticks(np.linspace(0, 1, 11))
ax.set_xlim(0, 1)
plt.tight_layout()
fig.savefig(os.path.join(FIG, "fase05_custo_vs_limiar_TREINO_OOF_DT_XGB.png"), dpi=150)
plt.close(fig)

# 5b. Gráficos separados: FN×5000 / FP×500 empilhados, por limiar (XGB)
for nome, df in CURVAS.items():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.fill_between(df["limiar"], 0, df["perda_principal_RS"], step="mid", alpha=0.75, label="Perda com calote (FN×R$5.000)")
    ax.fill_between(df["limiar"], df["perda_principal_RS"], df["custo_RS"], step="mid", alpha=0.75, label="Perda de oportunidade (FP×R$500)")
    ax.plot(df["limiar"], df["custo_RS"], color="black", linewidth=1.3)
    idx = df["custo_RS"].idxmin()
    ax.scatter([df.loc[idx,"limiar"]], [df.loc[idx,"custo_RS"]], marker="*", s=220, color="gold", edgecolor="black", zorder=5)
    ax.annotate(f"limiar={df.loc[idx,'limiar']:.2f}\nR${df.loc[idx,'custo_RS']:,.0f}",
                (df.loc[idx,"limiar"], df.loc[idx,"custo_RS"]),
                textcoords="offset points", xytext=(18,-32), fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9))
    ax.axhline(custo_atual_treino_rs, ls="--", color="crimson", alpha=0.8, label=f"Pol. Atual R${custo_atual_treino_rs:,.0f}")
    ax.set_xlabel("Limiar de probabilidade")
    ax.set_ylabel("Custo R$ (acumulado em componentes)")
    ax.set_title(f"Decomposição do custo por limiar — {nome}  (TREINO OOF)")
    ax.legend(loc="upper right")
    ax.set_xticks(np.linspace(0,1,11))
    ax.set_xlim(0,1)
    plt.tight_layout()
    safe = nome.split("(")[0].strip().replace(" ", "_").lower()
    fig.savefig(os.path.join(FIG, f"fase05_custo_decomposicao_{safe}_TREINO_OOF.png"), dpi=150)
    plt.close(fig)

# ============================================================
# 6. VALIDAÇÃO CEGA UMA VEZ NO HOLDOUT (não alteramos nada)
#    Aplicamos exatamente o threshold que foi aprendido no OOF treino.
# ============================================================
print("\n[PASSO 6] Validação CEGA UMA VEZ no HOLDOUT...")

# Carrega os pipes oficiais (refit 100% treino)
with open(os.path.join(MODELOS, "02_decisiontree_best_pipeline.pkl"), "rb") as f: pipe_dt  = pickle.load(f)
with open(os.path.join(MODELOS, "04_xgboost_pipeline.pkl"), "rb") as f:          pipe_xgb = pickle.load(f)

PIPES = {
    f"DecisionTree (d={BEST_DT_DEPTH})": pipe_dt,
    f"XGBoost best ({BEST_XGB['max_depth']}, lr={BEST_XGB['learning_rate']}, mcw={BEST_XGB['min_child_weight']})": pipe_xgb,
}

# Custo da política atual no HOLDOUT (baseline 100% aprovação)
TN_at_t, FP_at_t, FN_at_t, TP_at_t = confusion_matrix(y_test, np.zeros(len(y_test), dtype=int), labels=[0,1]).ravel()
custo_atual_holdout_rs = int(CUSTO_FN * FN_at_t + CUSTO_FP * FP_at_t)

holdout_tabela = []
for nome, df_treino in CURVAS.items():
    th_otimo_treino = df_treino.loc[df_treino["custo_RS"].idxmin(), "limiar"]
    proba = PIPES[nome].predict_proba(X_test)[:, 1]
    linha = calcular_custo_rs(y_test, proba, th_otimo_treino)
    economia = (1 - linha["custo_RS"] / custo_atual_holdout_rs) * 100
    holdout_tabela.append({
        "Modelo": nome,
        "Limiar (aprendido NO TREINO, aplicado cegamente)": f"{th_otimo_treino:.2f}",
        "Custo R$ Holdout (real cego)": f"R$ {linha['custo_RS']:,.2f}",
        "Custo por Cliente (R$ Holdout)": f"R$ {linha['custo_por_cliente_RS']:,.2f}",
        "FN Holdout / FP Holdout": f"{int(linha['n_FN']):,} / {int(linha['n_FP']):,}",
        "Taxa Aprovação": f"{100*linha['taxa_aprovacao']:.2f}%",
        "Inadimplência Carteira Aprovada": f"{linha['inadimplencia_carteira_aprovada_pct']:.2f}%",
        "Economia vs Pol. Atual Holdout (R$)": f"R$ {custo_atual_holdout_rs - int(linha['custo_RS']):,.2f}",
        "Economia vs Pol. Atual Holdout (%)": f"{economia:.2f}%",
    })
holdout_tabela.append({
    "Modelo": "🔥 Política ATUAL (aprova TODOS)",
    "Limiar (aprendido NO TREINO, aplicado cegamente)": "—",
    "Custo R$ Holdout (real cego)": f"R$ {custo_atual_holdout_rs:,.2f}",
    "Custo por Cliente (R$ Holdout)": f"R$ {round(custo_atual_holdout_rs/len(y_test),2):,.2f}",
    "FN Holdout / FP Holdout": f"{int(FN_at_t):,} / {int(FP_at_t):,}",
    "Taxa Aprovação": "100,00%",
    "Inadimplência Carteira Aprovada": f"{100*FN_at_t/(TN_at_t+FN_at_t):.2f}%",
    "Economia vs Pol. Atual Holdout (R$)": "R$ 0,00",
    "Economia vs Pol. Atual Holdout (%)": "0,00%",
})
df_holdout = pd.DataFrame(holdout_tabela)

print(f"\nPolítica ATUAL no HOLDOUT (37.500 clientes): CUSTO R$ = R$ {custo_atual_holdout_rs:,.2f}  "
      f"(FN={int(FN_at_t):,}, FP={int(FP_at_t):,})")
print()
print("VALIDAÇÃO CEGA HOLDOUT (threshold fixo, vindo do OOF treino):")
print("-" * 140)
print(df_holdout.to_string(index=False))

# ============================================================
# 7. Persistência: CSVs, pickles (thresholds para Streamlit)
# ============================================================
print("\n[PASSO 7] Persistindo artefatos...")

# Pickle: thresholds ótimos R$ (FN=5000, FP=500) aprendidos no OOF TREINO → p/ Streamlit
thresholds_reais_rs = {}
for nome, df in CURVAS.items():
    limiar = float(df.loc[df["custo_RS"].idxmin(), "limiar"])
    if "DecisionTree" in nome:    chave = "DecisionTree best"
    elif "XGBoost" in nome:       chave = "XGBoost best"
    else: chave = nome
    thresholds_reais_rs[chave] = limiar
thresholds_reais_rs["CUSTO_FN"] = CUSTO_FN
thresholds_reais_rs["CUSTO_FP"] = CUSTO_FP
thresholds_reais_rs["descricao"] = "Limiar aprendido SÓ no OOF treino (112.500), custo=FN*5000+FP*500, varredura 0,01 a 0,99."
with open(os.path.join(MODELOS, "thresholds_otimos_custo_REAIS_FN5000_FP500.pkl"), "wb") as f:
    pickle.dump(thresholds_reais_rs, f)
print(f"  pickle thresholds: {thresholds_reais_rs}")

# Salvando as curvas para auditoria
for nome, df in CURVAS.items():
    safe = nome.split("(")[0].strip().replace(" ", "_").lower()
    df.to_csv(os.path.join(REP, f"fase05_curva_custo_limiar_{safe}_TREINO_OOF_{TS}.csv"),
              index=False, encoding="utf-8")
df_tabela.to_csv(os.path.join(REP, f"fase05_tabela_comparativa_TREINO_OOF_{TS}.csv"),
                 index=False, encoding="utf-8")
df_holdout.to_csv(os.path.join(REP, f"fase05_validacao_cega_HOLDOUT_{TS}.csv"),
                  index=False, encoding="utf-8")

print("  ✓ CSVs da curva / tabela TREINO OOF / validação HOLDOUT salvos em reports/evaluations/")
print("  ✓ Gráficos salvos em reports/figures/")
print()
print("FIM.")
print(f"RESUMO (TREINO OOF):")
for nome, df in CURVAS.items():
    m = df.loc[df["custo_RS"].idxmin()]
    print(f"   · {nome}  →  limiar {m['limiar']:.2f}  |  Custo R$ {m['custo_RS']:,.2f}  |  Economia vs política atual {(1-m['custo_RS']/custo_atual_treino_rs)*100:.2f}%")
print(f"\nRESUMO (validação cega HOLDOUT):")
for row in holdout_tabela:
    if "Política" not in row["Modelo"]:
        print(f"   · {row['Modelo']}  →  {row['Economia vs Pol. Atual Holdout (%)']}  |  aprovação {row['Taxa Aprovação']}  |  inadimpl. aprovados {row['Inadimplência Carteira Aprovada']}")
