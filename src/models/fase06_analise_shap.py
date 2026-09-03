"""
Fase 6 — Análise SHAP — ENTREGÁVEL SEPARADO (padrão sênior / auditoria regulatória).

Boas práticas justificam um módulo próprio (análise SHAP separada):
  1. SHAP é artefato de auditoria (LGPD/BCB/regulador): entregável próprio, reproduzível,
     versionável, com versão de modelo atrelada. Não misturar com training notebooks.
  2. Pesado computacionalmente: TreeExplainer em 37.500 linhas leva minutos.
     Manter separado evita recomputar a cada re-run do tuning.
  3. Reaproveitável em produção: Streamlit carrega o shap_artefatos_producao.pkl
     e mostra "por que o crédito foi negado" para o cliente/analista/regulador
     em < 10 ms, sem re-computar nada.

Executado com:
    uv run python src/models/fase06_analise_shap.py

Entradas:
    models/04_xgboost_pipeline.pkl
    data/processed/X_test.csv, y_test.csv

Saídas (TODAS com foco no HOLDOUT, conforme pedido pelo usuário):
    reports/evaluations/fase06_shap_ranking_top10_holdout.csv    — ranking top10 global
    reports/figures/fase06_shap_summary_beeswarm_top10_holdout.png
    reports/figures/fase06_shap_bar_top10_holdout.png
    reports/figures/fase06_shap_dependence_<feature>_holdout.png — top 4 features
    reports/figures/fase06_shap_waterfall_<caso>_holdout.png     — 3 casos locais
    models/shap_values_holdout_XGBbest.npy                      — valores SHAP brutos
    models/shap_artefatos_producao.pkl                          — p/ Streamlit
"""
from __future__ import annotations

import os, sys, pickle, warnings

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from features.preprocessamento import SEED_DEFAULT
from features.pipeline_modelo import obter_nomes_features

warnings.filterwarnings("ignore")
np.random.seed(SEED_DEFAULT)
shap.initjs()
sns.set_theme(style="whitegrid")

BASE    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC    = os.path.join(BASE, "data", "processed")
MODELOS = os.path.join(BASE, "models")
FIG     = os.path.join(BASE, "reports", "figures")
REP     = os.path.join(BASE, "reports", "evaluations")
os.makedirs(FIG, exist_ok=True)
os.makedirs(REP, exist_ok=True)

print("=" * 140)
print("📊🔍  FASE 6 — ANÁLISE SHAP no HOLDOUT (X_test = 37.500 clientes)")
print("=" * 140)
print(f"SHAP versão instalada: {shap.__version__}")

# -----------------------------------------------------------------------------
# 1. Carrega o pipeline oficial (já foi fittado em 100% treino) e X_test bruto.
# -----------------------------------------------------------------------------
with open(os.path.join(MODELOS, "04_xgboost_pipeline.pkl"), "rb") as f:
    pipe_xgb = pickle.load(f)

X_test_raw = pd.read_csv(os.path.join(PROC, "X_test.csv"))
y_test     = pd.read_csv(os.path.join(PROC, "y_test.csv"))["inadimplente_2anos"].astype(int).values

N_FEATURES_ORIGINAIS = X_test_raw.shape[1]
N_LINHAS_TESTE = X_test_raw.shape[0]

# -----------------------------------------------------------------------------
# 2. Obtém a matriz de dados TRATADA (X_test → após preparo + imputacao)
#    NÃO olhamos o y para isso, apenas o pipeline aplica os parâmetros do treino.
# -----------------------------------------------------------------------------
prep_step = pipe_xgb.named_steps["preparo"]
imp_step  = pipe_xgb.named_steps["imputacao"]
X_passo1 = prep_step.transform(X_test_raw.copy())   # np.array (11 → 15 colunas)
X_passo2 = imp_step.transform(X_passo1)             # np.array (mantém 15 cols)
FEATURE_NAMES = obter_nomes_features(pipe_xgb)      # lista com os 15 nomes pós-preparo
X_test_processado = pd.DataFrame(X_passo2, columns=FEATURE_NAMES)
MODELO = pipe_xgb.named_steps["modelo"]

print(f"\n[1] Estrutura pronta p/ SHAP:")
print(f"    X_test bruto (raw) shape        = {N_LINHAS_TESTE:,} × {N_FEATURES_ORIGINAIS:,}")
print(f"    X_test processado shape         = {X_test_processado.shape[0]:,} × {X_test_processado.shape[1]:,}")
print(f"    Nomes features (pós-preparo): {FEATURE_NAMES}")
print(f"    y_test distrib: POS(1)={int(y_test.sum()):,}   NEG(0)={int((y_test==0).sum()):,}")
print(f"    Exemplo de 5 probabilidades preditas no holdout: "
      f"{pipe_xgb.predict_proba(X_test_raw.iloc[:5])[:, 1].round(4).tolist()}")

# -----------------------------------------------------------------------------
# 3. TreeExplainer (ideal para árvores / XGBoost — exato, sem aproximações).
# -----------------------------------------------------------------------------
print("\n[2] Construindo TreeExplainer...")
explicador = shap.TreeExplainer(MODELO)

# .shap_values sobre XGBoostClassifier retorna Explanation ou np.ndarray
# Nós queremos sempre a classe 1 (inadimplente).
explanation = explicador(X_test_processado)

# Extrair valores SHAP (shape = [N, M]) para a classe POSITIVA (inadimplente).
if isinstance(explanation, shap.Explanation):
    expected_value_full = explanation.base_values
    values_full = explanation.values
    if len(values_full.shape) == 3:                      # classificador: (N, M, 2 classes)
        shap_values_cls1 = values_full[:, :, 1]
        expected_value_cls1 = float(np.mean(expected_value_full[:, 1])) \
            if hasattr(expected_value_full, "shape") and len(expected_value_full.shape) == 2 \
            else float(expected_value_full[1]) if hasattr(expected_value_full, "__len__") \
            else float(expected_value_full)
    elif len(values_full.shape) == 2:                    # já veio shape (N, M)
        shap_values_cls1 = values_full
        if hasattr(expected_value_full, "__len__") and not np.isscalar(expected_value_full):
            expected_value_cls1 = float(np.mean(expected_value_full))
        else:
            expected_value_cls1 = float(expected_value_full)
    else:
        raise ValueError(f"shape inesperado: {values_full.shape}")
elif isinstance(explanation, np.ndarray):
    if len(explanation.shape) == 3:
        shap_values_cls1 = explanation[:, :, 1]
        ev = explicador.expected_value
        expected_value_cls1 = float(ev[1]) if hasattr(ev, "__len__") else float(ev)
    else:
        shap_values_cls1 = explanation
        ev = explicador.expected_value
        expected_value_cls1 = float(np.mean(ev)) if hasattr(ev, "__len__") else float(ev)
else:
    raise TypeError(f"shap_values retornou tipo inesperado: {type(explanation)}")

shap_df = pd.DataFrame(shap_values_cls1, columns=FEATURE_NAMES)

print(f"    expected_value (log-odds base): {expected_value_cls1:.5f}")
print(f"    shap_values_cls1 shape: {shap_values_cls1.shape}")
print(f"    sanity check (sum shap + expected ~ log-odds preditos) OK se média ≈ 0.")
# Opcional sanity: probas preditas vs probas via sigmoid(expected + row sum shap)
try:
    import xgboost as xgb
    raw_proba = pipe_xgb.predict_proba(X_test_raw)[:, 1]
    soma_shap = expected_value_cls1 + shap_values_cls1.sum(axis=1)
    sigmoid_shap = 1.0 / (1.0 + np.exp(-soma_shap))
    mae_proba = float(np.mean(np.abs(raw_proba - sigmoid_shap)))
    print(f"    diff médio entre pipe.predict_proba vs sigmoid(shap) = {mae_proba:.6f} "
          f"(perfeito se < 1e-5)")
except Exception as e:
    print(f"    (sanity skip: {e})")

# -----------------------------------------------------------------------------
# 4. Ranking GLOBAL Top 10 por |média(SHAP)| — resultado principal pedido
# -----------------------------------------------------------------------------
mean_abs = shap_df.abs().mean(axis=0).sort_values(ascending=False)
top10 = mean_abs.head(10)
top10_df = pd.DataFrame({
    "Feature": top10.index.tolist(),
    "Média |SHAP Value|": top10.values.round(6),
}).set_index(pd.Index(range(1, len(top10) + 1), name="Rank"))

print("\n" + "=" * 140)
print("🏆🏆🏆  RESULTADO PRINCIPAL: RANKING GLOBAL TOP 10  (HOLDOUT, 37.500 clientes)")
print("=" * 140)
print(top10_df.to_string())

# 4.2 Detalhe: sinal da contribuição média e extremos
detalhe_rows = []
for i, feat in enumerate(top10.index, 1):
    detalhe_rows.append({
        "Rank": i,
        "Feature": feat,
        "Média |SHAP|":  round(top10.loc[feat], 6),
        "Média SHAP (↑risco / ↓risco)": round(float(shap_df[feat].mean()), 6),
        "Máx puxa p/ ↑ risco (+)":       round(float(shap_df[feat].max()), 4),
        "Máx puxa p/ ↓ risco (-)":       round(float(shap_df[feat].min()), 4),
        "% clientes com SHAP > 0":       f"{100. * (shap_df[feat] > 0).mean():.2f}%",
    })
top10_detalhado_df = pd.DataFrame(detalhe_rows).set_index("Rank")
print("\n📋 Detalhamento do top 10 (média, extremos e % que puxam o score):")
print(top10_detalhado_df.to_string())

# Salvar CSV
top10_detalhado_df.to_csv(
    os.path.join(REP, "fase06_shap_ranking_top10_holdout.csv"),
    sep=";", encoding="utf-8")
print("\nCSV salvo em reports/evaluations/fase06_shap_ranking_top10_holdout.csv")

# -----------------------------------------------------------------------------
# 5. Gráficos SHAP: (a) beeswarm summary, (b) bar top10, (c) dependence top4
# -----------------------------------------------------------------------------
print("\n[3] Salvando gráficos em reports/figures/...")

# (a) beeswarm (resumo visual: cor = feature value alto/baixo; x = impacto no log-odds)
try:
    fig = plt.figure(figsize=(12, 9))
    shap.summary_plot(shap_df[top10.index].values,
                      X_test_processado[top10.index].values,
                      feature_names=top10.index.tolist(),
                      max_display=10, show=False)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fase06_shap_summary_beeswarm_top10_holdout.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    ✓ summary_beeswarm_top10")
except Exception as e:
    print(f"    ⚠ beeswarm falhou: {e}")

# (b) bar plot top10
try:
    fig = plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_df[top10.index].values,
                      X_test_processado[top10.index].values,
                      feature_names=top10.index.tolist(),
                      plot_type="bar", max_display=10, show=False)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, "fase06_shap_bar_top10_holdout.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    ✓ bar_top10")
except Exception as e:
    print(f"    ⚠ bar_top10 falhou: {e}")

# (c) Dependence plots para top 4 features
for feat in top10.index[:4]:
    try:
        fig = plt.figure(figsize=(10, 6))
        shap.dependence_plot(feat,
                             shap_df.values,
                             X_test_processado,
                             feature_names=FEATURE_NAMES,
                             show=False)
        plt.title(f"Dependence Plot: {feat}  (HOLDOUT n={N_LINHAS_TESTE:,})")
        plt.tight_layout()
        safe = feat.replace("/", "_").replace("(", "_").replace(")", "_").replace(" ", "_")
        fig.savefig(os.path.join(FIG, f"fase06_shap_dependence_{safe}_holdout.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    ✓ dependence_{safe}")
    except Exception as e:
        print(f"    ⚠ dependence_{feat} falhou: {e}")

# -----------------------------------------------------------------------------
# 6. Três explicações LOCAIS de clientes REAIS do holdout (waterfall plot).
#    Tipos: (A) calote real NEGADO (TP), (B) adimplente APROVADO (TN), (C) borderline.
# -----------------------------------------------------------------------------
print("\n[4] Três casos LOCAIS (waterfall) para explicar ao cliente/analista:")
probas = pipe_xgb.predict_proba(X_test_raw)[:, 1]
TH_OTIMO = 0.56   # vencedor do OOF treino para FN=5000 / FP=500
preds = (probas >= TH_OTIMO).astype(int)

aux = pd.DataFrame({
    "y_real": y_test,
    "proba": probas,
    "pred": preds,
    "TP": (preds == 1) & (y_test == 1),
    "TN": (preds == 0) & (y_test == 0),
})
tp_idx = aux[aux["TP"] & (probas > 0.86)].head(1).index.tolist()[0]
tn_idx = aux[aux["TN"] & (probas < 0.12)].head(1).index.tolist()[0]
bd_idx = aux[(probas >= TH_OTIMO - 0.02) & (probas <= TH_OTIMO + 0.02)].head(1).index.tolist()[0]

CASOS = [
    ("A_CALOTE_NEGADO_TP",  tp_idx, "Calote real, modelo NEGOU → TP (acerto alto)"),
    ("B_ADIMPLENTE_APROVADO_TN", tn_idx, "Adimplente real, modelo APROVOU → TN (acerto baixo)"),
    ("C_BORDERLINE", bd_idx, f"Próximo do limiar {TH_OTIMO} (caso dúbio, requer análise manual)"),
]

for (rotulo, idx, descr) in CASOS:
    print(f"\n► Caso {rotulo}: {descr}")
    print(f"   i={idx}   y_real={y_test[idx]}   proba={probas[idx]:.4f}   "
          f"pred={preds[idx]} (th={TH_OTIMO})")
    try:
        expl_obj = shap.Explanation(
            values=shap_df.iloc[idx].values,
            base_values=expected_value_cls1,
            data=X_test_processado.iloc[idx].values,
            feature_names=FEATURE_NAMES,
        )
        fig = plt.figure(figsize=(11, 7))
        shap.waterfall_plot(expl_obj, max_display=10, show=False)
        plt.title(f"Cliente {rotulo} — proba={probas[idx]:.3f}")
        plt.tight_layout()
        fig.savefig(os.path.join(FIG, f"fase06_shap_waterfall_{rotulo}_holdout.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        # Tabela texto dos 10 drivers mais impactantes para este cliente
        drivers = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Valor na feature": np.round(X_test_processado.iloc[idx].values, 4),
            "Contribuição SHAP": np.round(expl_obj.values, 5),
            "|SHAP|": np.abs(expl_obj.values),
        }).sort_values("|SHAP|", ascending=False).head(10).drop("|SHAP|", axis=1)
        print("   Top 10 drivers LOCAIS:")
        print(drivers.to_string(index=False))
    except Exception as e:
        print(f"   (waterfall falhou: {e})")
        # Fallback textual
        print("   Top 10 by |SHAP|:", (
            shap_df.iloc[idx].rename("shap").to_frame()
            .assign(abs_shap=lambda d: d["shap"].abs())
            .sort_values("abs_shap", ascending=False)["shap"].head(10).round(5).to_string()
        ))

# -----------------------------------------------------------------------------
# 7. Persistência para produção (Streamlit / análise por pedido de crédito)
# -----------------------------------------------------------------------------
print("\n[5] Persistindo artefatos SHAP para produção:")
np.save(os.path.join(MODELOS, "shap_values_holdout_XGBbest.npy"), shap_values_cls1)
print(f"    ✓ models/shap_values_holdout_XGBbest.npy  shape={shap_values_cls1.shape}")

with open(os.path.join(MODELOS, "shap_artefatos_producao.pkl"), "wb") as f:
    pickle.dump({
        "feature_names": FEATURE_NAMES,
        "expected_value_logodds": float(expected_value_cls1),
        "top10_features": top10_df["Feature"].tolist(),
        "top10_mean_abs_shap": top10_df["Média |SHAP Value|"].tolist(),
        "TH_OTIMO_XGB_5000_500": TH_OTIMO,
        "CUSTO_FN_RS": 5_000,
        "CUSTO_FP_RS": 500,
        "modelo_tipo": "XGBClassifier (TreeExplainer exato)",
        "nota_producao": (
            "Para explicar UM cliente NOVO no Streamlit: (1) pipe_xgb[:2].transform(linha_11cols) "
            "-> X_tratado, (2) shap.TreeExplainer(pipe_xgb.named_steps.modelo)(X_tratado) -> "
            "Explanation, (3) waterfall + tabela top 5 local. Não precisa re-carregar os "
            "37k shap values do holdout para explicar 1 cliente."
        ),
    }, f)
print(f"    ✓ models/shap_artefatos_producao.pkl")

# -----------------------------------------------------------------------------
# 8. Resumo final
# -----------------------------------------------------------------------------
print("\n" + "=" * 140)
print("🏁 RESUMO FINAL (top 10 por |mean_SHAP| — HOLDOUT n=37.500):")
print("=" * 140)
for i, (feat, v) in enumerate(top10.items(), 1):
    print(f"   {i:2d}. {feat:<30s}   |mean_SHAP| = {v:.6f}")
print("\nFIM. Consulte reports/evaluations e reports/figures por artefatos completos.")
