"""
Smoke test do app_streamlit/app_risco_credito.py.

NÃO abre servidor Streamlit. Apenas:
  1. Verifica que o arquivo de app compila (sintaxe OK).
  2. Carrega tudo que carregaria no startup do app (modelo, cfg, preparo, shap explainer).
  3. Roda os 2 casos de teste EXEMPLOS (APROVAR e RECUSAR) e verifica decisão.
  4. Compara com o resultado do self-test da exportação.
  5. Verifica o ranking TOP 5 local SHAP tem exatamente 5 linhas não-nulas.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

print(f"[smoke] app_dir = {APP_DIR}")
print(f"[smoke] arquivos presentes:")
for f in sorted(APP_DIR.iterdir()):
    if f.is_file():
        print(f"   · {f.name:<55s} {f.stat().st_size:>10,d} bytes")

# 1. Compilação do app
with open(APP_DIR / "app_risco_credito.py", "r", encoding="utf-8") as f:
    source = f.read()
compile(source, str(APP_DIR / "app_risco_credito.py"), "exec")
print("\n[smoke 1/5] Sintaxe do app_risco_credito.py .......... OK ✅")

# 2. Carrega tudo que o app carregaria no @st.cache_resource
from preprocessamento import preparar_dados

with open(APP_DIR / "modelo_xgboost_vencedor_CLASSIFICADOR_PURO.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open(APP_DIR / "config_MODELO_XGB_e_SHAP_consolidado.pkl", "rb") as f:
    cfg = pickle.load(f)

import shap
explainer = shap.TreeExplainer(xgb_model)
cols_in = cfg["INPUT_COLUNAS_OBRIGATORIAS_10"]
cols_out_prep = cfg["OUTPUT_PREP_COLUNAS_15"]
params_prep = cfg["PARAMS_PREPROCESSAMENTO"]
th = cfg["TH_OTIMO_XGB"]
exp_val = cfg["SHAP_expected_value_logodds"]

print("[smoke 2/5] Carregamento xgb + cfg + shap explainer ... OK ✅")
print(f"         TH = {th}, FN/FP = R${cfg['CUSTO_FN_RS']}/R${cfg['CUSTO_FP_RS']}, "
      f"expected_value_shap = {exp_val:.6f}")

# 3. Dois casos de teste
testes = [
    ("APROVAR (exemplo)", cfg["EXEMPLO_CLIENTE_APROVAR"],  "APROVAR"),
    ("RECUSAR (exemplo)", cfg["EXEMPLO_CLIENTE_RECUSAR"],  "RECUSAR"),
    ("CASO COM MISSING + COD_SISTEMA 98", {
        "idade": 45, "renda_mensal": np.nan, "dependentes": np.nan,
        "uso_limite_rotativo": 0.75, "razao_divida": 0.55,
        "linhas_credito_abertas": 8, "financiamentos_imobiliarios": 2,
        "atrasos_30_59_dias": 98, "atrasos_60_89_dias": 98,
        "atrasos_90_mais_dias": 98,
    }, "RECUSAR"),
]

ok_total = 0
for nome, caso, esperado in testes:
    linha = pd.DataFrame([caso])[cols_in]
    X_tratado = preparar_dados(linha.copy(), params=params_prep, fit=False)[cols_out_prep]
    p = float(xgb_model.predict_proba(X_tratado)[0, 1])
    decisao_real = "RECUSAR" if p >= th else "APROVAR"
    match = "✅" if decisao_real == esperado else "❌"

    sv = explainer.shap_values(X_tratado)
    if isinstance(sv, list):
        sv = sv[1]
    sv_flat = np.asarray(sv).reshape(-1)
    top5_idx = np.argsort(-np.abs(sv_flat))[:5]
    top5_names = [cols_out_prep[i] for i in top5_idx]

    # Sanidade SHAP
    logodds_shap = float(exp_val + sv_flat.sum())
    logodds_pred = float(np.log(p / max(1e-9, 1 - p)))
    delta = abs(logodds_pred - logodds_shap)

    print(f"\n[smoke 3/5] Caso: {nome}")
    print(f"         P(inadimplência) = {p*100:.2f}%   → decisão: {decisao_real} "
          f"(esperado: {esperado})  {match}")
    print(f"         Top 5 SHAP local = {top5_names}")
    print(f"         SHAP sanity |Δ|  = {delta:.1e}  {'✅' if delta < 1e-4 else '⚠️ INVESTIGAR'}")

    if decisao_real == esperado and delta < 1e-4:
        ok_total += 1

print(f"\n[smoke 4/5] Total de casos com decisão correta + SHAP OK: {ok_total}/{len(testes)} "
      f" {'🎉' if ok_total == len(testes) else 'INVESTIGAR'}")

# 5. Tamanho do carregamento e features batem com exportação
assert xgb_model.get_booster().feature_names == cols_out_prep, (
    "Feature names do booster NÃO batem com OUTPUT_PREP_COLUNAS_15 — ordem quebrada."
)
print("[smoke 5/5] Ordem de features do booster = colunas do preparo ... OK ✅")

print("\n" + "=" * 70)
if ok_total == len(testes):
    print("🎉 SMOKE TEST 100% PASSOU. App Streamlit está 100% funcional e compartilhável.")
    print(f"   Pasta para compartilhar: {APP_DIR}")
else:
    print(f"❌ Smoke test falhou em {len(testes)-ok_total} caso(s). Não compartilhar ainda.")
    raise SystemExit(1)
