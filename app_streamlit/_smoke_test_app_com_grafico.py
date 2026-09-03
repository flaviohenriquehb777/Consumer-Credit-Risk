"""
Smoke test estendido do app (com SHAP graph ALT renderizado, sem servidor Streamlit).
Executa o mesmo fluxo do app mas apenas com objetos python, valida:
  1. Sintaxe OK
  2. Tudo carrega OK (3 casos)
  3. Gráfico ALT de TOP10 SHAP local é criado e tem as 10 barras corretas
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from preprocessamento import preparar_dados

with open(APP_DIR / "modelo_xgboost_vencedor_CLASSIFICADOR_PURO.pkl","rb") as f:
    xgb_model = pickle.load(f)
with open(APP_DIR / "config_MODELO_XGB_e_SHAP_consolidado.pkl","rb") as f:
    cfg = pickle.load(f)

import altair as alt
import shap
explainer = shap.TreeExplainer(xgb_model)
cols_in = cfg["INPUT_COLUNAS_OBRIGATORIAS_10"]
cols_out_prep = cfg["OUTPUT_PREP_COLUNAS_15"]
params_prep = cfg["PARAMS_PREPROCESSAMENTO"]
th = cfg["TH_OTIMO_XGB"]
exp_val = cfg["SHAP_expected_value_logodds"]

# Compilação do app
with open(APP_DIR / "app_risco_credito.py","r",encoding="utf-8") as f:
    source = f.read()
compile(source, str(APP_DIR / "app_risco_credito.py"), "exec")
print("[smoke 1/4] Sintaxe do app ............ OK ✅")

testes = [
    ("APROVAR",  cfg["EXEMPLO_CLIENTE_APROVAR"],  "APROVAR"),
    ("RECUSAR",  cfg["EXEMPLO_CLIENTE_RECUSAR"],  "RECUSAR"),
    ("MISSING+98",{
        "idade": 45, "renda_mensal": np.nan, "dependentes": np.nan,
        "uso_limite_rotativo": 0.75, "razao_divida": 0.55,
        "linhas_credito_abertas": 8, "financiamentos_imobiliarios": 2,
        "atrasos_30_59_dias": 98, "atrasos_60_89_dias": 98, "atrasos_90_mais_dias": 98,
    }, "RECUSAR"),
]

ok = 0
for nome, caso, esperado in testes:
    linha = pd.DataFrame([caso])[cols_in]
    X_tratado = preparar_dados(linha.copy(), params=params_prep, fit=False)[cols_out_prep]
    p = float(xgb_model.predict_proba(X_tratado)[0, 1])
    dec = "RECUSAR" if p >= th else "APROVAR"
    sv = explainer.shap_values(X_tratado)
    if isinstance(sv, list): sv = sv[1]
    sv_flat = np.asarray(sv).reshape(-1)
    valores = [round(float(X_tratado.iloc[0,c]),4) for c in range(len(cols_out_prep))]
    shap_df = pd.DataFrame({
        "Feature": cols_out_prep,
        "Valor na Feature": valores,
        "Contribuição SHAP (log-odds)": [round(float(v),5) for v in sv_flat],
    })
    shap_df["|SHAP|"] = shap_df["Contribuição SHAP (log-odds)"].abs()

    # Monta gráfico igual ao app (TOP 10 por |SHAP|)
    shap_graf = (
        shap_df.sort_values("|SHAP|", ascending=True).tail(10)
        .assign(direcao=lambda d: d["Contribuição SHAP (log-odds)"].apply(
            lambda v: "aumenta risco (↑ calote)" if v>0 else
                      ("diminui risco (↓ calote)" if v<0 else "neutro")
        )).reset_index(drop=True)
    )
    faixa = float(max(
        abs(shap_graf["Contribuição SHAP (log-odds)"].min()),
        abs(shap_graf["Contribuição SHAP (log-odds)"].max()),
        0.01,
    ))
    bars = (alt.Chart(shap_graf)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusBottomLeft=4)
        .encode(
            x=alt.X("Contribuição SHAP (log-odds):Q",
                    scale=alt.Scale(domain=[-faixa*1.05, faixa*1.05])),
            y=alt.Y("Feature:N", sort=None),
            color=alt.Color("direcao:N",
                scale=alt.Scale(
                    domain=["aumenta risco (↑ calote)","diminui risco (↓ calote)","neutro"],
                    range=["#c0392b","#27ae60","#95a5a6"])),
            tooltip=["Feature:N","Valor na Feature:Q","Contribuição SHAP (log-odds):Q","direcao:N"],
        )
        .properties(height=340)
    )
    zero = (alt.Chart(pd.DataFrame({"zero":[0.0]}))
        .mark_rule(color="#2c3e50", strokeDash=[2,2], size=1.1).encode(x="zero:Q"))
    chart = bars + zero

    # Salva em JSON (valida que Altair renderizou). Camada 0 pode ter dados em
    # dataset + name ou inline values; aceitamos ambos.
    jspec = chart.to_dict()
    def conta_barras_na_camada(camada):
        if "data" in camada and "values" in camada["data"]:
            return len(camada["data"]["values"])
        nome_ds = camada.get("data", {}).get("name")
        if nome_ds and "datasets" in jspec and nome_ds in jspec["datasets"]:
            return len(jspec["datasets"][nome_ds])
        return -1
    tem_10_barras = conta_barras_na_camada(jspec["layer"][0]) == 10
    has_zero_line = "layer" in jspec and len(jspec["layer"]) == 2
    delta_shap = abs(float(np.log(p/max(1e-9,1-p)) - float(exp_val + sv_flat.sum())))

    status = "✅" if (dec==esperado and delta_shap<1e-4 and tem_10_barras and has_zero_line) else "❌"
    print(f"[smoke 2/4] Caso {nome:10s} | P={p*100:6.2f}% → {dec:8s} (esperado {esperado}) | "
          f"gráfico TOP10={tem_10_barras}/{has_zero_line} | |ΔSHAP|={delta_shap:.1e}  {status}")
    if dec==esperado and delta_shap<1e-4 and tem_10_barras and has_zero_line:
        ok += 1

print(f"[smoke 3/4] Resultados: {ok}/3 casos passaram.")
print(f"[smoke 4/4] Título do app contém 'XGBoost Campeão'? "
      f"({'SIM (ainda não removido) ❌' if 'XGBoost Campeão' in source else 'NÃO (removido) ✅'})")
if "(XGBoost Campeão)" in source:
    print("  ⚠️  Aviso: rastro de 'XGBoost Campeão' em outra parte que não o README.md.")

if ok == 3:
    print("\n🎉 SMOKE TEST EXTENDIDO 100% PASSOU. App com novos ajustes PRONTO.")
else:
    raise SystemExit(1)
