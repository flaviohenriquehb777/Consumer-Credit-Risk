"""
app_risco_credito.py — App Streamlit: Avaliação de Risco de Crédito
===================================================================

Como rodar (na pasta onde está este arquivo):
    1. pip install -r requirements.txt
    2. streamlit run app_risco_credito.py

Entrada: 10 campos do cliente (formulário).
Saída: 3 resultados por cliente:
  1. P(inadimplência 2 anos) — %
  2. Decisão: APROVAR / RECUSAR — com base no limiar 0,56 (aprendido OOF treino,
     custo FN = R$ 5.000 / FP = R$ 500).
  3. TOP 5 features que mais explicaram essa decisão para ESTE cliente (SHAP local)
     em TABELA + GRÁFICO de barras horizontais (fácil visualização).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# 0. Configuração da página (sempre o PRIMEIRO comando streamlit)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Avaliação de Risco de Crédito",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# 1. Carregamento dos artefatos — 1 única vez, cacheado pelo Streamlit
# ------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))


@st.cache_resource(show_spinner="Carregando modelo + SHAP...")
def carregar_tudo():
    """Carrega modelo + config + prepara SHAP Explainer.

    Retornos são imutáveis durante a sessão.
    """
    from preprocessamento import preparar_dados  # arquivo local na mesma pasta

    with open(APP_DIR / "modelo_xgboost_vencedor_CLASSIFICADOR_PURO.pkl", "rb") as f:
        xgb_model = pickle.load(f)

    with open(APP_DIR / "config_MODELO_XGB_e_SHAP_consolidado.pkl", "rb") as f:
        cfg = pickle.load(f)

    params_prep = cfg["PARAMS_PREPROCESSAMENTO"]
    colunas_entrada = cfg["INPUT_COLUNAS_OBRIGATORIAS_10"]
    colunas_saida_prep = cfg["OUTPUT_PREP_COLUNAS_15"]
    th = cfg["TH_OTIMO_XGB"]
    expected_value = cfg["SHAP_expected_value_logodds"]
    shap_top10_global = list(zip(
        cfg["SHAP_top10_features_global"],
        cfg["SHAP_top10_mean_abs_shap_global"],
    ))

    import shap
    explainer = shap.TreeExplainer(xgb_model)

    return dict(
        preparar_dados=preparar_dados,
        xgb_model=xgb_model,
        cfg=cfg,
        params_prep=params_prep,
        colunas_entrada=colunas_entrada,
        colunas_saida_prep=colunas_saida_prep,
        th=th,
        expected_value=expected_value,
        shap_top10_global=shap_top10_global,
        explainer=explainer,
    )


estado = carregar_tudo()
cfg = estado["cfg"]

# ------------------------------------------------------------------
# 2. Cabeçalho
# ------------------------------------------------------------------
st.title("🏦  Avaliação de Risco de Crédito")
st.markdown(
    f"""
    **Modelo:** {cfg["modelo_nome"]}

    Métricas OFICIAIS no HOLDOUT cego (37.500 clientes):
    - ROC AUC = **{cfg["metricas_oficiais_holdout_n37500"]["ROC_AUC"]:.5f}**
    - PR AUC  = **{cfg["metricas_oficiais_holdout_n37500"]["PR_AUC"]:.5f}**
    - Economia de custo vs política "aprova todos" = **{cfg["metricas_oficiais_holdout_n37500"]["economia_custo_pct_vs_aprova_todos"]:.2f}%**

    **Limiar de decisão (TH) = {estado["th"]:.2f}**  ·  Custo FN = R$ {cfg["CUSTO_FN_RS"]:,} / FP = R$ {cfg["CUSTO_FP_RS"]:,}
    """,
    unsafe_allow_html=False,
)

with st.expander("ℹ️ Como é a decisão?", expanded=False):
    TH_DISPLAY = float(estado["th"])
    st.write(
        f"- P(inadimplência) **≥ {TH_DISPLAY:.2f}**  →  **:red[RECUSAR CRÉDITO]** (risco alto).\n"
        f"- P(inadimplência) **< {TH_DISPLAY:.2f}**  →  **:green[APROVAR CRÉDITO]** (risco baixo).\n"
        f"\n Este limiar foi calculado na Fase 5 por grid 0,01→0,99 SOMENTE no OOF treino "
        f"(StratifiedKFold 5-fold, n=112.500 clientes) e validado 1 única vez no holdout. "
        f"Custo esperado = FN×R$5.000 + FP×R$500. Em hipótese nenhuma o limiar foi ajustado "
        f"olhando para o holdout (SEM LEAKAGE)."
    )

st.divider()

# ------------------------------------------------------------------
# 3. Botões auxiliares: carregar exemplos
# ------------------------------------------------------------------
col_b1, col_b2, col_b3 = st.columns([1, 1, 10])
with col_b1:
    if st.button("📥 Exemplo: APROVAR", help="Preenche um caso aprovável automaticamente."):
        for k, v in cfg["EXEMPLO_CLIENTE_APROVAR"].items():
            st.session_state[f"f__{k}"] = v
        st.rerun()
with col_b2:
    if st.button("📥 Exemplo: RECUSAR", help="Preenche um caso de alto risco (recusa)."):
        for k, v in cfg["EXEMPLO_CLIENTE_RECUSAR"].items():
            st.session_state[f"f__{k}"] = v
        st.rerun()

# ------------------------------------------------------------------
# 4. Formulário: 10 campos obrigatórios  (2 colunas para ficar mais compacto)
# ------------------------------------------------------------------
with st.form("formulario_cliente"):
    st.subheader("📝 Informações do cliente")
    c1, c2 = st.columns(2)

    with c1:
        idade = st.number_input(
            "Idade (anos)",
            min_value=18, max_value=110, value=st.session_state.get("f__idade", 40), step=1,
            help="18 a 110. Entrada BRUTA; nenhum tratamento prévio por parte do analista.",
        )
        renda_mensal = st.number_input(
            "Renda mensal (R$)",
            min_value=0.0, max_value=5_000_000.0,
            value=float(st.session_state.get("f__renda_mensal", 5_400.0)),
            step=100.0, format="%.2f",
            help="Deixe vazio / 0 se o cliente NÃO informou renda (o modelo trata o missing automaticamente).",
        )
        dependentes = st.number_input(
            "Dependentes (qtd pessoas)",
            min_value=0, max_value=20,
            value=int(st.session_state.get("f__dependentes", 2)), step=1,
        )
        uso_limite_rotativo = st.number_input(
            "Uso do limite rotativo (0 a 1, aceita >1 = estouro)",
            min_value=0.0, max_value=10.0,
            value=float(st.session_state.get("f__uso_limite_rotativo", 0.30)),
            step=0.01, format="%.4f",
            help="razão saldo_utilizado / limite_aprovado. 0.30 = 30% do limite usado. Valores >1 são estouro legítimos.",
        )
        razao_divida = st.number_input(
            "Razão dívida / renda  (Debt Ratio)",
            min_value=0.0, max_value=10.0,
            value=float(st.session_state.get("f__razao_divida", 0.40)),
            step=0.01, format="%.4f",
        )

    with c2:
        linhas_credito_abertas = st.number_input(
            "Linhas de crédito abertas (qtd)",
            min_value=0, max_value=60,
            value=int(st.session_state.get("f__linhas_credito_abertas", 5)), step=1,
        )
        financiamentos_imobiliarios = st.number_input(
            "Financiamentos imobiliários (qtd)",
            min_value=0, max_value=15,
            value=int(st.session_state.get("f__financiamentos_imobiliarios", 1)), step=1,
        )
        atrasos_30_59 = st.number_input(
            "Atrasos de 30 a 59 dias  (96/98 = código sistema)",
            min_value=0, max_value=98,
            value=int(st.session_state.get("f__atrasos_30_59_dias", 0)), step=1,
        )
        atrasos_60_89 = st.number_input(
            "Atrasos de 60 a 89 dias  (96/98 = código sistema)",
            min_value=0, max_value=98,
            value=int(st.session_state.get("f__atrasos_60_89_dias", 0)), step=1,
        )
        atrasos_90_mais = st.number_input(
            "Atrasos de 90+ dias  (96/98 = código sistema)",
            min_value=0, max_value=98,
            value=int(st.session_state.get("f__atrasos_90_mais_dias", 0)), step=1,
        )

    st.caption(
        "⚠️ Campos de atraso: **não arredonde / não normalize**. "
        "Se o sistema de origem retornar o código 96 ou 98, informe ASSIM — o "
        "preprocessamento detecta automaticamente e cria a flag "
        "`cod_sistema_atrasos` (melhor preditor de risco já medido)."
    )

    # ----- Botão de submit -----
    enviar = st.form_submit_button(
        "🔍  Avaliar risco deste cliente", type="primary", use_container_width=True,
    )

# ------------------------------------------------------------------
# 5. Computação: preparo → predict → SHAP
# ------------------------------------------------------------------
if enviar:
    preparar = estado["preparar_dados"]
    xgb_model = estado["xgb_model"]
    params_prep = estado["params_prep"]
    cols_in = estado["colunas_entrada"]
    cols_out_prep = estado["colunas_saida_prep"]
    th = estado["th"]
    explainer = estado["explainer"]
    expected_value = estado["expected_value"]

    # 5.a) Monta DataFrame 1 linha com as 10 colunas na ORDEM CORRETA
    linha = pd.DataFrame(
        [{
            "idade": idade,
            "renda_mensal": renda_mensal if renda_mensal > 0 else np.nan,
            "dependentes": dependentes,
            "uso_limite_rotativo": uso_limite_rotativo,
            "razao_divida": razao_divida,
            "linhas_credito_abertas": linhas_credito_abertas,
            "financiamentos_imobiliarios": financiamentos_imobiliarios,
            "atrasos_30_59_dias": atrasos_30_59,
            "atrasos_60_89_dias": atrasos_60_89,
            "atrasos_90_mais_dias": atrasos_90_mais,
        }]
    )[cols_in]

    # 5.b) Prepara → 15 colunas
    with st.spinner("Aplicando preparação dos dados (flags/caps/features novas)..."):
        X_tratado = preparar(linha.copy(), params=params_prep, fit=False)
        X_tratado = X_tratado[cols_out_prep]

    # 5.c) Predict probabilidade
    with st.spinner("Rodando modelo..."):
        p = float(xgb_model.predict_proba(X_tratado)[0, 1])
    risco_alto = p >= th
    decisao = "RECUSAR CRÉDITO" if risco_alto else "APROVAR CRÉDITO"
    cor = ":red[" if risco_alto else ":green["
    tag = cor + decisao + "]"

    # 5.d) SHAP LOCAL da QUELA LINHA (TreeExplainer exato → <10ms p/ 1 cliente)
    with st.spinner("Calculando explicabilidade SHAP local (top 5 drivers)..."):
        sv = explainer.shap_values(X_tratado)            # shape (1, 15)
        if isinstance(sv, list):
            sv = sv[1]                                     # compatibilidade versões antigas SHAP
        sv_flat = np.asarray(sv).reshape(-1)              # (15,)

        # Monta tabela SHAP local
        valores_features = [round(float(X_tratado.iloc[0, c]), 4) for c in range(len(cols_out_prep))]
        shap_df = pd.DataFrame({
            "Feature": cols_out_prep,
            "Valor na Feature": valores_features,
            "Contribuição SHAP (log-odds)": [round(float(v), 5) for v in sv_flat],
        })
        shap_df["|SHAP|"] = shap_df["Contribuição SHAP (log-odds)"].abs()
        shap_top5 = (
            shap_df.sort_values("|SHAP|", ascending=False)
            .drop(columns=["|SHAP|"])
            .head(5)
            .reset_index(drop=True)
        )
        shap_top5.index = shap_top5.index + 1
        shap_top5.index.name = "Ranking"

        # Interpretação textual: ↑ aumenta risco / ↓ diminui risco
        def interpreta(v: float) -> str:
            if v > 0:
                return "↑ aumenta risco de calote"
            elif v < 0:
                return "↓ diminui risco de calote"
            return "— neutro"

        shap_top5["Impacto no Score"] = shap_top5["Contribuição SHAP (log-odds)"].apply(interpreta)

        # Check de sanidade SHAP: expected + Σ shap ≈ logit(p). Mostra ao usuário só o delta.
        soma_shap = float(sv_flat.sum())
        logodds_pred = float(np.log(p / max(1e-9, 1 - p)))
        logodds_shap = float(expected_value + soma_shap)
        delta = abs(logodds_pred - logodds_shap)

    # ------------------------------------------------------------------
    # 6. Apresentação do resultado
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("🎯 Resultado da avaliação")

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Probabilidade de inadimplência (2 anos)",
        f"{p*100:.2f}%",
        delta=f"Limite = {th*100:.0f}%",
        delta_color="inverse",
        help="P(Y=1 | X) de o cliente ficar 90+ dias em atraso nos próximos 2 anos.",
    )
    m2.metric(
        "Decisão do modelo",
        tag,
        delta=(
            "Risco alto — acima do limiar"
            if risco_alto else
            "Risco baixo — abaixo do limiar"
        ),
        delta_color="off",
    )
    m3.metric(
        "Consistência SHAP (sanity check)",
        ("OK ✅" if delta < 1e-4 else "INVESTIGAR ⚠️"),
        delta=f"|Δ log-odds| = {delta:.1e}",
        help="|expected_value + Σ SHAP − logit(p)| < 1e-4 garante que a explicação "
             "reproduz o score. Se não, há bug de ordem de features no deploy.",
    )

    st.subheader("🧭 Principais causas da decisão (TOP 5 SHAP LOCAL deste cliente)")
    st.caption(
        "Interprete: valores **positivos** = aquele atributo AUMENTOU a chance de calote "
        "(empurrou a decisão para RECUSAR). Valores **negativos** = aquele atributo "
        "DIMINUIU a chance de calote (ajudou a APROVAR). Ranking é por impacto absoluto."
        " À esquerda: tabela detalhada. À direita: gráfico TOP 10 em barras horizontais."
    )

    def colorir_shap(val):
        try:
            v = float(val)
        except Exception:
            return ""
        if v > 0:
            return "color: #c0392b; font-weight: 600;"
        if v < 0:
            return "color: #27ae60; font-weight: 600;"
        return ""

    col_shap_tab, col_shap_fig = st.columns([1.1, 1.2], gap="large")
    with col_shap_tab:
        st.markdown("**Tabela TOP 5 detalhada**")
        st.dataframe(
            shap_top5.style.map(colorir_shap, subset=["Contribuição SHAP (log-odds)"]),
            use_container_width=True,
            height=220,
        )

    with col_shap_fig:
        import altair as alt
        st.markdown("**Gráfico TOP 10 — força do impacto no score**")
        shap_ordenado_graf = (
            shap_df
            .sort_values("|SHAP|", ascending=True)     # ascending True p/ barra horizontal TOP no topo
            .tail(10)
            .assign(
                direcao=lambda d: d["Contribuição SHAP (log-odds)"].apply(
                    lambda v: "aumenta risco (↑ calote)" if v > 0
                    else ("diminui risco (↓ calote)" if v < 0 else "neutro")
                )
            )
            .reset_index(drop=True)
        )
        faixa = float(max(
            abs(shap_ordenado_graf["Contribuição SHAP (log-odds)"].min()),
            abs(shap_ordenado_graf["Contribuição SHAP (log-odds)"].max()),
            0.01,
        ))
        grafico_shap = (
            alt.Chart(shap_ordenado_graf)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusBottomLeft=4)
            .encode(
                x=alt.X(
                    "Contribuição SHAP (log-odds):Q",
                    title="Contribuição SHAP (log-odds)  ⟵  reduz risco  ·  base  ·  aumenta risco  ⟶",
                    scale=alt.Scale(domain=[-faixa * 1.05, faixa * 1.05]),
                    axis=alt.Axis(grid=True),
                ),
                y=alt.Y("Feature:N", title=None, sort=None),
                color=alt.Color(
                    "direcao:N",
                    title="Direção do impacto",
                    scale=alt.Scale(
                        domain=["aumenta risco (↑ calote)", "diminui risco (↓ calote)", "neutro"],
                        range=["#c0392b", "#27ae60", "#95a5a6"],
                    ),
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=[
                    alt.Tooltip("Feature:N", title="Feature"),
                    alt.Tooltip("Valor na Feature:Q", title="Valor na Feature", format=",.4f"),
                    alt.Tooltip("Contribuição SHAP (log-odds):Q", title="Contribuição SHAP", format=",.4f"),
                    alt.Tooltip("direcao:N", title="Direção do impacto"),
                ],
            )
            .properties(height=340)
        )
        linha_base_zero = (
            alt.Chart(pd.DataFrame({"zero": [0.0]}))
            .mark_rule(color="#2c3e50", strokeDash=[2, 2], size=1.1)
            .encode(x="zero:Q")
        )
        st.altair_chart(grafico_shap + linha_base_zero, use_container_width=True, theme=None)

    with st.expander("🔧 Dados brutos preparados (15 colunas — auditável pelo time de risco)"):
        st.dataframe(X_tratado.T.rename(columns={0: "Valor"}), use_container_width=True, height=550)

    with st.expander("📊 Ranking GLOBAL Top 10 Features da base toda (histórico holdout 37.500)"):
        global_df = pd.DataFrame(
            estado["shap_top10_global"],
            columns=["Feature", "|mean SHAP| (impacto global)"],
        ).round(5)
        global_df.index = global_df.index + 1
        global_df.index.name = "Rank"
        st.dataframe(global_df, use_container_width=True, height=380)
        st.caption(
            "Este ranking NÃO serve para explicar o cliente INDIVIDUAL (use o TOP 5 acima). "
            "Serve apenas para a área de Risco entender o comportamento coletivo do modelo na base."
        )
