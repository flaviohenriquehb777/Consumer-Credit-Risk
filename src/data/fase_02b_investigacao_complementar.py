"""
Investigação complementar Fase 2:
  (a) Missing de renda e relação com inadimplência.
  (b) Código suspeito 98 nas colunas de atraso (mesmas linhas? taxa de inadimplência?).
SEED = 42
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.dpi"] = 100

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DADOS_PATH = os.path.join(BASE_DIR, "data", "raw", "credito_tratado.csv")
FIG_PATH = os.path.join(BASE_DIR, "reports", "figures")

df = pd.read_csv(DADOS_PATH)
ALVO = "inadimplente_2anos"

# ====================================================================
# ITEM A — Taxa de inadimplência: quem informou renda vs. quem NÃO informou
# ====================================================================
print("=" * 90)
print("ITEM A — INADIMPLÊNCIA: INFORMOU RENDA  vs.  NÃO INFORMOU RENDA")
print("=" * 90)

df["renda_informada_flag"] = ~df["renda_mensal"].isnull()   # True = informou

tabela_renda = (
    df.groupby("renda_informada_flag")[ALVO]
      .agg(
          Total="count",
          Inadimplentes="sum",
          Taxa_Inadimplencia=lambda s: (s.mean() * 100).round(2),
      )
      .reset_index()
)
tabela_renda["renda_informada_flag"] = tabela_renda["renda_informada_flag"].map(
    {True: "SIM — informou renda", False: "NÃO — missing de renda"}
)
tabela_renda.columns = ["Grupo", "N Clientes", "N Inadimplentes", "Taxa Inadimplência (%)"]
print("\nTabela comparativa:")
print(tabela_renda.to_string(index=False))

taxa_sim = tabela_renda.loc[tabela_renda["Grupo"].str.startswith("SIM"), "Taxa Inadimplência (%)"].values[0]
taxa_nao = tabela_renda.loc[tabela_renda["Grupo"].str.startswith("NÃO"), "Taxa Inadimplência (%)"].values[0]
diff_pp = taxa_nao - taxa_sim

print(f"\n>>> Taxa quem INFORMOU renda:   {taxa_sim:.2f}%")
print(f">>> Taxa quem NÃO informou renda: {taxa_nao:.2f}%")
print(f"\n>>> DIFERENÇA (não-informou − informou) = {diff_pp:.2f} pontos percentuais (pp).")
if diff_pp > 0:
    print(f">>> Ou seja: quem NÃO informa renda tem uma inadimplência {taxa_nao/taxa_sim:.2f}× MAIOR do que quem informa.")
else:
    print(f">>> Quem NÃO informa renda tem inadimplência {taxa_nao/taxa_sim:.2f}× menor.")

# Gráfico de barras
fig, ax = plt.subplots(figsize=(7, 4.8))
colors = ["#2ecc71", "#e74c3c"]
sns.barplot(x="Grupo", y="Taxa Inadimplência (%)", data=tabela_renda, ax=ax, palette=colors)
ax.set_title("Taxa de inadimplência por disponibilidade de renda_mensal", fontsize=13, pad=15)
for i, v in enumerate(tabela_renda["Taxa Inadimplência (%)"].values):
    n = tabela_renda["N Clientes"].iloc[i]
    ax.text(i, v + 0.08, f"{v:.2f}%\n(n={n:,})", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Taxa de inadimplência (%)")
ax.set_xlabel("")
plt.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "fase02_inadimplencia_renda_informada.png"), dpi=120)
print(f"\nGráfico salvo em: {os.path.join(FIG_PATH, 'fase02_inadimplencia_renda_informada.png')}")
plt.close(fig)

# ====================================================================
# ITEM B — Investigando código 98 nas colunas de atraso
# ====================================================================
print("\n" + "=" * 90)
print("ITEM B — INVESTIGANDO CÓDIGO 98 NAS 3 COLUNAS DE ATRASO")
print("=" * 90)

col_atrasos = [
    "atrasos_30_59_dias",
    "atrasos_60_89_dias",
    "atrasos_90_mais_dias",
]

print("\n>>> (B.1) Distribuição completa dos valores únicos de CADA coluna de atraso:")
for col in col_atrasos:
    vc = df[col].value_counts().sort_index()
    print(f"\n  · Coluna: {col}")
    print(f"    Valores únicos = {df[col].nunique()}")
    # Mostra valores >= 90 (suspeitos) sempre, e os mais frequentes
    top_5 = vc.head(5)
    print("    Top 5 valores mais frequentes:")
    for v, c in top_5.items():
        print(f"      valor={int(v):>3}  →  n={c:>7,}  ({c/len(df)*100:.3f}%)")
    suspeitos = vc[vc.index >= 90]
    if not suspeitos.empty:
        print("    Valores SUSPEITOS (>= 90 — impossível em 2 anos):")
        for v, c in suspeitos.items():
            print(f"      valor={int(v):>3}  →  n={c:>7,}  ({c/len(df)*100:.3f}%)")

print("\n>>> (B.2) Quantas linhas TÊM valor 98 em CADA UMA das colunas? Sobreposição?")
mask_98_30 = df["atrasos_30_59_dias"] == 98
mask_98_60 = df["atrasos_60_89_dias"] == 98
mask_98_90 = df["atrasos_90_mais_dias"] == 98
mask_98_qualquer = mask_98_30 | mask_98_60 | mask_98_90
mask_98_todas    = mask_98_30 & mask_98_60 & mask_98_90

print(f"  · Linhas com 98 em atrasos_30_59_dias : {mask_98_30.sum():>6,}  ({mask_98_30.mean()*100:.3f}%)")
print(f"  · Linhas com 98 em atrasos_60_89_dias : {mask_98_60.sum():>6,}  ({mask_98_60.mean()*100:.3f}%)")
print(f"  · Linhas com 98 em atrasos_90_mais_dias: {mask_98_90.sum():>6,}  ({mask_98_90.mean()*100:.3f}%)")
print(f"  · Linhas com 98 EM QUALQUER das 3    : {mask_98_qualquer.sum():>6,}  ({mask_98_qualquer.mean()*100:.3f}%)")
print(f"  · Linhas com 98 NAS 3 AO MESMO TEMPO : {mask_98_todas.sum():>6,}  ({mask_98_todas.mean()*100:.3f}%)")

# Mostra a combinação exata de quais colunas recebem 98 (matriz de sobreposição)
print("\n>>> (B.3) Sobreposição: combinação exata de colunas que contêm 98 (Venn simplificado):")
combos = (
    df.assign(
        tem_30=mask_98_30.astype(int),
        tem_60=mask_98_60.astype(int),
        tem_90=mask_98_90.astype(int),
    )
    .groupby(["tem_30", "tem_60", "tem_90"])
    .size()
    .reset_index(name="n_linhas")
    .sort_values("n_linhas", ascending=False)
)
combos["padrão"] = (
    combos["tem_30"].astype(str)
    + " | " + combos["tem_60"].astype(str)
    + " | " + combos["tem_90"].astype(str)
)
# Label legível
def legendar(r):
    cols = []
    if r["tem_30"]: cols.append("30-59")
    if r["tem_60"]: cols.append("60-89")
    if r["tem_90"]: cols.append("90+")
    return ", ".join(cols) if cols else "nenhuma (sadio)"

combos["legenda"] = combos.apply(legendar, axis=1)
combos["%"] = (combos["n_linhas"] / len(df) * 100).round(3)
print(combos[["legenda", "n_linhas", "%"]].to_string(index=False))

print("\n>>> (B.4) 10 exemplos de linhas que têm 98 em ALGUMA coluna de atraso (valores crus):")
col_exibir = [ALVO, "idade", "renda_mensal", "uso_limite_rotativo", "razao_divida"] + col_atrasos
exemplos = df.loc[mask_98_qualquer, col_exibir].head(10)
print(exemplos.to_string(index=False))

# ====================================================================
# ITEM C — Taxa de inadimplência das linhas com 98 vs. resto
# ====================================================================
print("\n" + "=" * 90)
print("ITEM C — TAXA DE INADIMPLÊNCIA: linhas COM 98 em alguma coluna de atraso vs. RESTO")
print("=" * 90)

grupos = pd.DataFrame({
    "grupo": np.where(mask_98_qualquer, "COM 98 em pelo menos 1 coluna de atraso", "RESTO (sem 98)"),
    ALVO:  df[ALVO].values,
})

tabela_98 = (
    grupos.groupby("grupo")[ALVO]
          .agg(N="count", Inadimplentes="sum", Taxa=lambda s: (s.mean() * 100).round(2))
          .reset_index()
)
tabela_98.columns = ["Grupo", "N Clientes", "N Inadimplentes", "Taxa Inadimplência (%)"]
print("\nComparação geral:")
print(tabela_98.to_string(index=False))

taxa_resto = tabela_98.loc[tabela_98["Grupo"].str.contains("RESTO"), "Taxa Inadimplência (%)"].values[0]
taxa_c98 = tabela_98.loc[tabela_98["Grupo"].str.contains("COM 98"), "Taxa Inadimplência (%)"].values[0]
diff_pp_98 = taxa_c98 - taxa_resto

print(f"\n>>> RESTO (sem 98 em atrasos) : {taxa_resto:.2f}%")
print(f">>> GRUPO COM 98 em atrasos   : {taxa_c98:.2f}%")
print(f">>> DIFERENÇA (com 98 − resto): {diff_pp_98:.2f} pp.")
print(f">>> Razão entre taxas: grupo com 98 é {taxa_c98/max(taxa_resto,1e-9):.2f}× o restante.")

# Detalhe: taxa por PADRÃO de 98 (qual(is) coluna(s) afetada(s))
grupos_detalhe = (
    combos.merge(
        (
            df.assign(legenda=df.index.map(lambda i: combos.loc[
                (combos["tem_30"] == (mask_98_30.iloc[i] * 1)) &
                (combos["tem_60"] == (mask_98_60.iloc[i] * 1)) &
                (combos["tem_90"] == (mask_98_90.iloc[i] * 1)),
                "legenda"
            ].iloc[0]))
        )
        .groupby("legenda")[ALVO]
        .agg(N="count", Inadimplentes="sum", Taxa=lambda s: (s.mean() * 100).round(2))
        .reset_index(),
        on="legenda",
        how="left",
    )
)
grupos_detalhe = grupos_detalhe[["legenda", "N", "Inadimplentes", "Taxa"]].copy()
grupos_detalhe.columns = ["Padrão (colunas com 98)", "N", "Inadimplentes", "Taxa Inadimplência (%)"]
# Para o padrão "nenhuma (sadio)" a taxa calculada bate com RESTO, ok.
print("\nTaxa detalhada por padrão de ocorrência do 98:")
print(grupos_detalhe.to_string(index=False))

# Gráfico
fig, ax = plt.subplots(figsize=(9, 4.8))
tabela_plot = grupos_detalhe.copy()
sns.barplot(x="Padrão (colunas com 98)", y="Taxa Inadimplência (%)", data=tabela_plot, ax=ax, palette="coolwarm")
ax.set_title("Taxa de inadimplência por padrão de 98 nas colunas de atraso", fontsize=13, pad=15)
for i, (v, n) in enumerate(zip(tabela_plot["Taxa Inadimplência (%)"].values, tabela_plot["N"].values)):
    ax.text(i, v + 0.08, f"{v:.2f}%\n(n={n:,})", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Taxa de inadimplência (%)")
ax.set_xlabel("")
plt.xticks(rotation=15)
plt.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "fase02_inadimplencia_codigo_98_atrasos.png"), dpi=120)
print(f"\nGráfico salvo em: {os.path.join(FIG_PATH, 'fase02_inadimplencia_codigo_98_atrasos.png')}")
plt.close(fig)

print("\nFIM das investigações complementares.")
