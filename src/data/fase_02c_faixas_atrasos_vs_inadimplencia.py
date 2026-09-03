"""
Análise visual: faixas (buckets) de atrasos históricos × taxa de inadimplência.
Para cada uma das 3 colunas de atraso:
  - agrupa por valor real (0, 1, 2, 3, 4+)
  - separa categoria "Código de sistema" para valores 96/98
  - plota volume por faixa (barras) e taxa de inadimplência (linha)
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

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.family"] = "DejaVu Sans"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DADOS_PATH = os.path.join(BASE_DIR, "data", "raw", "credito_tratado.csv")
FIG_PATH = os.path.join(BASE_DIR, "reports", "figures")

df = pd.read_csv(DADOS_PATH)
ALVO = "inadimplente_2anos"

# --- Definição das faixas ---
# Regra: valores 0-4 como categorias explícitas, 5-89 como "5+", e 96/98 como "Cód. sistema"
def bucketizar(v, max_val_real=4):
    if pd.isna(v):
        return "NA"
    if v in (96, 98):
        return "Cód. sistema (96/98)"
    if v <= max_val_real:
        return str(int(v))
    return f"{max_val_real+1}+"

col_atrasos = {
    "atrasos_30_59_dias":   "Atrasos 30–59 dias",
    "atrasos_60_89_dias":   "Atrasos 60–89 dias",
    "atrasos_90_mais_dias": "Atrasos ≥ 90 dias",
}
ORDEM_BUCKETS = ["0", "1", "2", "3", "4", "5+", "Cód. sistema (96/98)"]

# Paleta: Vermelho crescente com o atraso + cinza para cód sistema
PALETA_BARRAS = [
    "#2ecc71",   # 0
    "#b2f0a0",   # 1
    "#fff066",   # 2
    "#ffc145",   # 3
    "#ff8c42",   # 4
    "#ff5e5e",   # 5+
    "#8e44ad",   # cód de sistema
]

resultados = {}
for col, titulo in col_atrasos.items():
    bucket_col = f"{col}_faixa"
    df[bucket_col] = df[col].apply(bucketizar)

    tab = (
        df.groupby(bucket_col, dropna=False)[ALVO]
          .agg(n_clientes="count",
               n_inadimplentes="sum",
               taxa_inadimplencia=lambda s: s.mean() * 100)
          .reset_index()
    )
    tab.columns = ["Faixa", "N clientes", "N inadimplentes", "Taxa inadimplência (%)"]
    # Garante a ordem
    tab["ordem"] = tab["Faixa"].apply(lambda x: ORDEM_BUCKETS.index(x) if x in ORDEM_BUCKETS else 999)
    tab = tab.sort_values("ordem").drop(columns="ordem").reset_index(drop=True)

    # Arredonda p/ exibição
    tab_disp = tab.copy()
    tab_disp["Taxa inadimplência (%)"] = tab_disp["Taxa inadimplência (%)"].round(2)

    print("\n" + "=" * 90)
    print(f"COLUNA: {titulo}  ({col})")
    print("=" * 90)
    print(tab_disp.to_string(index=False))

    # Gráfico combinado: barras (volume) + linha (taxa %)
    fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
    ax2 = ax1.twinx()

    xs = list(range(len(tab)))
    labels = tab["Faixa"].tolist()

    bar_colors = [PALETA_BARRAS[ORDEM_BUCKETS.index(l)] if l in ORDEM_BUCKETS else "#999999"
                  for l in labels]
    bars = ax1.bar(xs, tab["N clientes"], color=bar_colors, alpha=0.9,
                   label="Nº de clientes", zorder=2)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, rotation=0, fontsize=10)
    ax1.set_ylabel("Nº de clientes (barras)", color="#555")
    ax1.tick_params(axis="y", labelcolor="#555")
    ax1.set_title(f"{titulo}: Nº de clientes × Taxa de inadimplência por faixa", fontsize=13, pad=15)

    # Rótulo de nª em cada barra
    for b in bars:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h, f"{h:,.0f}",
                 ha="center", va="bottom", fontsize=8, color="#333")

    # Linha da taxa
    linha, = ax2.plot(xs, tab["Taxa inadimplência (%)"], color="#c0392b",
                      marker="o", markersize=8, linewidth=2.2, label="Taxa de inadimplência", zorder=5)
    for i, t in enumerate(tab["Taxa inadimplência (%)"].values):
        ax2.annotate(f"{t:.1f}%", (xs[i], t),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=9.5, fontweight="bold", color="#c0392b")
    ax2.set_ylabel("Taxa de inadimplência % (linha)", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax2.set_ylim(0, max(60, tab["Taxa inadimplência (%)"].max() * 1.20))

    # Linha de referência = taxa média geral
    taxa_media = df[ALVO].mean() * 100
    ax2.axhline(taxa_media, color="#2c3e50", linestyle=":", linewidth=1.3, alpha=0.8,
                label=f"Taxa média geral = {taxa_media:.2f}%")

    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.95), fontsize=9, frameon=True)
    plt.tight_layout()
    figpath = os.path.join(FIG_PATH, f"fase02_faixas_{col}.png")
    fig.savefig(figpath, dpi=150, bbox_inches="tight")
    print(f">>> Gráfico salvo em: {figpath}")
    plt.close(fig)

    resultados[col] = tab_disp

# --- Gráfico conjunto (3 subplots lado a lado para comparação) ---
print("\n" + "=" * 90)
print("GRÁFICO CONJUNTO DAS 3 COLUNAS DE ATRASO")
print("=" * 90)

fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharey=False)
for idx, (col, titulo) in enumerate(col_atrasos.items()):
    ax = axes[idx]
    bucket_col = f"{col}_faixa"
    tab = resultados[col]
    xs = list(range(len(tab)))
    labels = tab["Faixa"].tolist()
    bar_colors = [PALETA_BARRAS[ORDEM_BUCKETS.index(l)] if l in ORDEM_BUCKETS else "#999999"
                  for l in labels]
    ax.bar(xs, tab["N clientes"], color=bar_colors, alpha=0.9, zorder=2)
    ax2 = ax.twinx()
    ax2.plot(xs, tab["Taxa inadimplência (%)"], color="#c0392b",
             marker="o", markersize=7, linewidth=2.0, zorder=5)
    for i, t in enumerate(tab["Taxa inadimplência (%)"].values):
        ax2.annotate(f"{t:.1f}%", (xs[i], t),
                     textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8.5, color="#c0392b", fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(f"{titulo}  ({col})", fontsize=11)
    ax.set_ylabel("Nº clientes", fontsize=9)
    ax2.set_ylabel("Inadimplência %", color="#c0392b", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    taxa_media = df[ALVO].mean() * 100
    ax2.axhline(taxa_media, color="#2c3e50", linestyle=":", linewidth=1.1, alpha=0.7)
fig.suptitle("Histórico de atrasos × taxa de inadimplência (por faixa)", fontsize=14, y=1.005)
plt.tight_layout()
figpath = os.path.join(FIG_PATH, "fase02_faixas_atrasos_conjunto.png")
fig.savefig(figpath, dpi=150, bbox_inches="tight")
print(f">>> Gráfico conjunto salvo em: {figpath}")
plt.close(fig)

# --- Correlação (Point-Biserial p/ colunas limpas; Spearman ignorando 96/98) ---
print("\n" + "=" * 90)
print("MEDIDAS DE ASSOCIAÇÃO (ignorando 96/98)")
print("=" * 90)

df_limpo_atrasos = df[
    (df["atrasos_30_59_dias"]  < 90) &
    (df["atrasos_60_89_dias"]  < 90) &
    (df["atrasos_90_mais_dias"] < 90)
].copy()

# Spearman com o alvo binário (equivale a point-biserial quando uma var é dicotômica)
corrs = {}
for col, titulo in col_atrasos.items():
    r = df_limpo_atrasos[[col, ALVO]].corr(method="spearman").iloc[0, 1]
    corrs[titulo] = r

tab_corr = pd.DataFrame(list(corrs.items()), columns=["Coluna de atraso", "Correlação Spearman com inadimplência"])
tab_corr = tab_corr.sort_values("Correlação Spearman com inadimplência", ascending=False)
tab_corr["Correlação Spearman com inadimplência"] = tab_corr["Correlação Spearman com inadimplência"].round(4)
print(tab_corr.to_string(index=False))

print("\nInterpretação rápida:")
print("  - ρ ≈ 0,00–0,10: associação desprezível")
print("  - ρ ≈ 0,10–0,20: associação fraca")
print("  - ρ ≈ 0,20–0,40: associação moderada")
print("  - ρ > 0,40: associação forte")
print("\nFIM da análise de faixas × inadimplência.")
