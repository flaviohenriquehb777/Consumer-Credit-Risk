"""
FASE 2 — Entendimento dos Dados
Projeto: Modelo de Risco de Crédito (Aurora Crédito Digital)
SEED = 42
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# --- Reprodutibilidade ---
SEED = 42
np.random.seed(SEED)

# --- Configurações de visualização ---
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.family"] = "DejaVu Sans"

# --- Caminhos ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DADOS_PATH = os.path.join(BASE_DIR, "data", "raw", "credito_tratado.csv")
FIG_PATH = os.path.join(BASE_DIR, "reports", "figures")

# ============================================================
# ITEM 1 — Carregar CSV, shape e primeiras linhas
# ============================================================
print("=" * 80)
print("ITEM 1 — CARREGAMENTO, SHAPE E PRIMEIRAS LINHAS")
print("=" * 80)

df = pd.read_csv(DADOS_PATH)

print(f"\nNúmero de LINHAS  (clientes):  {df.shape[0]:,}")
print(f"Número de COLUNAS (features): {df.shape[1]}")
print(f"\nMemória utilizada: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

print("\n>>> Primeiras 5 linhas da base:")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(df.head())

# ============================================================
# ITEM 2 — Tipos e dicionário de dados (significado, unidade)
# ============================================================
print("\n" + "=" * 80)
print("ITEM 2 — TIPOS DE DADOS, SIGNIFICADO E UNIDADE DE CADA COLUNA")
print("=" * 80)

dicionario = {
    "inadimplente_2anos":   ("Inteiro binário",  "1=inadimplente 90+ DPD em até 2 anos; 0=adimplente",   "Flag (0/1)"),
    "idade":                ("Inteiro",          "Idade do cliente no momento da avaliação de crédito", "Anos"),
    "renda_mensal":         ("Contínuo",         "Renda mensal informada ou estimada pelo cliente",   "Reais (R$)"),
    "dependentes":          ("Discreto",         "Número de dependentes declarados",                  "Pessoas"),
    "uso_limite_rotativo":  ("Contínuo",         "Proporção do limite rotativo já utilizada",         "% / 1 (0–1; pode >1 se estourado)"),
    "razao_divida":         ("Contínuo",         "Relação entre comprometimento financeiro e renda",  "Razão (sem unidade)"),
    "linhas_credito_abertas": ("Inteiro",       "Quantidade de linhas de crédito ativas no histórico","Unidades"),
    "financiamentos_imobiliarios": ("Inteiro", "Número de financiamentos imobiliários registrados", "Unidades"),
    "atrasos_30_59_dias":   ("Inteiro",          "Quantidade de episódios de atraso 30–59 dias",      "Unidades"),
    "atrasos_60_89_dias":   ("Inteiro",          "Quantidade de episódios de atraso 60–89 dias",      "Unidades"),
    "atrasos_90_mais_dias": ("Inteiro",          "Quantidade de episódios de atraso >= 90 dias",       "Unidades"),
}

tipos = df.dtypes.reset_index()
tipos.columns = ["Coluna", "Tipo Pandas"]
tipos["Significado"] = tipos["Coluna"].map(lambda c: dicionario.get(c, ("", "", ""))[1])
tipos["Classificação"] = tipos["Coluna"].map(lambda c: dicionario.get(c, ("", "", ""))[0])
tipos["Unidade"] = tipos["Coluna"].map(lambda c: dicionario.get(c, ("", "", ""))[2])

print("\n>>> Tabela completa:")
print(tipos.to_string(index=False))

# ============================================================
# ITEM 3 — Distribuição do alvo (inadimplente_2anos)
# ============================================================
print("\n" + "=" * 80)
print("ITEM 3 — DISTRIBUIÇÃO DO ALVO: inadimplente_2anos")
print("=" * 80)

alvo = "inadimplente_2anos"
contagem = df[alvo].value_counts().sort_index()
percentual = (contagem / len(df) * 100).round(2)

tabela_alvo = pd.DataFrame({
    "Classe": contagem.index.map({0: "Adimplente (0)", 1: "Inadimplente (1)"}),
    "Contagem": contagem.values,
    "Percentual (%)": percentual.values,
})
print(tabela_alvo.to_string(index=False))

n_adimplentes = contagem[0]
n_inadimplentes = contagem[1]
razao = n_adimplentes / n_inadimplentes if n_inadimplentes > 0 else np.inf
print(f"\n>>> Razão de desequilíbrio: 1 inadimplente para cada {razao:.2f} adimplentes.")
print(f">>> Representatividade classe positiva: {percentual[1]:.2f}%")

# Gráfico barras do alvo — salvar em reports/figures
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.countplot(x=alvo, data=df, ax=ax, palette=["#2ecc71", "#e74c3c"])
ax.set_title("Distribuição da variável-alvo inadimplente_2anos", fontsize=13, pad=15)
ax.set_xlabel("Classe")
ax.set_ylabel("Frequência")
ax.set_xticklabels(["Adimplente (0)", "Inadimplente (1)"])
for p in ax.patches:
    ax.annotate(f"{p.get_height():,}\n({p.get_height()/len(df)*100:.2f}%)",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "fase02_distribuicao_alvo.png"))
print(f"\nGráfico salvo em: {os.path.join(FIG_PATH, 'fase02_distribuicao_alvo.png')}")
plt.close(fig)

# ============================================================
# ITEM 4 — Acurácia do modelo "chuta sempre adimplente" (linha de base)
# ============================================================
print("\n" + "=" * 80)
print("ITEM 4 — ACURÁCIA DO MODELO DE UMA LINHA (chuta NINGUÉM DÁ CALOTE)")
print("=" * 80)

acuracia_baseline = (df[alvo] == 0).mean() * 100
predominante = 0 if contagem[0] >= contagem[1] else 1

print(f"\nRegra do modelo baseline: prever sempre '{predominante}' (Adimplente).")
print(f"\n>>> Acurácia do modelo de linha de base (tudo 0): {acuracia_baseline:.2f}%")
print(f">> Em outras palavras: chutar que ninguém calota já acerta {acuracia_baseline:.2f}% das vezes.")

# ============================================================
# ITEM 5 — Consequências para a escolha de métricas
# ============================================================
print("\n" + "=" * 80)
print("ITEM 5 — CONSEQUÊNCIAS PARA A ESCOLHA DE MÉTRICAS NESTE PROJETO")
print("=" * 80)

texto_metricas = r"""
Devido ao forte desequilíbrio de classes ({perc_pos:.2f}% positivas contra {perc_neg:.2f}% negativas),
a ACURÁCIA é uma métrica ENGANOSA: um modelo inútil (baseline de uma linha) já entrega {ac:.2f}%.
Se usarmos acurácia como critério, seremos tentados a produzir um modelo que quase nunca acusa
inadimplência — deixando passar muitos falsos negativos, que custam 10x mais caro.

MÉTRICAS QUE VAMOS USAR:

1) ROC AUC (critério técnico obrigatório ≥ 0,85)
   Mede: capacidade de ORDENAÇÃO do modelo — quanto maior, melhor ele separa clientes bons
         de maus pagadores independentemente do ponto de corte escolhido.
   Cuidado: pode ser "inflado" artificialmente em dados com pouquíssimos positivos; não
            diz nada sobre o custo econômico da decisão (não usa custos FN/FP).

2) PRECISÃO (Precision)
   Mede: dos clientes que o modelo MARCOU como "ruins", quantos realmente calotaram?
         (TP / (TP + FP)).
   Cuidado: foca apenas no lado do "negado" (FP); um modelo que nega só 1 pessoa com
            certeza terá 100% de precisão e não serve para nada.

3) REVOGAÇÃO / SENSAIBILIDADE (Recall / Sensitivity)
   Mede: de todos os VERDADEIROS calotes existentes, quantos o modelo conseguiu pegar?
         (TP / (TP + FN)).
   Cuidado: se você aumentar recall cegamente (baixar ponto de corte até quase tudo ser
            marcado como ruim), explode o número de FPs — muita gente boa negada.

4) F1-SCORE
   Mede: média harmônica entre Precisão e Revogação (2·P·R/(P+R)).
   Cuidado: trata FN e FP com o MESMO peso — mas aqui FN é 10× mais caro que FP.
            Usaremos apenas como comparativo, NÃO como decisório.

5) MATRIZ DE CONFUSÃO + CUSTO ESPERADO DA CARTEIRA (critério de NEGÓCIO)
   Mede: custo total = (N_FN × 10) + (N_FP × 1).
   Cuidado: depende DO PONTO DE CORTE escolhido pela área de Risco. É aqui que o modelo
            realmente se vira decisão economicamente defensável.

6) Brier Score (log loss / probabilidade calibrada)
   Mede: quão bem CALIBRADAS são as probabilidades — um cliente com 18% de score deve
         realmente ter ~18% de chance de calote.
   Cuidado: fundamental para pricing e para o analista confiar no número que aparece na tela.
""".format(
    perc_pos=percentual[1],
    perc_neg=percentual[0],
    ac=acuracia_baseline,
)
print(texto_metricas)

# ============================================================
# ITEM 6 — Tabela de valores NULOS por coluna
# ============================================================
print("=" * 80)
print("ITEM 6 — COLUNAS COM VALORES NULOS (quantidade e percentual)")
print("=" * 80)

nulos = df.isnull().sum()
nulos_pct = (nulos / len(df) * 100).round(2)
tabela_nulos = pd.DataFrame({
    "Coluna": nulos.index,
    "Qtd Nulos": nulos.values,
    "% Nulos": nulos_pct.values,
})
tabela_nulos = tabela_nulos[tabela_nulos["Qtd Nulos"] > 0].sort_values("% Nulos", ascending=False).reset_index(drop=True)

print("\nColunas que CONTÊM valores faltantes:")
print(tabela_nulos.to_string(index=False))

# Gráfico barras de nulos
if not tabela_nulos.empty:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x="% Nulos", y="Coluna", data=tabela_nulos, ax=ax, palette="Reds_r")
    ax.set_title("Percentual de valores nulos por coluna", fontsize=13, pad=15)
    ax.set_xlabel("% de Nulos")
    for i, v in enumerate(tabela_nulos["% Nulos"].values):
        ax.text(v + 0.1, i, f"{v}%  (n={tabela_nulos['Qtd Nulos'].iloc[i]:,})", va="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_PATH, "fase02_valores_nulos.png"))
    print(f"\nGráfico salvo em: {os.path.join(FIG_PATH, 'fase02_valores_nulos.png')}")
    plt.close(fig)

# ============================================================
# ITEM 7 — Podemos simplesmente descartar as linhas com nulos?
# ============================================================
print("\n" + "=" * 80)
print("ITEM 7 — PODEMOS DESCARTAR LINHAS COM NULOS? QUANTOS CLIENTES PERDERÍAMOS?")
print("=" * 80)

linhas_com_algum_nulo = df.isnull().any(axis=1).sum()
pct_linhas_nulo = (linhas_com_algum_nulo / len(df) * 100).round(2)

print(f"\nTotal de linhas (clientes):  {len(df):,}")
print(f"Linhas com PELO MENOS 1 nulo: {linhas_com_algum_nulo:,} ({pct_linhas_nulo}%)")
print(f"Linhas COMPLETAS (sem nulo): {len(df) - linhas_com_algum_nulo:,} ({(100 - pct_linhas_nulo):.2f}%)")
print("\n>>> Análise:")
print(f"  Simplesmente dropar as linhas com missing = jogar fora {linhas_com_algum_nulo:,} clientes ({pct_linhas_nulo}% da base).")
print(f"  Se o missing NÃO for completamente aleatório (MCAR), estamos introduzindo VIÉS de seleção.")
print(f"  Colunas 'renda_mensal' e 'dependentes' têm natureza informativa: quem NÃO declara renda já tem um sinal.")
print(f"  Estratégia recomendada: (a) flag binária 'renda_ausente' + (b) imputação mediana/mediana; mesmo para dependentes.")

# ============================================================
# ITEM 8 — Outros problemas detectados (estatísticas descritivas + outliers)
# ============================================================
print("\n" + "=" * 80)
print("ITEM 8 — OUTROS PROBLEMAS IDENTIFICADOS NA BASE")
print("=" * 80)

print("\n>>> ESTATÍSTICAS DESCRITIVAS (todas as colunas numéricas):")
desc = df.describe().T
desc["missing"] = df.isnull().sum()
desc["missing_pct"] = (df.isnull().sum() / len(df) * 100).round(2)
print(desc.round(2).to_string())

print("\n>>> CHECAGEM DE OUTLIERS EXTREMOS (valores > P99 e valores impossíveis):")
colunas_num = df.select_dtypes(include=np.number).columns.tolist()
if alvo in colunas_num:
    colunas_num.remove(alvo)

# P99 e P01
p99 = df[colunas_num].quantile(0.99)
p01 = df[colunas_num].quantile(0.01)
max_val = df[colunas_num].max()
min_val = df[colunas_num].min()

tabela_outliers = pd.DataFrame({
    "Coluna": colunas_num,
    "Mínimo": min_val.values,
    "P01": p01.values,
    "P99": p99.values,
    "Máximo": max_val.values,
    "P99_vs_Max_X": (max_val / p99.replace(0, np.nan)).values,
}).round(2)
print(tabela_outliers.to_string(index=False))

# Flags específicas
print("\n>>> PROBLEMAS ESPECÍFICOS DETECTADOS:")

# Idade
if df["idade"].min() < 16:
    print(f"  - Idade mínima = {df['idade'].min()} anos: improvável para cliente de crédito (valores inválidos / typos).")
if df["idade"].max() > 100:
    print(f"  - Idade máxima = {df['idade'].max()} anos: possível mas raro — merece tratamento (winzorização).")

# Uso limite rotativo
if df["uso_limite_rotativo"].max() > 2:
    print(f"  - uso_limite_rotativo máximo = {df['uso_limite_rotativo'].max():.2f}: extremamente acima do limite contratado (esperado 0–1, winzorizar).")
qtd_ultra_1 = (df["uso_limite_rotativo"] > 1).sum()
print(f"  - uso_limite_rotativo > 1 em {qtd_ultra_1:,} clientes ({qtd_ultra_1/len(df)*100:.2f}%): ultrapassaram o limite (sinal de risco, mas é informação real — não dropar).")

# Razão dívida
if df["razao_divida"].max() > 10:
    print(f"  - razao_divida máximo = {df['razao_divida'].max():.2f}: claramente outlier de digitação ou unidade trocada (receberia valor em R$ em vez de razão).")
qtd_rd_alta = (df["razao_divida"] > 5).sum()
print(f"  - razao_divida > 5 em {qtd_rd_alta:,} clientes ({qtd_rd_alta/len(df)*100:.2f}%): requer investigação e provável winzorização.")

# Renda
if df["renda_mensal"].max() > 500_000:
    print(f"  - renda_mensal máxima = R$ {df['renda_mensal'].max():,.0f}: valor extremo possível (higienizar via winsorização no P99).")

# Atrasos negativos
for col_atraso in ["atrasos_30_59_dias", "atrasos_60_89_dias", "atrasos_90_mais_dias"]:
    qtd_neg = (df[col_atraso] < 0).sum()
    if qtd_neg > 0:
        print(f"  - {col_atraso} contém {qtd_neg} valor(es) NEGATIVO(S): impossível — tratar como erro e limpar.")

# Dependentes negativos ou absurdos
qtd_dep_neg = (df["dependentes"] < 0).sum() if "dependentes" in df.columns else 0
qtd_dep_alto = (df["dependentes"] > 10).sum() if "dependentes" in df.columns else 0
if qtd_dep_neg > 0:
    print(f"  - dependentes negativos ({qtd_dep_neg}): limpar.")
if qtd_dep_alto > 0:
    print(f"  - dependentes > 10 ({qtd_dep_alto}): valor extremo (tamanho família rara).")

# Duplicatas
duplicatas = df.duplicated().sum()
print(f"  - Linhas duplicadas: {duplicatas:,} ({duplicatas/len(df)*100:.2f}% da base)")

print("\n>>> BOXPLOTS (colunas selecionadas) — salvos em reports/figures:")
# Boxplot colunas de atraso + uso_limite
cols_plot = [
    "idade",
    "uso_limite_rotativo",
    "razao_divida",
    "atrasos_30_59_dias",
    "atrasos_60_89_dias",
    "atrasos_90_mais_dias",
]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, c in enumerate(cols_plot):
    sns.boxplot(y=df[c].dropna(), ax=axes[i], color="#3498db", fliersize=2)
    axes[i].set_title(f"Boxplot — {c}", fontsize=11)
    axes[i].set_ylabel("")
plt.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "fase02_boxplots_outliers.png"))
print(f"  · Salvo em: {os.path.join(FIG_PATH, 'fase02_boxplots_outliers.png')}")
plt.close(fig)

# Histogramas
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, c in enumerate(cols_plot):
    sns.histplot(df[c].dropna(), kde=True, ax=axes[i], bins=40, color="#9b59b6")
    axes[i].set_title(f"Distribuição — {c}", fontsize=11)
    axes[i].set_xlabel("")
plt.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "fase02_histogramas.png"))
print(f"  · Salvo em: {os.path.join(FIG_PATH, 'fase02_histogramas.png')}")
plt.close(fig)

print("\nFIM DA FASE 2 — Entendimento dos Dados.")
