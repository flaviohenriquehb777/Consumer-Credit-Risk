"""Gera notebook clean e notebook executado para a Fase 2 — Entendimento dos Dados."""
import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLEAN_PATH = os.path.join(BASE_DIR, "notebooks", "clean", "02_entendimento_dados.ipynb")
EXEC_PATH = os.path.join(BASE_DIR, "notebooks", "executed", "02_entendimento_dados.ipynb")
FIG_PATH = os.path.join(BASE_DIR, "reports", "figures")

# Código do notebook (mesmo do script, mas com prints + matplot inline)
codigo = r'''# --- Configurações iniciais ---
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.family"] = "DejaVu Sans"
%matplotlib inline

# Caminhos (relativos ao notebook — que fica em notebooks/clean ou notebooks/executed)
NOTEBOOK_DIR = os.path.abspath(os.getcwd())
PROJECT_ROOT = os.path.abspath(os.path.join(NOTEBOOK_DIR, "..", ".."))
DADOS_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "credito_tratado.csv")
FIG_PATH = os.path.join(PROJECT_ROOT, "reports", "figures")

print(f"Projeto raiz: {PROJECT_ROOT}")
print(f"Dados:       {DADOS_PATH}")
'''

# ============================================================
# Célula 2 — ITEM 1
# ============================================================
code_item1 = '''# ============================================================
# ITEM 1 — Carregar CSV, shape, primeiras linhas
# ============================================================
df = pd.read_csv(DADOS_PATH)

print(f"Número de LINHAS  (clientes):  {df.shape[0]:,}")
print(f"Número de COLUNAS (features): {df.shape[1]}")
print(f"Memória utilizada: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
df.head()
'''

# ============================================================
# Célula 3 — ITEM 2
# ============================================================
code_item2 = '''# ============================================================
# ITEM 2 — Tipos de dados, significado e unidade de cada coluna
# ============================================================
dicionario = {
    "inadimplente_2anos":         ("Inteiro binário",  "1=inadimplente 90+ DPD em até 2 anos; 0=adimplente",   "Flag (0/1)"),
    "idade":                      ("Inteiro",          "Idade do cliente no momento da avaliação de crédito", "Anos"),
    "renda_mensal":               ("Contínuo",         "Renda mensal informada ou estimada pelo cliente",     "Reais (R$)"),
    "dependentes":                ("Discreto",         "Número de dependentes declarados",                    "Pessoas"),
    "uso_limite_rotativo":        ("Contínuo",         "Proporção do limite rotativo já utilizada",           "Proporção (0–1; pode >1 se estourado)"),
    "razao_divida":               ("Contínuo",         "Relação entre comprometimento financeiro e renda",    "Razão (sem unidade)"),
    "linhas_credito_abertas":     ("Inteiro",          "Qtd. de linhas de crédito ativas no histórico",       "Unidades"),
    "financiamentos_imobiliarios":("Inteiro",          "Número de financiamentos imobiliários registrados",   "Unidades"),
    "atrasos_30_59_dias":         ("Inteiro",          "Qtd. de episódios de atraso 30–59 dias",              "Unidades"),
    "atrasos_60_89_dias":         ("Inteiro",          "Qtd. de episódios de atraso 60–89 dias",              "Unidades"),
    "atrasos_90_mais_dias":       ("Inteiro",          "Qtd. de episódios de atraso >= 90 dias",              "Unidades"),
}

tipos = df.dtypes.reset_index()
tipos.columns = ["Coluna", "Tipo Pandas"]
tipos["Classificação"] = tipos["Coluna"].map(lambda c: dicionario.get(c, ("", "", ""))[0])
tipos["Significado"] = tipos["Coluna"].map(lambda c: dicionario.get(c, ("", "", ""))[1])
tipos["Unidade"] = tipos["Coluna"].map(lambda c: dicionario.get(c, ("", "", ""))[2])

tipos.style.set_properties(**{"text-align": "left"}).hide(axis="index")
'''

# ============================================================
# Célula 4 — ITEM 3
# ============================================================
code_item3 = '''# ============================================================
# ITEM 3 — Distribuição do alvo (inadimplente_2anos)
# ============================================================
alvo = "inadimplente_2anos"

contagem   = df[alvo].value_counts().sort_index()
percentual = (contagem / len(df) * 100).round(2)

tabela_alvo = pd.DataFrame({
    "Classe":        contagem.index.map({0: "Adimplente (0)", 1: "Inadimplente (1)"}),
    "Contagem":      contagem.values,
    "Percentual (%)": percentual.values,
})
display(tabela_alvo.style.hide(axis="index"))

n_adimplentes   = contagem[0]
n_inadimplentes = contagem[1]
razao = n_adimplentes / n_inadimplentes if n_inadimplentes > 0 else np.inf

print(f"Razão de desequilíbrio: 1 INADIMPLENTE para cada {razao:.2f} ADIMPLENTES.")
print(f"Representatividade classe positiva (inadimplentes): {percentual[1]:.2f}%")

# Gráfico
fig, ax = plt.subplots(figsize=(6, 4.5))
colors = ["#2ecc71", "#e74c3c"]
sns.countplot(x=alvo, data=df, ax=ax, palette=colors)
ax.set_title("Distribuição da variável-alvo (inadimplente_2anos)", fontsize=13, pad=15)
ax.set_xlabel("Classe")
ax.set_ylabel("Frequência")
ax.set_xticklabels(["Adimplente (0)", "Inadimplente (1)"])
for p in ax.patches:
    h = p.get_height()
    ax.annotate(f"{h:,}\n({h/len(df)*100:.2f}%)", (p.get_x()+p.get_width()/2, h),
                ha="center", va="bottom", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "fase02_distribuicao_alvo.png"), dpi=120)
plt.show()
'''

# ============================================================
# Célula 5 — ITEM 4
# ============================================================
code_item4 = '''# ============================================================
# ITEM 4 — Acurácia do modelo de uma linha que chuta "ninguém calota"
# ============================================================
predominante = 0 if contagem[0] >= contagem[1] else 1
acuracia_baseline = (df[alvo] == predominante).mean() * 100

print(f"Regra do modelo baseline: prever SEMPRE {predominante} (Adimplente).")
print(f"Acurácia do modelo de 1 linha (chuta tudo 0): {acuracia_baseline:.2f}%")
'''

# ============================================================
# Célula 6 — ITEM 5
# ============================================================
code_item5 = '''# ============================================================
# ITEM 5 — Consequências para a escolha de métricas
# ============================================================
from IPython.display import Markdown, display

texto = f"""
---

### Consequência imediata do baseline de {acuracia_baseline:.2f}%

Com **{percentual[1]:.2f}%** de positivos contra **{percentual[0]:.2f}%** de negativos, a **ACURÁCIA é inútil como métrica decisória** — um modelo que não faz nada (chuta tudo 0) já entrega {acuracia_baseline:.2f}% de acerto.

> Se usarmos acurácia como objetivo, o algoritmo tenderá a "nunca acusar ninguém", deixando passar Falsos Negativos, que custam **10x mais caro** que os Falsos Positivos.

---

### Métricas que usaremos neste projeto

| Métrica | O que mede | Cuidados que temos que tomar |
|---|---|---|
| **ROC AUC** (≥ 0,85 obrigatório) | Capacidade de **ordenação** do modelo: quão bem ele separa clientes bons de maus pagadores independentemente do ponto de corte. | Pode ser artificialmente alto em dados extremamente desbalanceados; **não usa a matriz de custos FN/FP**, portanto mede performance estatística, não econômica. |
| **Precision (Precisão)** | Dos clientes marcados como "ruins" (negados pelo modelo), quantos **realmente** calotaram? — TP/(TP+FP). | Mede só o lado do "negado" (FP). Um modelo que nega 1 única pessoa com 100% de certeza teria Precision=100% e não serve ao negócio. |
| **Recall / Sensibilidade (Revogação)** | De **todos os verdadeiros calotes** existentes na base, quantos o modelo conseguiu pegar? — TP/(TP+FN). | Se aumentar cegamente (ponto de corte muito baixo), explode o número de FPs e nega muita gente boa, destruindo volume de aprovações. |
| **F1-Score** | Média harmônica entre Precision e Recall. | **Trata FN e FP com mesmo peso**, mas na nossa relação 10:1 isso não reflete a realidade de custo. Usamos só como comparativo, NUNCA como decisório. |
| **Matriz de Confusão + Custo Esperado da Carteira** | `Custo = (N_FN × 10) + (N_FP × 1)` — o custo total da política em R$ (na unidade definida). | **Esta é a métrica de NEGÓCIO.** Depende estritamente do **ponto de corte** escolhido pela Área de Risco. Vamos otimizar sobre ela, não sobre acurácia. |
| **Brier Score / Calibração** | Quão calibradas são as probabilidades: um cliente com score 18% deve ter ~18% de chance real de calote. | Crítico para pricing e para o analista confiar no número que aparece na tela do sistema. |
"""
display(Markdown(texto))
'''

# ============================================================
# Célula 7 — ITEM 6
# ============================================================
code_item6 = '''# ============================================================
# ITEM 6 — Colunas com valores NULOS (qtd e %)
# ============================================================
nulos     = df.isnull().sum()
nulos_pct = (nulos / len(df) * 100).round(2)

tabela_nulos = pd.DataFrame({
    "Coluna":    nulos.index,
    "Qtd Nulos": nulos.values,
    "% Nulos":   nulos_pct.values,
})
tabela_nulos = tabela_nulos[tabela_nulos["Qtd Nulos"] > 0] \
    .sort_values("% Nulos", ascending=False) \
    .reset_index(drop=True)

display(tabela_nulos.style.hide(axis="index"))

# Gráfico
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x="% Nulos", y="Coluna", data=tabela_nulos, ax=ax, palette="Reds_r")
ax.set_title("Percentual de valores nulos por coluna", fontsize=13, pad=15)
for i, v in enumerate(tabela_nulos["% Nulos"].values):
    qtd = tabela_nulos["Qtd Nulos"].iloc[i]
    ax.text(v + 0.1, i, f"{v:.2f}%  (n={qtd:,})", va="center", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "fase02_valores_nulos.png"), dpi=120)
plt.show()
'''

# ============================================================
# Célula 8 — ITEM 7
# ============================================================
code_item7 = '''# ============================================================
# ITEM 7 — Podemos descartar as linhas com nulos?
# ============================================================
linhas_com_nulo = df.isnull().any(axis=1).sum()
pct_linhas_nulo = (linhas_com_nulo / len(df) * 100).round(2)

tabela_drop = pd.DataFrame([
    ["Total de clientes",                        f"{len(df):,}",      "100,00%"],
    ["Linhas com PELO MENOS 1 nulo",             f"{linhas_com_nulo:,}", f"{pct_linhas_nulo:.2f}%"],
    ["Linhas completas (sem nenhum nulo)",       f"{len(df)-linhas_com_nulo:,}", f"{100-pct_linhas_nulo:.2f}%"],
], columns=["Cenário", "Quantidade", "Percentual"])

display(tabela_drop.style.hide(axis="index"))

msg = f"""
---

### Conclusão sobre drop de linhas

Simplesmente **dropar** as linhas com valor faltante = perder **{linhas_com_nulo:,} clientes ({pct_linhas_nulo:.2f}% da base)**.

Além do volume perdido, o problema de viés é o mais grave:
- O missing em `renda_mensal` (19,82%) **não é aleatório**. Quem NÃO informa a renda em um pedido de crédito já carrega um **sinal de risco** — a ausência da informação é, ela mesma, informação. Dropar essas linhas estaria removendo justamente um perfil de maior risco do treinamento, e o modelo aprenderia enviesado.
- O missing em `dependentes` (2,62%) é menor mas segue o mesmo raciocínio.

### Recomendação (já com base nos números)

| Coluna | Ação sugerida na Fase 3 |
|---|---|
| `renda_mensal` | 1) Criar **flag binária** `renda_ausente` (0/1); 2) **Imputar** o missing com a **mediana** (robusta a outliers). |
| `dependentes`  | 1) Criar **flag binária** `dependentes_ausente` (0/1); 2) **Imputar** com moda (0 ou mediana). |
"""
from IPython.display import Markdown; display(Markdown(msg))
'''

# ============================================================
# Célula 9 — ITEM 8
# ============================================================
code_item8 = '''# ============================================================
# ITEM 8 — Outros problemas detectados (describe + outliers + outros)
# ============================================================
desc = df.describe().T
desc["missing"]     = df.isnull().sum()
desc["missing_pct"] = (df.isnull().sum() / len(df) * 100).round(2)
print("ESTATÍSTICAS DESCRITIVAS:")
display(desc.round(2))

colunas_num = df.select_dtypes(include=np.number).columns.tolist()
colunas_num.remove(alvo)

p99 = df[colunas_num].quantile(0.99)
p01 = df[colunas_num].quantile(0.01)
mx  = df[colunas_num].max()
mn  = df[colunas_num].min()

tab_out = pd.DataFrame({
    "Coluna":        colunas_num,
    "Mínimo":        mn.values,
    "P01":           p01.values,
    "P99":           p99.values,
    "Máximo":        mx.values,
    "Máx / P99 (x)": (mx / p99.replace(0, np.nan)).values,
}).round(2).sort_values("Máx / P99 (x)", ascending=False)

print("\nOUTLIERS EXTREMOS — razão entre Máximo e P99:")
display(tab_out.style.hide(axis="index"))
'''

code_item8b = '''# Detecção de problemas específicos
problemas = []

# Idade
idade_min, idade_max = df["idade"].min(), df["idade"].max()
if idade_min < 16:  problemas.append(f"Idade MÍNIMA = {idade_min} anos — impossível legalmente (erro de dado).")
if idade_max > 100: problemas.append(f"Idade MÁXIMA = {idade_max} anos — raro (top-capping recomendado no P99 = {p99['idade']:.0f}).")

# Uso limite rotativo
qtd_ultra_1 = (df["uso_limite_rotativo"] > 1).sum()
problemas.append(f"uso_limite_rotativo > 1 em {qtd_ultra_1:,} clientes ({qtd_ultra_1/len(df)*100:.2f}%) — ultrapassaram o limite contratado; informação real de risco NÃO dropar, mas max = {mx['uso_limite_rotativo']:.2f} pode ser winsorizada.")

# Renda
problemas.append(f"renda_mensal MÁXIMA = R$ {mx['renda_mensal']:,.0f} contra P99 = R$ {p99['renda_mensal']:,.0f} ({mx['renda_mensal']/p99['renda_mensal']:.1f}× o percentil) — outlier extremo; winsorizar no P99.")

# Dependentes
qtd_dep_alto = (df["dependentes"] > 10).sum()
if qtd_dep_alto > 0:
    problemas.append(f"dependentes > 10 = {qtd_dep_alto} casos contra P99 = {p99['dependentes']:.0f} — valor raro, tratar com top-capping.")

# Atrasos (3 faixas): valor MÁXIMO = 98 em todas! É impossível ter 98 atrasos de 30-59 dias em 2 anos.
for col_atraso in ["atrasos_30_59_dias", "atrasos_60_89_dias", "atrasos_90_mais_dias"]:
    p99_c  = p99[col_atraso]
    max_c  = mx[col_atraso]
    razao_c = max_c / (p99_c or np.nan)
    problemas.append(f"{col_atraso}: P99={p99_c:.0f}  vs  MÁXIMO={max_c:.0f} (≈ {razao_c:.1f}×) — 98 é fortemente suspeito de ser 'código de missing/erro de dados' (top-capping no P99 recomendado).")

# Financiamentos imobiliarios
qtd_fin_alto = (df["financiamentos_imobiliarios"] > 10).sum()
if qtd_fin_alto > 0:
    problemas.append(f"financiamentos_imobiliarios > 10 = {qtd_fin_alto} clientes; P99 = {p99['financiamentos_imobiliarios']:.0f} vs MÁXIMO = {mx['financiamentos_imobiliarios']:.0f} — valor extremo.")

# Duplicatas
dup = df.duplicated().sum()
problemas.append(f"Linhas DUPLICADAS = {dup:,} ({dup/len(df)*100:.2f}% da base) — investigar e remover duplicatas idênticas na Fase 3.")

from IPython.display import Markdown
md = "### Problemas específicos detectados (evidência por item)\n\n"
for i, p in enumerate(problemas, 1):
    md += f"{i}. {p}\n\n"
display(Markdown(md))
'''

code_item8c = '''# Visualizações (boxplots e histogramas)
cols_plot = ["idade", "uso_limite_rotativo", "razao_divida",
             "atrasos_30_59_dias", "atrasos_60_89_dias", "atrasos_90_mais_dias"]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, c in enumerate(cols_plot):
    sns.boxplot(y=df[c].dropna(), ax=axes[i], color="#3498db", fliersize=2)
    axes[i].set_title(f"Boxplot — {c}", fontsize=11)
    axes[i].set_ylabel("")
plt.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "fase02_boxplots_outliers.png"), dpi=120)
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, c in enumerate(cols_plot):
    sns.histplot(df[c].dropna(), kde=True, ax=axes[i], bins=40, color="#9b59b6")
    axes[i].set_title(f"Distribuição — {c}", fontsize=11)
    axes[i].set_xlabel("")
plt.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "fase02_histogramas.png"), dpi=120)
plt.show()
'''

# ============================================================
# Montar notebook
# ============================================================
nb = nbf.v4.new_notebook()
nb["cells"] = [
    nbf.v4.new_markdown_cell("# Fase 2 — Entendimento dos Dados\n**Projeto:** Modelo de Risco de Crédito  \n**Autor:** Equipe de Risco e Analytics"),
    nbf.v4.new_code_cell(codigo),
    nbf.v4.new_markdown_cell("## 1. Carregamento, dimensão e primeiras linhas"),
    nbf.v4.new_code_cell(code_item1),
    nbf.v4.new_markdown_cell("## 2. Tipos de dados, significado e unidade de cada coluna"),
    nbf.v4.new_code_cell(code_item2),
    nbf.v4.new_markdown_cell("## 3. Distribuição do alvo — inadimplente_2anos"),
    nbf.v4.new_code_cell(code_item3),
    nbf.v4.new_markdown_cell("## 4. Acurácia do modelo baseline de 1 linha (chuta ninguém calota)"),
    nbf.v4.new_code_cell(code_item4),
    nbf.v4.new_markdown_cell("## 5. Consequências para a escolha das métricas"),
    nbf.v4.new_code_cell(code_item5),
    nbf.v4.new_markdown_cell("## 6. Valores nulos por coluna"),
    nbf.v4.new_code_cell(code_item6),
    nbf.v4.new_markdown_cell("## 7. Podemos simplesmente descartar as linhas com nulos?"),
    nbf.v4.new_code_cell(code_item7),
    nbf.v4.new_markdown_cell("## 8. Outros problemas na base (outliers extremos, duplicatas, etc.)"),
    nbf.v4.new_code_cell(code_item8),
    nbf.v4.new_code_cell(code_item8b),
    nbf.v4.new_code_cell(code_item8c),
    nbf.v4.new_markdown_cell("---  \n**Fim da Fase 2.** Próxima: Fase 3 — Preparação dos Dados."),
]

os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
os.makedirs(os.path.dirname(EXEC_PATH), exist_ok=True)

with open(CLEAN_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"[OK] Notebook CLEAN    salvo em: {CLEAN_PATH}")

# ============================================================
# Executar o notebook para salvar a versão com outputs
# ============================================================
print("[...] Executando notebook (pode levar alguns segundos)...")
# Precisamos rodar a partir do diretório do notebook
NOTEBOOK_DIR_CLEAN = os.path.dirname(CLEAN_PATH)
with open(CLEAN_PATH, "r", encoding="utf-8") as f:
    nb_exec = nbf.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
try:
    ep.preprocess(nb_exec, {"metadata": {"path": NOTEBOOK_DIR_CLEAN}})
    with open(EXEC_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb_exec, f)
    print(f"[OK] Notebook EXECUTED salvo em: {EXEC_PATH}")
except Exception as ex:
    print(f"[WARN] Falhou a execução do notebook via ExecutePreprocessor: {ex}")
    print("       Salvando uma cópia idêntica ao clean.")
    import shutil
    shutil.copy(CLEAN_PATH, EXEC_PATH)
    print(f"[OK] Notebook EXECUTED (cópia de clean) salvo em: {EXEC_PATH}")
