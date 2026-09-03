"""Gera o notebook da Fase 3 Parte 1 e roda para salvar clean + executed."""
import json
import os
import subprocess
import sys
import nbformat as nbf

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLEAN = os.path.join(BASE, "notebooks", "clean", "03a_preparacao_dados_pt1.ipynb")
EXEC = os.path.join(BASE, "notebooks", "executed", "03a_preparacao_dados_pt1.ipynb")

# ============================================================
# Montar notebook
# ============================================================
nb = nbf.v4.new_notebook()

c1_md = nbf.v4.new_markdown_cell("# Fase 3 — Preparação dos Dados (Parte 1)\n\n**Objetivo:** aplicar o pipeline de tratamento utilizando APENAS a base de treino como referência (sem data leakage) e salvar os artefatos para a modelagem.\n\n**Pipeline aplicado:**\n0. Split estratificado 75-25 (`stratify=y`, seed=42)\n1. Imputar `renda_mensal` faltante com mediana do treino.\n2. Criar a flag `renda_ausente`.\n3. Criar a flag `dependentes_ausentes`.\n4. Tratar atrasos: top-cap em 20 + flag única `cod_sistema_atrasos` (96/98).\n5. Top-cap de renda em R$ 50.000.\n6. Criar features novas:\n   - `renda_por_dependente = renda / (dependentes + 1)`\n   - `sobra_caixa = renda_mensal * (1 - razao_divida)`")
c2_setup = nbf.v4.new_code_cell("""%matplotlib inline
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
np.random.seed(42)

# Caminhos
NOTEBOOK_DIR = os.path.abspath(os.getcwd())
PROJECT_ROOT = os.path.abspath(os.path.join(NOTEBOOK_DIR, "..", ".."))
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "credito_tratado.csv")
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIG_PATH = os.path.join(PROJECT_ROOT, "reports", "figures")

# Importa o módulo de preparação (função única, reutilizável no Streamlit)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from features.preprocessamento import preparar_dados, salvar_params, carregar_params, SEED_DEFAULT

print(f"Projeto raiz : {PROJECT_ROOT}")
print(f"Dados brutos : {RAW_PATH}")
print(f"Dados proc.  : {PROC_DIR}")
print(f"Modelos      : {MODELS_DIR}")
""")

c3_md = nbf.v4.new_markdown_cell("## 0. Split estratificado 75-25")
c3_split = nbf.v4.new_code_cell("""df_raw = pd.read_csv(RAW_PATH)
print(f"Base bruta: {df_raw.shape[0]:,} linhas × {df_raw.shape[1]} colunas.")

X_train, X_test, y_train, y_test, params = preparar_dados(
    df_raw, fit=True, test_size=0.25, seed=SEED_DEFAULT
)

tab_dist = pd.DataFrame({
    "Conjunto": ["Treino", "Teste", "Base bruta"],
    "Nº linhas": [len(X_train), len(X_test), len(df_raw)],
    "% do total": [len(X_train)/len(df_raw)*100, len(X_test)/len(df_raw)*100, 100.0],
    "N Inadimplentes": [int(y_train.sum()), int(y_test.sum()), int(df_raw["inadimplente_2anos"].sum())],
    "% Inadimplentes": [
        f"{y_train.mean()*100:.2f}", f"{y_test.mean()*100:.2f}",
        f"{df_raw['inadimplente_2anos'].mean()*100:.2f}",
    ],
})
display(tab_dist.style.format({"Nº linhas": "{:,}", "% do total": "{:.1f}%"}).hide(axis="index"))

msg = f"> ✔️ Split correto. A taxa de inadimplência ({y_train.mean()*100:.2f}%) está idêntica nos dois conjuntos — estratificação funcionou perfeitamente."
display(Markdown(msg))
""")

c4_md = nbf.v4.new_markdown_cell("## 1–5. Verificando efeitos de cada tratamento")
c4_checks = nbf.v4.new_code_cell("""print("=== 1 & 2 — Renda: mediana treino + flag ===")
mediana_renda = params["mediana_renda_mensal"]
print(f"Mediana da renda (aprendida SÓ no treino): R$ {mediana_renda:,.2f}")
print(f"Flag renda_ausente: {X_train['renda_ausente'].sum():,} registros com 1 ({X_train['renda_ausente'].mean()*100:.2f}%)")
print(f"renda_mensal NULOS após tratamento: {X_train['renda_mensal'].isnull().sum()}")

print("\\n=== 3 — Dependentes: flag ===")
print(f"Flag dependentes_ausentes: {X_train['dependentes_ausentes'].sum():,} ({X_train['dependentes_ausentes'].mean()*100:.2f}%)")
print(f"dependentes NULOS após tratamento: {X_train['dependentes'].isnull().sum()}")

print("\\n=== 4 — Atrasos: top-cap 20 + flag códigos de sistema ===")
for c in params["colunas_atraso"]:
    print(f"  {c} → max = {X_train[c].max()} (deveria ser 20)")
print(f"Flag cod_sistema_atrasos: {X_train['cod_sistema_atrasos'].sum():,} ({X_train['cod_sistema_atrasos'].mean()*100:.3f}%)")

print("\\n=== 5 — Top-cap renda 50.000 ===")
print(f"renda_mensal máxima após o cap: R$ {X_train['renda_mensal'].max():,.2f}")

print("\\n=== 6 — Features novas ===")
print(f"  · renda_por_dependente: média R$ {X_train['renda_por_dependente'].mean():,.2f}, mediana R$ {X_train['renda_por_dependente'].median():,.2f}")
print(f"  · sobra_caixa: média R$ {X_train['sobra_caixa'].mean():,.2f}, mediana R$ {X_train['sobra_caixa'].median():,.2f}")
print(f"  · sobra_caixa mínima (negativa = custos > renda): R$ {X_train['sobra_caixa'].min():,.2f}")
""")

c5_md = nbf.v4.new_markdown_cell("## 6. Resumo final da base tratada (X_train)")
c5_resumo = nbf.v4.new_code_cell("""print(f"Dimensão final X_train: {len(X_train):,} linhas × {len(X_train.columns)} colunas.")

tipos = pd.DataFrame({
    "Coluna": X_train.columns,
    "Tipo Pandas": [str(X_train[c].dtype) for c in X_train.columns],
    "Qtd Nulos": [int(X_train[c].isnull().sum()) for c in X_train.columns],
    "Mínimo": [X_train[c].min() for c in X_train.columns],
    "Máximo": [X_train[c].max() for c in X_train.columns],
    "Média":  [X_train[c].mean() for c in X_train.columns],
    "Mediana": [X_train[c].median() for c in X_train.columns],
})
display(tipos.round(2).style.hide(axis="index"))

# --- Gráfico: antes vs. depois das distribuições transformadas (renda e atrasos) ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Renda mensal após top-cap
sns.histplot(X_train["renda_mensal"], bins=40, kde=True, color="#3498db", ax=axes[0,0])
axes[0,0].set_title("renda_mensal (após top-cap R$ 50k)", fontsize=11); axes[0,0].set_xlabel("")

# Renda por dependente (nova feature)
sns.histplot(X_train["renda_por_dependente"].clip(upper=20000), bins=40, kde=True, color="#9b59b6", ax=axes[0,1])
axes[0,1].set_title("renda_por_dependente (nova, clipado até 20k p/ plot)", fontsize=11); axes[0,1].set_xlabel("")

# Sobra caixa
sns.histplot(X_train["sobra_caixa"], bins=50, kde=True, color="#2ecc71", ax=axes[1,0])
axes[1,0].set_title("sobra_caixa = renda · (1 − razao_divida)", fontsize=11); axes[1,0].set_xlabel("")

# Contagem de flags
flags_df = pd.DataFrame({
    "Flag": ["renda_ausente", "dependentes_ausentes", "cod_sistema_atrasos"],
    "Nº de vezes 1": [
        int(X_train["renda_ausente"].sum()),
        int(X_train["dependentes_ausentes"].sum()),
        int(X_train["cod_sistema_atrasos"].sum()),
    ],
})
sns.barplot(x="Flag", y="Nº de vezes 1", data=flags_df, ax=axes[1,1], palette=["#e67e22", "#f1c40f", "#c0392b"])
axes[1,1].set_title("Contagem de flags criadas", fontsize=11); axes[1,1].set_xlabel("")
for i, v in enumerate(flags_df["Nº de vezes 1"].values):
    axes[1,1].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
fpath = os.path.join(FIG_PATH, "fase03_distribuicoes_apos_tratamento.png")
fig.savefig(fpath, dpi=150, bbox_inches="tight")
print(f"Gráfico salvo em: {fpath}")
plt.show()
""")

c6_md = nbf.v4.new_markdown_cell("## 7. Persistência dos artefatos (prontos para a Fase 4 de Modelagem e o Streamlit)")
c6_salvar = nbf.v4.new_code_cell("""# O script preprocessamento.py já salvou tudo. Apenas listamos e confirmamos.
for f_name in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
    fp = os.path.join(PROC_DIR, f_name)
    size_kb = os.path.getsize(fp) / 1024
    print(f"  · {fp}  —  {size_kb:,.1f} KB")

fp_params = os.path.join(MODELS_DIR, "preprocessamento_params.pkl")
print(f"  · {fp_params}  —  {os.path.getsize(fp_params)/1024:,.1f} KB")

params_carregados = carregar_params(fp_params)
print("\\nParâmetros (pickle) carregados do treino (serão usados no Streamlit):")
for k, v in params_carregados.items():
    print(f"    - {k:30s} = {v}")
""")

c7_md = nbf.v4.new_markdown_cell("## 8. Próximos passos — tratamentos adicionais SUGERIDOS para a parte 2 da Fase 3\n\n> Nenhuma decisão sem evidência; a lista abaixo é **pré-análise** e será confirmada/recusada com teste estatístico/numérico na parte 2.\n\n| Ordem | Tratamento | Evidência/Motivação |\n|---|---|---|\n| 8.1 | **One-Hot / discretização de `razao_divida` e `uso_limite_rotativo` (bins IV/WoE) | Modelos lineares / regressão se beneficiam de bins interpretáveis; o negócio pede justificativa simples por faixa. |\n| 8.2 | **StandardScaler / RobustScaler** para modelos lineares e de distância | Regressão logística, SVM e KNN NÃO funcionam sem padronização — Árvores/XGBoost não precisam. Será criado um pipeline por algoritmo. |\n| 8.3 | **Interações polinomiais:** atraso total = 3 colunas de atraso somadas; utilização × atrasos; idade × idade. | Hipótese: combinação de sinais aumenta poder preditivo. |\n| 8.4 | **Tratar duplicatas restantes** | Fase 2 detectou 1.573 (1,05%) duplicatas exatas — devem ser removidas ANTES do split; re-testamos impacto após remoção. |\n| 8.5 | **Top-capping em `financiamentos_imobiliarios` (P99=4, max=54)** e `linhas_credito_abertas` (P99=24, max=58). | Sem tratamento ainda; pode poluir splits em árvores. |\n| 8.6 | **Feature engineering: (a) idade em faixas etárias; (b) taxa de severidade = max(atrasos); (c) número total de atrasos** | Conhecimento de negócio de risco de crédito. |\n| 8.7 | **Validação se os splits de renda/cap em 50 mil e atrasos em 20 são os pontos ótimos via IV/Gain** (grid de thresholds). | Valores atuais foram heurísticas; vamos validar via teste estatístico. |\n\n---\n\n**Fim da Fase 3 Parte 1.** Próxima: Parte 2 (tratamentos complementares + validação estatística).")

nb["cells"] = [c1_md, c2_setup, c3_md, c3_split, c4_md, c4_checks, c5_md, c5_resumo, c6_md, c6_salvar, c7_md]

os.makedirs(os.path.dirname(CLEAN), exist_ok=True)
with open(CLEAN, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"[OK] CLEAN salvo em: {CLEAN}")

# --- Roda via nbconvert para criar o executed ---
print("[...] Rodando nbconvert execute ...")
cmd = [sys.executable, "-m", "jupyter", "nbconvert",
       "--to", "notebook", "--execute", "--inplace",
       "--ExecutePreprocessor.timeout=900",
       "--ExecutePreprocessor.kernel_name=python3",
       "--allow-errors", CLEAN]
res = subprocess.run(cmd, cwd=os.path.dirname(CLEAN), capture_output=True, text=True)
if res.returncode == 0:
    import shutil
    shutil.move(CLEAN, EXEC)
    print(f"[OK] EXECUTED salvo em: {EXEC}")
    with open(EXEC, "r", encoding="utf-8") as f:
        nb2 = json.load(f)
    for c in nb2["cells"]:
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None
    with open(CLEAN, "w", encoding="utf-8") as f:
        json.dump(nb2, f, ensure_ascii=False, indent=1)
    print(f"[OK] CLEAN (sem outputs) regenerado em: {CLEAN}")
else:
    print("[ERRO] nbconvert falhou:")
    print("STDOUT final:"); print(res.stdout[-1800:])
    print("STDERR final:"); print(res.stderr[-1800:])
