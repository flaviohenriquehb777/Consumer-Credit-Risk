"""
Acrescenta uma seção de investigação complementar (missing renda + código 98)
ao notebook 02_entendimento_dados.ipynb. Salva clean e executed.
"""
import json
import os
import subprocess
import sys
import nbformat as nbf

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLEAN = os.path.join(BASE, "notebooks", "clean", "02_entendimento_dados.ipynb")
EXEC = os.path.join(BASE, "notebooks", "executed", "02_entendimento_dados.ipynb")

with open(CLEAN, "r", encoding="utf-8") as f:
    nb = json.load(f)

# --- Células a serem inseridas ANTES da última célula (markdown de encerramento) ---

md_itemA = nbf.v4.new_markdown_cell(
    "## 9. Investigação complementar: Missing de renda + código 98 nas colunas de atraso\n"
    "### 9.1 Inadimplência comparada: quem informou renda vs. quem NÃO informou"
)

code_itemA = nbf.v4.new_code_cell('''# ============================================================
# ITEM 9.1 — Missing de renda x taxa de inadimplência
# ============================================================
df["renda_informada_flag"] = ~df["renda_mensal"].isnull()

tabela_renda = (
    df.groupby("renda_informada_flag")[alvo]
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
display(tabela_renda.style.hide(axis="index"))

taxa_sim = tabela_renda.loc[tabela_renda["Grupo"].str.startswith("SIM"), "Taxa Inadimplência (%)"].values[0]
taxa_nao = tabela_renda.loc[tabela_renda["Grupo"].str.startswith("NÃO"), "Taxa Inadimplência (%)"].values[0]
diff_pp = taxa_nao - taxa_sim

print(f"Taxa quem INFORMOU renda:    {taxa_sim:.2f}%")
print(f"Taxa quem NÃO informou renda: {taxa_nao:.2f}%")
print(f"DIFERENÇA (não-informou − informou) = {diff_pp:.2f} pp.")
print(f"Razão entre taxas (não/informou)   = {taxa_nao/taxa_sim:.2f}×")

# Gráfico
fig, ax = plt.subplots(figsize=(7, 4.8))
sns.barplot(x="Grupo", y="Taxa Inadimplência (%)", data=tabela_renda, ax=ax, palette=["#2ecc71", "#e74c3c"])
ax.set_title("Taxa de inadimplência: renda informada vs. missing", fontsize=13, pad=15)
for i, (v, n) in enumerate(zip(tabela_renda["Taxa Inadimplência (%)"].values, tabela_renda["N Clientes"].values)):
    ax.text(i, v + 0.08, f"{v:.2f}%\\n(n={n:,})", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "fase02_inadimplencia_renda_informada.png"), dpi=120)
plt.show()
''')

md_itemB = nbf.v4.new_markdown_cell(
    "### 9.2 Código suspeito 98 nas 3 colunas de atraso — investigação completa"
)

code_itemB = nbf.v4.new_code_cell('''# ============================================================
# ITEM 9.2.1 — Distribuição de valores nas colunas de atraso (suspeitos >= 90)
# ============================================================
col_atrasos = [
    "atrasos_30_59_dias",
    "atrasos_60_89_dias",
    "atrasos_90_mais_dias",
]

for col in col_atrasos:
    vc = df[col].value_counts().sort_index()
    print(f"\\n>>> {col} — valores únicos = {df[col].nunique()}")
    print("    Top 5 mais frequentes:")
    for v, c in vc.head(5).items():
        print(f"      valor={int(v):>3}  →  n={c:>7,}  ({c/len(df)*100:.3f}%)")
    suspeitos = vc[vc.index >= 90]
    if not suspeitos.empty:
        print("    Valores SUSPEITOS (>= 90 — impossível em 2 anos):")
        for v, c in suspeitos.items():
            print(f"      valor={int(v):>3}  →  n={c:>7,}  ({c/len(df)*100:.3f}%)")
''')

code_itemC = nbf.v4.new_code_cell('''# ============================================================
# ITEM 9.2.2 — Sobreposição: as 264 linhas com 98 são AS MESMAS nas 3 colunas?
# ============================================================
mask_98_30 = df["atrasos_30_59_dias"] == 98
mask_98_60 = df["atrasos_60_89_dias"] == 98
mask_98_90 = df["atrasos_90_mais_dias"] == 98
mask_98_qualquer = mask_98_30 | mask_98_60 | mask_98_90
mask_98_todas    = mask_98_30 & mask_98_60 & mask_98_90

tabela_98_qtd = pd.DataFrame([
    ["Linhas com 98 em atrasos_30_59_dias",  mask_98_30.sum(),  (mask_98_30.mean()*100).round(3)],
    ["Linhas com 98 em atrasos_60_89_dias",  mask_98_60.sum(),  (mask_98_60.mean()*100).round(3)],
    ["Linhas com 98 em atrasos_90_mais_dias", mask_98_90.sum(), (mask_98_90.mean()*100).round(3)],
    ["Linhas com 98 EM QUALQUER das 3",      mask_98_qualquer.sum(), (mask_98_qualquer.mean()*100).round(3)],
    ["Linhas com 98 NAS 3 AO MESMO TEMPO",   mask_98_todas.sum(),   (mask_98_todas.mean()*100).round(3)],
], columns=["Medida", "N linhas", "%"])
display(tabela_98_qtd.style.hide(axis="index"))

# Combinações exatas
df_padrao = pd.DataFrame({
    "tem_30":  mask_98_30.astype(int),
    "tem_60":  mask_98_60.astype(int),
    "tem_90":  mask_98_90.astype(int),
})
df_padrao["Padrão"] = df_padrao.apply(lambda r:
    (["30-59"] if r["tem_30"] else []) +
    (["60-89"] if r["tem_60"] else []) +
    (["90+"]   if r["tem_90"] else []), axis=1
).apply(lambda x: ", ".join(x) if x else "nenhuma (sadio)")

combos = (df_padrao.groupby("Padrão")
                   .size()
                   .reset_index(name="n_linhas")
                   .sort_values("n_linhas", ascending=False)
)
combos["%"] = (combos["n_linhas"] / len(df) * 100).round(3)
print("\\n>>> Combinações de colunas que contêm 98 (são duas únicas):")
display(combos.style.hide(axis="index"))

print("\\n>>> 10 exemplos de linhas com 98 (para inspecionar):")
col_exibir = [alvo, "idade", "renda_mensal", "uso_limite_rotativo", "razao_divida"] + col_atrasos
display(df.loc[mask_98_qualquer, col_exibir].head(10))
''')

md_itemD = nbf.v4.new_markdown_cell(
    "### 9.3 Taxa de inadimplência do grupo com código 98 vs. resto da base"
)

code_itemD = nbf.v4.new_code_cell('''# ============================================================
# ITEM 9.3 — Inadimplência do grupo com 98 vs. RESTO
# ============================================================
grupo = pd.DataFrame({
    "grupo": np.where(mask_98_qualquer, "COM 98 em pelo menos 1 coluna de atraso", "RESTO (sem 98)"),
    alvo: df[alvo].values,
})

tab = (grupo.groupby("grupo")[alvo]
            .agg(N="count", Inadimplentes="sum",
                 Taxa=lambda s: (s.mean() * 100).round(2))
            .reset_index()
      )
tab.columns = ["Grupo", "N Clientes", "N Inadimplentes", "Taxa Inadimplência (%)"]
display(tab.style.hide(axis="index"))

taxa_resto = tab.loc[tab["Grupo"].str.contains("RESTO"), "Taxa Inadimplência (%)"].values[0]
taxa_c98  = tab.loc[tab["Grupo"].str.contains("COM 98"),  "Taxa Inadimplência (%)"].values[0]

print(f"RESTO (sem 98)          : {taxa_resto:.2f}%")
print(f"GRUPO COM 98 em atrasos : {taxa_c98:.2f}%")
print(f"DIFERENÇA (grupo 98 − resto) = {taxa_c98 - taxa_resto:.2f} pp")
print(f"Razão entre taxas         = {taxa_c98/taxa_resto:.2f}×")

# Detalhe por padrão
tab2 = df_padrao[[alvo] if False else ["Padrão"]].copy()
tab2[alvo] = df[alvo].values
tab2 = (tab2.groupby("Padrão")[alvo]
             .agg(N="count", Inadimplentes="sum",
                  Taxa=lambda s: (s.mean() * 100).round(2))
             .reset_index()
       )
tab2.columns = ["Padrão (colunas com 98)", "N", "Inadimplentes", "Taxa Inadimplência (%)"]
print("\\n>>> Detalhe por padrão de ocorrência do 98:")
display(tab2.style.hide(axis="index"))

# Gráfico
fig, ax = plt.subplots(figsize=(9, 4.8))
sns.barplot(x="Padrão (colunas com 98)", y="Taxa Inadimplência (%)", data=tab2, ax=ax, palette="coolwarm")
ax.set_title("Taxa de inadimplência por padrão de 98 nas colunas de atraso", fontsize=13, pad=15)
for i, (v, n) in enumerate(zip(tab2["Taxa Inadimplência (%)"].values, tab2["N"].values)):
    ax.text(i, v + 0.08, f"{v:.2f}%\\n(n={n:,})", ha="center", va="bottom", fontsize=9)
plt.xticks(rotation=15)
plt.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "fase02_inadimplencia_codigo_98_atrasos.png"), dpi=120)
plt.show()
''')

md_conclusoes = nbf.v4.new_markdown_cell(
    "### 9.4 Conclusões práticas (só com base nos números)\n"
    "\n"
    "**Missing de renda:**\n"
    "- Ao contrário da intuição inicial, quem **não informa** renda tem *menor* risco: **5,61%** contra **6,95%** de quem informa.\n"
    "- Diferença real de **−1,34 pp** (razão 0,81×). Logo, a informação de missing ainda é útil como flag, mas deve ser interpretada como 'perfil distinto' — não como 'perfil mais arriscado'.\n"
    "- Dropar essas 29.731 linhas continua **NÃO recomendado**: além de perder volume, perde-se um perfil que tem menor taxa de inadimplência e ajudaria o modelo a discriminar.\n"
    "\n"
    "**Código 98:**\n"
    "- **Padrão perfeito de código artificial:** as mesmas **264 linhas** apresentam 98 nas 3 colunas de atraso ao mesmo tempo. Também existem 5 linhas com 96 seguindo o mesmo padrão.\n"
    "- Esse grupo tem **54,17% de inadimplência** contra 6,60% do resto — diferença de **+47,57 pp**, razão **8,21×** mais risco.\n"
    "- **Ação recomendada para a Fase 3:** NÃO winsorize cegamente 98→P99 nas colunas de atraso. Isso apagaria o sinal preditivo mais forte de toda a base.\n"
    "  Faça antes: (1) criar **flag binária `cod_erro_atrasos_98`** (1 quando as 3 colunas têm 98) e (2) substituir o 98 por NaN ou por P99 nas colunas originais de atraso, deixando a flag carregar o sinal desse perfil altíssimo de risco."
)

# --- Inserir as novas células ANTES do último markdown ("Fim da Fase 2...") ---
novas_celulas = [md_itemA, code_itemA, md_itemB, code_itemB, code_itemC, md_itemD, code_itemD, md_conclusoes]

# Extrair última célula (encerramento)
ultima = nb["cells"].pop()  # remove a última (é "--- /nFim da Fase 2 /nPróxima...")

# Inseri as novas e depois recoloca a última
for c in novas_celulas:
    nb["cells"].append(json.loads(nbf.writes(c)))  # formato padrão nbformat
nb["cells"].append(ultima)

# Salva clean (sem outputs) — zera outputs/execution_count
for c in nb["cells"]:
    if c.get("cell_type") == "code":
        c["outputs"] = []
        c["execution_count"] = None
with open(CLEAN, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"[OK] Notebook CLEAN atualizado: {CLEAN}")

# --- Executa via CLI e gera executed ---
print("[...] Rodando nbconvert execute (vários minutos)...")
cmd = [sys.executable, "-m", "jupyter", "nbconvert",
       "--to", "notebook", "--execute", "--inplace",
       "--ExecutePreprocessor.timeout=900",
       "--ExecutePreprocessor.kernel_name=python3",
       "--allow-errors", CLEAN]
res = subprocess.run(cmd, cwd=os.path.dirname(CLEAN), capture_output=True, text=True)
if res.returncode == 0:
    import shutil
    shutil.move(CLEAN, EXEC)
    print(f"[OK] Notebook EXECUTADO gravado em: {EXEC}")
    # Regenera CLEAN sem outputs
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
    print("STDOUT:", res.stdout[-2000:])
    print("STDERR:", res.stderr[-2000:])
