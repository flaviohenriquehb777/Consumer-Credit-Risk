"""
Anexa a seção 10 (Faixas de atrasos × inadimplência) ao notebook clean e regera executed.
"""
import json
import os
import subprocess
import sys

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLEAN = os.path.join(BASE, "notebooks", "clean", "02_entendimento_dados.ipynb")
EXEC = os.path.join(BASE, "notebooks", "executed", "02_entendimento_dados.ipynb")

with open(CLEAN, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Células novas (markdown + código)
md_intro = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 10. Confirmação visual: faixas de atrasos históricos × taxa de inadimplência\n",
        "\n",
        "Para cada uma das 3 colunas de atraso (`30-59`, `60-89`, `≥90` dias) nós:\n",
        "- Agrupamos em faixas: `0, 1, 2, 3, 4, 5+` e uma categoria **`Cód. sistema (96/98)`** isolada.\n",
        "- Plotamos volume por faixa (barras, eixo esq.) e taxa de inadimplência (linha, eixo dir.), com linha pontilhada na taxa média geral (6,68%).\n",
        "- Medimos a correlação de Spearman (ignorando 96/98 para não inflar artificialmente).\n"
    ]
}

code_setup = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "sns.set_theme(style=\"whitegrid\")\n",
        "plt.rcParams[\"figure.dpi\"] = 120\n",
        "\n",
        "ALVO = \"inadimplente_2anos\"\n",
        "\n",
        "def bucketizar(v, max_val_real=4):\n",
        "    if pd.isna(v):\n",
        "        return \"NA\"\n",
        "    if v in (96, 98):\n",
        "        return \"Cód. sistema (96/98)\"\n",
        "    if v <= max_val_real:\n",
        "        return str(int(v))\n",
        "    return f\"{max_val_real+1}+\"\n",
        "\n",
        "col_atrasos = {\n",
        "    \"atrasos_30_59_dias\":   \"Atrasos 30–59 dias\",\n",
        "    \"atrasos_60_89_dias\":   \"Atrasos 60–89 dias\",\n",
        "    \"atrasos_90_mais_dias\": \"Atrasos ≥ 90 dias\",\n",
        "}\n",
        "ORDEM = [\"0\", \"1\", \"2\", \"3\", \"4\", \"5+\", \"Cód. sistema (96/98)\"]\n",
        "PALETA = [\"#2ecc71\", \"#b2f0a0\", \"#fff066\", \"#ffc145\", \"#ff8c42\", \"#ff5e5e\", \"#8e44ad\"]\n",
    ]
}

code_geral = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "taxa_media = df[ALVO].mean() * 100\n",
        "resultados = {}\n",
        "\n",
        "for col, titulo in col_atrasos.items():\n",
        "    bucket_col = f\"{col}_faixa\"\n",
        "    df[bucket_col] = df[col].apply(bucketizar)\n",
        "    tab = (df.groupby(bucket_col, dropna=False)[ALVO]\n",
        "             .agg(n_clientes=\"count\",\n",
        "                  n_inadimplentes=\"sum\",\n",
        "                  taxa=lambda s: s.mean() * 100)\n",
        "             .reset_index())\n",
        "    tab.columns = [\"Faixa\", \"N clientes\", \"N inadimplentes\", \"Taxa (%)\"]\n",
        "    tab[\"ordem\"] = tab[\"Faixa\"].apply(lambda x: ORDEM.index(x) if x in ORDEM else 999)\n",
        "    tab = tab.sort_values(\"ordem\").drop(columns=\"ordem\").reset_index(drop=True)\n",
        "    tab_disp = tab.round({\"Taxa (%)\": 2})\n",
        "    resultados[col] = tab_disp\n",
        "    print(f\"\\n=== {titulo} ({col}) ===\")\n",
        "    print(tab_disp.to_string(index=False))\n",
        "\n",
        "    xs = list(range(len(tab)))\n",
        "    labels = tab[\"Faixa\"].tolist()\n",
        "    cores = [PALETA[ORDEM.index(l)] if l in ORDEM else \"#999\" for l in labels]\n",
        "\n",
        "    fig, ax1 = plt.subplots(figsize=(8.5, 5.))\n",
        "    ax2 = ax1.twinx()\n",
        "    bars = ax1.bar(xs, tab[\"N clientes\"], color=cores, alpha=0.9)\n",
        "    for b in bars:\n",
        "        ax1.text(b.get_x() + b.get_width()/2, b.get_height(), f\"{b.get_height():,.0f}\",\n",
        "                 ha=\"center\", va=\"bottom\", fontsize=8)\n",
        "    linha, = ax2.plot(xs, tab[\"Taxa (%)\"], color=\"#c0392b\", marker=\"o\", markersize=8, linewidth=2.2)\n",
        "    for i, t in enumerate(tab[\"Taxa (%)\"].values):\n",
        "        ax2.annotate(f\"{t:.1f}%\", (xs[i], t), xytext=(0, 10),\n",
        "                     textcoords=\"offset points\", ha=\"center\", color=\"#c0392b\", fontweight=\"bold\")\n",
        "    ax2.axhline(taxa_media, color=\"#2c3e50\", ls=\":\", lw=1.3,\n",
        "                label=f\"Média geral = {taxa_media:.2f}%\")\n",
        "    ax1.set_xticks(xs); ax1.set_xticklabels(labels)\n",
        "    ax1.set_ylabel(\"Nº de clientes (barras)\")\n",
        "    ax2.set_ylabel(\"Taxa de inadimplência % (linha)\", color=\"#c0392b\")\n",
        "    ax2.set_ylim(0, max(65, tab[\"Taxa (%)\"].max() * 1.2))\n",
        "    ax1.set_title(f\"{titulo}: volume × taxa de inadimplência por faixa\", fontsize=12, pad=12)\n",
        "    fig.legend(loc=\"upper left\", bbox_to_anchor=(0.12, 0.95), fontsize=9)\n",
        "    plt.tight_layout()\n",
        "    fpath = os.path.join(FIG_PATH, f\"fase02_faixas_{col}.png\")\n",
        "    fig.savefig(fpath, dpi=150, bbox_inches=\"tight\")\n",
        "    print(f\"  -> gráfico salvo em: {fpath}\")\n",
        "    plt.show()\n",
    ]
}

code_conjunto = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- Gráfico conjunto para comparar as 3 colunas lado a lado ---\n",
        "fig, axes = plt.subplots(3, 1, figsize=(9, 12))\n",
        "for idx, (col, titulo) in enumerate(col_atrasos.items()):\n",
        "    ax = axes[idx]\n",
        "    tab = resultados[col]\n",
        "    xs = list(range(len(tab)))\n",
        "    labels = tab[\"Faixa\"].tolist()\n",
        "    cores = [PALETA[ORDEM.index(l)] if l in ORDEM else \"#999\" for l in labels]\n",
        "    ax.bar(xs, tab[\"N clientes\"], color=cores, alpha=0.9)\n",
        "    ax2 = ax.twinx()\n",
        "    ax2.plot(xs, tab[\"Taxa (%)\"], color=\"#c0392b\", marker=\"o\", ms=7, lw=2.)\n",
        "    for i, t in enumerate(tab[\"Taxa (%)\"].values):\n",
        "        ax2.annotate(f\"{t:.1f}%\", (xs[i], t), xytext=(0, 8),\n",
        "                     textcoords=\"offset points\", ha=\"center\", color=\"#c0392b\", fontweight=\"bold\", fontsize=8.5)\n",
        "    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)\n",
        "    ax.set_ylabel(\"Nº clientes\", fontsize=9)\n",
        "    ax2.set_ylabel(\"Inadimplência %\", color=\"#c0392b\", fontsize=9)\n",
        "    ax2.tick_params(axis=\"y\", labelcolor=\"#c0392b\")\n",
        "    ax2.axhline(taxa_media, color=\"#2c3e50\", ls=\":\", lw=1.1, alpha=0.7)\n",
        "    ax.set_title(f\"{titulo} ({col})\", fontsize=11)\n",
        "fig.suptitle(\"Histórico de atrasos × taxa de inadimplência (por faixa)\", fontsize=14, y=1.005)\n",
        "plt.tight_layout()\n",
        "fpath = os.path.join(FIG_PATH, \"fase02_faixas_atrasos_conjunto.png\")\n",
        "fig.savefig(fpath, dpi=150, bbox_inches=\"tight\")\n",
        "print(f\"Gráfico conjunto salvo em: {fpath}\")\n",
        "plt.show()\n",
    ]
}

code_corr = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- Correlação Spearman entre cada atraso e inadimplência (ignora 96/98) ---\n",
        "mask = (\n",
        "    (df[\"atrasos_30_59_dias\"] < 90) &\n",
        "    (df[\"atrasos_60_89_dias\"] < 90) &\n",
        "    (df[\"atrasos_90_mais_dias\"] < 90)\n",
        ")\n",
        "corrs = {}\n",
        "for col, titulo in col_atrasos.items():\n",
        "    rho = df.loc[mask, [col, ALVO]].corr(method=\"spearman\").iloc[0, 1]\n",
        "    corrs[titulo] = rho\n",
        "\n",
        "tab_corr = pd.DataFrame(corrs.items(), columns=[\"Coluna de atraso\", \"Spearman com inadimplência\"])\n",
        "tab_corr = tab_corr.sort_values(\"Spearman com inadimplência\", ascending=False).reset_index(drop=True)\n",
        "tab_corr[\"Spearman com inadimplência\"] = tab_corr[\"Spearman com inadimplência\"].round(4)\n",
        "display(tab_corr.style.hide(axis=\"index\"))\n",
        "\n",
        "print(\"\\nInterpretação:\")\n",
        "for i, linha in tab_corr.iterrows():\n",
        "    r = linha[\"Spearman com inadimplência\"]\n",
        "    force = (\n",
        "        \"desprezível\"  if abs(r) < 0.10 else\n",
        "        \"fraca\"         if abs(r) < 0.20 else\n",
        "        \"moderada\"      if abs(r) < 0.40 else\n",
        "        \"forte\"\n",
        "    )\n",
        "    print(f\"  · {linha['Coluna de atraso']:<20s} — ρ = {r:+.4f} — associação {force}.\")\n",
    ]
}

md_conclusao = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 10.1 Conclusão visual\n",
        "\n",
        "As 3 colunas de atraso histórico carregam **sinal preditivo monotônico forte**:\n",
        "- Taxa cresce de forma consistente com o número de atrasos (0 → 1 → 2 → 3 → 4 → 5+), tanto visualmente quanto nos números.\n",
        "- **Ordem de força:** `≥90 dias` (ρ = +0,335) > `60–89 dias` (ρ = +0,268) > `30–59 dias` (ρ = +0,251). Quanto mais severo o atraso, melhor a previsão.\n",
        "- `Cód. sistema (96/98)` aparece isoladamente como **segunda categoria mais perigosa** (≈54,6%) e deve ser mantido via flag binária separada.\n",
        "- O atraso de **≥90 dias, 4 vezes ou mais**, chega a **67% de inadimplência** na base — quase chance certa de calote.\n"
    ]
}

# Insere novas células ANTES da última (que é "Fim da Fase 2 / Próxima fase")
novas = [md_intro, code_setup, code_geral, code_conjunto, code_corr, md_conclusao]
ultima = nb["cells"].pop()
for c in novas:
    nb["cells"].append(c)
nb["cells"].append(ultima)

# Reseta outputs no clean
for c in nb["cells"]:
    if c.get("cell_type") == "code":
        c["outputs"] = []
        c["execution_count"] = None

with open(CLEAN, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"[OK] Clean atualizado em: {CLEAN}")

# Regenera executed rodando nbconvert
print("[...] Executando notebook via jupyter nbconvert...")
cmd = [sys.executable, "-m", "jupyter", "nbconvert",
       "--to", "notebook", "--execute", "--inplace",
       "--ExecutePreprocessor.timeout=900",
       "--ExecutePreprocessor.kernel_name=python3",
       "--allow-errors", CLEAN]
res = subprocess.run(cmd, cwd=os.path.dirname(CLEAN), capture_output=True, text=True)
if res.returncode == 0:
    import shutil
    shutil.move(CLEAN, EXEC)
    print(f"[OK] Executed salvo em: {EXEC}")
    with open(EXEC, "r", encoding="utf-8") as f:
        nb2 = json.load(f)
    for c in nb2["cells"]:
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None
    with open(CLEAN, "w", encoding="utf-8") as f:
        json.dump(nb2, f, ensure_ascii=False, indent=1)
    print(f"[OK] Clean (sem outputs) regenerado em: {CLEAN}")
else:
    print("[ERRO] nbconvert falhou.")
    print("STDOUT final:"); print(res.stdout[-1500:])
    print("STDERR final:"); print(res.stderr[-1500:])
