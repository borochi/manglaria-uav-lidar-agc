"""
Script 03 - Correlacion entre metricas LiDAR y AGC de campo
Entrada : metricas_lidar_principal.csv + correspondencia.csv (con agc_mg_c_ha)
Salida  : correlaciones_agc.csv, graficas de correlacion por epoca
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

RAIZ         = Path("/mnt/e/Modelos_IA/LiDar_3D_AGC")
REGION       = RAIZ / "Nayarit"
DIR_METRICAS = REGION / "metricas"
DIR_RESULT   = REGION / "resultados"
CSV_CORRESP  = REGION / "parcelas" / "correspondencia.csv"
CSV_METRICAS = DIR_METRICAS / "metricas_lidar_principal.csv"

DIR_RESULT.mkdir(parents=True, exist_ok=True)

METRICAS_MODELO = [
    "h_max", "h_mean", "h_median", "h_std", "h_cv",
    "h_p25", "h_p50", "h_p75", "h_p95", "h_p99",
    "h_skew", "h_kurt",
    "densidad_total", "densidad_veg",
    "fr_suelo", "fr_bajo", "fr_medio", "fr_alto",
    "rumple", "cobertura", "lad_max", "lad_mean",
    "int_mean", "int_std",
]


def correlacion_completa(df, metricas):
    """
    Calcula Pearson y Spearman de cada metrica vs AGC.
    Retorna DataFrame ordenado por |r_pearson|.
    """
    resultados = []
    for m in metricas:
        if m not in df.columns:
            continue
        vals = df[[m, "agc_mg_c_ha"]].dropna()
        if len(vals) < 5:
            continue
        r_p, p_p = pearsonr(vals[m],  vals["agc_mg_c_ha"])
        r_s, p_s = spearmanr(vals[m], vals["agc_mg_c_ha"])
        resultados.append({
            "metrica"   : m,
            "n"         : len(vals),
            "pearson_r" : round(r_p, 4),
            "pearson_p" : round(p_p, 4),
            "spearman_r": round(r_s, 4),
            "spearman_p": round(p_s, 4),
            "sig_pearson" : "***" if p_p < 0.001 else "**" if p_p < 0.01 else "*" if p_p < 0.05 else "",
            "sig_spearman": "***" if p_s < 0.001 else "**" if p_s < 0.01 else "*" if p_s < 0.05 else "",
        })
    df_r = pd.DataFrame(resultados)
    df_r["abs_pearson"] = df_r["pearson_r"].abs()
    df_r = df_r.sort_values("abs_pearson", ascending=False).drop(columns="abs_pearson")
    return df_r


def graficar_top_metricas(df, top_metricas, titulo, ruta_fig):
    """
    Scatterplot AGC vs cada una de las top metricas.
    """
    n  = len(top_metricas)
    nc = 3
    nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(15, nr * 4))
    axes = axes.flatten()

    for i, m in enumerate(top_metricas):
        ax  = axes[i]
        sub = df[[m, "agc_mg_c_ha", "parcela_id"]].dropna()
        ax.scatter(sub[m], sub["agc_mg_c_ha"],
                   alpha=0.7, edgecolors="white", linewidth=0.5,
                   color="#1d9e75", s=60)

        # Linea de tendencia
        z = np.polyfit(sub[m], sub["agc_mg_c_ha"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(sub[m].min(), sub[m].max(), 100)
        ax.plot(x_line, p(x_line), color="#0f6e56", linewidth=1.5, linestyle="--")

        r_p, p_p = pearsonr(sub[m], sub["agc_mg_c_ha"])
        r_s, _   = spearmanr(sub[m], sub["agc_mg_c_ha"])
        sig = "***" if p_p < 0.001 else "**" if p_p < 0.01 else "*" if p_p < 0.05 else "ns"

        ax.set_xlabel(m, fontsize=10)
        ax.set_ylabel("AGC (Mg C/ha)", fontsize=10)
        ax.set_title(f"r={r_p:.3f}{sig}  rho={r_s:.3f}  n={len(sub)}", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Ocultar ejes sobrantes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(titulo, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(ruta_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figura guardada: {ruta_fig.name}")


def graficar_heatmap_epocas(df_pivot, ruta_fig):
    """
    Heatmap de correlaciones Pearson por metrica x epoca.
    """
    import matplotlib.colors as mcolors
    fig, ax = plt.subplots(figsize=(10, len(df_pivot) * 0.45 + 2))
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    im = ax.imshow(df_pivot.values.astype(float),
                   cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(df_pivot.columns)))
    ax.set_xticklabels(df_pivot.columns, fontsize=10)
    ax.set_yticks(range(len(df_pivot.index)))
    ax.set_yticklabels(df_pivot.index, fontsize=9)

    # Valores en cada celda
    for i in range(len(df_pivot.index)):
        for j in range(len(df_pivot.columns)):
            val = df_pivot.values[i, j]
            if not np.isnan(float(val)):
                ax.text(j, i, f"{float(val):.2f}",
                        ha="center", va="center", fontsize=8,
                        color="black")

    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Correlacion Pearson: metricas LiDAR vs AGC por epoca",
                 fontsize=11, pad=12)
    plt.tight_layout()
    plt.savefig(ruta_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figura guardada: {ruta_fig.name}")


def main():
    # Cargar datos
    metr    = pd.read_csv(CSV_METRICAS)
    corresp = pd.read_csv(CSV_CORRESP)

    # AGC unico por parcela (tomar el primero, son todos iguales)
    agc = corresp.groupby("parcela_id")["agc_mg_c_ha"].first().reset_index()

    # Join metricas + AGC
    df = metr.merge(agc, on="parcela_id", how="left")
    n_sin_agc = df["agc_mg_c_ha"].isna().sum()
    if n_sin_agc > 0:
        print(f"AVISO: {n_sin_agc} registros sin AGC — se excluiran del analisis")

    df = df.dropna(subset=["agc_mg_c_ha"])
    print(f"Registros con AGC: {len(df)}")
    print(f"Parcelas unicas  : {df['parcela_id'].nunique()}")
    print(f"Epocas           : {sorted(df['epoca'].unique())}")
    print(f"AGC rango        : {df['agc_mg_c_ha'].min():.2f} - "
          f"{df['agc_mg_c_ha'].max():.2f} Mg C/ha")

    # ── 1. Correlacion global (todas las epocas juntas) ───────────────────────
    print("\n--- Correlacion global ---")
    corr_global = correlacion_completa(df, METRICAS_MODELO)
    print(corr_global[["metrica","n","pearson_r","sig_pearson",
                        "spearman_r","sig_spearman"]].head(15).to_string())

    ruta_corr = DIR_RESULT / "03_correlaciones_global.csv"
    corr_global.to_csv(ruta_corr, index=False)
    print(f"Guardado: {ruta_corr.name}")

    # Top 9 metricas para graficar
    top9 = corr_global["metrica"].head(9).tolist()
    graficar_top_metricas(
        df, top9,
        "Top 9 metricas LiDAR vs AGC — todas las epocas",
        DIR_RESULT / "03_scatterplots_global.png"
    )

    # ── 2. Correlacion por epoca ──────────────────────────────────────────────
    print("\n--- Correlacion por epoca ---")
    corr_epocas = {}
    for epoca in sorted(df["epoca"].unique()):
        df_ep = df[df["epoca"] == epoca]
        if len(df_ep) < 5:
            print(f"  {epoca}: muy pocas parcelas ({len(df_ep)}), omitida")
            continue
        corr_ep = correlacion_completa(df_ep, METRICAS_MODELO)
        corr_epocas[epoca] = corr_ep
        print(f"\n  {epoca} (n={len(df_ep)}):")
        print(corr_ep[["metrica","pearson_r","sig_pearson"]].head(5).to_string())

        ruta_ep = DIR_RESULT / f"03_correlaciones_{epoca}.csv"
        corr_ep.to_csv(ruta_ep, index=False)

    # ── 3. Heatmap comparativo entre epocas ──────────────────────────────────
    if len(corr_epocas) > 1:
        print("\n--- Heatmap por epoca ---")
        # Usar top 15 metricas del global
        top15 = corr_global["metrica"].head(15).tolist()
        pivot_data = {}
        for epoca, corr_ep in corr_epocas.items():
            corr_ep_idx = corr_ep.set_index("metrica")["pearson_r"]
            pivot_data[epoca] = corr_ep_idx.reindex(top15)
        df_pivot = pd.DataFrame(pivot_data)
        graficar_heatmap_epocas(
            df_pivot,
            DIR_RESULT / "03_heatmap_epocas.png"
        )

    # ── 4. Resumen final ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("TOP 5 METRICAS MAS CORRELACIONADAS CON AGC:")
    print("=" * 55)
    for _, row in corr_global.head(5).iterrows():
        print(f"  {row['metrica']:<20} "
              f"r={row['pearson_r']:+.3f}{row['sig_pearson']:<4} "
              f"rho={row['spearman_r']:+.3f}{row['sig_spearman']}")
    print("=" * 55)


if __name__ == "__main__":
    main()
