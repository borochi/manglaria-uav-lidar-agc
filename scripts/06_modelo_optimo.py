"""
Script 06 - Modelo optimo final: regresion lineal h_p99 + rumple
Entrena con todos los datos de Nayarit y guarda listo para inferencia
sobre nuevos vuelos sin datos de campo.
"""

import sys
sys.path.insert(0, "/mnt/e/Modelos_IA/LiDar_3D_AGC/scripts")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from modelo_base import cargar_datos, DIR_RESULT, DIR_MODELOS

FEATURES_OPTIMAS = ["h_p99", "rumple"]
NOMBRE           = "modelo_optimo_lineal"
DIR_MODELO       = DIR_MODELOS / NOMBRE


def validacion_loo_final(df, features):
    """LOO final para reporte."""
    X   = df[features].values
    y   = df["agc_mg_c_ha"].values
    ids = df["parcela_id"].values
    loo = LeaveOneOut()
    y_real, y_pred, ids_pred = [], [], []
    for train_idx, test_idx in loo.split(X):
        sc = MinMaxScaler()
        X_tr = sc.fit_transform(X[train_idx])
        X_te = sc.transform(X[test_idx])
        m = LinearRegression()
        m.fit(X_tr, y[train_idx])
        y_real.append(float(y[test_idx][0]))
        y_pred.append(float(m.predict(X_te)[0]))
        ids_pred.append(ids[test_idx[0]])
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    metricas = {
        "R2"  : round(r2_score(y_real, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_real, y_pred)), 4),
        "MAE" : round(np.mean(np.abs(y_real - y_pred)), 4),
        "r"   : round(pearsonr(y_real, y_pred)[0], 4),
        "bias": round(np.mean(y_pred - y_real), 4),
        "n"   : len(y_real),
    }
    df_pred = pd.DataFrame({
        "parcela_id": ids_pred,
        "agc_real"  : y_real,
        "agc_pred"  : y_pred,
        "error"     : y_pred - y_real,
    })
    return metricas, df_pred


def graficar_modelo_optimo(df_pred, metricas, coef, intercept, dir_result):
    """Grafica completa del modelo optimo para publicacion."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: real vs predicho
    ax = axes[0]
    ax.scatter(df_pred["agc_real"], df_pred["agc_pred"],
               alpha=0.8, color="#1d9e75", edgecolors="white",
               linewidth=0.5, s=80)
    lims = [
        min(df_pred["agc_real"].min(), df_pred["agc_pred"].min()) - 3,
        max(df_pred["agc_real"].max(), df_pred["agc_pred"].max()) + 3,
    ]
    ax.plot(lims, lims, "k--", linewidth=1.2, alpha=0.6, label="1:1")
    # Etiquetas de parcelas outliers
    umbral = metricas["RMSE"] * 1.5
    for _, row in df_pred.iterrows():
        if abs(row["error"]) > umbral:
            ax.annotate(row["parcela_id"],
                        (row["agc_real"], row["agc_pred"]),
                        fontsize=7, alpha=0.7,
                        xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("AGC real (Mg C/ha)", fontsize=11)
    ax.set_ylabel("AGC predicho (Mg C/ha)", fontsize=11)
    ax.set_title(
        f"R²={metricas['R2']:.3f}  RMSE={metricas['RMSE']:.2f} Mg C/ha\n"
        f"r={metricas['r']:.3f}  n={metricas['n']}",
        fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Panel 2: residuos vs predicho
    ax2 = axes[1]
    ax2.scatter(df_pred["agc_pred"], df_pred["error"],
                alpha=0.8, color="#185fa5", edgecolors="white",
                linewidth=0.5, s=80)
    ax2.axhline(0, color="k", linestyle="--", linewidth=1.2, alpha=0.6)
    ax2.axhline(metricas["RMSE"],  color="#ba7517",
                linestyle=":", linewidth=1, alpha=0.7, label=f"+RMSE")
    ax2.axhline(-metricas["RMSE"], color="#ba7517",
                linestyle=":", linewidth=1, alpha=0.7, label=f"-RMSE")
    ax2.set_xlabel("AGC predicho (Mg C/ha)", fontsize=11)
    ax2.set_ylabel("Residuo (predicho - real)", fontsize=11)
    ax2.set_title(f"Residuos | bias={metricas['bias']:.2f} Mg C/ha",
                  fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: distribucion de errores
    ax3 = axes[2]
    ax3.hist(df_pred["error"], bins=12, color="#534ab7",
             alpha=0.75, edgecolor="white", linewidth=0.5)
    ax3.axvline(0, color="k", linestyle="--", linewidth=1.2, alpha=0.6)
    ax3.axvline(metricas["bias"], color="#993556",
                linestyle="-", linewidth=1.5, label=f"bias={metricas['bias']:.2f}")
    ax3.set_xlabel("Error (Mg C/ha)", fontsize=11)
    ax3.set_ylabel("Frecuencia", fontsize=11)
    ax3.set_title("Distribucion de errores", fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.suptitle(
        "Modelo optimo: AGC = f(h_p99, rumple) — Manglares Nayarit\n"
        "Regresion lineal multiple, validacion LOO",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    ruta = dir_result / "06_modelo_optimo_final.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figura guardada: {ruta.name}")


def guardar_modelo_optimo(modelo, scaler, features, metricas, coef, intercept):
    """
    Guarda todo lo necesario para inferencia en nuevos vuelos.
    Incluye metadata completa para reproducibilidad.
    """
    DIR_MODELO.mkdir(parents=True, exist_ok=True)

    # Modelo y scaler
    with open(DIR_MODELO / "modelo.pkl", "wb") as f:
        pickle.dump(modelo, f)
    with open(DIR_MODELO / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Metadata completa
    meta = {
        "nombre"         : NOMBRE,
        "tipo"           : "LinearRegression",
        "features"       : features,
        "n_features"     : len(features),
        "coeficientes"   : {f: round(c, 6)
                            for f, c in zip(features, coef)},
        "intercepto"     : round(float(intercept), 6),
        "metricas_LOO"   : metricas,
        "dataset"        : "Nayarit_Marismas_Nacionales",
        "n_parcelas"     : 26,
        "n_registros"    : metricas["n"],
        "epocas"         : ["secas_2023","lluvias_2023",
                            "secas_2024","lluvias_2024"],
        "especies"       : ["Rhizophora mangle",
                            "Avicennia germinans",
                            "Laguncularia racemosa"],
        "unidad_AGC"     : "Mg C / ha",
        "notas"          : "Sitio afectado por huracan oct-2022. "
                           "Validar en sitios sin disturbio antes de extrapolar.",
        "version"        : "1.0",
    }
    with open(DIR_MODELO / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Ecuacion en texto plano para referencia rapida
    ecuacion = (
        f"AGC (Mg C/ha) = "
        f"{coef[0]:.4f} * h_p99 + "
        f"{coef[1]:.4f} * rumple + "
        f"{float(intercept):.4f}\n\n"
        f"Donde:\n"
        f"  h_p99  = percentil 99 de altura normalizada de vegetacion (m)\n"
        f"  rumple = ratio superficie real dosel / area proyectada (adimensional)\n\n"
        f"Metricas LOO: R2={metricas['R2']} RMSE={metricas['RMSE']} Mg C/ha "
        f"r={metricas['r']} n={metricas['n']}\n"
    )
    with open(DIR_MODELO / "ecuacion.txt", "w") as f:
        f.write(ecuacion)

    print(f"  Modelo guardado en: {DIR_MODELO}")
    print(f"  Ecuacion: {ecuacion.split(chr(10))[0]}")


def main():
    print("=" * 55)
    print("SCRIPT 06 - MODELO OPTIMO FINAL")
    print("=" * 55)

    df, _ = cargar_datos(epoca_filtro=None, incluir_disturbio=False)

    # Validacion LOO final
    print("\nValidacion LOO final...")
    metricas, df_pred = validacion_loo_final(df, FEATURES_OPTIMAS)
    print("Metricas LOO:")
    for k, v in metricas.items():
        print(f"  {k}: {v}")

    # Entrenamiento final con TODOS los datos
    print("\nEntrenando modelo final con todos los datos...")
    X  = df[FEATURES_OPTIMAS].values
    y  = df["agc_mg_c_ha"].values
    sc = MinMaxScaler()
    X_sc = sc.fit_transform(X)
    modelo_final = LinearRegression()
    modelo_final.fit(X_sc, y)

    coef      = modelo_final.coef_
    intercept = modelo_final.intercept_

    # Graficas
    print("\nGenerando figura...")
    graficar_modelo_optimo(df_pred, metricas, coef, intercept, DIR_RESULT)

    # Guardar predicciones LOO
    df_pred.to_csv(DIR_RESULT / "06_predicciones_optimo.csv", index=False)

    # Guardar modelo
    guardar_modelo_optimo(modelo_final, sc, FEATURES_OPTIMAS,
                          metricas, coef, intercept)

    print("\n" + "=" * 55)
    print("MODELO OPTIMO LISTO PARA INFERENCIA")
    print("=" * 55)
    print(f"  Features  : {FEATURES_OPTIMAS}")
    print(f"  R2 (LOO)  : {metricas['R2']}")
    print(f"  RMSE      : {metricas['RMSE']} Mg C/ha")
    print(f"  r         : {metricas['r']}")
    print(f"  Archivos  : {DIR_MODELO}")
    print("=" * 55)


if __name__ == "__main__":
    main()
