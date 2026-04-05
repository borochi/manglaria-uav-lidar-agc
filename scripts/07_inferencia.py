"""
Script 07 - Inferencia de AGC en nuevos vuelos sin datos de campo
Usa el modelo optimo entrenado con datos de Nayarit.

Uso:
    python 07_inferencia.py --metricas ruta/a/metricas.csv
                            --salida   ruta/a/resultados.csv
                            --region   NombreRegion

Si no se pasan argumentos usa los datos de Nayarit como prueba.
"""

import sys
import argparse
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, "/mnt/e/Modelos_IA/LiDar_3D_AGC/scripts")
from modelo_base import DIR_MODELOS, DIR_RESULT, DIR_METRICAS

DIR_MODELO = DIR_MODELOS / "modelo_optimo_lineal"


def cargar_modelo():
    """Carga modelo, scaler y metadata."""
    with open(DIR_MODELO / "modelo.pkl", "rb") as f:
        modelo = pickle.load(f)
    with open(DIR_MODELO / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(DIR_MODELO / "metadata.json") as f:
        meta = json.load(f)
    return modelo, scaler, meta


def validar_metricas(df, features):
    """
    Verifica que el CSV de metricas tiene las columnas necesarias
    y alerta si los valores estan fuera del rango de entrenamiento.
    """
    faltantes = [f for f in features if f not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas faltantes en el CSV: {faltantes}")

    # Rangos del dataset de entrenamiento (Nayarit)
    rangos = {
        "h_p99" : (0.5, 20.0),
        "rumple": (0.9, 3.5),
    }
    alertas = []
    for feat, (vmin, vmax) in rangos.items():
        if feat not in df.columns:
            continue
        fuera = df[(df[feat] < vmin) | (df[feat] > vmax)]
        if len(fuera) > 0:
            alertas.append(
                f"  ALERTA: {len(fuera)} parcelas con {feat} fuera del "
                f"rango de entrenamiento [{vmin}, {vmax}]"
            )
    if alertas:
        print("\nAVISOS DE EXTRAPOLACION:")
        for a in alertas:
            print(a)
        print("  Las predicciones fuera de rango son menos confiables.")
    return df


def predecir(df, modelo, scaler, features):
    """Aplica el modelo a un DataFrame de metricas."""
    X    = df[features].values
    X_sc = scaler.transform(X)
    pred = modelo.predict(X_sc)
    pred = np.clip(pred, 0, None)  # AGC no puede ser negativo
    return pred


def graficar_inferencia(df_pred, region, ruta_fig):
    """
    Grafica la distribucion de AGC predicho por parcela.
    Si hay AGC real disponible, grafica real vs predicho tambien.
    """
    tiene_real = "agc_real" in df_pred.columns and                   df_pred["agc_real"].notna().any()

    if tiene_real:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_dist = axes[1]
        ax_comp = axes[0]
        sub = df_pred.dropna(subset=["agc_real"])
        ax_comp.scatter(sub["agc_real"], sub["agc_pred"],
                        alpha=0.8, color="#1d9e75",
                        edgecolors="white", linewidth=0.5, s=80)
        lims = [
            min(sub["agc_real"].min(), sub["agc_pred"].min()) - 3,
            max(sub["agc_real"].max(), sub["agc_pred"].max()) + 3,
        ]
        ax_comp.plot(lims, lims, "k--", linewidth=1.2,
                     alpha=0.6, label="1:1")
        from sklearn.metrics import r2_score, mean_squared_error
        r2   = r2_score(sub["agc_real"], sub["agc_pred"])
        rmse = np.sqrt(mean_squared_error(sub["agc_real"], sub["agc_pred"]))
        ax_comp.set_xlabel("AGC real (Mg C/ha)")
        ax_comp.set_ylabel("AGC predicho (Mg C/ha)")
        ax_comp.set_title(f"Validacion externa\nR2={r2:.3f}  RMSE={rmse:.2f} Mg C/ha")
        ax_comp.legend()
        ax_comp.grid(True, alpha=0.3)
        print(f"\nValidacion externa:")
        print(f"  R2   : {r2:.4f}")
        print(f"  RMSE : {rmse:.4f} Mg C/ha")
        # Guardar metricas de validacion externa
        val_ext = {"region": region, "R2": round(r2,4),
                   "RMSE": round(rmse,4), "n": len(sub)}
        pd.DataFrame([val_ext]).to_csv(
            ruta_fig.parent / f"07_validacion_externa_{region}.csv",
            index=False)
    else:
        fig, ax_dist = plt.subplots(1, 1, figsize=(8, 5))

    # Distribucion de AGC predicho
    ax_dist.barh(df_pred["parcela_id"],
                 df_pred["agc_pred"].sort_values(),
                 color="#1d9e75", alpha=0.8, edgecolor="white")
    ax_dist.axvline(df_pred["agc_pred"].mean(), color="#0f6e56",
                    linestyle="--", linewidth=1.5,
                    label=f"media={df_pred['agc_pred'].mean():.1f}")
    ax_dist.set_xlabel("AGC predicho (Mg C/ha)")
    ax_dist.set_title(f"AGC estimado por parcela — {region}")
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3, axis="x")

    plt.suptitle(f"Inferencia de AGC — {region}", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(ruta_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figura: {ruta_fig.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Inferencia de AGC en nuevos vuelos LiDAR"
    )
    parser.add_argument("--metricas", type=str, default=None,
        help="Ruta al CSV de metricas LiDAR del nuevo sitio")
    parser.add_argument("--salida", type=str, default=None,
        help="Ruta donde guardar las predicciones")
    parser.add_argument("--region", type=str, default="NuevoSitio",
        help="Nombre de la region (para graficas y archivos)")
    parser.add_argument("--agc_real", type=str, default=None,
        help="Opcional: CSV con agc_mg_c_ha real para validacion externa")
    args = parser.parse_args()

    print("=" * 55)
    print("SCRIPT 07 - INFERENCIA DE AGC")
    print("=" * 55)

    # Cargar modelo
    modelo, scaler, meta = cargar_modelo()
    features = meta["features"]
    print(f"Modelo cargado: {meta['nombre']}")
    print(f"Entrenado con : {meta['dataset']}")
    print(f"Features      : {features}")
    print(f"Metricas LOO  : R2={meta['metricas_LOO']['R2']} "
          f"RMSE={meta['metricas_LOO']['RMSE']} Mg C/ha")

    # Cargar metricas del nuevo sitio
    if args.metricas:
        ruta_metr = Path(args.metricas)
        region    = args.region
    else:
        # Modo prueba: usa datos de Nayarit
        ruta_metr = DIR_METRICAS / "metricas_lidar_principal.csv"
        region    = "Nayarit_prueba"
        print("\nModo prueba: usando metricas de Nayarit")

    df = pd.read_csv(ruta_metr)
    print(f"\nParcelas a predecir: {len(df)}")

    # Validar columnas y rangos
    df = validar_metricas(df, features)

    # Predecir
    pred = predecir(df, modelo, scaler, features)

    # Armar resultado
    cols_id = ["parcela_id", "epoca"] if "epoca" in df.columns                else ["parcela_id"]
    df_pred = df[cols_id].copy()
    df_pred["agc_pred_mg_c_ha"] = pred.round(2)
    df_pred["h_p99_usado"]      = df["h_p99"].round(3)
    df_pred["rumple_usado"]     = df["rumple"].round(3)

    # Agregar AGC real si se provee para validacion externa
    if args.agc_real:
        agc_real = pd.read_csv(args.agc_real)
        df_pred  = df_pred.merge(
            agc_real[["parcela_id", "agc_mg_c_ha"]].rename(
                columns={"agc_mg_c_ha": "agc_real"}),
            on="parcela_id", how="left"
        )

    # Mostrar resumen
    print("\nPredicciones:")
    print(df_pred.to_string(index=False))
    print(f"\nAGC predicho — media : {pred.mean():.2f} Mg C/ha")
    print(f"AGC predicho — rango : {pred.min():.2f} - {pred.max():.2f} Mg C/ha")

    # Guardar
    if args.salida:
        ruta_sal = Path(args.salida)
    else:
        ruta_sal = DIR_RESULT / f"07_predicciones_{region}.csv"

    df_pred.to_csv(ruta_sal, index=False)
    print(f"\nPredicciones guardadas: {ruta_sal}")

    # Grafica
    df_graf = df_pred.rename(columns={"agc_pred_mg_c_ha": "agc_pred"})
    graficar_inferencia(
        df_graf, region,
        ruta_sal.parent / f"07_inferencia_{region}.png"
    )

    print("=" * 55)


if __name__ == "__main__":
    main()
