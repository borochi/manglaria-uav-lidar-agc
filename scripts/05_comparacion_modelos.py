"""
Script 05 - Comparacion de modelos: Regresion Lineal, Random Forest y TensorFlow
Mismas features, misma validacion LOO, mismas metricas.
Objetivo: encontrar el modelo optimo para el dataset de AGC en manglares.
"""

import sys
sys.path.insert(0, "/mnt/e/Modelos_IA/LiDar_3D_AGC/scripts")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr
from modelo_base import cargar_datos, DIR_RESULT, FEATURES_BASE

DIR_RESULT.mkdir(parents=True, exist_ok=True)


def loo_sklearn(df, features, modelo_sklearn, nombre):
    X   = df[features].values
    y   = df["agc_mg_c_ha"].values
    ids = df["parcela_id"].values
    loo = LeaveOneOut()
    y_real, y_pred, ids_pred = [], [], []
    total = len(df)
    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
        if fold % 5 == 0:
            print(f"  {nombre} LOO {fold+1}/{total}...", end="\r", flush=True)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = MinMaxScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)
        modelo_sklearn.fit(X_train_sc, y_train)
        pred = modelo_sklearn.predict(X_test_sc)[0]
        y_real.append(float(y_test[0]))
        y_pred.append(float(pred))
        ids_pred.append(ids[test_idx[0]])
    print()
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    metricas = {
        "modelo": nombre,
        "R2"    : round(r2_score(y_real, y_pred), 4),
        "RMSE"  : round(np.sqrt(mean_squared_error(y_real, y_pred)), 4),
        "MAE"   : round(np.mean(np.abs(y_real - y_pred)), 4),
        "r"     : round(pearsonr(y_real, y_pred)[0], 4),
        "bias"  : round(np.mean(y_pred - y_real), 4),
        "n"     : len(y_real),
    }
    df_pred = pd.DataFrame({
        "parcela_id": ids_pred,
        "agc_real"  : y_real,
        "agc_pred"  : y_pred,
        "error"     : y_pred - y_real,
    })
    return metricas, df_pred


def loo_solo_hmax(df):
    """Regresion lineal simple con solo h_p95 — baseline minimo."""
    X   = df[["h_p95"]].values
    y   = df["agc_mg_c_ha"].values
    ids = df["parcela_id"].values
    loo = LeaveOneOut()
    y_real, y_pred, ids_pred = [], [], []
    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)[0]
        y_real.append(float(y_test[0]))
        y_pred.append(float(pred))
        ids_pred.append(ids[test_idx[0]])
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    metricas = {
        "modelo": "Lineal_solo_h_p95",
        "R2"    : round(r2_score(y_real, y_pred), 4),
        "RMSE"  : round(np.sqrt(mean_squared_error(y_real, y_pred)), 4),
        "MAE"   : round(np.mean(np.abs(y_real - y_pred)), 4),
        "r"     : round(pearsonr(y_real, y_pred)[0], 4),
        "bias"  : round(np.mean(y_pred - y_real), 4),
        "n"     : len(y_real),
    }
    df_pred = pd.DataFrame({
        "parcela_id": ids_pred,
        "agc_real"  : y_real,
        "agc_pred"  : y_pred,
        "error"     : y_pred - y_real,
    })
    return metricas, df_pred


def importancia_features_rf(df, features):
    """Calcula importancia de features con Random Forest completo."""
    X = df[features].values
    y = df["agc_mg_c_ha"].values
    scaler = MinMaxScaler()
    X_sc   = scaler.fit_transform(X)
    rf = RandomForestRegressor(n_estimators=500, random_state=42)
    rf.fit(X_sc, y)
    imp = pd.DataFrame({
        "feature"   : features,
        "importancia": rf.feature_importances_,
    }).sort_values("importancia", ascending=False)
    return imp


def graficar_comparacion(resultados, dir_result):
    """Grafica comparativa de todos los modelos."""
    df_res = pd.DataFrame(resultados)
    df_res = df_res.sort_values("R2", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colores = ["#1d9e75", "#185fa5", "#ba7517", "#993556",
               "#534ab7", "#993c1d", "#3b6d11"]

    # R2
    ax = axes[0]
    bars = ax.barh(df_res["modelo"], df_res["R2"],
                   color=colores[:len(df_res)], alpha=0.85)
    ax.set_xlabel("R²")
    ax.set_title("R² (mayor es mejor)")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, df_res["R2"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    # RMSE
    ax = axes[1]
    bars = ax.barh(df_res["modelo"], df_res["RMSE"],
                   color=colores[:len(df_res)], alpha=0.85)
    ax.set_xlabel("RMSE (Mg C/ha)")
    ax.set_title("RMSE (menor es mejor)")
    for bar, val in zip(bars, df_res["RMSE"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    # r de Pearson
    ax = axes[2]
    bars = ax.barh(df_res["modelo"], df_res["r"],
                   color=colores[:len(df_res)], alpha=0.85)
    ax.set_xlabel("r de Pearson")
    ax.set_title("Correlacion real vs predicho")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, df_res["r"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle("Comparacion de modelos — AGC manglares Nayarit (LOO)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    ruta = dir_result / "05_comparacion_modelos.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figura guardada: {ruta.name}")


def graficar_importancia(imp_df, dir_result):
    """Grafica importancia de features del Random Forest."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colores = ["#1d9e75" if i < 5 else "#9fe1cb"
               for i in range(len(imp_df))]
    ax.barh(imp_df["feature"][::-1], imp_df["importancia"][::-1],
            color=colores[::-1], alpha=0.85)
    ax.set_xlabel("Importancia (Gini)")
    ax.set_title("Importancia de features — Random Forest\n(entrenado con todos los datos)",
                 fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    ruta = dir_result / "05_importancia_features.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figura guardada: {ruta.name}")


def main():
    print("=" * 55)
    print("SCRIPT 05 - COMPARACION DE MODELOS")
    print("=" * 55)

    df, features = cargar_datos(epoca_filtro=None, incluir_disturbio=False)

    modelos_sklearn = {
        "Lineal_multiple"  : LinearRegression(),
        "Ridge"            : Ridge(alpha=1.0),
        "Lasso"            : Lasso(alpha=0.1, max_iter=5000),
        "RandomForest"     : RandomForestRegressor(
                                n_estimators=500,
                                max_features=0.6,
                                min_samples_leaf=2,
                                random_state=42),
        "GradientBoosting" : GradientBoostingRegressor(
                                n_estimators=200,
                                max_depth=3,
                                learning_rate=0.05,
                                random_state=42),
    }

    resultados  = []
    predicciones = {}

    # Baseline: solo h_p95
    print("\nBaseline: regresion lineal solo h_p95")
    met, pred = loo_solo_hmax(df)
    resultados.append(met)
    predicciones["Lineal_solo_h_p95"] = pred
    print(f"  R2={met['R2']}  RMSE={met['RMSE']}  r={met['r']}")

    # Modelos sklearn
    for nombre, modelo in modelos_sklearn.items():
        print(f"\n{nombre}")
        met, pred = loo_sklearn(df, features, modelo, nombre)
        resultados.append(met)
        predicciones[nombre] = pred
        print(f"  R2={met['R2']}  RMSE={met['RMSE']}  r={met['r']}")

    # Agregar resultados TensorFlow del script 04b
    resultados.append({
        "modelo": "TensorFlow_B",
        "R2"    : 0.494,
        "RMSE"  : 12.012,
        "MAE"   : 8.9608,
        "r"     : 0.7035,
        "bias"  : 0.2098,
        "n"     : 48,
    })

    # Tabla resumen
    df_res = pd.DataFrame(resultados).sort_values("R2", ascending=False)
    print("\n" + "=" * 55)
    print("RESUMEN COMPARATIVO (ordenado por R2):")
    print("=" * 55)
    print(df_res[["modelo","n","R2","RMSE","MAE","r","bias"]].to_string(index=False))

    # Guardar tabla
    df_res.to_csv(DIR_RESULT / "05_comparacion_modelos.csv", index=False)

    # Graficas
    print("\nGenerando figuras...")
    graficar_comparacion(resultados, DIR_RESULT)

    # Importancia de features con RF
    print("Calculando importancia de features...")
    imp = importancia_features_rf(df, features)
    print(imp.to_string(index=False))
    imp.to_csv(DIR_RESULT / "05_importancia_features.csv", index=False)
    graficar_importancia(imp, DIR_RESULT)

    print("\n" + "=" * 55)
    mejor = df_res.iloc[0]
    print(f"MEJOR MODELO: {mejor['modelo']}")
    print(f"  R2   : {mejor['R2']}")
    print(f"  RMSE : {mejor['RMSE']} Mg C/ha")
    print(f"  r    : {mejor['r']}")
    print("=" * 55)


if __name__ == "__main__":
    main()
