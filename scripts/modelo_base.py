"""
Modulo base compartido por los tres modelos TensorFlow.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import pickle
import json

RAIZ         = Path("/mnt/e/Modelos_IA/LiDar_3D_AGC")
REGION       = RAIZ / "Nayarit"
DIR_METRICAS = REGION / "metricas"
DIR_MODELOS  = REGION / "modelos"
DIR_RESULT   = REGION / "resultados"
CSV_CORRESP  = REGION / "parcelas" / "correspondencia.csv"
CSV_METRICAS = DIR_METRICAS / "metricas_lidar_principal.csv"

DIR_MODELOS.mkdir(parents=True, exist_ok=True)
DIR_RESULT.mkdir(parents=True, exist_ok=True)

FEATURES_BASE = [
    "h_p99", "h_p95", "h_std", "lad_mean", "h_mean",
    "h_p75", "h_p50", "cobertura", "rumple", "fr_alto",
    "fr_suelo", "fr_bajo", "densidad_total", "h_cv", "h_skew",
]

FEATURES_CON_DISTURBIO = FEATURES_BASE + ["disturbio"]


def cargar_datos(epoca_filtro=None, incluir_disturbio=False):
    metr    = pd.read_csv(CSV_METRICAS)
    corresp = pd.read_csv(CSV_CORRESP)
    agc_dist = corresp.groupby("parcela_id").agg(
        agc_mg_c_ha=("agc_mg_c_ha", "first"),
        disturbio  =("disturbio",   "first"),
    ).reset_index()
    df = metr.merge(agc_dist, on="parcela_id", how="left")
    df = df.dropna(subset=["agc_mg_c_ha"])
    if epoca_filtro:
        df = df[df["epoca"] == epoca_filtro].copy()
    features = FEATURES_CON_DISTURBIO if incluir_disturbio else FEATURES_BASE
    features = [f for f in features if f in df.columns]
    df = df.dropna(subset=features)
    print(f"Dataset: {len(df)} registros | {len(features)} features")
    print(f"AGC: {df['agc_mg_c_ha'].min():.2f} - {df['agc_mg_c_ha'].max():.2f} Mg C/ha")
    return df, features


def construir_modelo(n_features, nombre="modelo"):
    tf.random.set_seed(42)
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(64, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear"),
    ], name=nombre)
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return modelo


def validacion_cruzada_loo(df, features, nombre_modelo, epochs=200):
    X    = df[features].values
    y    = df["agc_mg_c_ha"].values
    ids  = df["parcela_id"].values
    loo  = LeaveOneOut()
    y_real, y_pred, ids_pred = [], [], []
    total = len(df)
    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
        if fold % 5 == 0:
            print(f"  LOO fold {fold+1}/{total} ...", end="\r", flush=True)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_train_sc = scaler_X.fit_transform(X_train)
        X_test_sc  = scaler_X.transform(X_test)
        y_train_sc = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()
        modelo = construir_modelo(len(features), nombre_modelo)
        modelo.fit(
            X_train_sc, y_train_sc,
            epochs=epochs, batch_size=16, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                patience=20, restore_best_weights=True, monitor="loss")]
        )
        pred_sc = modelo.predict(X_test_sc, verbose=0)
        pred    = scaler_y.inverse_transform(pred_sc).ravel()[0]
        y_real.append(float(y_test[0]))
        y_pred.append(float(pred))
        ids_pred.append(ids[test_idx[0]])
    print()
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


def graficar_resultados(df_pred, metricas, titulo, ruta_fig):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(df_pred["agc_real"], df_pred["agc_pred"],
               alpha=0.7, color="#1d9e75", edgecolors="white",
               linewidth=0.5, s=70)
    lims = [
        min(df_pred["agc_real"].min(), df_pred["agc_pred"].min()) - 2,
        max(df_pred["agc_real"].max(), df_pred["agc_pred"].max()) + 2,
    ]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="1:1")
    ax.set_xlabel("AGC real (Mg C/ha)")
    ax.set_ylabel("AGC predicho (Mg C/ha)")
    ax.set_title(
        f"R2={metricas['R2']:.3f}  RMSE={metricas['RMSE']:.2f}  r={metricas['r']:.3f}  n={metricas['n']}",
        fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax2 = axes[1]
    ax2.scatter(df_pred["agc_pred"], df_pred["error"],
                alpha=0.7, color="#185fa5", edgecolors="white",
                linewidth=0.5, s=70)
    ax2.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("AGC predicho (Mg C/ha)")
    ax2.set_ylabel("Error (predicho - real)")
    ax2.set_title(f"Residuos | bias={metricas['bias']:.2f} Mg C/ha")
    ax2.grid(True, alpha=0.3)
    fig.suptitle(titulo, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ruta_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figura: {ruta_fig.name}")


def entrenar_modelo_final(df, features, nombre_modelo, epochs=300):
    X    = df[features].values
    y    = df["agc_mg_c_ha"].values
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_sc = scaler_X.fit_transform(X)
    y_sc = scaler_y.fit_transform(y.reshape(-1,1)).ravel()
    modelo = construir_modelo(len(features), nombre_modelo)
    historia = modelo.fit(
        X_sc, y_sc,
        epochs=epochs, batch_size=8, verbose=0,
        validation_split=0.15,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            patience=30, restore_best_weights=True, monitor="val_loss")]
    )
    return modelo, scaler_X, scaler_y, historia


def guardar_modelo(modelo, scaler_X, scaler_y, features, nombre, metricas_loo):
    dir_m = DIR_MODELOS / nombre
    dir_m.mkdir(parents=True, exist_ok=True)
    modelo.save(str(dir_m / "modelo.keras"))
    with open(dir_m / "scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    with open(dir_m / "scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)
    meta = {
        "nombre"      : nombre,
        "features"    : features,
        "n_features"  : len(features),
        "metricas_LOO": metricas_loo,
    }
    with open(dir_m / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Modelo guardado en: {dir_m}")
