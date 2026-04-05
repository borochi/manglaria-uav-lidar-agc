"""
Modelo B - Todas las epocas sin disturbio
"""
import sys
sys.path.insert(0, "/mnt/e/Modelos_IA/LiDar_3D_AGC/scripts")
from modelo_base import (
    cargar_datos, validacion_cruzada_loo,
    graficar_resultados, entrenar_modelo_final,
    guardar_modelo, DIR_RESULT
)

NOMBRE = "modelo_B_todas_epocas"

def main():
    print("=" * 55)
    print("MODELO B - Todas las epocas sin disturbio")
    print("=" * 55)
    df, features = cargar_datos(
        epoca_filtro=None,
        incluir_disturbio=False
    )
    print(f"Features ({len(features)}): {features}")
    print("Distribucion por epoca:")
    print(df["epoca"].value_counts().to_string())
    print("Iniciando validacion LOO...")
    metricas, df_pred = validacion_cruzada_loo(df, features, NOMBRE)
    print("Resultados LOO:")
    for k, v in metricas.items():
        print(f"  {k}: {v}")
    graficar_resultados(
        df_pred, metricas,
        "Modelo B - todas las epocas sin disturbio (LOO)",
        DIR_RESULT / "04b_resultados_modeloB.png"
    )
    df_pred.to_csv(DIR_RESULT / "04b_predicciones_modeloB.csv", index=False)
    print("Entrenando modelo final...")
    modelo, sc_X, sc_y, historia = entrenar_modelo_final(df, features, NOMBRE)
    guardar_modelo(modelo, sc_X, sc_y, features, NOMBRE, metricas)
    print("=" * 55)
    print("MODELO B COMPLETO")
    print(f"  R2   : {metricas['R2']}")
    print(f"  RMSE : {metricas['RMSE']} Mg C/ha")
    print(f"  r    : {metricas['r']}")
    print(f"  n    : {metricas['n']}")
    print("=" * 55)

if __name__ == "__main__":
    main()
