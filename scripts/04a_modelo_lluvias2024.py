"""
Modelo A - Solo lluvias_2024
"""
import sys
sys.path.insert(0, "/mnt/e/Modelos_IA/LiDar_3D_AGC/scripts")
from modelo_base import (
    cargar_datos, validacion_cruzada_loo,
    graficar_resultados, entrenar_modelo_final,
    guardar_modelo, DIR_RESULT
)

NOMBRE = "modelo_A_lluvias2024"

def main():
    print("=" * 55)
    print("MODELO A - Solo lluvias_2024")
    print("=" * 55)
    df, features = cargar_datos(
        epoca_filtro="lluvias_2024",
        incluir_disturbio=False
    )
    if len(df) < 5:
        print(f"ERROR: solo {len(df)} registros, insuficiente")
        return
    print(f"Features ({len(features)}): {features}")
    print("Iniciando validacion LOO...")
    metricas, df_pred = validacion_cruzada_loo(df, features, NOMBRE)
    print("Resultados LOO:")
    for k, v in metricas.items():
        print(f"  {k}: {v}")
    graficar_resultados(
        df_pred, metricas,
        "Modelo A - lluvias_2024 (LOO)",
        DIR_RESULT / "04a_resultados_modeloA.png"
    )
    df_pred.to_csv(DIR_RESULT / "04a_predicciones_modeloA.csv", index=False)
    print("Entrenando modelo final...")
    modelo, sc_X, sc_y, historia = entrenar_modelo_final(df, features, NOMBRE)
    guardar_modelo(modelo, sc_X, sc_y, features, NOMBRE, metricas)
    print("=" * 55)
    print("MODELO A COMPLETO")
    print(f"  R2   : {metricas['R2']}")
    print(f"  RMSE : {metricas['RMSE']} Mg C/ha")
    print(f"  r    : {metricas['r']}")
    print(f"  n    : {metricas['n']}")
    print("=" * 55)

if __name__ == "__main__":
    main()
