"""
Script 02 - Extraccion de metricas LiDAR por parcela
Entrada : .las recortados y clasificados (datos_procesados)
Salida  : metricas_lidar_todas.csv, metricas_lidar_principal.csv,
          metricas_lidar_comparativo.csv
"""

import laspy
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis

RAIZ         = Path("/mnt/e/Modelos_IA/LiDar_3D_AGC")
REGION       = RAIZ / "Nayarit"
DIR_PROC     = REGION / "datos_procesados"
DIR_METRICAS = REGION / "metricas"
DIR_RESULT   = REGION / "resultados"
CSV_LOG      = DIR_RESULT / "01_clip_log.csv"

DIR_METRICAS.mkdir(parents=True, exist_ok=True)


def calcular_rumple(x, y, z, res=1.0):
    x_min, y_min = x.min(), y.min()
    ci = np.floor((x - x_min) / res).astype(int)
    cj = np.floor((y - y_min) / res).astype(int)
    grilla = {}
    for i, j, zv in zip(ci, cj, z):
        clave = (i, j)
        if clave not in grilla or zv > grilla[clave]:
            grilla[clave] = zv
    if len(grilla) < 4:
        return 1.0
    sup_real = 0.0
    for (i, j), z_ij in grilla.items():
        for di, dj in [(1, 0), (0, 1)]:
            z_dx = grilla.get((i + di, j + dj))
            z_dy = grilla.get((i + di, j + 1 - dj))
            if z_dx is not None and z_dy is not None:
                v1 = np.array([res, 0,   z_dx - z_ij])
                v2 = np.array([0,   res, z_dy - z_ij])
                sup_real += np.linalg.norm(np.cross(v1, v2)) / 2
    sup_proy = len(grilla) * res * res
    return sup_real / sup_proy if sup_proy > 0 else 1.0


def calcular_cobertura(x, y, z, res=0.5):
    mask_veg = z > 0.5
    if mask_veg.sum() == 0:
        return 0.0
    x_min, y_min = x.min(), y.min()
    ci_v = np.floor((x[mask_veg] - x_min) / res).astype(int)
    cj_v = np.floor((y[mask_veg] - y_min) / res).astype(int)
    celdas_veg = set(zip(ci_v, cj_v))
    ci_t = np.floor((x - x_min) / res).astype(int)
    cj_t = np.floor((y - y_min) / res).astype(int)
    celdas_tot = set(zip(ci_t, cj_t))
    return len(celdas_veg) / len(celdas_tot)


def calcular_metricas(x, y, z_norm, intensidad, altura_vuelo_m):
    mask_veg = z_norm > 0.5
    z_veg    = z_norm[mask_veg]
    if len(z_veg) < 50:
        return None

    h_max    = float(np.max(z_veg))
    h_mean   = float(np.mean(z_veg))
    h_median = float(np.median(z_veg))
    h_std    = float(np.std(z_veg))
    h_cv     = h_std / h_mean if h_mean > 0 else 0.0
    h_p25    = float(np.percentile(z_veg, 25))
    h_p50    = float(np.percentile(z_veg, 50))
    h_p75    = float(np.percentile(z_veg, 75))
    h_p95    = float(np.percentile(z_veg, 95))
    h_p99    = float(np.percentile(z_veg, 99))
    h_skew   = float(skew(z_veg))
    h_kurt   = float(kurtosis(z_veg))

    n_total  = len(z_norm)
    n_veg    = len(z_veg)
    n_suelo  = n_total - n_veg
    area_parcela   = 400.0
    densidad_total = n_total / area_parcela
    densidad_veg   = n_veg   / area_parcela

    mask_bajo  = (z_norm > 0.5) & (z_norm <= 2.0)
    mask_medio = (z_norm > 2.0) & (z_norm <= 8.0)
    mask_alto  = z_norm > 8.0
    fr_suelo   = n_suelo          / n_total
    fr_bajo    = mask_bajo.sum()  / n_total
    fr_medio   = mask_medio.sum() / n_total
    fr_alto    = mask_alto.sum()  / n_total

    rumple    = calcular_rumple(x, y, z_norm, res=1.0)
    cobertura = calcular_cobertura(x, y, z_norm, res=0.5)

    estratos     = np.arange(0, h_max + 1, 1.0)
    lad_estratos = []
    for i in range(len(estratos) - 1):
        n_est = ((z_norm >= estratos[i]) & (z_norm < estratos[i+1])).sum()
        lad_estratos.append(n_est / n_total)
    lad_max  = float(max(lad_estratos))  if lad_estratos else 0.0
    lad_mean = float(np.mean(lad_estratos)) if lad_estratos else 0.0

    int_mean = float(np.mean(intensidad[mask_veg]))
    int_std  = float(np.std(intensidad[mask_veg]))

    return {
        "h_max"         : round(h_max,    3),
        "h_mean"        : round(h_mean,   3),
        "h_median"      : round(h_median, 3),
        "h_std"         : round(h_std,    3),
        "h_cv"          : round(h_cv,     4),
        "h_p25"         : round(h_p25,    3),
        "h_p50"         : round(h_p50,    3),
        "h_p75"         : round(h_p75,    3),
        "h_p95"         : round(h_p95,    3),
        "h_p99"         : round(h_p99,    3),
        "h_skew"        : round(h_skew,   4),
        "h_kurt"        : round(h_kurt,   4),
        "densidad_total": round(densidad_total, 2),
        "densidad_veg"  : round(densidad_veg,   2),
        "fr_suelo"      : round(fr_suelo,  4),
        "fr_bajo"       : round(fr_bajo,   4),
        "fr_medio"      : round(fr_medio,  4),
        "fr_alto"       : round(fr_alto,   4),
        "rumple"        : round(rumple,    4),
        "cobertura"     : round(cobertura, 4),
        "lad_max"       : round(lad_max,   4),
        "lad_mean"      : round(lad_mean,  4),
        "int_mean"      : round(int_mean,  2),
        "int_std"       : round(int_std,   2),
        "altura_vuelo_m": altura_vuelo_m,
        "n_puntos"      : n_total,
        "n_veg"         : n_veg,
    }


def main():
    log = pd.read_csv(CSV_LOG)
    log_ok = log[log["estado"] == "ok"].copy()

    # Identificar duplicados (misma parcela+epoca, diferente vuelo)
    conteo = log_ok.groupby(["parcela_id", "epoca"]).size().reset_index(name="n_vuelos")
    dups   = conteo[conteo["n_vuelos"] > 1]
    if len(dups) > 0:
        print(f"Parcelas con vuelos multiples en la misma epoca: {len(dups)}")
        print(dups.to_string())

    # Marcar principal (mayor n_puntos) y comparativo
    log_ok["es_principal"] = True
    for _, dup in dups.iterrows():
        mask    = (log_ok["parcela_id"] == dup["parcela_id"]) & \
                  (log_ok["epoca"]      == dup["epoca"])
        idx_grp = log_ok[mask].index
        idx_max = log_ok.loc[idx_grp, "n_puntos"].idxmax()
        for idx in idx_grp:
            if idx != idx_max:
                log_ok.at[idx, "es_principal"] = False

    metricas_todas = []
    total = len(log_ok)

    for n, (_, fila) in enumerate(log_ok.iterrows(), 1):
        pid            = fila["parcela_id"]
        epoca          = fila["epoca"]
        archivo_salida = fila["archivo_salida"]
        es_principal   = fila["es_principal"]
        altura_vuelo   = fila.get("altura_vuelo_m", None)

        print(f"[{n}/{total}] {pid} | {epoca} | {archivo_salida} ... ",
              end="", flush=True)

        ruta = DIR_PROC / epoca / archivo_salida
        if not ruta.exists():
            print("archivo no encontrado, OMITIDA")
            continue

        try:
            las = laspy.read(str(ruta))
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        x     = np.array(las.x)
        y     = np.array(las.y)
        z     = np.array(las.z)
        inten = np.array(las.intensity)
        cls   = np.array(las.classification)

        # Normalizar Z respecto al suelo clasificado
        z_suelo = z[cls == 2]
        if len(z_suelo) < 10:
            print("sin suelo suficiente, OMITIDA")
            continue
        z_ref  = np.percentile(z_suelo, 10)
        z_norm = np.clip(z - z_ref, 0, None)

        metr = calcular_metricas(x, y, z_norm, inten, altura_vuelo)
        if metr is None:
            print("sin vegetacion suficiente, OMITIDA")
            continue

        estado = "PRINCIPAL" if es_principal else "comparativo"
        print(f"OK ({estado}) | h_p95={metr['h_p95']}m | "
              f"cob={metr['cobertura']:.2f} | rumple={metr['rumple']:.3f}")

        metricas_todas.append({
            "parcela_id"  : pid,
            "epoca"       : epoca,
            "es_principal": es_principal,
            "archivo_salida": archivo_salida,
            **metr,
        })

    df_todas = pd.DataFrame(metricas_todas)
    df_princ = df_todas[df_todas["es_principal"]].copy()
    df_comp  = df_todas[~df_todas["es_principal"]].copy()

    ruta_todas = DIR_METRICAS / "metricas_lidar_todas.csv"
    ruta_princ = DIR_METRICAS / "metricas_lidar_principal.csv"
    ruta_comp  = DIR_METRICAS / "metricas_lidar_comparativo.csv"

    df_todas.to_csv(ruta_todas, index=False)
    df_princ.to_csv(ruta_princ, index=False)
    if len(df_comp) > 0:
        df_comp.to_csv(ruta_comp, index=False)

    print("\n" + "=" * 55)
    print(f"Metricas extraidas    : {len(df_todas)}")
    print(f"Registros principales : {len(df_princ)}")
    print(f"Registros comparativos: {len(df_comp)}")
    print(f"Archivos en           : {DIR_METRICAS}")
    print("=" * 55)
    print("\nResumen metricas principales:")
    cols = ["h_p95", "h_mean", "cobertura", "rumple", "densidad_total", "fr_alto"]
    print(df_princ[cols].describe().round(3).to_string())


if __name__ == "__main__":
    main()
