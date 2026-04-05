"""
Script 01 - Clip por parcela, correccion RTK y clasificacion suelo/vegetacion
Entrada : .las crudos (vuelo completo)
Salida  : .las recortados por parcela con clases 2=suelo 4=vegetacion
"""

import laspy
import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from shapely import contains_xy

RAIZ         = Path("/mnt/e/Modelos_IA/LiDar_3D_AGC")
REGION       = RAIZ / "Nayarit"
DIR_CRUDOS   = REGION / "datos_crudos"
DIR_SALIDA   = REGION / "datos_procesados"
DIR_PARCELAS = REGION / "parcelas"
SHP_PARCELAS = DIR_PARCELAS / "parcelas_manglaria.shp"
CSV_CORRESP  = DIR_PARCELAS / "correspondencia.csv"

UMBRAL_SUELO_M  = 0.5
RES_GRILLA_M    = 0.5
MIN_PUNTOS_PARC = 500


def corregir_offset(z):
    offset = np.percentile(z, 1)
    return z - offset, offset


def clip_poligono(las, poligono):
    x, y = np.array(las.x), np.array(las.y)
    minx, miny, maxx, maxy = poligono.bounds
    mask_bbox = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
    idx_cand  = np.where(mask_bbox)[0]
    if len(idx_cand) == 0:
        return np.zeros(len(x), dtype=bool)
    mask_poly  = contains_xy(poligono, x[idx_cand], y[idx_cand])
    mask_final = np.zeros(len(x), dtype=bool)
    mask_final[idx_cand[mask_poly]] = True
    return mask_final


def clasificar_suelo_vegetacion(x, y, z, res=0.5, umbral=0.5):
    clases   = np.full(len(z), 4, dtype=np.uint8)
    x_min, y_min = x.min(), y.min()
    celdas_x = np.floor((x - x_min) / res).astype(int)
    celdas_y = np.floor((y - y_min) / res).astype(int)
    clave    = celdas_x * 100000 + celdas_y
    df       = pd.DataFrame({"clave": clave, "z": z, "idx": np.arange(len(z))})
    idx_min  = df.groupby("clave")["z"].idxmin().values
    x_s, y_s, z_s = x[idx_min], y[idx_min], z[idx_min]
    z_mdt = griddata(
        points=np.column_stack([x_s, y_s]),
        values=z_s,
        xi=np.column_stack([x, y]),
        method="linear"
    )
    nan_mask = np.isnan(z_mdt)
    if nan_mask.any():
        arbol = cKDTree(np.column_stack([x_s, y_s]))
        _, idx_nn = arbol.query(np.column_stack([x[nan_mask], y[nan_mask]]))
        z_mdt[nan_mask] = z_s[idx_nn]
    altura_sobre_suelo = z - z_mdt
    clases[altura_sobre_suelo <= umbral] = 2
    return clases


def guardar_las_parcela(las_orig, mascara, clases, z_corr, ruta_sal):
    ruta_sal.parent.mkdir(parents=True, exist_ok=True)
    nuevo_las = laspy.LasData(header=laspy.LasHeader(
        point_format=las_orig.header.point_format,
        version=las_orig.header.version
    ))
    nuevo_las.x              = las_orig.x[mascara]
    nuevo_las.y              = las_orig.y[mascara]
    nuevo_las.z              = z_corr[mascara]
    nuevo_las.intensity      = las_orig.intensity[mascara]
    nuevo_las.classification = clases
    nuevo_las.write(str(ruta_sal))


def main():
    gdf     = gpd.read_file(SHP_PARCELAS)
    corresp = pd.read_csv(CSV_CORRESP)
    geom_idx = gdf.set_index("parcela_id")["geometry"].to_dict()
    altura_idx = corresp.set_index(["parcela_id", "epoca"])["altura_vuelo"].to_dict()

    log    = []
    grupos = corresp.groupby(["epoca", "archivo_las"])
    total  = len(grupos)

    for n, ((epoca, archivo), filas) in enumerate(grupos, 1):
        ruta_las = DIR_CRUDOS / epoca / archivo
        print(f"\n[{n}/{total}] {epoca} | {archivo}")
        try:
            las = laspy.read(str(ruta_las))
        except Exception as e:
            print(f"  ERROR leyendo: {e}")
            continue

        x = np.array(las.x)
        y = np.array(las.y)
        z = np.array(las.z)
        z_corr, offset = corregir_offset(z)
        print(f"  Offset RTK corregido: {offset:.3f} m")

        for _, fila in filas.iterrows():
            pid = fila["parcela_id"]
            print(f"  -> Parcela {pid} ... ", end="", flush=True)
            poligono = geom_idx.get(pid)
            if poligono is None:
                print("sin geometria, OMITIDA")
                continue

            mascara = clip_poligono(las, poligono)
            n_pts   = mascara.sum()

            if n_pts < MIN_PUNTOS_PARC:
                print(f"muy pocos puntos ({n_pts}), OMITIDA")
                log.append({"parcela_id": pid, "epoca": epoca,
                             "estado": "pocos_puntos", "n_puntos": int(n_pts),
                             "archivo_las": archivo, "archivo_salida": None,
                             "altura_vuelo_m": altura_idx.get((pid, epoca), None)})
                continue

            x_p = x[mascara]
            y_p = y[mascara]
            z_p = z_corr[mascara]
            clases = clasificar_suelo_vegetacion(x_p, y_p, z_p)
            n_suelo = int((clases == 2).sum())
            n_veg   = int((clases == 4).sum())

            ruta_sal  = DIR_SALIDA / epoca / f"{pid}.las"
            contador  = 2
            while ruta_sal.exists():
                ruta_sal = DIR_SALIDA / epoca / f"{pid}_v{contador}.las"
                contador += 1

            guardar_las_parcela(las, mascara, clases, z_corr, ruta_sal)

            print(f"OK | {n_pts:,} pts | suelo: {n_suelo:,} | veg: {n_veg:,} | -> {ruta_sal.name}")
            log.append({
                "parcela_id"    : pid,
                "epoca"         : epoca,
                "estado"        : "ok",
                "n_puntos"      : n_pts,
                "n_suelo"       : n_suelo,
                "n_veg"         : n_veg,
                "offset_rtk"    : round(float(offset), 4),
                "altura_vuelo_m": altura_idx.get((pid, epoca), None),
                "archivo_las"   : archivo,
                "archivo_salida": ruta_sal.name,
            })

    df_log   = pd.DataFrame(log)
    ruta_log = REGION / "resultados" / "01_clip_log.csv"
    ruta_log.parent.mkdir(parents=True, exist_ok=True)
    df_log.to_csv(ruta_log, index=False)

    ok    = (df_log["estado"] == "ok").sum()
    fallo = (df_log["estado"] != "ok").sum()
    print("\n" + "=" * 55)
    print(f"Procesadas : {len(df_log)}")
    print(f"Exitosas   : {ok}")
    print(f"Con fallas : {fallo}")
    print(f"Log        : {ruta_log}")
    print("=" * 55)


if __name__ == "__main__":
    main()
