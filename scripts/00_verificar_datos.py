"""
Script 00 - Verificacion de datos de entrada - Region Nayarit
Corre esto PRIMERO antes de procesar nada.
"""

import laspy
import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path

# ── Rutas del proyecto ────────────────────────────────────────────────────────
RAIZ          = Path("/mnt/e/Modelos_IA/LiDar_3D_AGC")
REGION        = RAIZ / "Nayarit"
DIR_CRUDOS    = REGION / "datos_crudos"
DIR_PARCELAS  = REGION / "parcelas"
SHP_PARCELAS  = DIR_PARCELAS / "parcelas_manglaria.shp"
CSV_CORRESP   = DIR_PARCELAS / "correspondencia.csv"

# ── 1. Verificar shapefile ────────────────────────────────────────────────────
print("=" * 55)
print("SHAPEFILE DE PARCELAS")
print("=" * 55)

gdf = gpd.read_file(SHP_PARCELAS)
print(f"Parcelas encontradas : {len(gdf)}")
print(f"CRS                  : {gdf.crs}")
print(f"Columnas             : {list(gdf.columns)}")
print(f"\nPrimeras filas:")
print(gdf[["parcela_id", "region"]].to_string())

if gdf.crs is None:
    print("\n ALERTA: el shapefile no tiene CRS asignado. Asignalo en QGIS.")
elif "32613" not in str(gdf.crs) and "utm" not in gdf.crs.name.lower():
    print(f"\n ALERTA: CRS inesperado ({gdf.crs}). Nayarit deberia ser EPSG:32613")
else:
    print("\n CRS UTM zona 13N (EPSG:32613) detectado correctamente")

# ── 2. Verificar CSV de correspondencia ───────────────────────────────────────
print("\n" + "=" * 55)
print("ARCHIVO DE CORRESPONDENCIA")
print("=" * 55)

corresp = pd.read_csv(CSV_CORRESP)
print(f"Filas totales        : {len(corresp)}")
print(f"Parcelas unicas      : {corresp['parcela_id'].nunique()}")
print(f"Epocas unicas        : {corresp['epoca'].unique()}")
print(f"\nConteo por epoca:")
print(corresp.groupby("epoca")["parcela_id"].count())

ids_shp = set(gdf["parcela_id"])
ids_csv = set(corresp["parcela_id"])
solo_en_csv = ids_csv - ids_shp
solo_en_shp = ids_shp - ids_csv

if solo_en_csv:
    print(f"\n En CSV pero NO en shapefile: {solo_en_csv}")
if solo_en_shp:
    print(f"\n En shapefile pero NO en CSV: {solo_en_shp}")
if not solo_en_csv and not solo_en_shp:
    print("\n Los parcela_id coinciden entre shapefile y CSV")

# ── 3. Verificar archivos .las ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("ARCHIVOS .LAS")
print("=" * 55)

encontrados, faltantes = [], []

for _, fila in corresp.iterrows():
    ruta = DIR_CRUDOS / fila["epoca"] / fila["archivo_las"]
    if ruta.exists():
        encontrados.append(str(ruta))
    else:
        faltantes.append(
            f"  {fila['parcela_id']} | {fila['epoca']} -> {ruta}"
        )

print(f"Archivos encontrados : {len(encontrados)}")
print(f"Archivos FALTANTES   : {len(faltantes)}")
if faltantes:
    print("\nArchivos que no se encuentran:")
    for f in faltantes:
        print(f)

# ── 4. Inspeccionar un .las de muestra ───────────────────────────────────────
if encontrados:
    print("\n" + "=" * 55)
    print("INSPECCION DE MUESTRA (primer .las)")
    print("=" * 55)

    muestra = encontrados[0]
    print(f"Archivo : {Path(muestra).name}")
    las = laspy.read(muestra)

    z       = np.array(las.z)
    clases  = np.unique(las.classification)

    print(f"Puntos totales  : {len(las.points):,}")
    print(f"Z minimo        : {z.min():.3f} m")
    print(f"Z maximo        : {z.max():.3f} m")
    print(f"Z rango         : {z.max() - z.min():.3f} m")
    print(f"Clases presentes: {clases}")

    area_bb = (
        (las.x.max() - las.x.min()) *
        (las.y.max() - las.y.min())
    )
    if area_bb > 0:
        densidad = len(las.points) / area_bb
        print(f"Densidad aprox. : {densidad:.1f} pts/m2")

    if z.min() < -5:
        print(f"\n Offset RTK detectado: Z min = {z.min():.2f} m")
        print(f"  Se corregira automaticamente en el procesado.")
    else:
        print(f"\n Valores Z parecen razonables")

print("\n" + "=" * 55)
print("Verificacion completa.")
print("=" * 55)
