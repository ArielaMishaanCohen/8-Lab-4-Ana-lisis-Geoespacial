# ===========================
#  Cálculo de índices CHL/agua con lectura robusta de GeoTIFF
#  Guarda matrices por fecha y promedios por fecha (sin interpolar NaN)
# ===========================

import numpy as np
import pandas as pd
import rasterio

# ---------- Parámetros de índices ----------
MNDWI_threshold = 0.42
NDWI_threshold  = 0.4
filter_UABS     = True     # filtrar urbano/suelo desnudo en WBI
EPS = 1e-9                 # evita divisiones exactas por cero


# ---------- Utilidades numéricas ----------
def nd_simple(num, den):
    """Divide num/den devolviendo NaN donde |den|≈0."""
    num = num.astype(np.float32, copy=False)
    den = den.astype(np.float32, copy=False)
    out = np.full_like(num, np.nan, dtype=np.float32)
    np.divide(num, den, out=out, where=np.abs(den) > EPS)
    return out

def normalize(b):
    """
    Normaliza banda a [0,1] con percentiles 2-98 ignorando <=0 y no finitos.
    Útil para visualización e índices robustos cuando las bandas no están en reflectancia física.
    Si trabajas en reflectancia (escala 0-1), puedes omitir esta función.
    """
    b = b.astype(np.float32, copy=False)
    valid = b[np.isfinite(b) & (b > 0)]
    if valid.size < 10:
        return np.zeros_like(b, dtype=np.float32)
    b_min, b_max = np.percentile(valid, (2, 98))
    if (b_max - b_min) < EPS:
        return np.zeros_like(b, dtype=np.float32)
    x = (b - b_min) / (b_max - b_min)
    return np.clip(x, 0, 1).astype(np.float32)


# ---------- Índices ----------
def calcular_ndvi(red, nir):
    return nd_simple(nir - red, nir + red)

def calcular_ndwi(nir, green):
    return nd_simple(green - nir, green + nir)

def calcular_savi(red, nir, L=0.428):
    return (1 + L) * nd_simple(nir - red, nir + red + L)

def FAI(a, b, c):
    # a ~ 665 nm (red), b ~ 783 nm, c ~ 865 nm (B8A). Ajusta si usas otras bandas.
    baseline = a + (c - a) * ((783 - 665) / (865 - 665))
    return b - baseline

def NDCI(a, b):
    # típicamente (R708 - R665)/(R708 + R665); aquí: a=red, b=B05 (~705-740 nm)
    return nd_simple(b - a, b + a)

def CHL(ndci):
    # Tu calibración original
    ndci = ndci.astype(np.float32, copy=False)
    return (826.57 * ndci**3 - 176.43 * ndci**2 + 19 * ndci + 4.071).astype(np.float32)


# ---------- Detección de agua (WBI, máscara 0/1) ----------
def wbi(r, g, b, nir, swir1, swir2):
    r = r.astype(np.float32); g = g.astype(np.float32); b = b.astype(np.float32)
    nir = nir.astype(np.float32); swir1 = swir1.astype(np.float32); swir2 = swir2.astype(np.float32)

    ndvi = nd_simple(nir - r, nir + r)
    mndwi = nd_simple(g - swir1, g + swir1)
    ndwi = nd_simple(g - nir, g + nir)
    ndwi_leaves = nd_simple(nir - swir1, nir + swir1)

    aweish  = b + 2.5*g - 1.5*(nir + swir1) - 0.25*swir2
    aweinsh = 4*(g - swir1) - (0.25*nir + 2.75*swir1)

    dbsi = nd_simple(swir1 - g, swir1 + g) - ndvi
    wii  = nd_simple(nir**2, r)             # puede salir NaN donde r≈0
    wri  = nd_simple(g + r, nir + swir1)
    puwi = 5.83*g - 6.57*r - 30.32*nir + 2.25

    den_uwi = np.abs(g - 1.1*r - 5.2*nir)   # puede ser 0
    uwi = nd_simple(g - 1.1*r - 5.2*nir + 0.4, den_uwi)

    usi = 0.25*nd_simple(g, r) - 0.57*nd_simple(nir, g) - 0.83*nd_simple(b, g) + 1

    ws = np.zeros_like(r, dtype=np.uint8)
    water_mask = (
        (mndwi > MNDWI_threshold) |
        (ndwi > NDWI_threshold)   |
        (aweinsh > 0.1879)        |
        (aweish  > 0.1112)        |
        (ndvi    < -0.2)          |
        (ndwi_leaves > 1)
    )
    ws[water_mask] = 1

    if filter_UABS:
        urban_mask = ((aweinsh <= -0.03) | (dbsi > 0)) & (ws == 1)
        ws[urban_mask] = 0

    return ws  # 0/1 con NaN donde las entradas tenían NaN


# ---------- Lectura robusta de GeoTIFF por bloques ----------
def read_tif_robusto(path):
    """
    Lee un GeoTIFF por bloques; rellena con NaN los tiles que fallen.
    Devuelve:
      arr: (count, H, W) float32 con NaN donde hubo error
      nodata: valor nodata del raster (puede ser None)
      bad_tiles: dict con lista de tiles dañados por banda
    """
    bad_tiles = {}
    with rasterio.open(path) as src:
        count, height, width = src.count, src.height, src.width
        arr = np.full((count, height, width), np.nan, dtype=np.float32)
        nodata = src.nodata

        for bidx in range(1, count + 1):
            bad_tiles[bidx] = []
            for ji, window in src.block_windows(bidx):
                try:
                    data = src.read(bidx, window=window, masked=False)
                    r0, r1 = window.row_off, window.row_off + window.height
                    c0, c1 = window.col_off, window.col_off + window.width
                    arr[bidx - 1, r0:r1, c0:c1] = data.astype(np.float32, copy=False)
                except Exception:
                    bad_tiles[bidx].append((ji, window))  # deja NaN en ese bloque y sigue
    return arr, nodata, bad_tiles


# ---------- Pipeline principal ----------
def procesar_lago(ruta_carpeta, fechas):
    """
    ruta_carpeta: carpeta con GeoTIFFs nombrados YYYY-MM-DD.tiff
    fechas: lista/iterable de strings 'YYYY-MM-DD' o timestamps

    Devuelve:
      df_mats: DataFrame indexado por fecha, columnas-objeto con matrices (RGB, NDVI, NDWI, SAVI, WBI, FAI, NDCI, CHL) y NoData
      df_stats: DataFrame con promedios por fecha (nanmean), y fracción de agua (WBI_frac)
    """
    # Índice de fechas (no asume continuidad diaria)
    fechas_idx = pd.to_datetime(sorted(set(pd.to_datetime(fechas))))
    # DF matrices (columnas object para guardar arrays)
    df_mats = pd.DataFrame(index=fechas_idx)
    for col in ["RGB","NDVI","NDWI","SAVI","WBI","FAI","NDCI","CHL"]:
        df_mats[col] = pd.Series([None]*len(df_mats), index=df_mats.index, dtype="object")
    df_mats["NoData"] = np.nan

    # DF estadísticas (promedios)
    df_stats = pd.DataFrame(
        index=fechas_idx,
        columns=["NDVI_mean","NDWI_mean","SAVI_mean","WBI_frac","FAI_mean","NDCI_mean","CHL_mean"],
        dtype=np.float32
    )

    for fecha in fechas_idx:
        ruta_tif = f"{ruta_carpeta}/{fecha.strftime('%Y-%m-%d')}.tiff"
        try:
            bandas, nodata, tiles_malos = read_tif_robusto(ruta_tif)
            # Log opcional
            total_malos = sum(len(v) for v in tiles_malos.values())
            if total_malos:
                print(f"{fecha.date()}: {total_malos} tiles dañados -> NaN")
        except Exception as e:
            print(f"[ERROR] {fecha.date()} al leer {ruta_tif}: {e}")
            # deja NaN en todo para esa fecha y continúa
            df_mats.at[fecha, "NoData"] = np.nan
            continue

        # --- Selección de bandas (ajusta a tu orden real) ---
        # Asumiendo: [0]=B02 blue, [1]=B03 green, [2]=B04 red, [3]=B05, [4]=B07, [5]=B08,
        #            [6]=B8A, [7]=B11, [8]=B12
        blue     = normalize(bandas[0])
        green    = normalize(bandas[1])
        red      = normalize(bandas[2])
        vnir_b05 = normalize(bandas[3])   # ~705-740nm
        vnir_b07 = normalize(bandas[4])
        nir      = normalize(bandas[5])   # B08
        vnir_b8a = normalize(bandas[6])   # ~865nm
        swir_b11 = normalize(bandas[7])
        swir_b12 = normalize(bandas[8])

        # --- Índices ---
        rgb   = np.dstack([red, green, blue]).astype(np.float32)
        ndvi  = calcular_ndvi(red, nir)
        ndwi  = calcular_ndwi(nir, green)
        savi  = calcular_savi(red, nir)
        wbi_m = wbi(red, green, blue, nir, swir_b11, swir_b12)   # 0/1
        fai   = FAI(red, vnir_b07, vnir_b8a)
        ndci  = NDCI(red, vnir_b05)
        chl   = CHL(ndci)

        # --- Guardar matrices por fecha ---
        df_mats.at[fecha, "NoData"] = nodata
        df_mats.at[fecha, "RGB"]  = rgb
        df_mats.at[fecha, "NDVI"] = ndvi
        df_mats.at[fecha, "NDWI"] = ndwi
        df_mats.at[fecha, "SAVI"] = savi
        df_mats.at[fecha, "WBI"]  = wbi_m
        df_mats.at[fecha, "FAI"]  = fai
        df_mats.at[fecha, "NDCI"] = ndci
        df_mats.at[fecha, "CHL"]  = chl

        # --- Estadísticas (sin rellenar NaN) ---
        df_stats.at[fecha, "NDVI_mean"] = np.nanmean(ndvi)
        df_stats.at[fecha, "NDWI_mean"] = np.nanmean(ndwi)
        df_stats.at[fecha, "SAVI_mean"] = np.nanmean(savi)
        # fracción de píxeles clasificados como agua (WBI=1) entre los válidos
        valid_wbi = wbi_m[np.isfinite(wbi_m)]
        df_stats.at[fecha, "WBI_frac"]  = np.mean(valid_wbi == 1) if valid_wbi.size else np.nan
        df_stats.at[fecha, "FAI_mean"]  = np.nanmean(fai)
        df_stats.at[fecha, "NDCI_mean"] = np.nanmean(ndci)
        df_stats.at[fecha, "CHL_mean"]  = np.nanmean(chl)

    return df_mats, df_stats



# ---------- Ejemplo de uso ----------
# Asume que 'fechas' es una lista de strings 'YYYY-MM-DD' (o timestamps) disponibles en tu entorno.
#ruta 'imgs1' contiene archivos nombrados exactamente como esas fechas con extensión .tiff

# fechas = ["2025-04-18", "2025-04-23", ...]
df_mats_atitlan, df_stats_atitlan = procesar_lago("imgs1", fechas)

# Si quieres procesar otra carpeta (p.ej. Amatitlán):
# df_mats_amatitlan, df_stats_amatitlan = procesar_lago("imgs2", fechas)









## Función buena para procesar lago

def procesar_lago(ruta_carpeta, fechas):
    """
    Procesa los TIFF de un lago.
    Devuelve:
      df_mats: matrices por fecha (con todas las fechas entre min y max)
      df_stats: promedios por fecha (NaN si no hay imagen)
    """
    # --- Rango completo de fechas ---
    start_date = pd.to_datetime(min(fechas))
    end_date   = pd.to_datetime(max(fechas))
    fechas_idx = pd.date_range(start=start_date, end=end_date, freq="D")  # TODAS las fechas

    # --- DataFrame de matrices ---
    df_mats = pd.DataFrame(index=fechas_idx)
    for col in ["RGB","NDVI","NDWI","SAVI","WBI","FAI","NDCI","CHL"]:
        df_mats[col] = pd.Series([None]*len(df_mats), index=df_mats.index, dtype="object")
    df_mats["NoData"] = np.nan

    # --- DataFrame de estadísticas ---
    df_stats = pd.DataFrame(
        index=fechas_idx,
        columns=["NDVI_mean","NDWI_mean","SAVI_mean","WBI_frac","FAI_mean","NDCI_mean","CHL_mean"],
        dtype=np.float32
    )

    # --- Loop solo sobre fechas que tienen archivo ---
    fechas_disponibles = pd.to_datetime(sorted(set(pd.to_datetime(fechas))))
    for fecha in fechas_disponibles:
        ruta_tif = f"{ruta_carpeta}/{fecha.strftime('%Y-%m-%d')}.tiff"
        try:
            bandas, nodata, tiles_malos = read_tif_robusto(ruta_tif)
            total_malos = sum(len(v) for v in tiles_malos.values())
            if total_malos:
                print(f"{fecha.date()}: {total_malos} tiles dañados -> NaN")
        except Exception as e:
            print(f"[ERROR] {fecha.date()} al leer {ruta_tif}: {e}")
            df_mats.at[fecha, "NoData"] = np.nan
            continue

        # --- Normalización de bandas ---
        blue     = normalize(bandas[0])
        green    = normalize(bandas[1])
        red      = normalize(bandas[2])
        vnir_b05 = normalize(bandas[3])
        vnir_b07 = normalize(bandas[4])
        nir      = normalize(bandas[5])
        vnir_b8a = normalize(bandas[6])
        swir_b11 = normalize(bandas[7])
        swir_b12 = normalize(bandas[8])

        # --- Cálculo de índices ---
        rgb   = np.dstack([red, green, blue]).astype(np.float32)
        ndvi  = calcular_ndvi(red, nir)
        ndwi  = calcular_ndwi(nir, green)
        savi  = calcular_savi(red, nir)
        wbi_m = wbi(red, green, blue, nir, swir_b11, swir_b12)
        fai   = FAI(red, vnir_b07, vnir_b8a)
        ndci  = NDCI(red, vnir_b05)
        chl   = CHL(ndci)

        # --- Guardar matrices ---
        df_mats.at[fecha, "NoData"] = nodata
        df_mats.at[fecha, "RGB"]  = rgb
        df_mats.at[fecha, "NDVI"] = ndvi
        df_mats.at[fecha, "NDWI"] = ndwi
        df_mats.at[fecha, "SAVI"] = savi
        df_mats.at[fecha, "WBI"]  = wbi_m
        df_mats.at[fecha, "FAI"]  = fai
        df_mats.at[fecha, "NDCI"] = ndci
        df_mats.at[fecha, "CHL"]  = chl

        # --- Guardar estadísticas ---
        df_stats.at[fecha, "NDVI_mean"] = np.nanmean(ndvi)
        df_stats.at[fecha, "NDWI_mean"] = np.nanmean(ndwi)
        df_stats.at[fecha, "SAVI_mean"] = np.nanmean(savi)
        valid_wbi = wbi_m[np.isfinite(wbi_m)]
        df_stats.at[fecha, "WBI_frac"]  = np.mean(valid_wbi == 1) if valid_wbi.size else np.nan
        df_stats.at[fecha, "FAI_mean"]  = np.nanmean(fai)
        df_stats.at[fecha, "NDCI_mean"] = np.nanmean(ndci)
        df_stats.at[fecha, "CHL_mean"]  = np.nanmean(chl)

    return df_mats, df_stats



## código para verificar
import os
import numpy as np
import pandas as pd

# 1) Normaliza tu lista de fechas esperadas (las 29 que tienes)
expected = pd.to_datetime(sorted(set(pd.to_datetime(fechas))))
print("Fechas esperadas:", len(expected))  # debería ser 29

# 2) ¿Qué días tienen estadísticas (al menos una métrica no-NaN)?
present_stats = atitlan_stats.index[~atitlan_stats.isna().all(axis=1)]
print("Fechas con estadísticas:", len(present_stats))

# 3) Diferencias: faltantes y sobrantes
missing_in_stats = expected.difference(present_stats)
extra_in_stats = present_stats.difference(expected)

print("Faltan en stats:", len(missing_in_stats), list(missing_in_stats.strftime("%Y-%m-%d"))[:10])
print("Sobraron en stats:", len(extra_in_stats), list(extra_in_stats.strftime("%Y-%m-%d"))[:10])

# 4) Marcador útil en el DF de estadísticas
atitlan_stats["processed"] = ~atitlan_stats.isna().all(axis=1)

# 5) Cheque alterno con df_mats: ¿hay matrices guardadas ese día?
def has_any_matrix(f):
    row = atitlan_mats.loc[f]
    return any(isinstance(row[c], np.ndarray) for c in ["NDVI","NDWI","SAVI","WBI","FAI","NDCI","CHL","RGB"])

present_mats = pd.DatetimeIndex([f for f in expected if has_any_matrix(f)])
print("Fechas con matrices guardadas:", len(present_mats))

missing_in_mats = expected.difference(present_mats)
print("Faltan en mats:", len(missing_in_mats), list(missing_in_mats.strftime("%Y-%m-%d"))[:10])

# 6) (Opcional) Verifica que los archivos existen en disco
missing_files = [d.strftime("%Y-%m-%d") for d in expected
                 if not os.path.exists(f'imgs1/{d.strftime("%Y-%m-%d")}.tiff')]
print("Archivos faltantes en carpeta:", len(missing_files), missing_files[:10])

# 7) (Opcional) Chequea si alguna matriz quedó totalmente NaN (archivo existía pero datos no válidos)
def all_nan_matrix(f, col="NDVI"):
    arr = atitlan_mats.at[f, col]
    return (arr is None) or (not np.isfinite(arr).any())

suspects_all_nan = [f.strftime("%Y-%m-%d") for f in expected if all_nan_matrix(f, "NDVI")]
print("Fechas con NDVI vacío o todo NaN:", len(suspects_all_nan), suspects_all_nan[:10])





## Interpolación
import numpy as np
import pandas as pd

# ----- helpers de interpolación en tiempo -----

def _interp_stack_time(stack_3d):
    """
    Interpola a lo largo del eje temporal T para cada píxel.
    stack_3d: array (T, H, W) con NaNs.
    Devuelve: array (T, H, W) sin NaNs internos (bordes extrapolados por copia del extremo).
    """
    T, H, W = stack_3d.shape
    t = np.arange(T, dtype=np.float32)
    X = stack_3d.reshape(T, -1)                      # (T, H*W)
    Y = np.empty_like(X)

    for j in range(X.shape[1]):
        y = X[:, j]
        m = np.isfinite(y)
        if m.sum() == 0:
            Y[:, j] = y  # todo NaN, se queda NaN
        elif m.sum() == 1:
            # un único valor -> extiende constante
            val = y[m][0]
            Y[:, j] = np.full(T, val, dtype=np.float32)
        else:
            Y[:, j] = np.interp(t, t[m], y[m])      # interpola y copia bordes
    return Y.reshape(T, H, W)

def _interp_stack_time_binary(stack_3d, thresh=0.5):
    """
    Interpola binarios (0/1) como flotantes y aplica umbral al final.
    """
    Y = _interp_stack_time(stack_3d.astype(np.float32))
    return (Y >= thresh).astype(np.uint8)

def _build_stack_from_df(df_mats, col):
    """
    Convierte una columna-objeto (arrays HxW por fecha) a un stack (T,H,W)
    usando el primer array válido para tomar la forma.
    """
    # encuentra primera matriz válida para la forma
    first = next((a for a in df_mats[col].values if isinstance(a, np.ndarray)), None)
    if first is None:
        raise ValueError(f"No hay ninguna matriz válida en la columna {col}.")
    if first.ndim != 2:
        raise ValueError(f"Se esperaba 2D en {col}, obtuve shape {first.shape}.")
    H, W = first.shape
    T = len(df_mats.index)
    stack = np.full((T, H, W), np.nan, dtype=np.float32)
    for i, f in enumerate(df_mats.index):
        a = df_mats.at[f, col]
        if isinstance(a, np.ndarray) and a.shape == (H, W):
            stack[i] = a.astype(np.float32, copy=False)
    return stack

def _write_stack_to_df(df_mats, col, stack):
    """Escribe de vuelta el stack (T,H,W) en la columna-objeto por fecha."""
    for i, f in enumerate(df_mats.index):
        df_mats.at[f, col] = stack[i]


# ===== 1) INTERPOLACIÓN EN MATRICES =====
# Crea una copia para no perder el original
atitlan_mats_interp = atitlan_mats.copy()

# Columnas escalares por píxel (interpolación lineal en el tiempo)
scalar_cols = ["NDVI","NDWI","SAVI","FAI","NDCI","CHL"]
for col in scalar_cols:
    try:
        stack = _build_stack_from_df(atitlan_mats_interp, col)
        stack_i = _interp_stack_time(stack)
        _write_stack_to_df(atitlan_mats_interp, col, stack_i)
    except ValueError as e:
        print(f"[Aviso] {col}: {e}")

# Columna binaria (agua): interpolo y umbralizo
try:
    stack = _build_stack_from_df(atitlan_mats_interp, "WBI")
    stack_i = _interp_stack_time_binary(stack, thresh=0.5)
    _write_stack_to_df(atitlan_mats_interp, "WBI", stack_i)
except ValueError as e:
    print(f"[Aviso] WBI: {e}")

# NOTA: Si quisieras interpolar RGB por canal, avísame y te paso una función 3 canales.


# ===== 2) INTERPOLACIÓN EN ESTADÍSTICAS =====
# Copia de stats
atitlan_stats_interp = atitlan_stats.copy()

# Asegúrate de tener índice DateTimeIndex con frecuencia temporal
atitlan_stats_interp = atitlan_stats_interp.sort_index()

# Interpolación temporal para todas las columnas numéricas
# 'time' usa la posición temporal del índice; limit_direction='both' rellena bordes
num_cols = atitlan_stats_interp.select_dtypes(include=[np.number]).columns
atitlan_stats_interp[num_cols] = (
    atitlan_stats_interp[num_cols]
    .interpolate(method='time', limit_direction='both')
)

# (Opcional) Si prefieres no extrapolar fuera del rango observado, quita limit_direction='both'.


# ===== 3) RESULTADOS =====
# - atitlan_mats_interp: mismo esquema que atitlan_mats pero con gaps temporales rellenos
# - atitlan_stats_interp: mismas columnas que atitlan_stats, sin NaNs internos
