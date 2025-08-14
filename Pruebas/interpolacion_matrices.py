# ----- helpers de interpolación en tiempo -----

def _interp_stack_time(stack_3d):

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
    for i, f in enumerate(df_mats.index):
        df_mats.at[f, col] = stack[i]





# interpolación para matrices
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