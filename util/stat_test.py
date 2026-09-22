import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.stats.multitest import multipletests


# ---------- 1. PRIMARIO: prueba de signos global ----------


def prueba_signos(df):
    """
    df: una fila por (dataset, variante, ventana)
    con delta = fuzzy - baseline.
    """
    victorias = int((df.delta > 0).sum())
    n = len(df)

    r = stats.binomtest(victorias, n, p=0.5, alternative="greater")

    return victorias, n, r.pvalue


# ---------- 2. PRIMARIO: U invertida por permutacion ----------


def test_u_invertida(ventanas, deltas, n_perm=10000, seed=42):
    """
    Ajusta delta = a + b*T + c*T^2
    y testea c < 0 por permutacion.
    """
    rng = np.random.default_rng(seed)

    T = np.asarray(ventanas, float)
    d = np.asarray(deltas, float)

    X = np.column_stack([np.ones_like(T), T, T**2])

    c_obs = np.linalg.lstsq(X, d, rcond=None)[0][2]

    nulos = np.empty(n_perm)

    for i in range(n_perm):
        nulos[i] = np.linalg.lstsq(X, rng.permutation(d), rcond=None)[0][2]

    # Unilateral: c < 0
    p = (np.sum(nulos <= c_obs) + 1) / (n_perm + 1)

    return c_obs, p


# ---------- 3. PRIMARIO: interaccion por permutacion sobre F ----------


def F_interaccion(Y):
    """
    Y: array (n_sujetos, n_metodos, n_ventanas).
    F de la interaccion.
    """
    gm = Y.mean()

    m_m = Y.mean(axis=(0, 2), keepdims=True)

    m_v = Y.mean(axis=(0, 1), keepdims=True)

    cel = Y.mean(axis=0, keepdims=True)

    ss_int = Y.shape[0] * ((cel - m_m - m_v + gm) ** 2).sum()

    resid = Y - cel
    ss_res = (resid**2).sum()

    nm = Y.shape[1]
    nv = Y.shape[2]
    ns = Y.shape[0]

    gl_int = (nm - 1) * (nv - 1)

    gl_res = (ns - 1) * (nm * nv - 1) - gl_int

    return (ss_int / gl_int) / (ss_res / max(gl_res, 1))


def perm_interaccion(Y, n_perm=10000, seed=42):
    """
    Permuta etiquetas de metodo DENTRO de cada sujeto
    (respeta el pareado).
    """
    rng = np.random.default_rng(seed)

    F_obs = F_interaccion(Y)

    nulos = np.empty(n_perm)

    for i in range(n_perm):
        Yp = Y.copy()

        for s in range(Y.shape[0]):
            Yp[s] = Y[s][rng.permutation(Y.shape[1])]

        nulos[i] = F_interaccion(Yp)

    p = (np.sum(nulos >= F_obs) + 1) / (n_perm + 1)

    return F_obs, p


# ---------- 4. Tamano de efecto y IC ----------


def rango_biserial(x, y):
    """
    Correlacion rango-biserial pareada,
    companera correcta del Wilcoxon.
    """
    d = np.asarray(x) - np.asarray(y)

    d = d[d != 0]

    if len(d) == 0:
        return 0.0

    r = stats.rankdata(np.abs(d))

    Wp = r[d > 0].sum()
    Wm = r[d < 0].sum()

    return (Wp - Wm) / (Wp + Wm)


def hodges_lehmann(x, y, alpha=0.05):
    """
    Estimador HL pareado: mediana de los promedios de Walsh.
    IC por inversion de la distribucion de rangos con signo de Wilcoxon.
    La calibracion exacta supone diferencias independientes de una
    distribucion continua y simetrica; no ajusta la distribucion por empates.
    Por discrecion, la cobertura nominal es al menos 1-alpha bajo ese modelo.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    n = len(d)
    walsh = np.array([(d[i] + d[j]) / 2 for i in range(n) for j in range(i, n)])
    walsh.sort()
    walsh.sort()

    # Estimador Hodges-Lehmann
    hl = np.median(walsh)

    ranks = np.arange(1, n + 1)
    W_plus = []
    for mask in range(2**n):
        w = 0
        for i in range(n):
            if mask & (1 << i):
                w += ranks[i]
        W_plus.append(w)

    W_plus = np.asarray(W_plus)

    # ---------------------------------------------------------
    # Valor crítico
    # ---------------------------------------------------------

    possible_w = np.arange(0, n * (n + 1) // 2 + 1)

    cdf = np.array([np.mean(W_plus <= w) for w in possible_w])

    valid = possible_w[cdf <= alpha / 2]

    if len(valid) == 0:
        k = -1
    else:
        k = valid[-1]

    # ---------------------------------------------------------
    # IC usando Walsh averages ordenadas
    # ---------------------------------------------------------

    N = len(walsh)

    lower_index = k + 1
    upper_index = N - k - 2

    lower_index = max(0, lower_index)
    upper_index = min(N - 1, upper_index)

    lo = walsh[lower_index]
    hi = walsh[upper_index]
    return hl, lo, hi


# ---------- 5. Wilcoxon + Holm por familia ----------


def familia_wilcoxon(pares, etiquetas):
    """
    pares: lista de (x, y) por comparacion.
    Devuelve tabla con Holm.
    """
    filas = []

    for (x, y), et in zip(pares, etiquetas):

        st, p = stats.wilcoxon(x, y, alternative="two-sided", method="exact")

        hl, lo, hi = hodges_lehmann(x, y)

        filas.append(
            dict(
                comparacion=et,
                W=st,
                p_bruto=p,
                delta=hl,
                ic_lo=lo,
                ic_hi=hi,
                r_rb=rango_biserial(x, y),
            )
        )

    df = pd.DataFrame(filas)

    df["p_holm"] = multipletests(df.p_bruto, method="holm")[1]

    return df
