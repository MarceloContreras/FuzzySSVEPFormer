"""Export Nakanishi accuracy mean +/- SE by method and time window."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.e7 import ARCHIVOS, RESULTADOS_DIR, BASELINES, FUZZY, cargar_resultados


def estrellas(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def agregar_significancia(tabla, archivo):
    pruebas = pd.read_csv(archivo)
    requeridas = {"dataset", "ventana", "modelo_x", "modelo_y", "p_holm"}

    if not requeridas.issubset(pruebas.columns):
        raise ValueError(
            f"Missing Wilcoxon columns: {sorted(requeridas - set(pruebas.columns))}"
        )

    pruebas = pruebas.loc[
        pruebas["dataset"].eq("nakanishi")
        & pruebas["modelo_x"].isin(FUZZY)
        & pruebas["modelo_y"].isin(BASELINES)
    ].copy()
    pruebas["ventana"] = pd.to_numeric(pruebas["ventana"], errors="raise").round(12)
    pruebas["p_holm"] = pd.to_numeric(pruebas["p_holm"], errors="raise")

    if not np.isfinite(pruebas[["ventana", "p_holm"]].to_numpy()).all():
        raise ValueError("Wilcoxon windows and p-values must be finite.")

    if not pruebas["p_holm"].between(0, 1).all():
        raise ValueError("p_holm must be between 0 and 1.")

    claves = ["modelo_x", "modelo_y", "ventana"]
    if pruebas.duplicated(claves).any():
        raise ValueError("Duplicate Wilcoxon comparisons for the same window.")
    pvalores = pruebas.set_index(claves)["p_holm"]

    for metodo in BASELINES:
        for ventana in tabla.columns:
            marcas = ""
            for fuzzy, posicion, color in (
                ("f1ssvepformer", "^", "blue"),
                ("f2ssvepformer", "_", "red"),
            ):
                clave = (fuzzy, metodo, round(float(ventana), 12))
                if clave not in pvalores.index:
                    raise ValueError(f"Missing Wilcoxon comparison: {clave}")
                stars = estrellas(pvalores.loc[clave])

                if stars:
                    marcas += (
                        posicion + r"{\text{\textcolor{" + color + "}{" + stars + "}}}"
                    )
            tabla.loc[metodo, ventana] = tabla.loc[metodo, ventana][:-1] + marcas + "$"
    return tabla


def generar_tabla(results_dir=RESULTADOS_DIR, decimals=2, wilcoxon=None):
    filas = []
    referencia = None
    for metodo, archivo in ARCHIVOS.items():
        df = cargar_resultados(Path(results_dir) / archivo, "nakanishi", metodo)
        sujetos = df.groupby(["ventana", "sujeto"])["acc"].mean()

        if referencia is None:
            referencia = sujetos.index
        elif not sujetos.index.equals(referencia):
            raise ValueError(f"{metodo}: subjects/windows do not match.")
        resumen = sujetos.groupby(level="ventana").agg(["mean", "std", "count"])

        if (resumen["count"] < 2).any():
            raise ValueError("At least two subjects per window are required.")
        resumen["se"] = resumen["std"] / np.sqrt(resumen["count"])
        filas.append(
            pd.Series(
                {
                    ventana: f"${row['mean']:.{decimals}f} \\pm {row['se']:.{decimals}f}$"
                    for ventana, row in resumen.iterrows()
                },
                name=metodo,
            )
        )

    tabla = pd.DataFrame(filas).sort_index(axis=1)
    nota = ""

    if wilcoxon is not None:
        tabla = agregar_significancia(tabla, wilcoxon)
        nota = (
            r" Blue superscripts compare each baseline with F1; red subscripts with F2. "
            r"Stars indicate two-sided differences using Holm-adjusted p-values: "
            r"$*p<0.05$, $**p<0.01$, $***p<0.001$. "
            r"They do not indicate the direction of the difference."
        )

    tabla.columns = [f"{v:g}" for v in tabla.columns]
    tabla.index.name = "Method"
    latex = tabla.to_latex(
        escape=False,
        column_format="l" + "c" * len(tabla.columns),
        caption=(
            r"Nakanishi accuracy (\%): mean $\pm$ standard error across subjects, "
            r"after averaging repetitions within each subject. "
            r"Columns indicate time windows in seconds." + nota
        ),
        label="tab:nakanishi-accuracy",
    )
    return tabla, latex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTADOS_DIR)
    parser.add_argument("--output", type=Path, default=Path("accuracy_nakanishi.tex"))
    parser.add_argument("--decimals", type=int, choices=range(7), default=2)
    parser.add_argument(
        "--wilcoxon",
        type=Path,
        default=Path("wilcoxon_nakanishi.csv"),
        help="CSV containing per-window Holm-adjusted p-values.",
    )
    args = parser.parse_args()
    _, latex = generar_tabla(args.results_dir, args.decimals, args.wilcoxon)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex, encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
