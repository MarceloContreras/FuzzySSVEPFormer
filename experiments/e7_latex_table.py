"""Export dataset accuracy or ITR mean +/- SE by method and time window."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.e7 import (
    RESULTADOS_ROOT,
    DATASETS,
    BASELINES,
    FUZZY,
    cargar_resultados,
    descubrir_archivos,
)

from util.itr import itr


def estrellas(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def agregar_significancia(tabla, archivo, dataset="nakanishi"):
    pruebas = pd.read_csv(archivo)
    requeridas = {"dataset", "ventana", "modelo_x", "modelo_y", "p_holm"}

    if not requeridas.issubset(pruebas.columns):
        raise ValueError(
            f"Missing Wilcoxon columns: {sorted(requeridas - set(pruebas.columns))}"
        )

    pruebas = pruebas.loc[
        pruebas["dataset"].str.lower().eq(dataset.lower())
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


def generar_tabla(
    results_dir=None,
    decimals=2,
    wilcoxon=None,
    dataset="nakanishi",
    archivos=None,
    metric="acc",
):
    if metric not in ("acc", "itr"):
        raise ValueError(f"Unsupported metric: {metric}")
    dataset = dataset.lower()
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    results_dir = (
        RESULTADOS_ROOT / dataset if results_dir is None else Path(results_dir)
    )
    if archivos is None:
        archivos = descubrir_archivos(results_dir, dataset, (*BASELINES, *FUZZY))
    faltantes = set((*BASELINES, *FUZZY)) - archivos.keys()
    if faltantes:
        raise ValueError(f"Missing models in file mapping: {sorted(faltantes)}")
    filas = []
    referencia = None
    for metodo in (*BASELINES, *FUZZY):
        df = cargar_resultados(results_dir / archivos[metodo], dataset, metodo)
        if metric == "itr":
            if not df["acc"].between(0, 100).all() or not df["ventana"].gt(0).all():
                raise ValueError(
                    f"{metodo}: ITR requires accuracy in [0, 100] and Time > 0."
                )
            # Convert each repetition before averaging, matching e7.py.
            df["itr"] = itr(df["acc"] / 100.0, DATASETS[dataset], df["ventana"])
        sujetos = df.groupby(["ventana", "sujeto"])[metric].mean()

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
        tabla = agregar_significancia(tabla, wilcoxon, dataset)
        nota = (
            r" Blue superscripts compare each baseline with F1; red subscripts with F2. "
            r"Stars indicate two-sided differences using Holm-adjusted p-values: "
            r"$*p<0.05$, $**p<0.01$, $***p<0.001$. "
            r"They do not indicate the direction of the difference."
        )

    tabla.columns = [f"{v:g}" for v in tabla.columns]
    tabla.index.name = "Method"
    metric_title = "ITR (bits/min)" if metric == "itr" else r"accuracy (\%)"
    metric_name = "itr" if metric == "itr" else "accuracy"
    latex = tabla.to_latex(
        escape=False,
        column_format="l" + "c" * len(tabla.columns),
        caption=(
            ("Nakanishi" if dataset == "nakanishi" else "UTEC")
            + f" {metric_title}"
            + r": mean $\pm$ standard error across subjects, "
            r"after averaging repetitions within each subject. "
            r"Columns indicate time windows in seconds." + nota
        ),
        label=f"tab:{dataset}-{metric_name}",
    )
    return tabla, latex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=str.lower, choices=DATASETS, default="nakanishi"
    )
    parser.add_argument(
        "--metric",
        choices=("acc", "itr"),
        default="acc",
        help="Metric to export; ITR is calculated from the raw accuracy CSVs.",
    )
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--files-map",
        type=Path,
        help="JSON {model: accuracy.csv}, as in e7.py; paths relative to --results-dir.",
    )
    parser.add_argument("--decimals", type=int, choices=range(7), default=2)
    parser.add_argument(
        "--wilcoxon",
        type=Path,
        default=None,
        help="CSV containing Holm-adjusted p-values for the selected metric.",
    )
    args = parser.parse_args()
    archivos = None
    if args.files_map is not None:
        with args.files_map.open() as file:
            archivos = json.load(file)
        if not isinstance(archivos, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in archivos.items()
        ):
            parser.error("--files-map requires a JSON object {model: accuracy.csv}.")
    metric_name = "itr" if args.metric == "itr" else "accuracy"
    suffix = "_itr" if args.metric == "itr" else ""
    args.output = args.output or Path(f"{metric_name}_{args.dataset}.tex")
    wilcoxon = args.wilcoxon or Path(f"wilcoxon_{args.dataset}{suffix}.csv")
    _, latex = generar_tabla(
        args.results_dir,
        args.decimals,
        wilcoxon,
        dataset=args.dataset,
        archivos=archivos,
        metric=args.metric,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex, encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
