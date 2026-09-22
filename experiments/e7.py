"""Comparaciones pareadas por sujeto entre modelos para Nakanishi."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util.stat_test import familia_wilcoxon


RESULTADOS_DIR = Path(
    "/home/marcelo/Documents/Tesis/run_pod_data/results/raw/nakanishi"
)
# Para agregar un metodo, registrar su CSV y agregarlo a una lista de comparacion.
ARCHIVOS = {
    "CNN": "CNN_e50_Nakanishi.csv",
    "Deformer": "Deformer_e400_NAKANISHI.csv",
    "EEGNet": "EEGNet_e600_NAKANISHI.csv",
    "SSVEPNet": "SSVEPNet_e500_NAKANISHI.csv",
    "TRCA": "TRCA_Nakanishi.csv",
    "ssvepformer": "ssvepformer_e100_NAKANISHI.csv",
    "f1ssvepformer": "f1ssvepformer_C_e100_NAKANISHI.csv",
    "f2ssvepformer": "f2ssvepformer_C_e100_NAKANISHI.csv",
}
BASELINES = ("CNN", "Deformer", "EEGNet", "SSVEPNet", "TRCA", "ssvepformer")
FUZZY = ("f1ssvepformer", "f2ssvepformer")


def cargar_resultados(archivo, dataset, modelo):
    df = pd.read_csv(archivo)
    if df.empty or "Time" not in df or len(df.columns) < 2:
        raise ValueError(f"{modelo}: CSV vacio o sin Time/columnas de sujetos.")
    df = df.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(f"{modelo}: el CSV contiene NaN o valores no finitos.")
    # Cada fila repetida dentro de Time corresponde a una repeticion/seed.
    df["seed"] = df.groupby("Time").cumcount() + 1
    df = df.melt(id_vars=["Time", "seed"], var_name="sujeto", value_name="acc")
    df["dataset"] = dataset
    df["modelo"] = modelo
    df = df.rename(columns={"Time": "ventana"})
    return df[["dataset", "modelo", "ventana", "sujeto", "seed", "acc"]]


def comparar_metodos(
    resultados_dir=RESULTADOS_DIR, archivos=None, baselines=BASELINES, fuzzy=FUZZY
):
    """Retorna resumen descriptivo y Wilcoxon; delta positivo favorece fuzzy.

    Cada ventana tiene su propia familia Holm de contrastes fuzzy-baseline.
    El control de error es por ventana, no sobre todas las ventanas juntas.
    No se incluyen comparaciones entre baselines ni entre los dos fuzzy.
    """
    archivos = ARCHIVOS if archivos is None else archivos
    if not baselines or not fuzzy or set(baselines) & set(fuzzy):
        raise ValueError("Se requieren listas no vacias y disjuntas de modelos.")
    if len(set(baselines)) != len(baselines) or len(set(fuzzy)) != len(fuzzy):
        raise ValueError("Las listas de modelos no deben contener duplicados.")
    modelos = {}
    for modelo in (*fuzzy, *baselines):
        df = cargar_resultados(
            Path(resultados_dir) / archivos[modelo], "nakanishi", modelo
        )
        modelos[modelo] = df.groupby(
            ["dataset", "modelo", "ventana", "sujeto"], as_index=False
        ).agg(
            acc=("acc", "mean"), sd_semillas=("acc", "std"), n_semillas=("acc", "size")
        )

    por_sujeto = pd.concat(modelos.values(), ignore_index=True)
    tabla = por_sujeto.groupby(["dataset", "modelo", "ventana"], as_index=False).agg(
        media=("acc", "mean"),
        se=("acc", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        sd_semillas_media=("sd_semillas", "mean"),
        n_sujetos=("sujeto", "size"),
        n_semillas_min=("n_semillas", "min"),
        n_semillas_max=("n_semillas", "max"),
    )

    claves = ["dataset", "ventana", "sujeto"]
    pares, etiquetas, resumen = [], [], []
    for modelo_x in fuzzy:
        for modelo_y in baselines:
            comparaciones = modelos[modelo_x][claves + ["acc"]].merge(
                modelos[modelo_y][claves + ["acc"]],
                on=claves,
                how="outer",
                suffixes=("_x", "_y"),
                validate="one_to_one",
                indicator=True,
            )
            if not comparaciones["_merge"].eq("both").all():
                faltantes = comparaciones.loc[
                    comparaciones["_merge"] != "both", claves + ["_merge"]
                ]
                raise ValueError(
                    f"{modelo_x} vs {modelo_y}: sujetos/ventanas sin pareja:\n"
                    f"{faltantes.to_string(index=False)}"
                )
            for (dataset, ventana), grupo in comparaciones.groupby(
                ["dataset", "ventana"]
            ):
                grupo = grupo.sort_values("sujeto")
                if len(grupo) < 2:
                    raise ValueError(
                        f"{dataset}, {ventana}: se requieren al menos 2 sujetos."
                    )
                pares.append((grupo["acc_x"].to_numpy(), grupo["acc_y"].to_numpy()))
                etiquetas.append(
                    f"{dataset}: {modelo_x} vs {modelo_y}, ventana={ventana}"
                )
                resumen.append(
                    dict(
                        dataset=dataset,
                        ventana=ventana,
                        modelo_x=modelo_x,
                        modelo_y=modelo_y,
                        n_sujetos=len(grupo),
                    )
                )

    # Cada llamada corrige solo los p-valores de una ventana (12 por defecto).
    metadata = pd.DataFrame(resumen)
    resultados = []
    for _, grupo in metadata.groupby(["dataset", "ventana"], sort=True):
        indices = grupo.index.tolist()
        pruebas = familia_wilcoxon(
            [pares[i] for i in indices],
            [etiquetas[i] for i in indices],
        )
        pruebas = pd.concat([grupo.reset_index(drop=True), pruebas], axis=1)
        pruebas["n_tests_holm"] = len(indices)
        resultados.append(pruebas)
    resultado = pd.concat(resultados, ignore_index=True)
    return tabla, resultado


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTADOS_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    tabla, resultado = comparar_metodos(args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(args.output_dir / "tabla_cuerpo.csv", index=False)
    resultado.to_csv(args.output_dir / "wilcoxon_nakanishi.csv", index=False)
    print(resultado.to_string(index=False))


if __name__ == "__main__":
    main()
