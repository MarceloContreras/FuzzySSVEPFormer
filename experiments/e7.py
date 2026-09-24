import argparse
import json
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util.stat_test import familia_wilcoxon
from util.itr import itr


RESULTADOS_ROOT = Path("/home/marcelo/Documents/Tesis/results/report/accuracy")
RESULTADOS_DIR = RESULTADOS_ROOT / "nakanishi"
DATASETS = {"nakanishi": 12, "utec": 4}
PREFIJOS = {
    "f1ssvepformer": "f1ssvepformer_C",
    "f2ssvepformer": "f2ssvepformer_C",
}
BASELINES = ("CNN", "Deformer", "EEGNet", "SSVEPNet", "TRCA", "ssvepformer")
FUZZY = ("f1ssvepformer", "f2ssvepformer")


def descubrir_archivos(resultados_dir, dataset, modelos):
    """Busca CSV de accuracy; no incluye reportes Noise/TimeVarying/ITR."""
    archivos = {}
    candidatos = sorted(Path(resultados_dir).glob("*.csv"))

    for modelo in modelos:
        prefijo = PREFIJOS.get(modelo, modelo)
        patron = re.compile(
            rf"{re.escape(prefijo)}(?:_e[0-9]+)?_{re.escape(dataset)}\.csv",
            re.IGNORECASE,
        )
        matches = [p.name for p in candidatos if patron.fullmatch(p.name)]

        if len(matches) != 1:
            raise ValueError(
                f"{modelo} ({dataset}): se esperaba un CSV en {resultados_dir}; "
                f"encontrados: {matches}. Use --files-map para elegir los archivos."
            )
        archivos[modelo] = matches[0]
    return archivos


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
    resultados_dir=None,
    archivos=None,
    baselines=BASELINES,
    fuzzy=FUZZY,
    metrica="acc",
    dataset="nakanishi",
):
    """Retorna resumen descriptivo y Wilcoxon; delta positivo favorece fuzzy.

    ITR se calcula por repeticion en bits/min (clases del dataset, gaze=0.5 s).
    Cada metrica/ventana tiene su propia familia Holm de contrastes fuzzy-baseline.
    El control de error es por ventana, no sobre todas las ventanas juntas.
    No se incluyen comparaciones entre baselines ni entre los dos fuzzy.
    """
    if metrica not in ("acc", "itr"):
        raise ValueError("La metrica debe ser acc o itr.")
    dataset = dataset.lower()

    if dataset not in DATASETS:
        raise ValueError(f"Dataset no soportado: {dataset}.")
    resultados_dir = (
        RESULTADOS_ROOT / dataset if resultados_dir is None else Path(resultados_dir)
    )

    if not baselines or not fuzzy or set(baselines) & set(fuzzy):
        raise ValueError("Se requieren listas no vacias y disjuntas de modelos.")

    if len(set(baselines)) != len(baselines) or len(set(fuzzy)) != len(fuzzy):
        raise ValueError("Las listas de modelos no deben contener duplicados.")

    if archivos is None:
        archivos = descubrir_archivos(resultados_dir, dataset, (*fuzzy, *baselines))

    modelos = {}
    for modelo in (*fuzzy, *baselines):
        df = cargar_resultados(Path(resultados_dir) / archivos[modelo], dataset, modelo)
        if metrica == "itr":
            if not df["acc"].between(0, 100).all() or not df["ventana"].gt(0).all():
                raise ValueError(
                    f"{modelo}: ITR requiere accuracy en [0, 100] y Time > 0."
                )
            df["itr"] = itr(df["acc"] / 100.0, DATASETS[dataset], df["ventana"])

        modelos[modelo] = df.groupby(
            ["dataset", "modelo", "ventana", "sujeto"], as_index=False
        ).agg(
            **{
                metrica: (metrica, "mean"),
                "sd_semillas": (metrica, "std"),
                "n_semillas": (metrica, "size"),
            }
        )

    por_sujeto = pd.concat(modelos.values(), ignore_index=True)

    tabla = por_sujeto.groupby(["dataset", "modelo", "ventana"], as_index=False).agg(
        media=(metrica, "mean"),
        se=(metrica, lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        sd_semillas_media=("sd_semillas", "mean"),
        n_sujetos=("sujeto", "size"),
        n_semillas_min=("n_semillas", "min"),
        n_semillas_max=("n_semillas", "max"),
    )

    claves = ["dataset", "ventana", "sujeto"]
    pares, etiquetas, resumen = [], [], []
    for modelo_x in fuzzy:
        for modelo_y in baselines:
            comparaciones = modelos[modelo_x][claves + [metrica]].merge(
                modelos[modelo_y][claves + [metrica]],
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
                pares.append(
                    (grupo[f"{metrica}_x"].to_numpy(), grupo[f"{metrica}_y"].to_numpy())
                )
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
    parser.add_argument(
        "--dataset", type=str.lower, choices=DATASETS, default="nakanishi"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directorio de CSV; por defecto results/raw/<dataset> en RESULTADOS_ROOT.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--files-map",
        type=Path,
        help="JSON {modelo: archivo.csv}; rutas relativas a --results-dir o absolutas.",
    )

    args = parser.parse_args()
    archivos = None

    if args.files_map is not None:
        with args.files_map.open() as file:
            archivos = json.load(file)
        if not isinstance(archivos, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in archivos.items()
        ):
            parser.error("--files-map requiere un objeto JSON {modelo: archivo.csv}.")
        faltantes = set((*BASELINES, *FUZZY)) - archivos.keys()
        if faltantes:
            parser.error(f"Faltan modelos en --files-map: {sorted(faltantes)}")

    analisis = {}
    for metrica in ("acc", "itr"):
        analisis[metrica] = comparar_metodos(
            args.results_dir, archivos=archivos, metrica=metrica, dataset=args.dataset
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metrica, (tabla, resultado) in analisis.items():
        sufijo = "_itr" if metrica == "itr" else ""
        tabla.to_csv(
            args.output_dir / f"tabla_cuerpo_{args.dataset}{sufijo}.csv", index=False
        )
        resultado.to_csv(
            args.output_dir / f"wilcoxon_{args.dataset}{sufijo}.csv", index=False
        )
        print(
            f"\n{args.dataset}: {metrica} ({'bits/min' if metrica == 'itr' else '%'})"
        )
        print(resultado.to_string(index=False))


if __name__ == "__main__":
    main()
