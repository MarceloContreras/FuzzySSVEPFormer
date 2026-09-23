import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd


def extraer_fuzzy(ckpt_path):
    """
    Devuelve los parametros aprendidos del FIB tras aplicar las
    reparametrizaciones sigmoid del paper (Ecs. 12 y 13).
    """
    sd = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" in sd:
        sd = sd["state_dict"]

    out = {}

    for k, v in sd.items():
        if k.endswith("beta"):
            out["beta_raw"] = v.detach().cpu().numpy()
        elif k.endswith("h"):
            out["h_raw"] = v.detach().cpu().numpy()
        elif k.endswith("sigma"):
            out["sig_up"] = v.detach().cpu().numpy()
        elif k.endswith("alpha"):
            out["sig_lo_raw"] = v.detach().cpu().numpy()
        elif k.endswith("mu"):
            out["v"] = v.detach().cpu().numpy()

    sig = lambda x: 1.0 / (1.0 + np.exp(-x))

    beta = sig(out["beta_raw"])  # Ec. 12
    h = sig(out["h_raw"])  # Ec. 12
    s_up = out["sig_up"]
    s_lo = 0.5 * s_up * sig(out["sig_lo_raw"]) + 0.5  # Ec. 13

    return dict(
        beta=beta,
        h=h,
        sigma_up=s_up,
        sigma_lo=s_lo,
        fou_width=(s_up - s_lo),
        centroids=out["v"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract fuzzy parameters from all f2ssvepformer_C checkpoints."
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "checkpoints",
        help="Checkpoint root containing the utec and nakanishi folders.",
    )
    parser.add_argument("--output", type=Path, default=Path("E3_parametros_fuzzy.csv"))
    args = parser.parse_args()

    checkpoints = []
    for dataset in ("utec", "nakanishi"):
        folders = sorted(
            path
            for path in args.weights_path.iterdir()
            if path.is_dir() and path.name.lower() == dataset
        )
        if not folders:
            raise FileNotFoundError(f"Missing {dataset} folder in {args.weights_path}")
        paths = sorted(
            path
            for folder in folders
            for path in folder.rglob("f2ssvepformer_C_*")
            if path.is_file() and path.suffix.lower() in {".pt", ".pth"}
        )
        if not paths:
            raise FileNotFoundError(
                f"No f2ssvepformer_C checkpoints found for {dataset}"
            )
        print(f"{dataset.upper()}: {len(paths)} checkpoints")
        checkpoints.extend((dataset.upper(), path) for path in paths)

    filas = []
    for dataset, path in checkpoints:
        p = extraer_fuzzy(path)
        filas.append(
            dict(
                dataset=dataset,
                checkpoint=str(path),
                beta=float(np.mean(p["beta"])),
                h_med=float(np.median(p["h"])),
                fou_med=float(np.median(p["fou_width"])),
                fou_p10=float(np.percentile(p["fou_width"], 10)),
                fou_p90=float(np.percentile(p["fou_width"], 90)),
                sigma_med=float(np.median(p["sigma_up"])),
            )
        )

    df = pd.DataFrame(filas)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")

    # Diagnostico rapido de saturacion
    for ds, g in df.groupby("dataset"):
        frac_sat = ((g.beta < 0.05) | (g.beta > 0.95)).mean()
        print(
            f"{ds}: beta medio={g.beta.mean()} "
            f"frac saturado={frac_sat} "
            f"ancho FOU mediano={g.fou_med}"
        )


if __name__ == "__main__":
    main()
