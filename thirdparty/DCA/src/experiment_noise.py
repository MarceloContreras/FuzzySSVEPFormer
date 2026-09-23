import numpy as np
from dca.DCA import DCA
from dca.schemes import (
    DCALoggers,
    DelaunayGraphParams,
    ExperimentDirs,
    GeomCAParams,
    HDBSCANParams,
    REData,
)
import typer

app = typer.Typer()


@app.command()
def main(
    cleanup: int = 0,
    model: str = typer.Option("f1ssvepformer_C", help="Model of transformer"),
    sub: int = typer.Option(1, help="Subject"),
    noise: float = typer.Option(0.01, help="Noise"),
    dataset: str = typer.Option("NAKANISHI", help="Dataset type"),
    seed: int = typer.Option(1, help="seed"),
):

    # Set path to the output folders
    experiment_path = f"output/{model}_{dataset}_S{sub}_n{noise}_seed{seed}"
    experiment_id = "template_id1"

    # Load data
    R = np.load(
        f"/home/marcelo/Documents/Tesis/results/embeddings/e1/{dataset}/X_{model}_S{sub}_n0_seed{seed}.npy"
    )
    E = np.load(
        f"/home/marcelo/Documents/Tesis/results/embeddings/e1/{dataset}/X_{model}_S{sub}_n{noise}_seed{seed}.npy"
    )

    # Generate input parameters
    data_config = REData(R=R, E=E)
    experiment_config = ExperimentDirs(
        experiment_dir=experiment_path,
        experiment_id=experiment_id,
    )
    graph_config = DelaunayGraphParams()
    hdbscan_config = HDBSCANParams()
    geomCA_config = GeomCAParams()

    # Initialize loggers
    exp_loggers = DCALoggers(experiment_config.logs_dir)

    # Run DCA
    dca = DCA(
        experiment_config,
        graph_config,
        hdbscan_config,
        geomCA_config,
        loggers=exp_loggers,
    )
    dca_scores = dca.fit(data_config)

    if cleanup:
        dca.cleanup()  # Optional cleanup

    return dca_scores


if __name__ == "__main__":
    typer.run(main)
