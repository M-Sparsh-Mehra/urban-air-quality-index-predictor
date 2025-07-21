import mlflow
import yaml

def get_best_configs(experiment_name="ESN_hyperparams", save_path="/workspaces/urban-air-quality-index-predictor/config/best_esn_configs.yaml"):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(experiment.experiment_id)

    best_configs = {}

    for run in runs:
        run_data = run.data
        run_tags = run.data.tags
        run_params = run.data.params
        pollutant = run_tags.get("pollutant")

        if pollutant not in best_configs or float(run_data.metrics["val_rmse"]) < best_configs[pollutant]["val_rmse"]:
            best_configs[pollutant] = {
                "n_reservoir": int(run_params["n_reservoir"]),
                "sparsity": float(run_params["sparsity"]),
                "spectral_radius": float(run_params["spectral_radius"]),
                "forecast_horizon": int(run_params["forecast_horizon"]),
                "val_rmse": float(run_data.metrics["val_rmse"]),
                "run_id": run.info.run_id
            }

    with open(save_path, "w") as f:
        yaml.dump(best_configs, f)

    print(f"Saved best configs to {save_path}")
