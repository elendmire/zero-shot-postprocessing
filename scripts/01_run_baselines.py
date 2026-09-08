"""Real baseline comparison: raw ensemble vs EMOS, on the real Germany t2m test set."""
import numpy as np

from zeropp.config import load_experiment_config
from zeropp.data.splits import load_test, load_train
from zeropp.eval.scores import crps_from_quantiles
from zeropp.models.emos import EMOS


def main() -> None:
    quantile_levels = load_experiment_config().quantile_levels

    train_df = load_train()
    test_df = load_test()

    emos = EMOS(quantile_levels=quantile_levels).fit(train_df)

    # Group by (station_id, valid_time, step_hours) to get one ensemble per forecast instance
    grouped = test_df.groupby(["station_id", "valid_time", "step_hours"])
    ens_means, ens_vars, obs_values = [], [], []
    for _, group in grouped:
        ens_means.append(group["t2m_forecast"].mean())
        ens_vars.append(group["t2m_forecast"].var())
        obs_values.append(group["t2m_obs"].iloc[0])

    ens_means = np.array(ens_means).reshape(-1, 1)
    ens_vars = np.array(ens_vars).reshape(-1, 1)
    obs_values = np.array(obs_values).reshape(-1, 1)

    # Raw ensemble: use the empirical quantiles of each ensemble directly
    raw_quantiles = []
    for _, group in grouped:
        raw_quantiles.append(np.quantile(group["t2m_forecast"].to_numpy(), quantile_levels))
    raw_quantiles = np.array(raw_quantiles).reshape(len(obs_values), 1, len(quantile_levels))

    emos_preds = emos.predict_quantiles({"ens_mean": ens_means, "ens_var": ens_vars})

    raw_crps = crps_from_quantiles(obs_values, raw_quantiles, quantile_levels)
    emos_crps = crps_from_quantiles(obs_values, emos_preds, quantile_levels)

    print(f"Raw ensemble mean CRPS: {raw_crps.mean():.4f}")
    print(f"EMOS mean CRPS:         {emos_crps.mean():.4f}")
    print(f"EMOS improvement over raw ensemble: {(1 - emos_crps.mean()/raw_crps.mean())*100:.1f}%")


if __name__ == "__main__":
    main()
