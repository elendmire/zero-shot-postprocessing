import math

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

from zeropp.models.base import Postprocessor


def _safe_scale(std: float) -> float:
    """Replace a zero or non-finite standard deviation by 1."""
    return std if np.isfinite(std) and std != 0 else 1.0


def _gaussian_crps_torch(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = (y - mu) / sigma
    normal = torch.distributions.Normal(0.0, 1.0)
    cdf = normal.cdf(z)
    pdf = torch.exp(normal.log_prob(z))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


class _DRNNet(nn.Module):
    def __init__(self, n_stations: int, embedding_dim: int, hidden_dim: int, n_leads: int = 0):
        super().__init__()
        self.station_embedding = nn.Embedding(n_stations, embedding_dim)
        self.lead_embedding = nn.Embedding(n_leads, embedding_dim) if n_leads > 0 else None
        in_dim = 2 + embedding_dim * (2 if n_leads > 0 else 1)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        # zero-initialised output layer: the untrained network returns the raw-ensemble Gaussian
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, ens_mean, ens_var, station_idx, lead_idx=None):
        parts = [ens_mean.unsqueeze(-1), ens_var.unsqueeze(-1), self.station_embedding(station_idx)]
        if self.lead_embedding is not None:
            if lead_idx is None:
                raise ValueError("this DRN was trained with a lead-time embedding; lead_idx is required")
            parts.append(self.lead_embedding(lead_idx))
        out = self.net(torch.cat(parts, dim=-1))
        return out[..., 0], out[..., 1]  # delta_mu, delta_log_sigma2 (standardized-target space)


class DRN(Postprocessor):
    """Distributional regression network (Rasp and Lerch 2018).

    Gaussian forecast like EMOS, with a neural network and a station embedding (and a lead-time embedding if
    the training data have a step_hours column) instead of a linear model, trained on the Gaussian CRPS with
    Adam. The network predicts corrections to the raw-ensemble Gaussian in standardized units (two ReLU
    layers of 64 units, zero-initialised output layer); inputs and target are standardized with the
    training-set mean and standard deviation.

    10 % of the training cases (year_idx, time_idx; random rows if these columns are missing) are held out
    for early stopping on the validation CRPS; the best weights, starting with the initial raw-ensemble
    Gaussian, are restored. Training runs for at most n_epochs epochs but at least min_updates optimizer
    steps; validation is checked every max(100, steps_per_epoch // 4) steps. Training sets with fewer than
    200 rows or 2 cases are fit for fallback_steps steps without validation. Stations or leads unseen in
    fit() use embedding index 0. predict_quantiles needs "station_id" (and "step_hours") in X.
    loss_history_ and val_loss_history_ hold the losses at each validation check, n_updates_ the number of
    steps.
    """

    def __init__(
        self,
        quantile_levels: list[float],
        embedding_dim: int = 4,
        hidden_dim: int = 64,
        n_epochs: int = 100,
        lr: float = 1e-3,
        seed: int = 0,
        device: str | None = None,
        batch_size: int = 4096,
        patience: int = 5,
        val_fraction: float = 0.1,
        fallback_steps: int = 200,
        min_updates: int = 2000,
    ):
        self.quantile_levels = quantile_levels
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.fallback_steps = fallback_steps
        self.min_updates = min_updates
        self._net = None
        self._station_to_idx: dict | None = None
        self._lead_to_idx: dict | None = None
        self._ens_mean_loc = 0.0
        self._ens_mean_scale = 1.0
        self._ens_var_loc = 0.0
        self._ens_var_scale = 1.0
        self._obs_loc = 0.0
        self._obs_scale = 1.0
        self.loss_history_: list[float] = []
        self.val_loss_history_: list[float] = []
        self.n_updates_: int = 0

    def _standardize(self, ens_mean_np: np.ndarray, ens_var_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            (ens_mean_np - self._ens_mean_loc) / self._ens_mean_scale,
            (ens_var_np - self._ens_var_loc) / self._ens_var_scale,
        )

    def _unstandardize_output(self, mu_std: torch.Tensor, log_sigma2_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # rescale the standardized output to Kelvin
        mu = mu_std * self._obs_scale + self._obs_loc
        sigma = torch.exp(0.5 * log_sigma2_std) * self._obs_scale
        return mu, sigma

    def _base_params(self, ens_mean_std: torch.Tensor, ens_var_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw-ensemble Gaussian in standardized units, to which the network adds its corrections."""
        ens_mean = ens_mean_std * self._ens_mean_scale + self._ens_mean_loc
        ens_var = ens_var_std * self._ens_var_scale + self._ens_var_loc
        base_mu = (ens_mean - self._obs_loc) / self._obs_scale
        var_floor = 1e-6 * self._obs_scale**2
        base_log_s2 = torch.log(torch.clamp(ens_var, min=var_floor) / self._obs_scale**2)
        return base_mu, base_log_s2

    def _split_indices(self, train, n: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (train_idx, val_idx) or None if the set is too small to split."""
        rng = np.random.default_rng(self.seed)
        if n < 200:
            return None
        if "year_idx" in train.columns and "time_idx" in train.columns:
            case_codes = train.groupby(["year_idx", "time_idx"], sort=False).ngroup().to_numpy()
            n_cases = int(case_codes.max()) + 1
            if n_cases < 2:
                return None
            n_val_cases = min(max(1, int(round(self.val_fraction * n_cases))), n_cases - 1)
            val_cases = rng.choice(n_cases, size=n_val_cases, replace=False)
            is_val = np.isin(case_codes, val_cases)
        else:
            n_val = max(1, int(round(self.val_fraction * n)))
            is_val = np.zeros(n, dtype=bool)
            is_val[rng.choice(n, size=n_val, replace=False)] = True
        return np.flatnonzero(~is_val), np.flatnonzero(is_val)

    def _residual_forward(self, ens_mean_std, ens_var_std, station_idx, lead_idx=None):
        d_mu, d_log_s2 = self._net(ens_mean_std, ens_var_std, station_idx, lead_idx)
        base_mu, base_log_s2 = self._base_params(ens_mean_std, ens_var_std)
        return base_mu + d_mu, base_log_s2 + d_log_s2

    def _batch_loss(self, tensors: tuple, idx: torch.Tensor) -> torch.Tensor:
        ens_mean, ens_var, station_idx, lead_idx, obs = tensors
        mu_std, log_sigma2_std = self._residual_forward(
            ens_mean[idx], ens_var[idx], station_idx[idx], None if lead_idx is None else lead_idx[idx]
        )
        mu, sigma = self._unstandardize_output(mu_std, log_sigma2_std)
        return _gaussian_crps_torch(mu, sigma, obs[idx])

    def _mean_loss(self, tensors: tuple, idx: torch.Tensor) -> float:
        self._net.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(idx), 65536):
                chunk = idx[start : start + 65536]
                total += float(self._batch_loss(tensors, chunk).sum().item())
                count += len(chunk)
        self._net.train()
        return total / max(count, 1)

    def fit(self, train) -> "DRN":
        stations = sorted(train["station_id"].unique())
        self._station_to_idx = {s: i for i, s in enumerate(stations)}
        use_lead = "step_hours" in train.columns
        if use_lead:
            leads = sorted(train["step_hours"].unique())
            self._lead_to_idx = {s: i for i, s in enumerate(leads)}
        else:
            self._lead_to_idx = None

        ens_mean_raw = train["ens_mean"].to_numpy()
        ens_var_raw = train["ens_var"].to_numpy()
        # standardization constants from the training data, reused at predict time
        self._ens_mean_loc = float(ens_mean_raw.mean())
        self._ens_mean_scale = _safe_scale(float(ens_mean_raw.std()))
        self._ens_var_loc = float(ens_var_raw.mean())
        self._ens_var_scale = _safe_scale(float(ens_var_raw.std()))
        ens_mean_std, ens_var_std = self._standardize(ens_mean_raw, ens_var_raw)

        obs_raw = train["t2m_obs"].to_numpy()
        self._obs_loc = float(obs_raw.mean())
        self._obs_scale = _safe_scale(float(obs_raw.std()))

        dev = torch.device(self.device)
        n = len(train)
        tensors = (
            torch.tensor(ens_mean_std, dtype=torch.float32, device=dev),
            torch.tensor(ens_var_std, dtype=torch.float32, device=dev),
            torch.tensor(train["station_id"].map(self._station_to_idx).to_numpy(), dtype=torch.long, device=dev),
            torch.tensor(train["step_hours"].map(self._lead_to_idx).to_numpy(), dtype=torch.long, device=dev)
            if use_lead
            else None,
            torch.tensor(obs_raw, dtype=torch.float32, device=dev),
        )

        split = self._split_indices(train, n)
        if split is None:
            train_idx_np, val_idx = np.arange(n), None
        else:
            train_idx_np, val_idx_np = split
            val_idx = torch.tensor(val_idx_np, dtype=torch.long, device=dev)
        train_idx = torch.tensor(train_idx_np, dtype=torch.long, device=dev)
        n_train = len(train_idx)
        bs = max(1, min(int(self.batch_size), max(256, n_train // 20)))

        self.loss_history_ = []
        self.val_loss_history_ = []
        self.n_updates_ = 0
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        # seed network initialisation without changing the global torch RNG state
        with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
            torch.manual_seed(self.seed)
            self._net = _DRNNet(
                len(stations), self.embedding_dim, self.hidden_dim, len(self._lead_to_idx) if use_lead else 0
            ).to(dev)
            optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
            self._net.train()

            def run_epoch(max_steps: int | None = None) -> float:
                perm = train_idx[torch.randperm(n_train, generator=generator).to(dev)]
                total, count = 0.0, 0
                for start in range(0, n_train, bs):
                    if max_steps is not None and max_steps <= 0:
                        break
                    idx = perm[start : start + bs]
                    optimizer.zero_grad()
                    loss = self._batch_loss(tensors, idx).mean()
                    loss.backward()
                    optimizer.step()
                    self.n_updates_ += 1
                    total += float(loss.item()) * len(idx)
                    count += len(idx)
                    if max_steps is not None:
                        max_steps -= 1
                return total / max(count, 1)

            if val_idx is None:
                # too few rows to split: fixed number of steps, no validation
                while self.n_updates_ < self.fallback_steps:
                    self.loss_history_.append(run_epoch(self.fallback_steps - self.n_updates_))
            else:
                steps_per_epoch = max(1, math.ceil(n_train / bs))
                max_updates = max(self.n_epochs * steps_per_epoch, self.min_updates)
                check_every = max(100, steps_per_epoch // 4)
                # the initial weights (raw-ensemble Gaussian) are the first checkpoint
                best_val = self._mean_loss(tensors, val_idx)
                best_state = {k: v.detach().clone() for k, v in self._net.state_dict().items()}
                bad_checks = 0
                run_loss, run_count = 0.0, 0
                done = max_updates <= 0
                while not done:
                    perm = train_idx[torch.randperm(n_train, generator=generator).to(dev)]
                    for start in range(0, n_train, bs):
                        idx = perm[start : start + bs]
                        optimizer.zero_grad()
                        loss = self._batch_loss(tensors, idx).mean()
                        loss.backward()
                        optimizer.step()
                        self.n_updates_ += 1
                        run_loss += float(loss.item()) * len(idx)
                        run_count += len(idx)
                        if self.n_updates_ % check_every == 0 or self.n_updates_ >= max_updates:
                            self.loss_history_.append(run_loss / max(run_count, 1))
                            run_loss, run_count = 0.0, 0
                            val = self._mean_loss(tensors, val_idx)
                            self.val_loss_history_.append(val)
                            if val < best_val:
                                best_val, bad_checks = val, 0
                                best_state = {k: v.detach().clone() for k, v in self._net.state_dict().items()}
                            else:
                                bad_checks += 1
                            if self.n_updates_ >= max_updates or (
                                bad_checks >= self.patience and self.n_updates_ >= self.min_updates
                            ):
                                done = True
                                break
                self._net.load_state_dict(best_state)
            self._net.eval()

        return self

    def predict_quantiles(self, X: dict) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("predict_quantiles called before fit()")

        ens_mean = np.asarray(X["ens_mean"])
        ens_var = np.asarray(X["ens_var"])
        station_id = X["station_id"]
        n_leads = ens_mean.shape[1]
        dev = torch.device(self.device)

        ens_mean_std, ens_var_std = self._standardize(ens_mean, ens_var)

        station_idx_arr = np.array(
            [[self._station_to_idx.get(s, 0)] * n_leads for s in station_id], dtype=np.int64
        )

        lead_idx = None
        if self._lead_to_idx is not None:
            if "step_hours" not in X:
                raise ValueError("this DRN was trained with step_hours; predict_quantiles X must include 'step_hours'")
            sh = np.broadcast_to(np.asarray(X["step_hours"]), ens_mean.shape)
            lead_arr = np.vectorize(lambda v: self._lead_to_idx.get(v, 0), otypes=[np.int64])(sh)
            lead_idx = torch.tensor(lead_arr, dtype=torch.long, device=dev)

        self._net.eval()
        with torch.no_grad():
            mu_std, log_sigma2_std = self._residual_forward(
                torch.tensor(ens_mean_std, dtype=torch.float32, device=dev),
                torch.tensor(ens_var_std, dtype=torch.float32, device=dev),
                torch.tensor(station_idx_arr, dtype=torch.long, device=dev),
                lead_idx,
            )
            mu, sigma = self._unstandardize_output(mu_std, log_sigma2_std)

        mu_np = mu.cpu().numpy()
        sigma_np = sigma.cpu().numpy()
        quantiles = [mu_np + sigma_np * norm.ppf(q) for q in self.quantile_levels]
        return np.stack(quantiles, axis=-1)
