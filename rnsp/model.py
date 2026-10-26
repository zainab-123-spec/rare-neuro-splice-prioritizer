"""Cross-validated splice-variant models and single-rule baselines.

The comparison is deliberate. StratifiedKFold mixes variants from the same
gene into train and test, so a model can memorize gene-specific ClinVar
ascertainment. GroupKFold holds out every variant of a gene together. That is
the number to quote when asking whether the features generalize to a
neurological gene the model has not seen.

Metrics are computed from out-of-fold probabilities. A model refit on the
full cohort is saved only for the explorer and is not the benchmark.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rnsp.config import FEATURE_SETS, MODEL_FEATURES_FULL, PROCESSED_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

MODEL_NAMES = (
    "logistic_regression",
    "hist_gradient_boosting",
    "gradient_boosting",
)
SUBSETS = ("all", "canonical_dinucleotide", "hard_3_50", "deep_gt_50", "criteria_provided")


def make_estimator(name: str) -> Pipeline:
    """Build an unfitted pipeline for one model name."""
    if name == "logistic_regression":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
    if name == "hist_gradient_boosting":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_depth=3,
                        learning_rate=0.08,
                        max_iter=200,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
    if name == "gradient_boosting":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=120,
                        max_depth=3,
                        learning_rate=0.06,
                        subsample=0.9,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
    raise ValueError(f"unknown model {name}")


def balanced_weights(labels: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights that sum, in expectation, like a balanced sample."""
    y = np.asarray(labels).astype(int)
    n_obs = len(y)
    n_pos = max(int(y.sum()), 1)
    n_neg = max(n_obs - int(y.sum()), 1)
    return np.where(y == 1, n_obs / (2 * n_pos), n_obs / (2 * n_neg)).astype(float)


def subset_mask(frame: pd.DataFrame, subset: str) -> np.ndarray:
    """Boolean mask for an evaluation slice."""
    if subset == "all":
        return np.ones(len(frame), dtype=bool)
    if subset == "canonical_dinucleotide":
        return frame["region_class"].eq("canonical_dinucleotide").to_numpy()
    if subset == "hard_3_50":
        return frame["region_class"].eq("hard_3_50").to_numpy()
    if subset == "deep_gt_50":
        return frame["region_class"].eq("deep_gt_50").to_numpy()
    if subset == "criteria_provided":
        return frame["criteria_provided"].eq(1).to_numpy()
    raise ValueError(subset)


def distance_rule_scores(distance: np.ndarray) -> np.ndarray:
    """Higher score means closer to a junction. This rule is not fitted."""
    return 1.0 / (1.0 + np.asarray(distance, dtype=float))


def dinucleotide_rule_scores(flags: np.ndarray) -> np.ndarray:
    """Canonical ±1/±2 rule: 1 if the SNV changes that dinucleotide, else 0."""
    return np.asarray(flags, dtype=float)


def crossval_oof(
    frame: pd.DataFrame,
    feature_names: list[str] | tuple[str, ...],
    model_name: str,
    split: str,
    *,
    n_splits: int = 5,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Out-of-fold positive-class probabilities. Rows that were never tested stay NaN."""
    y = frame["label"].to_numpy(dtype=int)
    x_matrix = frame.loc[:, list(feature_names)].to_numpy(dtype=float)
    oof = np.full(len(frame), np.nan)
    if groups is None:
        groups = frame["gene"].to_numpy()
    splitter = list(_iter_splits(x_matrix, y, groups, split, n_splits))
    for train_idx, test_idx in splitter:
        if len(np.unique(y[train_idx])) < 2:
            continue
        estimator = make_estimator(model_name)
        fit_kwargs: dict[str, Any] = {}
        if model_name != "logistic_regression":
            fit_kwargs["model__sample_weight"] = balanced_weights(y[train_idx])
        estimator.fit(x_matrix[train_idx], y[train_idx], **fit_kwargs)
        oof[test_idx] = estimator.predict_proba(x_matrix[test_idx])[:, 1]
    return oof


def _iter_splits(
    x_matrix: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    split: str,
    n_splits: int,
) -> Any:
    if split == "group_kfold_gene":
        n_groups = len(np.unique(groups))
        folds = min(n_splits, n_groups)
        if folds < 2:
            return
        yield from GroupKFold(n_splits=folds).split(x_matrix, y, groups)
        return
    if split == "stratified_kfold":
        n_pos = int(y.sum())
        n_neg = int(len(y) - y.sum())
        folds = min(n_splits, n_pos, n_neg)
        if folds < 2:
            return
        yield from StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED).split(
            x_matrix, y
        )
        return
    raise ValueError(split)


def summarize_predictions(
    y: np.ndarray,
    scores: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = RANDOM_SEED,
    min_per_class: int = 8,
) -> dict[str, Any] | None:
    """ROC-AUC, PR-AUC, and Brier score, with percentile bootstrap intervals."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    y = y[valid]
    scores = scores[valid]
    n_pos = int(y.sum()) if len(y) else 0
    n_neg = int(len(y) - n_pos)
    if n_pos < min_per_class or n_neg < min_per_class:
        return None
    roc = float(roc_auc_score(y, scores))
    pr = float(average_precision_score(y, scores))
    # Brier needs scores in [0, 1]. Ranking-only scores are clipped for this metric
    # and the clipping is recorded so a reader can see it.
    clipped = False
    probabilities = scores
    if np.nanmin(scores) < 0 or np.nanmax(scores) > 1:
        probabilities = np.clip(scores, 0, 1)
        clipped = True
    brier = float(brier_score_loss(y, probabilities))
    roc_ci = _bootstrap(y, scores, roc_auc_score, n_bootstrap, seed)
    pr_ci = _bootstrap(y, scores, average_precision_score, n_bootstrap, seed + 1)
    return {
        "n": int(len(y)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "roc_auc": roc,
        "pr_auc": pr,
        "brier": brier,
        "roc_auc_ci95": roc_ci,
        "pr_auc_ci95": pr_ci,
        "brier_scores_clipped": clipped,
    }


def _bootstrap(
    y: np.ndarray,
    scores: np.ndarray,
    metric: Any,
    n_bootstrap: int,
    seed: int,
) -> list[float] | None:
    if n_bootstrap <= 0:
        return None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n_obs = len(y)
    for _ in range(n_bootstrap):
        index = rng.integers(0, n_obs, n_obs)
        if len(np.unique(y[index])) < 2:
            continue
        values.append(float(metric(y[index], scores[index])))
    if len(values) < max(20, n_bootstrap // 5):
        return None
    low, high = np.quantile(values, [0.025, 0.975])
    return [float(low), float(high)]


def evaluate(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_bootstrap: int = 1000,
    models: tuple[str, ...] = MODEL_NAMES,
    feature_sets: tuple[str, ...] = ("full", "mechanism", "motif"),
) -> tuple[dict[str, Any], pd.DataFrame, Pipeline]:
    """Run the benchmark and attach out-of-fold scores to a copy of ``frame``."""
    missing = [name for name in MODEL_FEATURES_FULL if name not in frame.columns]
    if missing:
        raise ValueError(f"variant table is missing features: {missing}")
    scored = frame.reset_index(drop=True).copy()
    y = scored["label"].to_numpy(dtype=int)
    evaluations: list[dict[str, Any]] = []
    oof_cache: dict[tuple[str, str, str], np.ndarray] = {}

    for model_name in models:
        for feature_set in feature_sets:
            for split in ("group_kfold_gene", "stratified_kfold"):
                logger.info("CV %s / %s / %s", model_name, feature_set, split)
                oof = crossval_oof(
                    scored, FEATURE_SETS[feature_set], model_name, split, n_splits=n_splits
                )
                oof_cache[(model_name, feature_set, split)] = oof
                _store_oof(scored, oof, f"oof_{_short(model_name)}_{_short(split)}_{feature_set}")
                _collect_subset_metrics(
                    evaluations,
                    scored,
                    y,
                    oof,
                    model=model_name,
                    feature_set=feature_set,
                    split=split,
                    protocol="train_all_eval_subset",
                    n_bootstrap=n_bootstrap,
                )

    _hard_refits(scored, evaluations, models, feature_sets, n_splits, n_bootstrap)
    _baselines(scored, evaluations, n_splits, n_bootstrap)

    coefficients = _logistic_coefficients(scored)
    importance = _permutation_importance(scored, n_splits=n_splits)
    deployment_name = _choose_deployment(evaluations)
    deployment = _fit_full(scored, deployment_name, FEATURE_SETS["full"])
    scored["deployment_model"] = deployment_name
    scored["deployment_score"] = deployment.predict_proba(
        scored.loc[:, list(FEATURE_SETS["full"])].to_numpy(dtype=float)
    )[:, 1]

    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_variants": int(len(scored)),
        "n_genes": int(scored["gene"].nunique()),
        "n_pathogenic": int(y.sum()),
        "n_benign": int((y == 0).sum()),
        "prevalence": float(y.mean()),
        "n_splits_requested": n_splits,
        "feature_sets": {name: list(columns) for name, columns in FEATURE_SETS.items()},
        "evaluations": evaluations,
        "logistic_coefficients": coefficients,
        "permutation_importance": importance,
        "deployment_model": deployment_name,
        "deployment_features": list(FEATURE_SETS["full"]),
        "notes": [
            "ROC-AUC, PR-AUC, and Brier score are computed from out-of-fold predictions.",
            "GroupKFold holds out every variant of a gene. StratifiedKFold does not.",
            "The canonical dinucleotide rule and the distance rule are not trained.",
            "deployment_score is a refit on the full research cohort and is not the benchmark.",
            "Gene-level expression features can proxy ascertainment. Compare the mechanism and motif feature sets.",
        ],
    }
    payload["calibration"] = _calibration_payload(scored)
    return payload, scored, deployment


def save_evaluation(
    payload: dict[str, Any],
    scored: pd.DataFrame,
    estimator: Pipeline,
    out_dir: Path | None = None,
) -> Path:
    """Write metrics, scored variants, and the deployment model."""
    destination = Path(out_dir) if out_dir else PROCESSED_DIR
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    scored.to_csv(destination / "variants_scored.csv", index=False)
    joblib.dump(
        {
            "estimator": estimator,
            "model_name": payload["deployment_model"],
            "feature_names": payload["deployment_features"],
        },
        destination / "model.joblib",
    )
    return destination / "metrics.json"


def load_metrics(path: Path | None = None) -> dict[str, Any]:
    """Read ``metrics.json`` produced by :func:`save_evaluation`."""
    metrics_path = path or (PROCESSED_DIR / "metrics.json")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _hard_refits(
    scored: pd.DataFrame,
    evaluations: list[dict[str, Any]],
    models: tuple[str, ...],
    feature_sets: tuple[str, ...],
    n_splits: int,
    n_bootstrap: int,
) -> None:
    """Retrain inside the ±3..±50 bp slice so canonical sites cannot dominate training."""
    mask = subset_mask(scored, "hard_3_50")
    hard = scored.loc[mask].reset_index(drop=True)
    if hard["gene"].nunique() < 5 or hard["label"].nunique() < 2:
        evaluations.append(
            {
                "model": "all",
                "feature_set": "all",
                "split": "group_kfold_gene",
                "subset": "hard_3_50",
                "protocol": "train_on_subset",
                "skipped": True,
                "reason": "hard subset does not have 5 genes and both classes",
                "n": int(len(hard)),
            }
        )
        return
    y = hard["label"].to_numpy(dtype=int)
    for model_name in models:
        for feature_set in feature_sets:
            for split in ("group_kfold_gene", "stratified_kfold"):
                oof = crossval_oof(
                    hard, FEATURE_SETS[feature_set], model_name, split, n_splits=n_splits
                )
                if (
                    model_name == "hist_gradient_boosting"
                    and feature_set == "motif"
                    and split == "group_kfold_gene"
                ):
                    column = np.full(len(scored), np.nan)
                    column[np.flatnonzero(mask)] = oof
                    scored["oof_hgb_group_motif_hard"] = column
                summary = summarize_predictions(y, oof, n_bootstrap=n_bootstrap)
                record: dict[str, Any] = {
                    "model": model_name,
                    "feature_set": feature_set,
                    "split": split,
                    "subset": "hard_3_50",
                    "protocol": "train_on_subset",
                }
                if summary is None:
                    record["skipped"] = True
                    record["reason"] = "too few variants in one class after out-of-fold scoring"
                else:
                    record.update(summary)
                evaluations.append(record)


def _baselines(
    scored: pd.DataFrame,
    evaluations: list[dict[str, Any]],
    n_splits: int,
    n_bootstrap: int,
) -> None:
    y = scored["label"].to_numpy(dtype=int)
    distance = distance_rule_scores(scored["abs_dist_to_splice"].to_numpy(dtype=float))
    dinucleotide = dinucleotide_rule_scores(
        scored["disrupts_canonical_dinucleotide"].to_numpy(dtype=float)
    )
    scored["oof_distance_rule"] = distance
    scored["oof_dinucleotide_rule"] = dinucleotide
    for name, scores in (
        ("distance_rule", distance),
        ("dinucleotide_rule", dinucleotide),
    ):
        _collect_subset_metrics(
            evaluations,
            scored,
            y,
            scores,
            model=name,
            feature_set="rule",
            split="not_applicable",
            protocol="fixed_rule",
            n_bootstrap=n_bootstrap,
        )
    for split in ("group_kfold_gene", "stratified_kfold"):
        oof = crossval_oof(
            scored, ["abs_dist_to_splice"], "logistic_regression", split, n_splits=n_splits
        )
        scored[f"oof_distance_logistic_{_short(split)}"] = oof
        _collect_subset_metrics(
            evaluations,
            scored,
            y,
            oof,
            model="distance_logistic",
            feature_set="abs_dist_to_splice",
            split=split,
            protocol="train_all_eval_subset",
            n_bootstrap=n_bootstrap,
        )


def _collect_subset_metrics(
    evaluations: list[dict[str, Any]],
    scored: pd.DataFrame,
    y: np.ndarray,
    scores: np.ndarray,
    *,
    model: str,
    feature_set: str,
    split: str,
    protocol: str,
    n_bootstrap: int,
) -> None:
    for subset in SUBSETS:
        mask = subset_mask(scored, subset)
        summary = summarize_predictions(y[mask], scores[mask], n_bootstrap=n_bootstrap)
        record: dict[str, Any] = {
            "model": model,
            "feature_set": feature_set,
            "split": split,
            "subset": subset,
            "protocol": protocol,
        }
        if summary is None:
            record["skipped"] = True
            record["n"] = int(mask.sum())
            record["reason"] = "fewer than 8 variants in one class, or no finite scores"
        else:
            record.update(summary)
        evaluations.append(record)


def _logistic_coefficients(frame: pd.DataFrame) -> list[dict[str, float | str]]:
    estimator = _fit_full(frame, "logistic_regression", FEATURE_SETS["full"])
    coef = estimator.named_steps["model"].coef_.ravel()
    names = list(FEATURE_SETS["full"])
    order = np.argsort(np.abs(coef))[::-1]
    return [{"feature": names[index], "coefficient": float(coef[index])} for index in order]


def _permutation_importance(
    frame: pd.DataFrame, *, n_splits: int, n_repeats: int = 5
) -> list[dict[str, float | str]]:
    """Mean drop in test ROC-AUC when one feature is shuffled inside each held-out gene fold."""
    y = frame["label"].to_numpy(dtype=int)
    names = list(FEATURE_SETS["full"])
    x_matrix = frame.loc[:, names].to_numpy(dtype=float)
    groups = frame["gene"].to_numpy()
    drops = np.zeros((len(names),))
    counts = 0
    rng = np.random.default_rng(RANDOM_SEED)
    for train_idx, test_idx in _iter_splits(x_matrix, y, groups, "group_kfold_gene", n_splits):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        estimator = make_estimator("hist_gradient_boosting")
        estimator.fit(
            x_matrix[train_idx],
            y[train_idx],
            model__sample_weight=balanced_weights(y[train_idx]),
        )
        baseline = roc_auc_score(y[test_idx], estimator.predict_proba(x_matrix[test_idx])[:, 1])
        for feature_index in range(len(names)):
            losses: list[float] = []
            for _ in range(n_repeats):
                shuffled = x_matrix[test_idx].copy()
                shuffled[:, feature_index] = rng.permutation(shuffled[:, feature_index])
                score = roc_auc_score(y[test_idx], estimator.predict_proba(shuffled)[:, 1])
                losses.append(baseline - score)
            drops[feature_index] += float(np.mean(losses))
        counts += 1
    if counts == 0:
        return []
    drops /= counts
    order = np.argsort(drops)[::-1]
    return [{"feature": names[index], "mean_auc_drop": float(drops[index])} for index in order]


def _fit_full(
    frame: pd.DataFrame, model_name: str, feature_names: tuple[str, ...] | list[str]
) -> Pipeline:
    estimator = make_estimator(model_name)
    y = frame["label"].to_numpy(dtype=int)
    x_matrix = frame.loc[:, list(feature_names)].to_numpy(dtype=float)
    fit_kwargs: dict[str, Any] = {}
    if model_name != "logistic_regression":
        fit_kwargs["model__sample_weight"] = balanced_weights(y)
    estimator.fit(x_matrix, y, **fit_kwargs)
    return estimator


def _choose_deployment(evaluations: list[dict[str, Any]]) -> str:
    candidates = [
        row
        for row in evaluations
        if row.get("protocol") == "train_all_eval_subset"
        and row.get("feature_set") == "full"
        and row.get("split") == "group_kfold_gene"
        and row.get("subset") == "all"
        and row.get("model") in MODEL_NAMES
        and "roc_auc" in row
    ]
    if not candidates:
        return "hist_gradient_boosting"
    best = max(candidates, key=lambda row: float(row["roc_auc"]))
    return str(best["model"])


def _calibration_payload(frame: pd.DataFrame) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    y = frame["label"].to_numpy(dtype=int)
    for column, key in (
        ("oof_hgb_group_full", "hist_gradient_boosting_group_full"),
        ("oof_lr_group_full", "logistic_regression_group_full"),
    ):
        if column not in frame.columns:
            continue
        scores = frame[column].to_numpy(dtype=float)
        valid = np.isfinite(scores)
        if valid.sum() < 20 or len(np.unique(y[valid])) < 2:
            continue
        fraction, mean_pred = calibration_curve(
            y[valid], scores[valid], n_bins=8, strategy="quantile"
        )
        payload[key] = {
            "fraction_positive": [float(value) for value in fraction],
            "mean_predicted": [float(value) for value in mean_pred],
        }
    return payload


def _store_oof(frame: pd.DataFrame, values: np.ndarray, column: str) -> None:
    # Keep the curves used in the figures. Other combinations stay in metrics.json.
    keep = column.endswith("_full") or column.endswith("_motif")
    if keep and ("group" in column or "stratified" in column):
        frame[column] = values


def _short(name: str) -> str:
    return {
        "logistic_regression": "lr",
        "hist_gradient_boosting": "hgb",
        "gradient_boosting": "gbm",
        "group_kfold_gene": "group",
        "stratified_kfold": "stratified",
    }.get(name, name)


def load_model(path: Path | None = None) -> dict[str, Any]:
    """Load the deployment bundle written by :func:`save_evaluation`."""
    model_path = path or (PROCESSED_DIR / "model.joblib")
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "estimator" not in bundle:
        raise ValueError(f"{model_path} is not a deployment bundle")
    return bundle


def metric_lookup(
    metrics: dict[str, Any],
    *,
    model: str,
    feature_set: str,
    split: str,
    subset: str,
    protocol: str,
) -> dict[str, Any] | None:
    """Find one evaluation row. Returns None when that slice was not run."""
    for row in metrics.get("evaluations", []):
        if (
            row.get("model") == model
            and row.get("feature_set") == feature_set
            and row.get("split") == split
            and row.get("subset") == subset
            and row.get("protocol") == protocol
        ):
            return row
    return None


def format_metric(row: dict[str, Any] | None, key: str = "roc_auc") -> str:
    """Format a metric for a table. Skipped slices stay blank rather than inventing a number."""
    if row is None or row.get("skipped") or key not in row:
        return "—"
    value = float(row[key])
    interval = row.get(f"{key}_ci95")
    if key == "brier" or not interval:
        return f"{value:.3f}"
    return f"{value:.3f} ({interval[0]:.3f}–{interval[1]:.3f})"
