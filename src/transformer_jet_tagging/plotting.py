"""
plotting.py
===========
Visualization module for the GN2 jet flavour tagging pipeline.

Generates plots of input and output variables,
reading directly from the HDF5 file and/or a DataLoader.
"""

import logging
from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from sklearn.metrics import confusion_matrix

from ._constants import FLAVOUR_COLORS, FLAVOUR_LABELS

matplotlib.use("Agg")
hep.style.use(hep.style.ATLAS)
logger = logging.getLogger("GN2.plotting  ")

ATLAS_RLABEL = r"$\sqrt{s}=13.6$ TeV, $t\bar{t}$ simulation" + "\n" + r"$20<p_T<250$ GeV," \
               r"$\left|\eta\right|<2.5$"

def _load_jet_data(
    h5_path: str | Path,
    jet_indices: np.ndarray,
    jet_vars: list[str],
    jet_flavour: str,
    jet_flavour_map: dict[int, int],
) -> dict[str, np.ndarray]:
    """
    Load jet-level variables and labels from HDF5 for a subset of jets.

    Args:
        h5_path (str | Path): Path to HDF5 file.
        jet_indices (np.ndarray): Sorted jet indices to read.
        jet_vars (list): Jet variable names.
        jet_flavour (str): Name of the flavour field in HDF5.
        jet_flavour_map (dict): Raw label - class index mapping.

    Returns:
        dict:
            var_name (np.ndarray): shape ``(n_jets,)`` for each jet variable.
            "label" (np.ndarray): shape ``(n_jets,)`` integer class index for each jet.

    Raises:
        FileNotFoundError: if the specified file does not exist.
        KeyError: if expected datasets or fields are missing in the HDF5 file.
    """
    logger.debug("Loading jet data for %s jets ...", f"{len(jet_indices):,}")

    h5_path = Path(h5_path)
    sorted_idx = np.sort(jet_indices)
    data: dict[str, np.ndarray] = {}

    try:
        with h5py.File(h5_path, "r") as f:
            jets = f["jets"][sorted_idx]
            for var in jet_vars:
                data[var] = jets[var].astype(np.float32)
            raw_labels = jets[jet_flavour].astype(int)
            data["label"] = np.array(
                [jet_flavour_map.get(label, 0) for label in raw_labels], dtype=np.int32
            )
    except FileNotFoundError:
        logger.error("HDF5 file not found: %s", h5_path)
        raise
    except KeyError as e:
        logger.error("Missing dataset in HDF5 file: %s", e)
        raise

    return data


def _load_track_data(
    h5_path: str | Path,
    jet_indices: np.ndarray,
    track_vars: list[str],
    jet_flavour: str,
    jet_flavour_map: dict[int, int],
    max_jets: int = 50_000,
) -> dict[str, np.ndarray]:
    """
    Load valid track-level variables from HDF5, flattened across jets.

    Args:
        h5_path (str | Path): Path to HDF5 file.
        jet_indices (np.ndarray): Sorted jet indices.
        track_vars (list): Track variable names.
        jet_flavour (str): Flavour field name.
        jet_flavour_map (dict): Raw label - class index.
        max_jets (int): Cap on jets to read (memory guard).

    Returns:
        dict:
            var_name (np.ndarray): shape ``(n_tracks,)`` for each track variable.
            "label" (np.ndarray): shape ``(n_tracks,)`` integer class index for each track's jet.
        
    Raises:
        FileNotFoundError: if the specified file does not exist.
        KeyError: if expected datasets or fields are missing in the HDF5 file.
    """
    logger.debug("Loading track data (up to %s jets) ...", f"{max_jets:,}")

    h5_path = Path(h5_path)
    sorted_idx = np.sort(jet_indices[:max_jets])
    data_lists: dict[str, list] = {v: [] for v in track_vars}
    data_lists["label"] = []

    try:
        with h5py.File(h5_path, "r") as f:
            jets_raw   = f["jets"][sorted_idx]
            tracks_raw = f["tracks"][sorted_idx]
            raw_labels = jets_raw[jet_flavour].astype(int)
            labels     = np.array(
                [jet_flavour_map.get(label, 0) for label in raw_labels], dtype=np.int32
            )

            has_valid = "valid" in tracks_raw.dtype.names
            for i in range(len(sorted_idx)):
                if has_valid:
                    valid_mask = tracks_raw["valid"][i].astype(bool)
                else:
                    valid_mask = np.ones(tracks_raw.shape[1], dtype=bool)

                n = valid_mask.sum()
                if n == 0:
                    continue
                for var in track_vars:
                    data_lists[var].append(tracks_raw[var][i][valid_mask].astype(np.float32))
                data_lists["label"].append(np.full(n, labels[i], dtype=np.int32))
    except FileNotFoundError:
        logger.error("HDF5 file not found: %s", h5_path)
        raise
    except KeyError as e:
        logger.error("Missing dataset in HDF5 file: %s", e)
        raise

    return {
        k: np.concatenate(v) if len(v) > 0 else np.array(
            [], dtype=np.int32 if k == "label" else np.float32
        )for k, v in data_lists.items()
    }


def plot_jet_variables(
    jet_data: dict[str, np.ndarray],
    jet_vars: list[str],
    output_dir: str | Path,
) -> None:
    """
    Plot per-flavour distributions of jet-level variables.
    pt is shown both raw (linear) and log-transformed.

    Args:
        jet_data (dict): Output of _load_jet_data().
        jet_vars (list): Variable names to plot.
        output_dir (str | Path): Directory where PNGs are saved.
    """
    logger.info("Plotting jet variables ...")
    output_dir = Path(output_dir)
    labels  = jet_data["label"]
    classes = sorted(FLAVOUR_LABELS.keys())

    # build plot list: for pt add a log version
    plot_list = []
    for var in jet_vars:
        plot_list.append((var, False))
        if var == "pt":
            plot_list.append(("pt", True))

    n_cols = 2
    n_rows = (len(plot_list) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, (var, do_log) in zip(axes, plot_list, strict=False):
        values = jet_data[var].copy()
        labels_var = labels.copy()

        valid = np.isfinite(values)
        values     = values[valid]
        labels_var = labels_var[valid]

        if do_log:
            values = np.log(np.clip(values, 1e-6, None))
            finite = np.isfinite(values)
            values     = values[finite]
            labels_var = labels_var[finite]

        if values.size == 0:
            continue

        lo, hi = np.min(values), np.max(values)
        bins = np.linspace(lo, hi, 60)

        for cls in classes:
            mask = labels_var == cls
            ax.hist(
                values[mask],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.5,
                color=FLAVOUR_COLORS[cls],
                label=FLAVOUR_LABELS[cls],
            )

        ax.set_xlabel(var,loc='center',fontsize=14)
        ax.set_ylabel("Entries")
        ax.legend()
        hep.atlas.label(ax=ax, data=True, loc=1, rlabel=ATLAS_RLABEL)

    # hide unused axes
    for ax in axes[len(plot_list):]:
        ax.set_visible(False)

    fig.tight_layout()
    out = output_dir / "jet_variables.pdf"
    fig.savefig(out)
    plt.close(fig)

    logger.info("Saved: %s", out)


def plot_track_variables(
    track_data: dict[str, np.ndarray],
    track_vars: list[str],
    output_dir: str | Path,
    vars_per_page: int = 6,
) -> None:
    """
    Plot per-flavour distributions of track-level variables.
    Variables are split across multiple pages if needed.

    Args:
        track_data (dict): Output of ``_load_track_data()``.
        track_vars (list): Variable names to plot.
        output_dir (str | Path): Directory where PNGs are saved.
        vars_per_page (int): Max variables per figure (default ``6``).
    """
    logger.info("Plotting track variables ...")
    output_dir = Path(output_dir)
    classes = sorted(FLAVOUR_LABELS.keys())

    for page, start in enumerate(range(0, len(track_vars), vars_per_page)):
        page_vars = track_vars[start : start + vars_per_page]
        n_cols = 3
        n_rows = (len(page_vars) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(-1)

        for ax, var in zip(axes, page_vars, strict=False):
            values = track_data[var]
            labels_var = track_data["label"].copy()

            valid_mask = np.isfinite(values)
            values     = values[valid_mask]
            labels_var = labels_var[valid_mask]

            if values.size == 0:
                continue
            lo, hi = np.min(values), np.max(values)
            bins = np.linspace(lo, hi, 60)

            for cls in classes:
                mask = labels_var == cls
                ax.hist(
                    values[mask],
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=1.5,
                    color=FLAVOUR_COLORS[cls],
                    label=FLAVOUR_LABELS[cls],
                )

            ax.set_xlabel(var,loc='center',fontsize=14)
            ax.set_ylabel("Entries")
            ax.set_yscale("log")
            ax.legend()
            hep.atlas.label(ax=ax, data=True, loc=1, rlabel=ATLAS_RLABEL)

        for ax in axes[len(page_vars):]:
            ax.set_visible(False)

        fig.tight_layout()
        out = output_dir / f"track_variables_page{page + 1}.pdf"
        fig.savefig(out)
        plt.close(fig)

        logger.info("Saved: %s", out)


def plot_label_distribution(
    labels: np.ndarray,
    output_dir: str | Path,
) -> None:
    """
    Plot the class-label distribution for a dataset split (train / val / test).

    Unmapped jets (label ``-1``) are silently ignored.

    Args:
        labels (np.ndarray): integer array of class indices.
            Jets with label ``-1`` are dropped before plotting.
        output_dir (str | Path): directory where the PDF is saved.

    Raises:
        ValueError: if *labels* is not a 1-D array.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Plotting label distribution ...")

    labels = np.asarray(labels).ravel()

    if labels.ndim != 1:
        raise ValueError("`labels` must be a 1-D array.")

    # drop unmapped jets
    labels = labels[labels != -1]
    if labels.size == 0:
        logger.warning(
            "plot_label_distribution: all labels are -1, nothing to plot."
        )
        return

    classes = sorted(np.unique(labels).tolist())
    counts  = np.array([int((labels == c).sum()) for c in classes])
    total   = counts.sum()
    names   = [FLAVOUR_LABELS.get(c, f"unknown ({c})") for c in classes]
    colors  = [FLAVOUR_COLORS.get(c, "grey")           for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bars = ax.bar(names, counts, color=colors, edgecolor="black", linewidth=0.8)
    for num_bar, count in zip(bars, counts, strict=False):
        pct = 100.0 * count / total
        ax.text(
            num_bar.get_x() + num_bar.get_width() / 2.,
            num_bar.get_height() * 1.01,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=11,
        )
    ax.set_xlabel("Jet flavour", fontsize=14)
    ax.set_ylabel("Number of jets", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    hep.atlas.label(ax=ax, data=True, loc=2, rlabel=ATLAS_RLABEL)

    ax = axes[1]
    _, _, autotexts = ax.pie(
        counts,
        labels     = names,
        colors     = colors,
        autopct    = "%1.1f%%",
        startangle = 90,
        textprops  = dict(fontsize=11),
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title("Class fractions", fontsize=14)

    fig.tight_layout()
    out = output_dir / "label_distribution.pdf"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved: %s", out)


def _corr_matrix(data_dict, vars_list):
    """
    Compute correlation matrix for the specified variables.
    (Non-finite values are replaced with the column mean before correlation)

    Args:
        data_dict (dict): dict of variable name to ``np.ndarray``.
        vars_list (list): list of variable names to include in the matrix.
    
    Returns:
        np.ndarray: shape ``(len(vars_list), len(vars_list))``, correlation matrix.
    """
    mat = np.column_stack([data_dict[v].astype(np.float32) for v in vars_list])
    # replace inf/nan with column mean
    col_means = np.nanmean(mat, axis=0)
    inds = np.where(~np.isfinite(mat))
    mat[inds] = col_means[inds[1]]
    return np.corrcoef(mat, rowvar=False)


def _draw_heatmap(ax, corr, labels, title):
    """
    Draw a heatmap of the correlation matrix with annotations.

    Args:
        ax: matplotlib axis to draw on.
        corr: 2D array of correlation coefficients.
        labels: list of variable names for axes.
        title: title of the plot.
    
    Returns:
        im: image object from imshow (for colorbar).
    """
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=11)
    # annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)
    return im


def plot_correlations(
    jet_data: dict[str, np.ndarray],
    track_data: dict[str, np.ndarray],
    jet_vars: list[str],
    track_vars: list[str],
    output_dir: str | Path,
) -> None:
    """
    Plot Pearson correlation matrices for jet and track variables.

    Args:
        jet_data (dict): Output of ``_load_jet_data()``.
        track_data (dict): Output of ``_load_track_data()``.
        jet_vars (list): Jet variable names.
        track_vars (list): Track variable names.
        output_dir (str | Path): Directory where PNGs are saved.
    """
    logger.info("Plotting correlations ...")
    output_dir = Path(output_dir)

    # jet correlation
    if len(jet_vars) >= 2:
        # add log_pt as an extra column
        jet_data_ext = dict(jet_data)
        jet_data_ext["log(pt)"] = np.log(np.clip(jet_data["pt"], 1e-6, None))
        jet_vars_ext = ["log(pt)"] + [v for v in jet_vars if v != "pt"]

        corr_jet = _corr_matrix(jet_data_ext, jet_vars_ext)
        fig, ax = plt.subplots(figsize=(max(5, len(jet_vars_ext)), max(4, len(jet_vars_ext))))
        im = _draw_heatmap(ax, corr_jet, jet_vars_ext, "Jet variables - Correlation")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out = output_dir / "correlation_jet.pdf"
        fig.savefig(out)
        plt.close(fig)

        logger.info("Saved: %s", out)

    # track correlation
    if len(track_vars) >= 2:
        corr_track = _corr_matrix(track_data, track_vars)
        n = len(track_vars)
        fig, ax = plt.subplots(figsize=(max(8, n * 0.55), max(7, n * 0.55)))
        im = _draw_heatmap(ax, corr_track, track_vars, "Track variables - Correlation")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out = output_dir / "correlation_track.pdf"
        fig.savefig(out)
        plt.close(fig)

        logger.info("Saved: %s", out)


def plot_statistics(
    h5_path: str | Path,
    jet_vars: list[str],
    track_vars: list[str],
    jet_flavour: str,
    jet_flavour_map: dict[int, int],
    jet_indices: np.ndarray,
    output_dir: str = "outputs/plots",
    n_jets_track: int = 50_000,
) -> None:
    """
    Generate all plots and save them to ``plot_dir``.

    Args:
        h5_path (str | Path): Path to HDF5 file.
        jet_vars (list): Jet variable names.
        track_vars (list): Track variable names.
        jet_flavour (str): Flavour field name in HDF5.
        jet_flavour_map (dict): Raw label - class index.
        jet_indices (np.ndarray): Jet indices to use (e.g. ``train_indices``).
        output_dir (str): Directory for output PNGs.
        n_jets_track (int): Max jets for track plots (memory guard).
    """
    h5_path = Path(h5_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jet_data = _load_jet_data(h5_path, jet_indices, jet_vars, jet_flavour, jet_flavour_map)

    track_data = _load_track_data(h5_path, jet_indices, track_vars, jet_flavour,
                                  jet_flavour_map, max_jets = n_jets_track)

    plot_jet_variables(jet_data, jet_vars, output_dir)

    plot_track_variables(track_data, track_vars, output_dir)

    plot_label_distribution(jet_data["label"], output_dir)

    plot_correlations(jet_data, track_data, jet_vars, track_vars, output_dir)

    logger.info("All plots saved to '%s/'", output_dir)


def plot_learning_curves(
    history: dict[str, list[float]],
    output_dir: str | Path,
) -> None:
    """
    Plot training and validation loss curves + LR schedule.

    Args:
        history (dict): keys ``"train_loss"``, ``"val_loss"``, ``"lr"``.
        output_dir (str | Path): Directory where the PDF is saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Plotting learning curve ...")

    with plt.rc_context({"axes.autolimit_mode": "data"}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # loss
        ax = axes[0]
        ax.plot(history["train_loss"], c='r', ls='-', label="Train", linewidth=1.5)
        ax.plot(history["val_loss"], c='b', ls='--', label="Validation", linewidth=1.5)
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel("Loss (CE)", fontsize=14)
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        hep.atlas.label(ax=ax, data=False, loc=2, rlabel=ATLAS_RLABEL)

        # lr
        ax = axes[1]
        ax.plot(history["lr"], color="darkorange", linewidth=1.5)
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel("Learning Rate", fontsize=14)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        hep.atlas.label(ax=ax, data=False, loc=2, rlabel=ATLAS_RLABEL)

        fig.tight_layout()
        out = output_dir / "learning_curves.pdf"
        fig.savefig(out)
        plt.close(fig)

    logger.info("Saved: %s", out)


def plot_score_distributions(
    proba: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path,
) -> None:
    """
    Plot softmax score distributions for each output node.

    One figure is produced with one panel per class (P_b, P_c, P_u, P_tau).
    Inside each panel, the distribution is shown separately for every
    true-label class, allowing direct reading of signal/background separation.

    Args:
        proba (np.ndarray): shape ``(N, n_classes)``, softmax probabilities.
        labels (np.ndarray): shape ``(N,)``, true class labels.
        output_dir (str | Path): directory where the PDF is saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Plotting output score distributions ...")

    n_classes = proba.shape[1]
    classes   = sorted(FLAVOUR_LABELS.keys())
    bins      = np.linspace(0, 1, 50)

    nrows = 2
    ncols = int(n_classes / nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten()
    if n_classes == 1:
        axes = [axes]

    for label_idx, ax in enumerate(axes):
        label_name = FLAVOUR_LABELS.get(label_idx, f"class {label_idx}")
        for cls in classes:
            mask = labels == cls
            if mask.sum() == 0:
                continue
            ax.hist(
                proba[mask, label_idx],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.5,
                color=FLAVOUR_COLORS[cls],
                label=FLAVOUR_LABELS[cls],
            )
        ax.set_xlabel(f"P({label_name})", fontsize=14)
        ax.set_ylabel("Normalised entries", fontsize=14)
        ax.set_yscale("log")
        ax.legend()
        hep.atlas.label(ax=ax, data=True, loc=1, rlabel=ATLAS_RLABEL)

    fig.tight_layout()
    out = output_dir / "score_distributions.pdf"
    fig.savefig(out)
    plt.close(fig)

    logger.info("Saved: %s", out)


def plot_discriminant(
    discriminant_scores: np.ndarray,
    labels: np.ndarray,
    discriminant_type: str,
    output_dir: str | Path,
) -> None:
    """
    Plot the distribution of a discriminant (D_b or D_c) per flavour class.

    Args:
        discriminant_scores (np.ndarray): discriminant values for all jets.
        labels (np.ndarray): true class labels.
        discriminant_type (str): name of the discriminant (e.g. "b" or "c", used for axis
            label and filename).
        output_dir (str | Path): output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Plotting discriminant distributions ...")

    classes = sorted(FLAVOUR_LABELS.keys())
    finite = np.isfinite(discriminant_scores)
    discriminant_scores = discriminant_scores[finite]
    labels = labels[finite]

    lo, hi = np.min(discriminant_scores), np.max(discriminant_scores)
    bins = np.linspace(lo, hi, 60)

    fig, ax = plt.subplots(figsize=(7, 5))
    for cls in classes:
        mask = labels == cls
        if mask.sum() == 0:
            continue
        ax.hist(
            discriminant_scores[mask],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            color=FLAVOUR_COLORS[cls],
            label=FLAVOUR_LABELS[cls],
        )
    ax.set_xlabel(f"D_{discriminant_type}", fontsize=14)
    ax.set_ylabel("Normalised entries", fontsize=14)
    ax.set_yscale("log")
    ax.legend()
    hep.atlas.label(ax=ax, data=True, loc=1, rlabel=ATLAS_RLABEL)
    fig.tight_layout()
    out = output_dir / f"discriminant_d{discriminant_type}.pdf"
    fig.savefig(out)
    plt.close(fig)

    logger.info("Saved: %s", out)


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    output_dir: str | Path,
) -> None:
    """
    Plot and save a normalised confusion matrix.

    Args:
        labels (np.ndarray): true class labels.
        preds (np.ndarray): predicted class labels.
        output_dir (str | Path): output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Plotting confusion matrix ...")

    classes     = sorted(FLAVOUR_LABELS.keys())
    class_names = [FLAVOUR_LABELS[c] for c in classes]
    conf_mat    = confusion_matrix(labels, preds, labels=classes, normalize="true")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf_mat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted class", fontsize=14)
    ax.set_ylabel("True class", fontsize=14)

    # write values in cells (white text for high values, black for low)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val   = conf_mat[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, color=color)

    hep.atlas.label(ax=ax, data=True, loc=1, rlabel=ATLAS_RLABEL)
    fig.tight_layout()
    out = output_dir / "confusion_matrix.pdf"
    fig.savefig(out)
    plt.close(fig)

    logger.info("Saved: %s", out)


def _roc_rejection(
    scores: np.ndarray,
    labels: np.ndarray,
    signal_class: int,
    bg_class: int,
    n_points: int = 200,
    eff_range: list = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate signal efficiency and background rejection for ROC curve.

    Args:
        scores (np.ndarray): Discriminant scores for all jets.
        labels (np.ndarray): True class labels for all jets.
        signal_class (int): Class index of the signal (e.g. ``b-jets``).
        bg_class (int): Class index of the background (e.g. ``c-jets``).
        n_points (int): Number of points on the ROC curve.
        eff_range (list): Minimum and maximum signal efficiency to include in the output
            (default ``[0.5, 1.0]``).

    Returns:
        eff (np.ndarray): Signal efficiency values.
        rej (np.ndarray): Background rejection values (``1 / bg efficiency``).
    """
    if eff_range is None:
        eff_range = [0.5, 1.0]
    thresholds = np.linspace(scores.min(), scores.max(), n_points)
    total_sig  = (labels == signal_class).sum()
    total_bg   = (labels == bg_class).sum()

    eff, rej = [], []
    for thr in thresholds:
        tagged_sig = ((scores >= thr) & (labels == signal_class)).sum()
        tagged_bg  = ((scores >= thr) & (labels == bg_class)).sum()
        sig_eff = tagged_sig / total_sig if total_sig > 0 else 0
        bg_eff  = tagged_bg  / total_bg  if total_bg  > 0 else 0
        eff.append(sig_eff)
        rej.append(1. / bg_eff if bg_eff > 0 else np.nan)

    eff = np.array(eff)
    rej = np.array(rej)

    mask = (eff >= eff_range[0]) & (eff <= eff_range[1]) & np.isfinite(rej)
    return eff[mask], rej[mask]


def _plot_roc(
    scores: np.ndarray,
    labels: np.ndarray,
    signal_class: int,
    bg_classes: list[tuple[int, str, str]],
    discriminant_type: str,
    output_dir: str | Path,
    eff_range: list = None,
) -> None:
    """
    Plot a ROC curve (signal efficiency vs background rejection).

    Args:
        scores (np.ndarray): discriminant scores for all jets.
        labels (np.ndarray): true class labels.
        signal_class (int): index of the signal class.
        bg_classes (list): list of ``(class_index, linestyle, legend_label)`` tuples.
        discriminant_type (str): name of the discriminant (e.g. "b" or "c", used for
            axis label and filename).
        output_dir (str | Path): output directory.
        eff_range (list): Minimum and maximum signal efficiency to include in the plot
            (default ``[0.5, 1.0]``).
    """
    if eff_range is None:
        eff_range = [0.5, 1.0]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Plotting ROC curve (%s-tag) ...", discriminant_type)

    fig, ax = plt.subplots(figsize=(7, 6))

    for bg_cls, linestyle, bg_label in bg_classes:
        eff, rej = _roc_rejection(scores, labels, signal_class, bg_cls, eff_range=eff_range)
        if eff.size == 0:
            logger.warning("No valid ROC points for `class %d` vs `class %d`.",
                           signal_class, bg_cls)
            continue
        ax.plot(eff, rej, linestyle=linestyle, linewidth=1.8, label=bg_label)

    ax.set_xlabel(f"{discriminant_type}-jet tagging efficiency", fontsize=14)
    ax.set_ylabel("Background rejection", fontsize=14)
    ax.set_yscale("log")
    ax.set_xlim(*eff_range)
    ax.legend()
    ax.grid(True, alpha=0.3)
    hep.atlas.label(ax=ax, data=True, loc=1, rlabel=ATLAS_RLABEL)
    ax.set_ylim(bottom=1.0)
    fig.tight_layout()
    out = output_dir / f"roc_d{discriminant_type}.pdf"
    fig.savefig(out)
    plt.close(fig)

    logger.info("Saved: %s", out)


def plot_all_roc(
    results: dict[str, np.ndarray],
    output_dir: str | Path,
) -> None:
    """
    Plot ROC curves (D_b and D_c) on the full test set.

    Args:
        results (dict): output of ``run_inference``.
        output_dir (str | Path): output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    b_cls, c_cls, u_cls, tau_cls = 0, 1, 2, 3

    _plot_roc(
        scores = results["db"],
        labels = results["labels"],
        signal_class = b_cls,
        bg_classes = [
            (c_cls,   "-",  "c-jet rejection"),
            (u_cls,   "-.", "light-jet rejection"),
            (tau_cls, "--", r"$\tau$-jet rejection"),
        ],
        discriminant_type = "b",
        output_dir = output_dir,
        eff_range = [0.6, 1.],
    )

    _plot_roc(
        scores = results["dc"],
        labels = results["labels"],
        signal_class = c_cls,
        bg_classes = [
            (b_cls,   "-",  "b-jet rejection"),
            (u_cls,   "-.", "light-jet rejection"),
            (tau_cls, "--", r"$\tau$-jet rejection"),
        ],
        discriminant_type = "c",
        output_dir = output_dir,
        eff_range = [0.1, 0.6],
    )


if __name__ == "__main__":
    pass
