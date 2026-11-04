import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Literal
from collections.abc import Mapping, Sequence
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import inspect
from mpl_toolkits.axes_grid1 import make_axes_locatable

def local_outliers(
    adata,
    metric: str = "total_counts",
    direction: Literal["lower", "higher", "both"] = "lower",
    n_neighbors: int = 36,
    samples: str = "sample_id", 
    log: bool = True,
    cutoff: float = 3.0,
    coords: Optional[np.ndarray] = None,
    spatial_key: str = "spatial"
) -> None:
    """
    Detect local outliers in spatial transcriptomics data based on quality metrics.
    
    Local outliers are defined as spots with low/high quality metrics compared to 
    their surrounding neighbors, based on a modified z-score statistic.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object containing spatial transcriptomics data
    metric : str, default "total_counts"
        Column name in adata.obs for QC metric to use for outlier detection
        (e.g., "total_counts", "n_genes_by_counts", "pct_counts_mt")
    direction : {"lower", "higher", "both"}, default "lower"
        Direction of outlier detection:
        - "lower": detect spots with unusually low values
        - "higher": detect spots with unusually high values  
        - "both": detect spots with unusually low or high values
    n_neighbors : int, default 36
        Number of nearest neighbors to use for outlier detection
    samples : str, default "sample_id"
        Column name in adata.obs for sample IDs (for multi-sample datasets)
    log : bool, default True
        Whether to log1p transform the metric values before analysis
    cutoff : float, default 3.0
        Z-score cutoff for outlier detection (absolute value)
    coords : np.ndarray, optional
        Custom coordinates matrix for neighborhood detection (n_spots × n_dims).
        If None, uses spatial coordinates from adata.obsm[spatial_key].
        Can be PCA, UMAP, or any other coordinate system.
    spatial_key : str, default "spatial"
        Key in adata.obsm containing spatial coordinates (if coords is None)
        
    Returns
    -------
    None
        Modifies adata.obs in place, adding columns:
        - f"{metric}_outliers": boolean array indicating outliers
        - f"{metric}_z": modified z-scores for each spot
        - f"{metric}_log": log-transformed values (if log=True)
        
    Examples
    --------
    >>> import scanpy as sc
    >>> import pandas as pd
    >>> 
    >>> # Load spatial data
    >>> adata = sc.datasets.visium_sge(sample="V1_Breast_Cancer_Block_A_Section_1")
    >>> adata.var_names_unique()
    >>> 
    >>> # Calculate QC metrics
    >>> adata.var["mt"] = adata.var_names.str.startswith("MT-")
    >>> sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    >>> 
    >>> # Detect local outliers based on total counts
    >>> local_outliers(adata, 
    ...                metric="total_counts",
    ...                direction="lower",
    ...                log=True)
    >>> 
    >>> # Check results
    >>> print(f"Found {adata.obs['total_counts_outliers'].sum()} outliers")
    >>> 
    >>> # Example with custom coordinates (PCA)
    >>> sc.tl.pca(adata, n_comps=50)
    >>> pca_coords = adata.obsm["X_pca"][:, :10]  # Use first 10 PCs
    >>> local_outliers(adata,
    ...                metric="n_genes_by_counts", 
    ...                coords=pca_coords)
    """
    
    # ===== Validity checks =====
    if not hasattr(adata, 'obs'):
        raise TypeError("Input must be an AnnData object")
        
    if direction not in ["lower", "higher", "both"]:
        raise ValueError("'direction' must be one of 'lower', 'higher', or 'both'")
        
    if not isinstance(n_neighbors, int) or n_neighbors <= 0:
        raise ValueError("'n_neighbors' must be a positive integer")
        
    if not isinstance(cutoff, (int, float)):
        raise ValueError("'cutoff' must be a numeric value")
        
    # Check if required columns exist
    if metric not in adata.obs.columns:
        raise KeyError(f"Metric '{metric}' not found in adata.obs")
        
    if samples not in adata.obs.columns:
        raise KeyError(f"Sample column '{samples}' not found in adata.obs") 
        
    # Validate coordinates
    if coords is not None:
        if not isinstance(coords, np.ndarray) or coords.ndim != 2:
            raise ValueError("'coords' must be a 2D numpy array")
        if coords.shape[0] != adata.n_obs:
            raise ValueError("'coords' must have same number of rows as observations")
    else:
        if spatial_key not in adata.obsm.keys():
            raise KeyError(f"Spatial coordinates '{spatial_key}' not found in adata.obsm")
    
    # ===== Main function =====
    # Create working copy of metadata
    metadata = adata.obs.copy()
    
    # Log transform if requested
    if log:
        metric_log = f"{metric}_log"
        metadata[metric_log] = np.log1p(metadata[metric])
        metric_to_use = metric_log
    else:
        metric_to_use = metric
        
    # Get unique sample IDs
    unique_sample_ids = metadata[samples].unique()
    
    # Initialize result columns
    metric_outliers = f"{metric}_outliers"
    metric_z = f"{metric}_z"
    
    metadata[metric_outliers] = False
    metadata[metric_z] = 0.0
    
    # Process each sample separately
    for sample in unique_sample_ids:
        # Get indices for current sample
        sample_mask = metadata[samples] == sample
        sample_indices = np.where(sample_mask)[0]
        
        if len(sample_indices) == 0:
            continue
            
        # Get coordinates for neighborhood analysis
        if coords is not None:
            neighborhood_coords = coords[sample_indices]
        else:
            neighborhood_coords = adata.obsm[spatial_key][sample_indices]
            
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1 because includes self
        nbrs.fit(neighborhood_coords)
        distances, indices = nbrs.kneighbors(neighborhood_coords)
        
        # Remove self from neighbors (first column is always self with distance 0)
        neighbor_indices = indices[:, 1:]  # Exclude self
        
        # Calculate modified z-scores for each spot
        sample_values = metadata.loc[sample_mask, metric_to_use].values
        mod_z_scores = np.zeros(len(sample_indices))
        
        for i, neighbors in enumerate(neighbor_indices):
            # Get neighbor values (map back to original sample indices)
            neighbor_vals = sample_values[neighbors]
            
            # Calculate modified z-score using neighborhood statistics
            mod_z_scores[i] = _modified_z_score(sample_values[i], neighbor_vals)
            
        # Handle non-finite values
        mod_z_scores = np.where(np.isfinite(mod_z_scores), mod_z_scores, 0.0)
        
        # Identify outliers based on direction and cutoff
        if direction == "higher":
            outliers = mod_z_scores > cutoff
        elif direction == "lower": 
            outliers = mod_z_scores < -cutoff
        else:  # "both"
            outliers = (mod_z_scores > cutoff) | (mod_z_scores < -cutoff)
            
        # Update metadata for this sample
        metadata.loc[sample_mask, metric_outliers] = outliers
        metadata.loc[sample_mask, metric_z] = mod_z_scores
        
    # Update adata.obs with results
    adata.obs[metric_outliers] = metadata[metric_outliers]
    adata.obs[metric_z] = metadata[metric_z]
    
    if log:
        adata.obs[metric_log] = metadata[metric_log]
        
    print(f"Detected {metadata[metric_outliers].sum()} outliers out of {len(metadata)} spots")


def _modified_z_score(value: float, neighbors: np.ndarray) -> float:
    """
    Calculate modified z-score based on neighborhood values.
    
    This implements a robust z-score calculation using median and median 
    absolute deviation (MAD) instead of mean and standard deviation.
    
    Parameters
    ----------
    value : float
        Value to calculate z-score for
    neighbors : np.ndarray
        Array of neighbor values
        
    Returns
    -------
    float
        Modified z-score
    """
    if len(neighbors) == 0:
        return 0.0
        
    # Calculate median and MAD
    median = np.median(neighbors)
    mad = np.median(np.abs(neighbors - median))
    
    # Avoid division by zero
    if mad == 0:
        return 0.0
        
    # Modified z-score formula: 0.6745 is the 75th percentile of standard normal
    mod_z = 0.6745 * (value - median) / mad
    
    return mod_z


def _groupwise_modified_z_score(values: np.ndarray, group_labels: Sequence) -> np.ndarray:
    """
    Compute a robust modified z-score for each value relative to its group median and MAD.

    This is the cluster/cell-type analogue of the local neighborhood statistic:
    cells are compared against other cells in the same cluster/cell type, rather than
    against spatial neighbors.
    """
    values = np.asarray(values, dtype=float)
    group_labels = np.asarray(group_labels)
    z_scores = np.zeros(values.shape[0], dtype=float)

    for label in np.unique(group_labels[~pd.isna(group_labels)]):
        mask = group_labels == label
        if not np.any(mask):
            continue

        group_values = values[mask]
        median = np.median(group_values)
        mad = np.median(np.abs(group_values - median))
        if not np.isfinite(mad) or mad == 0:
            z_scores[mask] = 0.0
            continue

        z_scores[mask] = 0.6745 * (group_values - median) / mad

    return np.where(np.isfinite(z_scores), z_scores, 0.0)


def qc_outliers_by_group(
    adata,
    metric: str = "total_counts",
    groupby: str = "leiden",
    samples: str = "sample_id",
    direction: Literal["lower", "higher", "both"] = "lower",
    cutoff: float = 3.0,
    log: bool = True,
    return_df: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Detect QC outliers at the cluster or cell-type level using a robust modified z-score.

    For each sample and each group (e.g. Leiden cluster or cell type), this compares
    each cell's QC metric against the median/MAD distribution of the same group.
    """
    if not hasattr(adata, "obs"):
        raise TypeError("Input must be an AnnData object")

    if direction not in ["lower", "higher", "both"]:
        raise ValueError("'direction' must be one of 'lower', 'higher', or 'both'")

    if metric not in adata.obs.columns:
        raise KeyError(f"Metric '{metric}' not found in adata.obs")

    if samples not in adata.obs.columns:
        raise KeyError(f"Sample column '{samples}' not found in adata.obs")

    if groupby not in adata.obs.columns:
        raise KeyError(f"Grouping column '{groupby}' not found in adata.obs")

    metadata = adata.obs.copy()

    if log:
        metric_log = f"{metric}_log"
        metadata[metric_log] = np.log1p(metadata[metric].astype(float))
        metric_to_use = metric_log
    else:
        metric_to_use = metric

    outlier_col = f"{metric}_{groupby}_outliers"
    z_col = f"{metric}_{groupby}_z"
    metadata[outlier_col] = False
    metadata[z_col] = 0.0

    for sample in pd.unique(metadata[samples].dropna()):
        sample_mask = metadata[samples].to_numpy() == sample
        if not np.any(sample_mask):
            continue

        sample_group_labels = metadata.loc[sample_mask, groupby].to_numpy()
        sample_values = metadata.loc[sample_mask, metric_to_use].to_numpy(dtype=float)
        sample_z = _groupwise_modified_z_score(sample_values, sample_group_labels)

        if direction == "higher":
            sample_outliers = sample_z > cutoff
        elif direction == "lower":
            sample_outliers = sample_z < -cutoff
        else:
            sample_outliers = (sample_z > cutoff) | (sample_z < -cutoff)

        metadata.loc[sample_mask, z_col] = sample_z
        metadata.loc[sample_mask, outlier_col] = sample_outliers

    adata.obs[outlier_col] = metadata[outlier_col].to_numpy()
    adata.obs[z_col] = metadata[z_col].to_numpy()
    if log:
        adata.obs[metric_log] = metadata[metric_log].to_numpy()

    print(
        f"Detected {int(metadata[outlier_col].sum())} group-level outliers for '{metric}' "
        f"across '{groupby}' groups out of {len(metadata)} cells"
    )

    if return_df:
        return metadata[[outlier_col, z_col] + ([metric_log] if log else [])]
    return None


def calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True):
    """
    Calculate basic QC metrics for spatial transcriptomics data.
    
    This is a simplified version that calculates commonly used QC metrics.
    For full functionality, use scanpy.pp.calculate_qc_metrics.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    percent_top : list of int, optional
        Calculate percentage of counts in top N genes
    log1p : bool, default False
        Whether to log1p transform count data
    inplace : bool, default True
        Whether to modify adata in place
        
    Returns
    -------
    AnnData or None
        Modified adata if inplace=False, else None
    """
    if not inplace:
        adata = adata.copy()
        
    # Basic QC metrics
    adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()
    adata.obs['n_genes_by_counts'] = np.array((adata.X > 0).sum(axis=1)).flatten()
    
    # Mitochondrial gene percentages if available
    if 'mt' in adata.var.columns:
        mt_mask = adata.var['mt'].values
        mt_counts = np.array(adata[:, mt_mask].X.sum(axis=1)).flatten()
        adata.obs['pct_counts_mt'] = (mt_counts / adata.obs['total_counts']) * 100
    
    if not inplace:
        return adata

def _process_vmin_vmax(values, vmin=None, vmax=None):
    """
    Process vmin and vmax parameters in the same way as scanpy.
    
    Parameters
    ----------
    values : array-like
        The values to be plotted
    vmin : Union[str, float, Callable, None]
        Lower limit of color scale
    vmax : Union[str, float, Callable, None]
        Upper limit of color scale
        
    Returns
    -------
    tuple
        Processed (vmin, vmax) values
    """
    # Filter out NaN values for processing
    finite_values = values[np.isfinite(values)]
    
    # If no finite values, return None for both
    if len(finite_values) == 0:
        return None, None
    
    def _process_limit(limit, values):
        if limit is None:
            return None
        elif isinstance(limit, str):
            if limit.startswith('p'):
                # Handle percentile format like 'p1.5' -> 1.5th percentile
                try:
                    percentile = float(limit[1:])
                    return np.percentile(values, percentile)
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid percentile format: {limit}")
            else:
                raise ValueError(f"Invalid string format for limit: {limit}")
        elif callable(limit):
            # Handle function
            return limit(values)
        else:
            # Handle numeric value
            return float(limit)
    
    processed_vmin = _process_limit(vmin, finite_values)
    processed_vmax = _process_limit(vmax, finite_values)
    
    return processed_vmin, processed_vmax


def _resolve_color_limit(limit, key=None):
    """Resolve vmin/vmax inputs following scanpy.pl.spatial semantics."""
    if limit is None:
        return None

    if isinstance(limit, Mapping):
        if key is None:
            return None
        return limit.get(key, None)

    if isinstance(limit, Sequence) and not isinstance(limit, (str, bytes)):
        if len(limit) == 0:
            return None
        if len(limit) == 1:
            return limit[0]
        raise ValueError("When providing a sequence for vmin/vmax, it must contain exactly one value for this plot.")

    return limit

def plot_qc_metrics(
    adata,
    sample_id_col="sample_id",
    sample=None,
    metric="detected",
    outliers=None,
    outlier_colors=None,
    point_size=10,
    colors=("white", "black"),
    stroke=1,
    spatial_key="spatial",
    ax=None,
    return_fig=False,
    vmin=None,
    vmax=None,
    log_scale: bool = False,
    log_base: float = 10.0,
    linthresh: float = 1.0,
    cbar_position: str = "right",
    rasterized: bool = False
):
    """
    Plot QC metrics for a single sample in an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing spatial transcriptomics data.
    sample_id_col : str, default "sample_id"
        Column in adata.obs containing sample IDs.
    sample : str or int, optional
        Sample to plot. If None, uses the first unique sample.
    metric : str, default "detected"
        QC metric column in adata.obs to visualize.
    outliers : str, Sequence[str], Mapping[str, str], optional
        One or more columns in adata.obs indicating outliers (boolean). If a mapping,
        the keys are column names and the values are colors used for each column.
    outlier_colors : Mapping[str, str] or Sequence[str], optional
        Custom colors for outlier columns when ``outliers`` is provided as a single
        column name or a sequence. When a sequence is supplied, colors are matched in
        order; otherwise they are looked up by column name. Falls back to a palette
        if not provided.
    point_size : int, default 10
        Size of points in the plot.
    colors : tuple or list, default ("white", "black")
        Colors for the metric gradient.
    stroke : int, default 1
        Border thickness for outlier points.
    spatial_key : str, default "spatial"
        Key in adata.obsm for spatial coordinates.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates a new figure and axes.
    return_fig : bool, default False
        Whether to return the figure and axes objects.
    vmin : Union[str, float, Callable, Mapping, Sequence, None], optional (default: None)
        Lower limit of the color scale. Accepts numeric values, percentile strings
        (e.g., ``"p1.5"``), callables, or a Mapping keyed by metric name.
    vmax : Union[str, float, Callable, Mapping, Sequence, None], optional (default: None)
        Upper limit of the color scale with the same semantics as ``vmin``.
    log_scale : bool, default False
        If True, use a logarithmic color scale (LogNorm for all-positive data,
        SymLogNorm otherwise).
    log_base : float, default 10.0
        Base for the logarithmic scale (passed to SymLogNorm or LogNorm).
    linthresh : float, default 1.0
        Linear threshold for SymLogNorm when data contains non-positive values.
    cbar_position : str, default "right"
        Position of the colorbar. One of "right", "left", "top", or "bottom".
    rasterized : bool, default False
        If True, rasterize the scatter plot for smaller file sizes in vector formats.
    Notes
    -----
    The spatial axes are rendered without tick marks or labels, and the colorbar is
    positioned using an axes divider to maintain equal height with the scatter plot.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        Only returned if return_fig=True or if ax was None.
    """
    # Subset to the specified sample
    if sample is None:
        sample = adata.obs[sample_id_col].unique()[0]
    mask = adata.obs[sample_id_col] == sample
    df = adata.obs.loc[mask].copy()
    coords = adata.obsm[spatial_key][mask]

    # Prepare color mapping for metric
    metric_vals = df[metric].values
    if len(colors) == 2:
        cmap = sns.light_palette(colors[1], as_cmap=True)
    else:
        cmap = sns.color_palette(colors, as_cmap=True)

    # Process vmin and vmax
    resolved_vmin = _resolve_color_limit(vmin, metric)
    resolved_vmax = _resolve_color_limit(vmax, metric)
    processed_vmin, processed_vmax = _process_vmin_vmax(metric_vals, resolved_vmin, resolved_vmax)

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
        created_fig = True
    else:
        fig = ax.figure
        created_fig = False

    # Prepare normalization (log or linear)
    norm = None
    supports_base = {
        LogNorm: "base" in inspect.signature(LogNorm.__init__).parameters,
        SymLogNorm: "base" in inspect.signature(SymLogNorm.__init__).parameters,
    }
    if log_scale:
        finite_vals = metric_vals[np.isfinite(metric_vals)]
        positive_vals = finite_vals[finite_vals > 0]
        # determine vmin/vmax for lognorm if needed
        vmin_for_norm = processed_vmin
        vmax_for_norm = processed_vmax
        if vmin_for_norm is None and len(finite_vals) > 0:
            vmin_for_norm = float(np.min(finite_vals))
        if vmax_for_norm is None and len(finite_vals) > 0:
            vmax_for_norm = float(np.max(finite_vals))

        if len(positive_vals) > 0 and len(positive_vals) == len(finite_vals):
            # all finite values are positive -> use LogNorm
            # ensure vmin > 0
            if vmin_for_norm is None or vmin_for_norm <= 0:
                vmin_for_norm = float(np.min(positive_vals))
            if vmax_for_norm is None:
                vmax_for_norm = float(np.max(positive_vals))
            lognorm_kwargs = dict(vmin=vmin_for_norm, vmax=vmax_for_norm)
            if supports_base[LogNorm]:
                lognorm_kwargs["base"] = log_base
            elif log_base != 10.0:
                warnings.warn(
                    "Current Matplotlib LogNorm version does not support 'base'; using default base.",
                    UserWarning,
                )
            norm = LogNorm(**lognorm_kwargs)
        else:
            # data contains non-positive values -> use SymLogNorm
            # SymLogNorm accepts vmin/vmax as None
            symlog_kwargs = dict(linthresh=linthresh, vmin=processed_vmin, vmax=processed_vmax)
            if supports_base[SymLogNorm]:
                symlog_kwargs["base"] = log_base
            elif log_base != 10.0:
                warnings.warn(
                    "Current Matplotlib SymLogNorm version does not support 'base'; using default base.",
                    UserWarning,
                )
            norm = SymLogNorm(**symlog_kwargs)

    # Build scatter kwargs
    scatter_kwargs = dict(
        x=coords[:, 0],
        y=coords[:, 1],
        c=metric_vals,
        cmap=cmap,
        s=point_size,
        edgecolor="none",
        rasterized=rasterized
    )
    if norm is not None:
        scatter_kwargs["norm"] = norm
    else:
        scatter_kwargs["vmin"] = processed_vmin
        scatter_kwargs["vmax"] = processed_vmax

    sc = ax.scatter(**scatter_kwargs)

    # Highlight outliers if provided
    if outliers is not None:
        outlier_specs = []

        if isinstance(outliers, str):
            outlier_columns = [outliers]
            color_lookup = {}
        elif isinstance(outliers, Mapping):
            for column, color in outliers.items():
                outlier_specs.append((column, color))
            outlier_columns = []
        elif isinstance(outliers, Sequence) and not isinstance(outliers, (str, bytes)):
            outlier_columns = list(outliers)
            color_lookup = {}
        else:
            raise TypeError("'outliers' must be a string, sequence of strings, or mapping of column to color")

        if outlier_specs and outlier_colors is not None:
            warnings.warn("'outlier_colors' is ignored when 'outliers' is supplied as a mapping with colors.", UserWarning)

        if not outlier_specs:
            color_lookup = {}
            if outlier_colors is not None:
                if isinstance(outlier_colors, Mapping):
                    color_lookup = dict(outlier_colors)
                elif isinstance(outlier_colors, Sequence) and not isinstance(outlier_colors, (str, bytes)):
                    if len(outlier_colors) < len(outlier_columns):
                        raise ValueError("Not enough colors provided for the number of outlier columns")
                    color_lookup = {col: color for col, color in zip(outlier_columns, outlier_colors)}
                else:
                    raise TypeError("'outlier_colors' must be a mapping or sequence of color specifications")

            fallback_palette = []
            if outlier_columns and len(outlier_columns) > 1:
                fallback_palette = sns.color_palette("husl", len(outlier_columns))
            for idx, column in enumerate(outlier_columns):
                color = color_lookup.get(column)
                if color is None:
                    if len(outlier_columns) == 1:
                        color = "red"
                    else:
                        color = fallback_palette[idx % len(fallback_palette)] if fallback_palette else "red"
                outlier_specs.append((column, color))

        valid_outlier_columns = [column for column, _ in outlier_specs if column in df.columns]
        single_valid_outlier = len(valid_outlier_columns) == 1

        for column, color in outlier_specs:
            if column not in df.columns:
                warnings.warn(f"Outlier column '{column}' not found in adata.obs; skipping.", UserWarning)
                continue

            outlier_mask = df[column].astype(bool)
            if not np.any(outlier_mask):
                continue

            label = "Outlier" if single_valid_outlier else str(column)
            ax.scatter(
                coords[outlier_mask, 0], coords[outlier_mask, 1],
                facecolors="none",
                edgecolors=color,
                s=point_size + stroke * 5,
                linewidths=stroke,
                label=label,
                rasterized=rasterized
            )

    ax.set_title(f"Sample: {sample}")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    previous_cbar_ax = getattr(ax, "_plot_qc_metrics_cbar", None)
    if previous_cbar_ax is not None:
        try:
            previous_cbar_ax.remove()
        except ValueError:
            if previous_cbar_ax in fig.axes:
                fig.delaxes(previous_cbar_ax)

    if cbar_position not in {"right", "left", "top", "bottom"}:
        raise ValueError("cbar_position must be one of 'right', 'left', 'top', or 'bottom'")

    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes(cbar_position, size="5%", pad=0.1)
    cbar_orientation = "horizontal" if cbar_position in {"top", "bottom"} else "vertical"
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation=cbar_orientation)
    cbar.set_label(metric)
    ax._plot_qc_metrics_cbar = cbar_ax
    ax.invert_yaxis()
    ax.legend()
    
    # Only call tight_layout if we created the figure
    if created_fig:
        plt.tight_layout()

    if return_fig or created_fig:
        return fig, ax

# Example usage and testing
if __name__ == "__main__":
    # This would typically be run in a Jupyter notebook with real data
    print("Local outliers detection function loaded successfully!")
    print("\nExample usage:")
    print("""
    import scanpy as sc
    
    # Load data
    adata = sc.datasets.visium_sge()
    
    # Calculate QC metrics  
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    
    # Detect outliers
    local_outliers(adata, metric="total_counts", direction="lower")
    
    # Check results
    print(f"Outliers detected: {adata.obs['total_counts_outliers'].sum()}")
    """)
