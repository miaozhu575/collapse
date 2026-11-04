# collapse

A focused Python package for cell-level fragment deduplication, coordinate collision detection, and group-aware quality control in single-cell and spatial ATAC-seq data.

---

## Core Capabilities

The `collapse` library is applicable to both single-cell and spatial epigenomics datasets (e.g. 10x Genomics, sci-ATAC-seq, microfluidics, Slide-tags, and spatial chromatin profiling). It centers on two primary functions:
1. **Coordinate Collision Ratio Calculation (`collapse.count`)**: Streaming and memory-efficient estimation of duplicate/shared fragment coordinates across single cells or spatial spots without requiring massive RAM.
2. **Cluster-Aware Quality Control (`collapse.localOutliers`)**: Adaptive outlier detection that accounts for cell-type and spatial domain heterogeneity, preventing genuine high/low accessibility clusters from being filtered out.

---

## Installation

### Requirements
- Python >= 3.9
- Key libraries: `numpy`, `pandas`, `scipy`, `polars`, `scikit-learn`, `matplotlib`, `seaborn`, `scanpy`

### From Source

```bash
# Clone the repository and navigate to the collapse directory
cd collapse

# Install in standard mode
pip install .

# Or install in editable / development mode
pip install -e .
```

---

## Core Modules

### 1. `collapse.count`: Fragment Collision Quantification
- **`calc_collision_ratio()`**: Calculates the frequency and ratio of shared-coordinate fragments (identical chromosome, start, and end coordinates across distinct cell barcodes) for each cell barcode.
  - **Memory-efficient streaming**: $O(\text{barcodes})$ memory usage rather than loading all fragments into memory. Easily handles multi-gigabyte files with hundreds of millions of reads within tens of megabytes of RAM.
  - **Customizable filters**: Supports chromosome filtering (e.g., excluding `chrM`), minimum cell barcode counts, and minimum sharing thresholds.
- **`read_bc_and_counts_from_fragments_file()`**: Fast extraction of cell barcode occurrences and counts per fragment.
- **`get_top_cell_barcodes()`**: Identifies top cell barcodes ranked by total fragment count.
- **`groupCommand` / `groupCommandUMI`**: Generates grouping commands for fragments.

### 2. `collapse.localOutliers`: Cell Quality Control & Outlier Filtering
- **`qc_outliers_by_group()`**: Detects quality control outliers at the cluster or cell-type level using a robust modified z-score based on median and median absolute deviation (MAD). This accommodates natural biological variations in chromatin accessibility across different cell types.
- **`local_outliers()`**: Spatial neighborhood-based anomaly detection using k-nearest neighbors and robust modified z-scores (median/MAD).
- **`plot_qc_metrics()`**: Visualizes QC metric distributions, scatter relationships, and outlier boundaries across cell groups.

---

## Usage Examples

### 1. Calculate Coordinate Collision Ratio

```python
import collapse

# Stream through a sorted fragment file and compute collision metrics
df_collision = collapse.calc_collision_ratio(
    fragments="fragments.tsv.gz",
    exclude_chromosome="chrM"      # Remove mitochondrial fragments
)

print(df_collision.head())
# Output columns:
# - CB: Cell barcode
# - frequency_count: Total fragments observed for this barcode
# - shared_count: Number of fragments sharing coordinates with other cells
# - ratio: Collision ratio (shared_count / frequency_count)
# - reads_count: Total reads (if 5th column available)
# - shared_reads_count: Reads belonging to collided fragments
# - reads_ratio: Fraction of shared reads
```

### 2. Cluster-Aware QC Outlier Detection

```python
import scanpy as sc
import collapse

# Load single-cell ATAC-seq dataset
adata = sc.read_h5ad("scatac_processed.h5ad")

# Detect outliers within each cell type using modified z-score (median/MAD)
collapse.qc_outliers_by_group(
    adata,
    groupby="cell_type",
    metrics="Collision_ratio",
    cutoff=3.0                       # threshold
)

# Outlier flags are stored in adata.obs
print(adata.obs["Collision_ratio_cell_type_outliers"].value_counts())
```

### 3. Spatial Local QC Outlier Detection
```python
import scanpy as sc
import collapse

# Load spatial ATAC-seq dataset
adata = sc.read_h5ad("spatac_processed.h5ad")

# Detect local outliers using modified z-score (median/MAD)
collapse.local_outliers(
    adata,
    metric="Collision_ratio",
    direction="higher",
    n_neighbors=36,
)

# Visualize local outliers
collapse.plot_qc_metrics(
    adata,
    metric="Collision_ratio",
    outliers="Collision_ratio_outliers",
)
```

---

## License

This project is licensed under the MIT License.
