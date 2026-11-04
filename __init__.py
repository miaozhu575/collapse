__version__ = "0.1.0"

from .count import (
    read_bc_and_counts_from_fragments_file,
    groupCommand,
    groupCommandUMI,
    get_top_cell_barcodes,
    calc_collision_ratio,
)
from .localOutliers import (
    local_outliers,
    qc_outliers_by_group,
    plot_qc_metrics,
)
