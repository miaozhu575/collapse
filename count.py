import gzip
import re
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd
import polars as pl


def read_bc_and_counts_from_fragments_file(
    fragments_bed_filename: str,
    umi: bool = False,
    single_end: bool = False,
    chromosome: str = None,
    cellbarcode: list = None,
    **kwargs,
) -> "pl.DataFrame":
    """
    Read cell barcode (column 4) and counts per fragment (column 5 or 6 if UMI is True) from fragments BED file.
    Cell barcodes will appear more than once as they have counts per fragment, but as
    the fragment locations are not needed, they are not returned.

    Parameters
    ----------
    fragments_bed_filename: str
        Fragments BED filename.
    umi: bool, optional
        Whether the file includes UMI (six columns with UMI in the 5th column).
    chromosome: str, optional
        Regex pattern to filter specific chromosome(s). If None, read all chromosomes.
        Examples: "^chrM$", "^chr([1-9]|1[0-9]|2[0-2])$", "^chr[XY]$", etc.

    Returns
    -------
    pl.DataFrame
        Polars dataframe with cell barcode and count per fragment.
    """
    # Set the correct open function depending if the fragments BED file is gzip compressed or not.
    open_fn = gzip.open if fragments_bed_filename.endswith((".bgz", ".gz")) else open

    skip_rows = 0
    nbr_columns = 0

    with open_fn(fragments_bed_filename, "rt") as fragments_bed_fh:
        for line in fragments_bed_fh:
            # Remove newlines and spaces.
            line = line.strip()

            if not line or line.startswith("#"):
                # Count number of empty lines and lines which start with a comment before the actual data.
                skip_rows += 1
            else:
                # Get number of columns from the first real BED entry.
                nbr_columns = len(line.split("\t"))

                # Stop reading the BED file.
                break

    expected_columns = 6 if umi else 5
    if nbr_columns < expected_columns:
        raise ValueError(
            f'Fragments BED file needs to have at least {expected_columns} columns. "{fragments_bed_filename}" contains only '
            f"{nbr_columns} columns."
        )

    # Define columns based on UMI presence
    if umi:
        columns = [
            "column_1",
            "column_2",
            "column_3",
            "column_4",
            "column_5",
            "column_6",
        ]
        new_columns = [
            "Chromosome",
            "Start",
            "End",
            "CellBarcode",
            "UMI",
            "FragmentCount",
        ]
        if single_end:
            dtypes = [
                pl.Categorical,
                pl.Int32,
                pl.UInt32,
                pl.Categorical,
                pl.Categorical,
                pl.UInt32,
            ]
        else:
            dtypes = [
                pl.Categorical,
                pl.UInt32,
                pl.UInt32,
                pl.Categorical,
                pl.Categorical,
                pl.UInt32,
            ]
    else:
        columns = ["column_1", "column_2", "column_3", "column_4", "column_5"]
        new_columns = ["Chromosome", "Start", "End", "CellBarcode", "FragmentCount"]
        if single_end:
            dtypes = [pl.Categorical, pl.Int32, pl.UInt32, pl.Categorical, pl.UInt32]
        else:
            dtypes = [pl.Categorical, pl.UInt32, pl.UInt32, pl.Categorical, pl.UInt32]

    # Read cell barcode and counts per fragment from fragments BED file.
    fragments_df = pl.read_csv(
        fragments_bed_filename,
        has_header=False,
        skip_rows=skip_rows,
        separator="\t",
        use_pyarrow=False,
        n_threads=6,
        columns=columns,
        new_columns=new_columns,
        dtypes=dtypes,
        **kwargs,
    )

    # Filter by chromosome if specified
    if chromosome is not None:
        # Compile regex pattern for efficiency
        chromosome_pattern = re.compile(chromosome)
        fragments_df = fragments_df.filter(
            pl.col("Chromosome").cast(pl.Utf8).str.contains(chromosome_pattern.pattern)
        )

    # Filter by cell barcode if specified
    if cellbarcode is not None:
        fragments_df = fragments_df.filter(
            pl.col("CellBarcode").cast(pl.Utf8).is_in(cellbarcode)
        )

    return fragments_df


def groupCommand(
    fragments, whitelist_cells=None, max_lines=0, verbose=False, chromosome=None
):
    """
    Count fragments in a fragments BED file.

    Parameters
    ----------
    fragments: str
        Fragments BED filename.
    whitelist_cells: list, optional
        List of cell barcodes to include in the count.
    max_lines: int, optional
        Maximum number of lines to read.
    verbose: bool, optional
        Whether to print verbose output.
    chromosome: str or list, optional
        Specific chromosome(s) to count fragments for. Can be a single chromosome string
        or a list of chromosomes. If None, count all chromosomes.
    """
    # opening gzipped compressed stream
    fileHandler = gzip.open(fragments, "rb")

    # determine if we read the whole file or the first n lines
    read_part = False
    if max_lines > 0:
        read_part = True

    # return empty DataFrame if it can't find the file
    if fileHandler is None:
        print("can't open file")
        return pd.DataFrame()

    # Hash Map storing the count data
    index_hash = {}

    has_whitelist = False
    num_whitelist_cells = 0
    if whitelist_cells is not None:
        has_whitelist = True
        for i, cell in enumerate(whitelist_cells):
            index_hash[cell] = i

        num_whitelist_cells = len(index_hash)
        if verbose:
            print(f"Found {num_whitelist_cells} cell barcodes")

    # metadata from fragments file
    freq_count, mono_nuc_count, nuc_free_count, read_count = [], [], [], []
    # if we have external cells reserve the size
    if has_whitelist:
        freq_count = [0] * num_whitelist_cells
        mono_nuc_count = [0] * num_whitelist_cells
        nuc_free_count = [0] * num_whitelist_cells
        read_count = [0] * num_whitelist_cells

    # looping over the fragments file
    line_counter = 1
    for line in fileHandler:
        line = line.decode().strip()
        if line[0] == "#":
            continue

        fields = line.split("\t")
        if len(fields) < 5:
            print("Error: fragment file has too few columns")
            fileHandler.close()
            return pd.DataFrame()

        # Check chromosome filter if specified
        if chromosome is not None:
            # Handle both single chromosome string and list of chromosomes
            if isinstance(chromosome, str):
                if fields[0] != chromosome:
                    continue
            elif isinstance(chromosome, (list, tuple)):
                if fields[0] not in chromosome:
                    continue

        cb_seq = fields[3]
        start = int(fields[1])
        end = int(fields[2])
        reads = int(fields[4])

        nuc_free_inc = 1 if end - start < 147 else 0
        mono_inc = 1 if 147 < end - start < 294 else 0

        # index of the cellular barcode
        cb_idx = len(index_hash)
        if cb_seq in index_hash:
            cb_idx = index_hash[cb_seq]

            # increment the count
            freq_count[cb_idx] += 1
            if mono_inc > 0:
                mono_nuc_count[cb_idx] += mono_inc
            if nuc_free_inc > 0:
                nuc_free_count[cb_idx] += nuc_free_inc
            read_count[cb_idx] += reads
        elif not has_whitelist:
            # extracting the index
            index_hash[cb_seq] = cb_idx

            # increment the count
            freq_count.append(1)
            mono_nuc_count.append(mono_inc)
            nuc_free_count.append(nuc_free_inc)
            read_count.append(reads)

        line_counter += 1
        if read_part and line_counter > max_lines:
            break

        if verbose and line_counter % 1000000 == 0:
            print(f"Done Processing {line_counter / 1000000} million lines")

    # Cleanup
    fileHandler.close()

    # extracting cellular barcode names
    keys = [None] * len(index_hash)
    for k, v in index_hash.items():
        keys[v] = k

    # creating data frame and returning
    cb_groups = pd.DataFrame(
        {
            "CB": keys,
            "frequency_count": freq_count,
            "mononucleosomal": mono_nuc_count,
            "nucleosome_free": nuc_free_count,
            "reads_count": read_count,
        }
    )

    return cb_groups


def groupCommandUMI(
    fragments, whitelist_cells=None, max_lines=0, verbose=False, chromosome=None
):
    """
    Count fragments in a fragments BED file with UMI.

    Parameters
    ----------
    fragments: str
        Fragments BED filename.
    whitelist_cells: list, optional
        List of cell barcodes to include in the count.
    max_lines: int, optional
        Maximum number of lines to read.
    verbose: bool, optional
        Whether to print verbose output.
    chromosome: str or list, optional
        Specific chromosome(s) to count fragments for. Can be a single chromosome string
        or a list of chromosomes. If None, count all chromosomes.
    """
    # opening gzipped compressed stream
    fileHandler = gzip.open(fragments, "rb")

    # determine if we read the whole file or the first n lines
    read_part = False
    if max_lines > 0:
        read_part = True

    # return empty DataFrame if it can't find the file
    if fileHandler is None:
        print("can't open file")
        return pd.DataFrame()

    # Hash Map storing the count data
    index_hash = {}

    has_whitelist = False
    num_whitelist_cells = 0
    if whitelist_cells is not None:
        has_whitelist = True
        for i, cell in enumerate(whitelist_cells):
            index_hash[cell] = i

        num_whitelist_cells = len(index_hash)
        if verbose:
            print(f"Found {num_whitelist_cells} cell barcodes")

    # metadata from fragments file
    freq_count, mono_nuc_count, nuc_free_count, read_count = [], [], [], []
    # Add dictionary to store unique UMIs for each cell
    unique_umis = {}

    # if we have external cells reserve the size
    if has_whitelist:
        freq_count = [0] * num_whitelist_cells
        mono_nuc_count = [0] * num_whitelist_cells
        nuc_free_count = [0] * num_whitelist_cells
        read_count = [0] * num_whitelist_cells
        # Initialize empty sets for each whitelisted cell
        for cell in whitelist_cells:
            unique_umis[cell] = set()

    # looping over the fragments file
    line_counter = 1
    for line in fileHandler:
        line = line.decode().strip()
        if line[0] == "#":
            continue

        fields = line.split("\t")
        if len(fields) < 5:
            print("Error: fragment file has too few columns")
            fileHandler.close()
            return pd.DataFrame()

        # Check chromosome filter if specified
        if chromosome is not None:
            # Handle both single chromosome string and list of chromosomes
            if isinstance(chromosome, str):
                if fields[0] != chromosome:
                    continue
            elif isinstance(chromosome, (list, tuple)):
                if fields[0] not in chromosome:
                    continue

        cb_seq = fields[3]
        start = int(fields[1])
        end = int(fields[2])
        reads = int(fields[5])
        umi = fields[4]

        nuc_free_inc = 1 if end - start < 147 else 0
        mono_inc = 1 if 147 < end - start < 294 else 0

        # index of the cellular barcode
        cb_idx = len(index_hash)
        if cb_seq in index_hash:
            cb_idx = index_hash[cb_seq]

            # increment the count
            freq_count[cb_idx] += 1
            if mono_inc > 0:
                mono_nuc_count[cb_idx] += mono_inc
            if nuc_free_inc > 0:
                nuc_free_count[cb_idx] += nuc_free_inc
            read_count[cb_idx] += reads
            # Add UMI to set for this cell
            unique_umis[cb_seq].add(umi)

        elif not has_whitelist:
            # extracting the index
            index_hash[cb_seq] = cb_idx

            # increment the count
            freq_count.append(1)
            mono_nuc_count.append(mono_inc)
            nuc_free_count.append(nuc_free_inc)
            read_count.append(reads)
            # Initialize set with first UMI for this cell
            unique_umis[cb_seq] = {umi}

        line_counter += 1
        if read_part and line_counter > max_lines:
            break

        if verbose and line_counter % 1000000 == 0:
            print(f"Done Processing {line_counter / 1000000} million lines")

    # Cleanup
    fileHandler.close()

    # extracting cellular barcode names
    keys = [None] * len(index_hash)
    for k, v in index_hash.items():
        keys[v] = k

    # Create list of UMI counts in same order as keys
    umi_counts = [len(unique_umis[k]) for k in keys]

    # creating data frame and returning
    cb_groups = pd.DataFrame(
        {
            "CB": keys,
            "frequency_count": freq_count,
            "mononucleosomal": mono_nuc_count,
            "nucleosome_free": nuc_free_count,
            "reads_count": read_count,
            "unique_umi_count": umi_counts,  # Add UMI counts to output
        }
    )

    return cb_groups


def get_top_cell_barcodes(
    fragments,
    top_n,
    *,
    umi=False,
    whitelist_cells=None,
    min_counts=None,
    chromosome=None,
    max_lines=0,
    verbose=False,
    save_path=None
):
    """Return the top ``top_n`` cell barcodes ranked by fragment count.

    Parameters
    ----------
    fragments : str
        Path to the fragments BED file (gzip-compressed or plain text).
    top_n : int
        Number of barcodes to return. Must be positive.
    umi : bool, optional
        When True, use :func:`groupCommandUMI` to consider UMI-aware fragments.
        Defaults to False and uses :func:`groupCommand`.
    whitelist_cells : Iterable[str], optional
        Optional whitelist of barcodes to consider when tallying fragments.
    chromosome : str or list, optional
        Restrict counting to specific chromosome(s), passed through to the
        underlying counting function.
    max_lines : int, optional
        Limit the number of fragment lines processed. ``0`` reads the entire file.
    verbose : bool, optional
        Enable progress logging when counting fragments.

    Returns
    -------
    list[str]
        List of barcode strings sorted by decreasing fragment count. Returns an
        empty list when no fragments are available or ``top_n`` <= 0.
    """

    try:
        top_n_int = int(top_n)
    except (TypeError, ValueError):
        raise ValueError("top_n must be convertible to an integer") from None

    if top_n_int <= 0:
        return []

    counter_fn = groupCommandUMI if umi else groupCommand
    counts_df = counter_fn(
        fragments,
        whitelist_cells=whitelist_cells,
        max_lines=max_lines,
        verbose=verbose,
        chromosome=chromosome,
    )

    if counts_df.empty:
        return []

    # Ensure non-null barcodes before sorting
    filtered_df = counts_df[counts_df["CB"].notna()].copy()
    if min_counts is not None:
        filtered_df = filtered_df[filtered_df["frequency_count"] >= min_counts]

    if filtered_df.empty:
        return []

    ranked_df = filtered_df.sort_values(
        by="frequency_count", ascending=False, kind="mergesort"
    )

    if top_n_int < filtered_df.shape[0]:
        ranked_df = ranked_df.head(top_n_int)

    if save_path is not None:
        ranked_df.to_csv(save_path, sep="\t", index=False)

    return ranked_df["CB"].tolist()


def _iter_fragment_lines(
    fragments: Union[str, Path],
    assume_sorted: bool = True,
    sort_memory: str = "2G",
    sort_threads: int = 4,
):
    """Yield raw lines from fragments file, streaming through Unix sort if unsorted."""
    fragments_path = Path(fragments)
    if not fragments_path.exists():
        raise FileNotFoundError(f"Fragments file not found: {fragments}")

    is_gz = fragments_path.suffix in (".gz", ".bgz") or any(
        s in (".gz", ".bgz") for s in fragments_path.suffixes
    )

    if not assume_sorted:
        sort_cmd = (
            f"LC_ALL=C sort -k1,1 -k2,2n -k3,3n -S {shlex.quote(str(sort_memory))} "
            f"--parallel={int(sort_threads)}"
        )
        if is_gz:
            cmd_str = f"gzip -dc {shlex.quote(str(fragments_path))} | {sort_cmd}"
        else:
            cmd_str = f"{sort_cmd} {shlex.quote(str(fragments_path))}"

        proc = subprocess.Popen(
            cmd_str,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=65536,
        )
        try:
            for line in proc.stdout:
                yield line
        finally:
            if proc.stdout:
                proc.stdout.close()
            ret = proc.wait()
            if ret != 0:
                err = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(f"Sort pipeline failed (exit {ret}): {err}")
    else:
        if is_gz:
            proc = None
            try:
                proc = subprocess.Popen(
                    ["gzip", "-dc", str(fragments_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=65536,
                )
                for line in proc.stdout:
                    yield line
                proc.stdout.close()
                ret = proc.wait()
                if ret != 0:
                    raise RuntimeError(f"gzip -dc exited with {ret}")
            except Exception:
                if proc is not None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                with gzip.open(
                    fragments_path, "rt", encoding="utf-8", errors="replace"
                ) as gz_f:
                    for line in gz_f:
                        yield line
        else:
            with open(
                fragments_path,
                "rt",
                encoding="utf-8",
                errors="replace",
                buffering=65536,
            ) as f:
                for line in f:
                    yield line


def calc_collision_ratio(
    fragments: Union[str, Path],
    *,
    whitelist_cells: Optional[Iterable[str]] = None,
    whitelist_only: bool = False,
    chromosome: Optional[Union[str, Iterable[str]]] = None,
    exclude_chromosome: Optional[Union[str, Iterable[str]]] = None,
    distinct_barcodes: bool = True,
    min_sharing: int = 2,
    assume_sorted: bool = True,
    sort_memory: str = "2G",
    sort_threads: int = 4,
    max_lines: int = 0,
    verbose: bool = False,
    progress_interval: int = 10_000_000,
    min_counts: Optional[int] = None,
    sort_by: Optional[str] = None,
    as_polars: bool = False,
    save_path: Optional[Union[str, Path]] = None,
) -> Union[pd.DataFrame, "pl.DataFrame"]:
    """
    Calculate the percentage and ratio of share-coordinate fragments (with the same chr, start, and end)
    for each cell barcode from a fragment file.

    Memory-efficient streaming algorithm designed for massive (multi-gigabyte/hundreds of millions of lines)
    scATAC-seq fragment files. The memory consumption is O(number of cell barcodes) rather than
    O(number of fragments), using only tens of megabytes of RAM even on files with hundreds of millions of lines.

    Parameters
    ----------
    fragments : str or Path
        Path to the fragments BED/TSV file (plain text or gzip/bgzip compressed).
    whitelist_cells : Iterable[str], optional
        List or set of cell barcodes to keep in the final output. If None, all cell barcodes
        present in the fragment file are returned.
    whitelist_only : bool, default False
        When True, only fragments from `whitelist_cells` are considered when determining
        whether a coordinate is shared.
        When False (default), all fragments across the file (including non-whitelisted/background barcodes)
        contribute to coordinate sharing, but the output metrics are reported for `whitelist_cells`.
    chromosome : str or Iterable[str], optional
        Specific chromosome(s) to include. Can be a single string (e.g., "chr1") or an iterable
        of chromosomes. If None, all chromosomes are considered.
    exclude_chromosome : str or Iterable[str], optional
        Specific chromosome(s) to exclude (e.g., "chrM", ["chrM", "chrMT"]). Useful for removing
        mitochondrial reads prior to collision ratio calculation.
    distinct_barcodes : bool, default True
        If True, a coordinate (chr, start, end) is considered shared only if it appears across at least
        `min_sharing` distinct cell barcodes.
        If False, a coordinate is considered shared if it appears at least `min_sharing` times total
        (even if from the same cell barcode).
    min_sharing : int, default 2
        Minimum number of occurrences (or distinct barcodes) required to consider a coordinate shared.
    assume_sorted : bool, default True
        Whether the fragment file is sorted by genomic coordinates (chr, start, end), which is standard
        for 10x Genomics and tabix-indexed fragment files.
        When True, processes the file in a single pass with minimal memory usage. If an unsorted coordinate
        order is detected, raises ValueError.
        When False, streams the file through external Unix sort (`sort -k1,1 -k2,2n -k3,3n -S sort_memory`),
        maintaining bounded memory usage.
    sort_memory : str, default "2G"
        Memory buffer hint passed to Unix `sort -S` when `assume_sorted=False`.
    sort_threads : int, default 4
        Number of parallel threads passed to Unix `sort` when `assume_sorted=False`.
    max_lines : int, default 0
        Maximum number of fragment lines to read (0 means read entire file). Useful for testing/sampling.
    verbose : bool, default False
        Whether to log progress while processing large fragment files.
    progress_interval : int, default 10_000_000
        Number of lines between progress reports when `verbose=True`.
    min_counts : int, optional
        Filter out barcodes with fewer than `min_counts` total fragments from the final output.
    sort_by : str, optional
        Column name to sort the final output by (descending), e.g. "frequency_count" or "percentage".
    as_polars : bool, default False
        If True and Polars is installed, return a `polars.DataFrame`; otherwise return a `pandas.DataFrame`.
    save_path : str or Path, optional
        If provided, save the resulting DataFrame to this file path (supports .tsv, .csv, .gz).

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        DataFrame with per-cell barcode metrics:
        - `CB`: Cell barcode
        - `frequency_count`: Total number of fragments for this barcode
        - `shared_count`: Number of share-coordinate fragments for this barcode
        - `ratio`: Fraction of share-coordinate fragments (`shared_count / frequency_count`)
        - `reads_count`: (if 5th column available) Total read counts for this barcode
        - `shared_reads_count`: (if 5th column available) Read counts in share-coordinate fragments
        - `reads_ratio`: (if 5th column available) Fraction of shared reads
    """
    if min_sharing < 2:
        raise ValueError("min_sharing must be at least 2")

    whitelist = None
    if whitelist_cells is not None:
        whitelist = set(str(c) for c in whitelist_cells if c is not None)

    include_chroms = None
    if chromosome is not None:
        if isinstance(chromosome, str):
            include_chroms = {chromosome}
        else:
            include_chroms = set(chromosome)

    exclude_chroms = None
    if exclude_chromosome is not None:
        if isinstance(exclude_chromosome, str):
            exclude_chroms = {exclude_chromosome}
        else:
            exclude_chroms = set(exclude_chromosome)

    total_frags = defaultdict(int)
    shared_frags = defaultdict(int)
    total_reads = defaultdict(int)
    shared_reads = defaultdict(int)
    has_reads_col = False

    barcode_order = []
    seen_barcodes = set()
    if whitelist is not None:
        for bc in whitelist_cells:
            if bc is not None:
                bc_str = str(bc)
                if bc_str not in seen_barcodes:
                    seen_barcodes.add(bc_str)
                    barcode_order.append(bc_str)

    current_chrom = None
    current_start = -1
    current_end = -1
    current_entries = []
    seen_chroms = set()
    line_count = 0

    def _flush_coord():
        nonlocal current_entries
        if not current_entries:
            return

        if distinct_barcodes:
            if len(current_entries) < min_sharing:
                is_shared = False
            else:
                bcs = {item[0] for item in current_entries}
                is_shared = len(bcs) >= min_sharing
        else:
            is_shared = len(current_entries) >= min_sharing

        for item_bc, item_reads in current_entries:
            if whitelist is not None and item_bc not in whitelist:
                continue
            total_frags[item_bc] += 1
            total_reads[item_bc] += item_reads
            if is_shared:
                shared_frags[item_bc] += 1
                shared_reads[item_bc] += item_reads

        current_entries = []

    for raw_line in _iter_fragment_lines(
        fragments,
        assume_sorted=assume_sorted,
        sort_memory=sort_memory,
        sort_threads=sort_threads,
    ):
        line_count += 1
        if max_lines > 0 and line_count > max_lines:
            break

        if verbose and line_count % progress_interval == 0:
            print(f"Done processing {line_count / 1_000_000:.1f} million lines")

        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        fields = line.split("\t")
        if len(fields) < 4:
            continue

        chrom = fields[0]

        if (include_chroms is not None and chrom not in include_chroms) or (
            exclude_chroms is not None and chrom in exclude_chroms
        ):
            if current_chrom is not None and chrom != current_chrom:
                _flush_coord()
                if assume_sorted:
                    seen_chroms.add(current_chrom)
                current_chrom = None
                current_start = -1
                current_end = -1
            continue

        try:
            start = int(fields[1])
            end = int(fields[2])
        except ValueError:
            continue

        bc = fields[3]
        if whitelist_only and whitelist is not None and bc not in whitelist:
            continue

        reads = 1
        if len(fields) >= 5:
            try:
                reads = int(fields[4])
                has_reads_col = True
            except ValueError:
                if len(fields) >= 6:
                    try:
                        reads = int(fields[5])
                        has_reads_col = True
                    except ValueError:
                        reads = 1

        if (chrom != current_chrom) or (start != current_start) or (end != current_end):
            _flush_coord()

            if assume_sorted:
                if chrom == current_chrom:
                    if start < current_start or (
                        start == current_start and end < current_end
                    ):
                        raise ValueError(
                            f"Fragment file '{fragments}' is not sorted by genomic coordinates: "
                            f"encountered {chrom}:{start}-{end} after {current_chrom}:{current_start}-{current_end}. "
                            "Please pass assume_sorted=False to sort on the fly, or sort with 'sort -k1,1 -k2,2n -k3,3n'."
                        )
                elif current_chrom is not None:
                    if chrom in seen_chroms:
                        raise ValueError(
                            f"Fragment file '{fragments}' has non-contiguous chromosome records: "
                            f"chromosome '{chrom}' encountered again after '{current_chrom}'. "
                            "Please pass assume_sorted=False to sort on the fly, or sort with 'sort -k1,1 -k2,2n -k3,3n'."
                        )
                    seen_chroms.add(current_chrom)

            current_chrom = chrom
            current_start = start
            current_end = end

        current_entries.append((bc, reads))
        if whitelist is None and bc not in seen_barcodes:
            seen_barcodes.add(bc)
            barcode_order.append(bc)

    _flush_coord()

    tot_frags_list = [total_frags.get(bc, 0) for bc in barcode_order]
    sh_frags_list = [shared_frags.get(bc, 0) for bc in barcode_order]
    ratios = [
        (sh / tot) if tot > 0 else 0.0
        for sh, tot in zip(sh_frags_list, tot_frags_list)
    ]

    data = {
        "CB": barcode_order,
        "frequency_count": tot_frags_list,
        "shared_count": sh_frags_list,
        "ratio": ratios,
    }

    if has_reads_col:
        tot_r_list = [total_reads.get(bc, 0) for bc in barcode_order]
        sh_r_list = [shared_reads.get(bc, 0) for bc in barcode_order]
        r_ratios = [
            (sh / tot) if tot > 0 else 0.0
            for sh, tot in zip(sh_r_list, tot_r_list)
        ]
        data["reads_count"] = tot_r_list
        data["shared_reads_count"] = sh_r_list
        data["reads_ratio"] = r_ratios

    result_df = pd.DataFrame(data)

    if min_counts is not None:
        result_df = result_df[result_df["frequency_count"] >= min_counts].copy()
        result_df.reset_index(drop=True, inplace=True)

    if sort_by is not None:
        if sort_by in result_df.columns:
            result_df = result_df.sort_values(by=sort_by, ascending=False).reset_index(drop=True)
        else:
            raise KeyError(
                f"sort_by column '{sort_by}' not in result columns: {list(result_df.columns)}"
            )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        sep = "\t" if save_path.name.endswith((".tsv", ".tsv.gz")) else ","
        result_df.to_csv(save_path, sep=sep, index=False)

    if as_polars:
        if pl is None:
            raise ImportError(
                "polars is not installed. Please install polars to return a polars DataFrame."
            )
        return pl.from_pandas(result_df)

    return result_df