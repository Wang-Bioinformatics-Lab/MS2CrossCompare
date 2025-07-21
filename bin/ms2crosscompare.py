#!/usr/bin/env python3
"""
MS2CrossCompare - Cross-comparison of mass spectrometry data between two folders

This script compares all spectra from folder1 against all spectra from folder2,
using the indexing approach from gnps_index.py for efficient computation.
"""

import os
import glob
import argparse
import time
import numpy as np
from typing import List, Tuple, Dict
import sys

# Import functions from gnps_index.py
from gnps_index import (
    parse_files_in_parallel,
    create_index,
    find_bin_range,
    compute_all_pairs,
    calculate_exact_score_GNPS,
    calculate_exact_score_GNPS_multi_charge,
    SHIFTED_OFFSET,
    MINMATCHES,
    TOPPRODUCTS,
    ADJACENT_BINS
)
from numba.typed import List as NumbaList
import numba


@numba.njit
def compute_cross_comparisons(folder1_spectra, folder2_spectra, shared_entries, shifted_entries, tolerance, threshold, scoring_func, minmatches=6, max_shift=1000.0):
    results = NumbaList()
    for query_idx in range(len(folder1_spectra)):
        query_spec = folder1_spectra[query_idx]
        precursor_mz_query = query_spec[2]
        upper_bounds = np.zeros(len(folder2_spectra), dtype=np.float32)
        match_counts = np.zeros(len(folder2_spectra), dtype=np.int32)
        for peak_idx in range(len(query_spec[0])):
            mz = query_spec[0][peak_idx]
            intensity = query_spec[1][peak_idx]
            precursor_mz = query_spec[2]
            shared_bin = np.int64(round(mz / tolerance))
            shifted_bin = np.int64(round((precursor_mz - mz + SHIFTED_OFFSET) / tolerance))
            for entries, bin_val in [(shared_entries, shared_bin), (shifted_entries, shifted_bin)]:
                for delta in ADJACENT_BINS:
                    target_bin = bin_val + delta
                    start, end = find_bin_range(entries, target_bin)
                    pos = start
                    while pos < end and entries[pos][0] == target_bin:
                        spec_idx = entries[pos][1]
                        precursor_mz_target = folder2_spectra[spec_idx][2]
                        if abs(precursor_mz_query - precursor_mz_target) > max_shift:
                            pos += 1
                            continue
                        upper_bounds[spec_idx] += intensity * entries[pos][4]
                        match_counts[spec_idx] += 1
                        pos += 1
        candidates = NumbaList()
        for target_idx in range(len(folder2_spectra)):
            if (upper_bounds[target_idx] >= threshold and 
                match_counts[target_idx] >= minmatches):
                candidates.append((target_idx, upper_bounds[target_idx]))
        for target_idx, _ in candidates:
            target_spec = folder2_spectra[target_idx]
            score, shared, shifted = scoring_func(query_spec, target_spec, tolerance)
            if score >= threshold:
                results.append((query_idx, target_idx, score))
    return results


def get_all_spectra_files(folder_path: str) -> List[str]:
    """Get all mzML and mgf files from a folder recursively"""
    mzml_files = glob.glob(os.path.join(folder_path, "**/*.mzML"), recursive=True)
    mgf_files = glob.glob(os.path.join(folder_path, "**/*.mgf"), recursive=True)
    return mzml_files + mgf_files


def parse_files_with_metadata(file_paths: List[str], threads: int = 1, enable_peak_filtering: bool = False) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Parse files and return both spectra and metadata
    Returns: (parsed_spectra, metadata) where metadata contains (cluster_id, scan_id, file_path)
    """
    parsed_spectra = []
    metadata = []
    
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        
        # Parse this file
        from gnps_index import parse_one_file
        file_spectra = parse_one_file(file_path, enable_peak_filtering)
        
        # Create metadata for each spectrum in this file
        for i, (scan_id, spectrum) in enumerate(file_spectra):
            # Create cluster_id as filename_scan
            # Follow the same logic as gnps_index.py:
            # - Use scan_id if available (from MGF scans field or mzML scan number)
            # - Only use index if scan_id is -1 (no scan information available)
            if scan_id == -1:  # No scan information available, use index
                cluster_id = f"{file_name}_{i + 1}"
            else:  # Use the actual scan number from file
                cluster_id = f"{file_name}_{scan_id}"
            
            metadata.append((cluster_id, scan_id, file_path))
            parsed_spectra.append((scan_id, spectrum))
    
    return parsed_spectra, metadata


def create_spectrum_metadata(parsed_spectra: List[Tuple], file_paths: List[str]) -> List[Tuple]:
    """
    Create metadata for each spectrum including filename and scan information
    Returns: List of (cluster_id, scan_id, file_path) tuples
    """
    # This function is now deprecated, use parse_files_with_metadata instead
    raise DeprecationWarning("Use parse_files_with_metadata instead")


def cross_compare_folders(
    folder1_path: str,
    folder2_path: str,
    tolerance: float = 0.01,
    threshold: float = 0.7,
    threads: int = 1,
    alignment_strategy: str = "index_single_charge",
    enable_peak_filtering: bool = False,
    minmatches: int = 6,
    max_shift: float = 1000.0
) -> List[Tuple]:
    """
    Compare all spectra from folder1 against all spectra from folder2
    
    Returns: List of (cluster1, cluster2, delta_mz, cosine_score) tuples
    """
    
    # Get all files from both folders
    print(f"Scanning folder1: {folder1_path}")
    folder1_files = get_all_spectra_files(folder1_path)
    print(f"Found {len(folder1_files)} files in folder1")
    
    print(f"Scanning folder2: {folder2_path}")
    folder2_files = get_all_spectra_files(folder2_path)
    print(f"Found {len(folder2_files)} files in folder2")
    
    if not folder1_files:
        raise ValueError(f"No mzML or mgf files found in folder1: {folder1_path}")
    if not folder2_files:
        raise ValueError(f"No mzML or mgf files found in folder2: {folder2_path}")
    
    # Parse all files with metadata
    print("Parsing folder1 files...")
    t0 = time.time()
    folder1_spectra, folder1_metadata = parse_files_with_metadata(folder1_files, threads, enable_peak_filtering)
    t1 = time.time()
    print(f"Parsed {len(folder1_spectra)} spectra from folder1 in {t1-t0:.2f} seconds")
    
    print("Parsing folder2 files...")
    t0 = time.time()
    folder2_spectra, folder2_metadata = parse_files_with_metadata(folder2_files, threads, enable_peak_filtering)
    t1 = time.time()
    print(f"Parsed {len(folder2_spectra)} spectra from folder2 in {t1-t0:.2f} seconds")
    
    # Convert to numba-compatible format
    print("Converting to numba format...")
    folder1_numba = NumbaList()
    folder2_numba = NumbaList()
    
    for scan_id, spectrum in folder1_spectra:
        folder1_numba.append((
            spectrum[0].astype(np.float32),
            spectrum[1].astype(np.float32),
            np.float32(spectrum[2]),
            np.int32(spectrum[3])
        ))
    
    for scan_id, spectrum in folder2_spectra:
        folder2_numba.append((
            spectrum[0].astype(np.float32),
            spectrum[1].astype(np.float32),
            np.float32(spectrum[2]),
            np.int32(spectrum[3])
        ))
    
    # Build indexes for folder2 (target)
    print("Building indexes for folder2...")
    if alignment_strategy == "index_multi_charge":
        shared_idx = create_index(folder2_numba, False, tolerance, SHIFTED_OFFSET)
        shifted_idx = create_index(folder2_numba, True, tolerance, SHIFTED_OFFSET)
        scoring_func = calculate_exact_score_GNPS_multi_charge
    else:
        shared_idx = create_index(folder2_numba, False, tolerance, SHIFTED_OFFSET)
        shifted_idx = create_index(folder2_numba, True, tolerance, SHIFTED_OFFSET)
        scoring_func = calculate_exact_score_GNPS
    
    # Compare each spectrum from folder1 against all spectra in folder2
    print("Computing cross-comparisons...")
    t0 = time.time()
    
    # Use the optimized cross-comparison function
    matches = compute_cross_comparisons(folder1_numba, folder2_numba, shared_idx, shifted_idx,
                                       tolerance, threshold, scoring_func, minmatches, max_shift)
    
    # Convert matches to results with metadata
    results = []
    for query_idx, target_idx, score in matches:
        # Get metadata
        query_cluster = folder1_metadata[query_idx][0]
        target_cluster = folder2_metadata[target_idx][0]
        
        # Calculate delta m/z
        query_spec = folder1_numba[query_idx]
        target_spec = folder2_numba[target_idx]
        delta_mz = query_spec[2] - target_spec[2]
        
        results.append((query_cluster, target_cluster, delta_mz, score))
    
    t1 = time.time()
    print(f"Computed {len(results)} matches in {t1-t0:.2f} seconds")
    
    return results


def write_results(results: List[Tuple], output_file: str):
    """Write results to TSV file with filename and scan split at the last underscore."""
    print(f"Writing results to {output_file}...")
    with open(output_file, 'w') as f:
        f.write("set1_filename\tset1_scan\tset2_filename\tset2_scan\tdelta_mz\tcosine\n")
        for cluster1, cluster2, delta_mz, cosine in results:
            # Split cluster_id at the last underscore to get filename and scan
            if "_" in cluster1:
                idx = cluster1.rfind("_")
                set1_filename = cluster1[:idx]
                set1_scan = cluster1[idx+1:]
            else:
                set1_filename = cluster1
                set1_scan = ""
            if "_" in cluster2:
                idx = cluster2.rfind("_")
                set2_filename = cluster2[:idx]
                set2_scan = cluster2[idx+1:]
            else:
                set2_filename = cluster2
                set2_scan = ""
            f.write(f"{set1_filename}\t{set1_scan}\t{set2_filename}\t{set2_scan}\t{delta_mz:.3f}\t{cosine:.4f}\n")
    print(f"Wrote {len(results)} results to {output_file}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Cross-compare mass spectrometry data between two folders"
    )
    parser.add_argument("folder1", help="Path to first folder containing mzML/mgf files")
    parser.add_argument("folder2", help="Path to second folder containing mzML/mgf files")
    parser.add_argument("-o", "--output", default="cross_compare_results.tsv",
                        help="Output TSV file (default: cross_compare_results.tsv)")
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="Fragment mass tolerance (default: 0.01)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Similarity score threshold (default: 0.7)")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of processing threads (default: 1)")
    parser.add_argument("--alignment_strategy", type=str, default="index_single_charge",
                        choices=["index_single_charge", "index_multi_charge"],
                        help="Alignment strategy to use (default: index_single_charge)")
    parser.add_argument("--enable_peak_filtering", type=str, default="no",
                    help="Enable peak filtering: yes/no (default: no)")
    parser.add_argument("--minmatches", type=int, default=6, help="Minimum number of matched peaks (default: 6)")
    parser.add_argument("--max_shift", type=float, default=1000.0, help="Maximum allowed precursor m/z difference for comparison (default: 1000.0)")
    return parser.parse_args()


def main():
    args = parse_arguments()

    args.enable_peak_filtering = str(args.enable_peak_filtering).lower() in ["true", "1", "yes"]
    
    print("MS2CrossCompare - Cross-comparison of mass spectrometry data")
    print(f"Folder1: {args.folder1}")
    print(f"Folder2: {args.folder2}")
    print(f"Output: {args.output}")
    print(f"Tolerance: {args.tolerance}")
    print(f"Threshold: {args.threshold}")
    print(f"Threads: {args.threads}")
    print(f"Alignment strategy: {args.alignment_strategy}")
    print(f"Peak filtering: {args.enable_peak_filtering}")
    print(f"Minimum matches: {args.minmatches}")
    print(f"Maximum allowed precursor m/z difference: {args.max_shift}")
    print("-" * 50)
    
    try:
        # Perform cross-comparison
        results = cross_compare_folders(
            folder1_path=args.folder1,
            folder2_path=args.folder2,
            tolerance=args.tolerance,
            threshold=args.threshold,
            threads=args.threads,
            alignment_strategy=args.alignment_strategy,
            enable_peak_filtering=args.enable_peak_filtering,
            minmatches=args.minmatches,
            max_shift=args.max_shift
        )
        
        # Write results
        write_results(results, args.output)
        
        print("Done!")
        print(f"Total matches found: {len(results)}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
