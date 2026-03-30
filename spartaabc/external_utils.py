"""
External Statistics Utilities for SpartaV2.
Helper functions for extracting and processing data for external (Python-based) statistics.
Implements the same unique gap counting logic as MSAStats C++ code.
"""
import numpy as np
from typing import List, Dict, Tuple

def fill_unique_gaps_map(msa: List[str]) -> Dict[Tuple[int, int], List[int]]:
    """
    Replicate C++ fillUniqueGapsMap() logic exactly.
    
    This function implements the same algorithm as the C++ code in
    MsaStatsCalculator::fillUniqueGapsMap().
    
    Args:
        msa: List of aligned sequences (strings)
    
    Returns:
        Dictionary with key=(start_pos, end_pos), value=[length, count]
        where length is the gap length and count is how many sequences
        have this exact gap pattern.
    """
    unique_indel_map = {}
    num_sequences = len(msa)
    
    if num_sequences == 0:
        return unique_indel_map
    
    msa_length = len(msa[0]) if msa[0] else 0
    
    # Validate all sequences have same length
    for j, seq in enumerate(msa):
        if len(seq) != msa_length:
            raise ValueError(f"MSA sequence {j} has length {len(seq)}, expected {msa_length}. All sequences must be aligned (same length).")
    
    for j in range(num_sequences):
        previous_is_indel = 0
        curr_start_indel_point = -1
        curr_end_indel_point = -1
        
        for i in range(msa_length):
            if msa[j][i] == '-' and previous_is_indel == 0:
                # Start of new gap
                previous_is_indel = 1
                curr_start_indel_point = i
                curr_end_indel_point = i
                
            elif msa[j][i] == '-' and previous_is_indel == 1:
                # Continue gap
                curr_end_indel_point += 1
                
            else:
                # We're on a character (not gap)
                if curr_start_indel_point == -1:
                    previous_is_indel = 0
                    continue
                
                # We have an indel - put in map
                curr_pair = (curr_start_indel_point, curr_end_indel_point)
                curr_length = curr_end_indel_point - curr_start_indel_point + 1
                
                if curr_pair not in unique_indel_map:
                    # New entry
                    unique_indel_map[curr_pair] = [curr_length, 1]
                else:
                    # Already exists - raise counter
                    unique_indel_map[curr_pair][1] += 1
                
                previous_is_indel = 0
                curr_start_indel_point = -1
                curr_end_indel_point = -1
        
        # Handle gap at end of sequence
        if curr_start_indel_point != -1:
            curr_pair = (curr_start_indel_point, curr_end_indel_point)
            curr_length = curr_end_indel_point - curr_start_indel_point + 1
            
            if curr_pair not in unique_indel_map:
                unique_indel_map[curr_pair] = [curr_length, 1]
            else:
                unique_indel_map[curr_pair][1] += 1
    
    return unique_indel_map


def get_unique_gap_lengths(msa: List[str]) -> np.ndarray:
    """
    Get unique gap lengths, fully consistent with C++ MsaStatsCalculator logic.

    Matches the two-step C++ flow in recomputeStats():
      1. trimMSAFromAllIndelPositionAndgetSummaryStatisticsFromIndelCounter()
         — removes columns where every sequence has a gap
      2. fillUniqueGapsMap()
         — collects unique (start, end) gap positions across all sequences

    Args:
        msa: List of aligned sequences (no FASTA headers)

    Returns:
        numpy array of unique gap lengths (one per unique alignment position)
    """
    if not msa or not msa[0]:
        return np.array([], dtype=np.intp)

    n_seqs = len(msa)
    seq_len = len(msa[0])

    # Build 2D boolean gap mask — all MSA characters are ASCII
    msa_bytes = b''.join(seq.encode() for seq in msa)
    if len(msa_bytes) != n_seqs * seq_len:
        raise ValueError(
            "MSA sequences have different lengths. All sequences must be aligned."
        )
    msa_array = np.frombuffer(msa_bytes, dtype=np.uint8).reshape(n_seqs, seq_len)
    is_gap = (msa_array == ord('-'))

    # Step 1: trim all-gap columns
    # C++: removes positions where _indelCounter[j] == _numberOfSequences
    all_gap_cols = np.all(is_gap, axis=0)
    is_gap = is_gap[:, ~all_gap_cols]
    if is_gap.shape[1] == 0:
        return np.array([], dtype=np.intp)

    # Step 2: detect gap-start (+1) and gap-end (-1) transitions
    # Pad with a False column on each side so boundary gaps are captured correctly
    padded = np.concatenate([
        np.zeros((n_seqs, 1), dtype=np.int8),
        is_gap.astype(np.int8),
        np.zeros((n_seqs, 1), dtype=np.int8)
    ], axis=1)
    transitions = np.diff(padded, axis=1)

    _, start_cols = np.where(transitions == 1)
    _, end_cols   = np.where(transitions == -1)
    end_cols = end_cols - 1  # shift to last column of each gap run

    if len(start_cols) == 0:
        return np.array([], dtype=np.intp)

    # Step 3: deduplicate (start, end) pairs
    # Equivalent to C++ map<pair<int,int>, vector<int>> — keys are unique positions
    gap_positions    = np.stack([start_cols, end_cols], axis=1)
    unique_positions = np.unique(gap_positions, axis=0)

    # length = end - start + 1, matches C++: currLength = currEndIndelPoint - currStartIndelPoint + 1
    return unique_positions[:, 1] - unique_positions[:, 0] + 1


def compute_gap_density(msa: List[str]) -> float:
    """
    Compute gap density on the trimmed MSA.

    gap_density = total '-' characters / (n_seqs × trimmed_length)

    Uses the same all-gap column trimming step as get_unique_gap_lengths,
    so this value is consistent with the other extended statistics.

    Args:
        msa: List of aligned sequences (no FASTA headers)

    Returns:
        float in [0, 1] — proportion of the trimmed alignment that is gaps
    """
    if not msa or not msa[0]:
        return 0.0

    n_seqs = len(msa)
    seq_len = len(msa[0])

    msa_bytes = b''.join(seq.encode() for seq in msa)
    if len(msa_bytes) != n_seqs * seq_len:
        return 0.0

    msa_array = np.frombuffer(msa_bytes, dtype=np.uint8).reshape(n_seqs, seq_len)
    is_gap = (msa_array == ord('-'))

    # Trim all-gap columns — same step as get_unique_gap_lengths
    all_gap_cols = np.all(is_gap, axis=0)
    is_gap_trimmed = is_gap[:, ~all_gap_cols]

    total = is_gap_trimmed.size
    if total == 0:
        return 0.0

    return float(np.sum(is_gap_trimmed) / total)


def calculate_extended_stats(msa: List[str]) -> List[float]:
    """
    Calculate extended statistics: quantiles of unique gap lengths.
    
    Uses the same unique gap counting logic as C++ MSAStats.
    
    Args:
        msa: List of aligned sequences (strings)
    
    Returns:
        List[float]: [q25, q50, q70] - 25th, 50th (median), and 70th percentiles
                     of unique gap lengths
    """
    unique_gap_lengths = get_unique_gap_lengths(msa)
    
    if len(unique_gap_lengths) == 0:
        return [0.0, 0.0, 0.0]
    
    # Calculate quantiles using numpy
    # Use 'nearest' method to get actual values from the data (not interpolated)
    quantiles = np.percentile(unique_gap_lengths, [25, 50, 70], method='nearest')
    
    return quantiles.tolist()

