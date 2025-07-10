import pandas as pd
import numpy as np

def encode_with_intermediate_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes a SNP DataFrame using the user's proposed two-step strategy.

    1.  Applies a global map to assign unique integers to each homozygote.
    2.  Iterates through columns to map the identified homozygote codes to 0 and 2.
    """
    # Step 1: Apply the global intermediate mapping. This is very fast.
    intermediate_map = {
        'AG': 1, 'GA': 1, 'AC': 1, 'CA': 1, 'AT': 1, 'TA': 1, # Heterozygotes -> 1
        'CG': 1, 'GC': 1, 'GT': 1, 'TG': 1, 'DI': 1, 'ID': 1, # Heterozygotes -> 1
        'AA': 3, 'CC': 4, 'DD': 5, 'GG': 6, 'II': 7, 'TT': 8, # Homozygotes -> unique integers >         
        'NN': -1 # Missing -> -1
    }
    encoded_df = df.replace(intermediate_map)

    # Step 2: Apply the final column-wise mapping rule.
    # This loop is fast because it operates on simple integer columns.
    for col in encoded_df.columns:
        # Find the unique homozygote codes (values > 2) in the column.
        hom_codes = [code for code in encoded_df[col].unique() if code > 2]

        if len(hom_codes) == 2:
            # The larger code becomes the alternate (2).
            # The smaller code becomes the reference (0).
            # Note: This correctly maps heterozygotes (1) without change.
            replacement_map = {max(hom_codes): 2, min(hom_codes): 0}
            encoded_df[col] = encoded_df[col].replace(replacement_map)

        elif len(hom_codes) == 1:
            # This is a monomorphic SNP; its one homozygote type is the reference (0).
            encoded_df[col] = encoded_df[col].replace({hom_codes[0]: 0})

    return encoded_df

def get_overlap_partitions(*datasets):
    """
    Calculates all mutually exclusive partitions for any number of input sets.

    This function dynamically handles 2, 3, or more sets, generating all the
    unique "only" and "common" regions.

    Args:
        *datasets: A variable number of tuples, where each tuple contains
                   a name (str) and a set object.
                   Example: ('Train', {1, 2}), ('Test', {2, 3})

    Returns:
        A dictionary where keys describe the partition (e.g., 'Train-Only',
        'Common (T&V)') and values are the sets of elements unique to that
        partition.
    """

    names = [d[0] for d in datasets]
    sets = [d[1] for d in datasets]
    num_sets = len(sets)
    partitions = {}

    # Iterate through all possible non-empty combinations of sets (from 1 to 2^n - 1)
    # Each integer 'i' represents a unique combination (a bitmask).
    for i in range(1, 1 << num_sets):
        sets_in_combo = []
        names_in_combo = []
        sets_not_in_combo = []

        for j in range(num_sets):
            # Check if the j-th bit is set in the bitmask 'i'
            if (i >> j) & 1:
                sets_in_combo.append(sets[j])
                names_in_combo.append(names[j])
            else:
                sets_not_in_combo.append(sets[j])

        # Find elements that are in ALL sets of the current combination
        intersection_of_combo = set.intersection(*sets_in_combo)

        # Exclude elements that appear in ANY of the other sets
        union_of_others = set.union(*sets_not_in_combo) if sets_not_in_combo else set()
        final_partition = intersection_of_combo - union_of_others

        # --- Generate a descriptive key based on the user's preferred style ---
        # Abbreviate 'Test' to 's' for the key to maintain compatibility
        key_name_initials = [n[0] if n.lower() != 'test' else 's' for n in names_in_combo]

        if len(names_in_combo) == 1:
            key = f'{names_in_combo[0]}-Only'
        elif len(names_in_combo) == num_sets:
            key = 'Common (All)'
        else:
            key = f'Common ({ "&".join(key_name_initials) })'

        partitions[key] = final_partition

    return partitions

def split_by_environment(df, split_col='Environment', train_ratio=0.7, seed=42):
    """
    Splits a DataFrame into training and validation sets based on unique values in a specified column.
    Returns:
        tuple: A tuple containing two DataFrames: (train_df, val_df).
    """

    unique_envs = df[split_col].unique()
    np.random.seed(seed)
    np.random.shuffle(unique_envs)

    train_count = int(len(unique_envs) * train_ratio)

    
    # Select the environments for each set
    train_envs = set(unique_envs[:train_count])
    val_envs = set(unique_envs[train_count : ])
    
    # Create the new DataFrames based on the environment split
    train_df = df[df[split_col].isin(train_envs)].copy()
    val_df = df[df[split_col].isin(val_envs)].copy()

    print(f"Training environments: {len(train_envs)} ({len(train_df)} rows)")
    print(f"Validation environments: {len(val_envs)} ({len(val_df)} rows)")
    return train_df, val_df