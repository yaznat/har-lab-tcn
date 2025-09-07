import pandas as pd
import numpy as np
from glob import glob

def build_temporal_dataset(
    directory_path: str, 
    temporal_dim: int, 
    test_fraction: float = 0.2,
    filter_files_by_column_count: int = None,
    activity_types: list = None,
    excluded_columns: list= None
):
    """
    Parses csv data from a directory of files into a dataset with train and test data. **Returns:** X_train, y_train, X_test, y_test <p>
    Each file is assumed to contain data collected for one person: label classes are split by file. <p>
    Reserves the last `test_fraction` percentage of each contiguous activity sequence for testing. <p>
    Measurements are grouped into sequences according to temporal_dim, with one label per sequence.
    """
    # "time", "person", and "activity" are not properly numerical measurements
    columns_to_exclude = ["time", "person", "activity"] + (excluded_columns or [])

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    label = 0

    # Load file by file since each file is one class
    for file_name in sorted(glob(directory_path + "/*.csv")):
        raw = pd.read_csv(file_name)
        
        # Skip files that don't match desired column count
        if filter_files_by_column_count is not None:
            if raw.shape[1] != filter_files_by_column_count:
                print(f"Skipping {file_name}: column count {raw.shape[1]} != {filter_files_by_column_count}")
                continue
        
        # Filter by activity types if specified
        if activity_types is not None:
            raw = raw[raw['activity'].isin(activity_types)]
            if len(raw) == 0:
                print(f"Skipping {file_name}: no data for activities {activity_types}")
                continue
        
        # Group by activity within this person's data
        activities = raw['activity'].unique()
        person_X_train, person_X_test = [], []
        person_y_train, person_y_test = [], []
        
        for activity in activities:
            # Acquire all data for this activity
            activity_data = raw[raw['activity'] == activity].copy()
            # Exclude columns as specified
            activity_data = activity_data.drop(columns=columns_to_exclude)
            
            # Calculate split point
            total_rows = len(activity_data)
            num_sequences = total_rows // temporal_dim
                
            # Truncate to avoid remainder in temporal windows
            activity_data = activity_data[:num_sequences * temporal_dim]
            
            # Temporal split: first part for training, last part for testing
            train_sequences = int(num_sequences * (1 - test_fraction))
            train_end = train_sequences * temporal_dim
                
            # Slice data into train and test
            train_data = activity_data[:train_end]
            test_data = activity_data[train_end:]

            
            if len(train_data) > 0:
                # Reshape data to (temporal_dim, features)
                train_sequences_shaped = train_data.to_numpy().reshape(-1, temporal_dim, activity_data.shape[1])

                person_X_train.append(train_sequences_shaped)
                # Add one label per sequence
                person_y_train.append(np.full(len(train_sequences_shaped), label))
                
            if len(test_data) > 0:
                # Reshape data to (temporal_dim, features)
                test_sequences_shaped = test_data.to_numpy().reshape(-1, temporal_dim, activity_data.shape[1])

                person_X_test.append(test_sequences_shaped)
                # Add one label per sequence
                person_y_test.append(np.full(len(test_sequences_shaped), label))
           
        # Add this person's data if they have any
        if person_X_train:
            X_train.extend(person_X_train)
            y_train.extend(person_y_train)
        
        if person_X_test:
            X_test.extend(person_X_test)
            y_test.extend(person_y_test)
        
        # New file, new person - increment label by 1
        label += 1
    
    # Concatenate all data
    X_train = np.concatenate(X_train, axis=0) if X_train else np.array([])
    y_train = np.concatenate(y_train, axis=0) if y_train else np.array([])
    X_test = np.concatenate(X_test, axis=0) if X_test else np.array([])
    y_test = np.concatenate(y_test, axis=0) if y_test else np.array([])
    
    print(f"Train: {len(X_train)} sequences, Test: {len(X_test)} sequences")
    
    return X_train, y_train, X_test, y_test


