import os
import json
import random
import argparse
from collections import defaultdict
from glob import glob

def parse_filename(filename):
    """
    Given a filename e.g. '3P50A03R02.dat' or '6P01A05R1.dat',
    parse out the activity label, subject ID, activity type, repetition, etc.
    We only need the first character (K) to identify the activity class.
    If your naming convention is strictly K PXX AYY RZ .dat,
    for example, '3P50A03R02.dat':
      - The first character '3' indicates the activity (in your set {1..6})
    E.g. command:
    python /project_ghent/Mostafa/ActivityRecognition/DistInference/src/DistInference/five_fold_split.py \
  --data_roots "/project_ghent/Mostafa/ActivityRecognition/1 December 2017 Dataset" \
               "/project_ghent/Mostafa/ActivityRecognition/2 March 2017 Dataset" \
               "/project_ghent/Mostafa/ActivityRecognition/3 June 2017 Dataset" \
               "/project_ghent/Mostafa/ActivityRecognition/4 July 2018 Dataset" \
               "/project_ghent/Mostafa/ActivityRecognition/5 February 2019 UoG Dataset" \
               "/project_ghent/Mostafa/ActivityRecognition/6 February 2019 NG Homes Dataset" \
               "/project_ghent/Mostafa/ActivityRecognition/7 March 2019 West Cumbria Dataset" \
  --k 5 \
  --seed 42 \
  --output_path "five_folds.json"
    """
    base = os.path.basename(filename)
    # Remove extension
    base = os.path.splitext(base)[0]
    # The first character K is the activity code
    # e.g., '3' => fall or pick up object etc. (based on your provided mapping).
    # You can further parse if needed, but for now let's assume '3' is the label
    activity_class = base[0]  
    return activity_class

def collect_dat_files(root_directories):
    """
    root_directories: List of strings representing your data folders, e.g.
       [
         '/project_ghent/Mostafa/ActivityRecognition/1 December 2017 Dataset',
         '/project_ghent/Mostafa/ActivityRecognition/2 March 2017 Dataset',
         ...
       ]
    Returns a list of all .dat file paths.
    """
    all_dat_files = []
    for root_dir in root_directories:
        dat_files = glob(os.path.join(root_dir, "*.dat"))
        all_dat_files.extend(dat_files)
    return all_dat_files

def stratified_k_fold_split(files, k=5, seed=42):
    """
    Given a list of file paths, each with an activity class, 
    produce a stratified k-fold split.

    Returns:
       A list of length k, where each element is a dict:
         {"train": [...list of file paths...], "test": [...list of file paths...] }
    """
    random.seed(seed)

    # Group files by activity
    activity_to_files = defaultdict(list)
    for f in files:
        activity_class = parse_filename(f)
        activity_to_files[activity_class].append(f)

    # For each activity class, shuffle the file list
    for act_class in activity_to_files:
        random.shuffle(activity_to_files[act_class])

    # Create k empty folds (each a dict with "train" and "test")
    folds = []
    for _ in range(k):
        folds.append({"train": [], "test": []})

    # Distribute files across folds in a round-robin fashion or chunked approach
    # so that each fold has a roughly equal proportion of each activity.
    for act_class, file_list in activity_to_files.items():
        # number of files for this activity
        n = len(file_list)
        fold_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
        start_idx = 0
        for i, size in enumerate(fold_sizes):
            test_files_for_fold = file_list[start_idx : start_idx + size]
            start_idx += size
            # For the i-th fold, we set these as test, the rest as train
            # Actually, for cross-validation, each fold has a unique test set
            # Typically, you'd want to keep fold i's test files out of i's train
            # So we'll assign test_files_for_fold to folds[i]['test']
            # and the rest to folds[i]['train'] at the end. 
            folds[i]["test"].extend(test_files_for_fold)

    # Now, for each fold, the train set is all files *not* in the fold’s test set
    all_files_set = set(files)
    for i in range(k):
        test_set_i = set(folds[i]["test"])
        folds[i]["train"] = list(all_files_set - test_set_i)

    return folds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_path", type=str, default="folds.json", 
                        help="Where to save the split info")
    parser.add_argument("--data_roots", nargs='+', required=True,
                        help="List of root directories containing .dat files.")
    args = parser.parse_args()

    # 1) Collect all dat files
    all_files = collect_dat_files(args.data_roots)
    print(f"Collected {len(all_files)} .dat files from directories: {args.data_roots}")

    # 2) Generate the folds
    folds = stratified_k_fold_split(all_files, k=args.k, seed=args.seed)

    # 3) Save to JSON (or any other format)
    with open(args.output_path, "w") as f:
        json.dump(folds, f, indent=2)
    print(f"Saved {args.k}-fold split info to {args.output_path}")


if __name__ == "__main__":
    main()
