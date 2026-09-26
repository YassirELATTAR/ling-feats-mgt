from elfen_extractor import main
import polars as pl
import argparse
import os
import glob
import re 


# With the help of copilot, we create a batch processing script that can be run in parallel on different servers.
def batch_extract_features(input_base_dir, output_base_dir='features', combined=True, normalize='token'):
    
    # 1. Use glob to find all .csv files recursively
    # The input_base_dir is now the starting point for the partition (e.g., 'data/human/cmv')
    search_path = os.path.join(input_base_dir, '**', '*.csv')
    csv_files = glob.glob(search_path, recursive=True)

    if not csv_files:
        print(f"No CSV files found in '{input_base_dir}' or its subdirectories.")
        return

    print(f"Found {len(csv_files)} files in partition '{input_base_dir}'. Starting batch extraction...")

    # Ensure the output base directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    # # Find the path component *before* the first folder (e.g., the parent of 'data')
    # data_root_parent = os.path.dirname(os.path.abspath(input_base_dir)).split(os.sep)[-1]
    
    # Determine the absolute path to the 'data' root for reliable string replacement.
    # If the user provides 'data/human/cmv', we look for 'data' in the path.
    data_root_prefix = os.path.join('data', '')

    # 2. Iterate and process each file
    for input_file_path in csv_files:
        print(f"\nProcessing input file: {input_file_path}")
        
        # --- A. Construct the full mirrored path string ---
        # 1. Replace the 'data/' prefix with 'features/'. This ensures the full structure.
        #    e.g., 'data/human/cmv/samples.csv' -> 'features/human/cmv/samples.csv'
        output_path_with_file = input_file_path.replace(data_root_prefix, os.path.join(output_base_dir, ''), 1)

        # 2. Get the file name base (e.g., 'samples')
        file_name_base = os.path.splitext(os.path.basename(input_file_path))[0]
        
        # 3. Construct the output directory path by replacing the file name 
        #    with a directory of the same name.
        #    e.g., 'features/human/cmv/samples.csv' -> 'features/human/cmv/samples/'
        output_dir = os.path.join(os.path.dirname(output_path_with_file), file_name_base)
        
        # Create the specific output directory for this CSV's features
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Features will be saved to: {output_dir}")

        # --- B. Configure and run the extraction script ---
        try:
            args = argparse.Namespace(
                input=input_file_path,
                output=output_dir + os.sep,
                combined=combined,
                normalize=normalize
            )
            
            # Run the extraction script
            main(args)
            
            print(f"Successfully extracted features for {input_file_path}")

        except Exception as e:
            print(f"Error processing {input_file_path}: {e}")

    print("\nBatch feature extraction complete.")

# # Set the partition for THIS specific script instance
# # INPUT_PARTITION = 'data/cmv/Human'
# # OUTPUT_ROOT = 'cmv_features'
# for INPUT_PARTITION in [
#    # 'data/train/AI/eli5',
#    # 'data/train/AI/hswag',
#    # 'data/train/AI/sci_gen',
#    # 'data/train/AI/xsum',
#     'data/train/AI/roct',
#     'data/train/AI/wp',
#     'data/train/AI/squad',
#     ]:
    
#     batch_extract_features(
#         input_base_dir=INPUT_PARTITION,
#         output_base_dir=OUTPUT_ROOT,
#         combined=False,
#         normalize='token'
#     )



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, nargs='+', help='one or more input dirs, e.g. data/train/AI/wp')
    p.add_argument('--output', required=True, help='output root, e.g. data_features_prepared')
    p.add_argument('--combined', action='store_true')
    p.add_argument('--normalize', default='token',
                   choices=['none', 'token', 'standard', 'ratio', 'rescale'])
    a = p.parse_args()
    for part in a.input:
        batch_extract_features(part, a.output, combined=a.combined, normalize=a.normalize)