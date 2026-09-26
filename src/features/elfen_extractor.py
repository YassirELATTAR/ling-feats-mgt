import polars as pl
import os
import argparse
from elfen import Extractor

def main(args):
    
    input_file = args.input
    output_folder = args.output
    create_combined = args.combined
    normalize_method = args.normalize
    


    # List of feature groups
    feature_groups = [
        "surface",
        "lexical_richness",
        "emotion",
        "psycholinguistic",
        "readability",
        "morphological",
        "pos",
        "dependency",
        "semantic",
        "entities",
        "information",
    ]
    
    print(f"\n=== Processing dataset: {input_file} ===")
    print(f"Normalization method: {normalize_method}")
    
    # Read the dataset
    df = pl.read_csv(input_file)
    
    # Create output folder
    dataset_name = os.path.splitext(os.path.basename(input_file))[0]
    dataset_output_folder = os.path.join(output_folder, f"{dataset_name}_features")
    os.makedirs(dataset_output_folder, exist_ok=True)
    
    # Columns to exclude from saved files (except 'id' which we keep)
    exclude_columns = {'nlp', 'id', 'comment_id', 'post_id'}
    
    # Track all previously extracted features to avoid duplicates
    all_previous_features = set()

    # Initialize extractor according to Max notes
    new_extractor = Extractor(data=df, model='en_core_web_lg')
    
    # Store all features for combined file if requested
    combined_features_data = None
    
    # Loop through each feature group
    for group in feature_groups:
        print(f"Extracting {group} features...")
        
        # # Create clean data without preprocessing columns for fresh extractor
        # clean_data = df.select([col for col in df.columns if col not in ['nlp']])
        # group_extractor = Extractor(data=clean_data)
        
        # Extract features for this group
        new_extractor.extract_feature_group(feature_group=group)

        if normalize_method == 'token' and 'n_tokens' not in new_extractor.data.columns:
            # Special case for psycholguistic features as they don't have n_tokens required for token normalization
            print(f"Skipping token normalization for {group} features as they don't have n_tokens,\nusing standard notrmalization instead.")
            new_extractor.normalize("all")
        else:
            # Apply normalization if specified
            if normalize_method == 'token':
                new_extractor.token_normalize("all") ## The new method that the study was based on
            elif normalize_method == 'standard':
                new_extractor.normalize("all")
            elif normalize_method == 'ratio':
                new_extractor.ratio_normalize("all", "token")
            elif normalize_method == 'rescale':
                new_extractor.rescale("all")
        
        print(f"-----------------------------------\n\nExtracted features for {group} are:")
        print(new_extractor.data.columns)
        
        # Get columns that are safe for CSV (only basic types)
        safe_columns = []
        for col in new_extractor.data.columns:
            dtype = new_extractor.data[col].dtype
            # Only keep simple data types that CSV can handle
            if dtype in [pl.String, pl.Utf8, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                        pl.Float32, pl.Float64, pl.Boolean]:
                safe_columns.append(col)
        
        # Filter columns: keep 'id' + new features only (exclude text, nlp, n_tokens)
        columns_to_save = []
        for col in safe_columns:
            if col in ['id', 'comment_id', 'doc_id']:
                columns_to_save.append(col)
            elif col not in exclude_columns and col not in all_previous_features:
                columns_to_save.append(col)
                all_previous_features.add(col) # Track this feature for future groups
        
        # Select only the columns we want to save
        data_to_save = new_extractor.data.select(columns_to_save)
        
        # Save individual group file
        output_file = os.path.join(dataset_output_folder, f"{group}_features.csv")
        data_to_save.write_csv(output_file)
        print(f"Saved {len(columns_to_save)} columns to {group}_features.csv")
        
        # Prepare data for combined file if requested
        if create_combined:
            if combined_features_data is None:
                combined_features_data = data_to_save
            else:
                # Join with existing data on 'icomment_id'
                combined_features_data = combined_features_data.join(data_to_save, on='doc_id', how='left')
    
    # Save combined features file if requested
    if create_combined and combined_features_data is not None:
        combined_output_file = os.path.join(dataset_output_folder, "all_features_combined.csv")
        combined_features_data.write_csv(combined_output_file)
        print(f"\nSaved combined features file: all_features_combined.csv")
        print(f"Combined file contains {len(combined_features_data.columns)} columns")
    
    print(f"All feature groups extracted and saved for {dataset_name}!")

if __name__ == "__main__":   

    
    # Using the command line arguments parser:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract linguistic features using Elfen toolkit')
    parser.add_argument('--input', '-i', required=True, help='Input file path')
    parser.add_argument('--output', '-o', required=True, help='Output directory path')
    parser.add_argument('--combined', '-c', action='store_true', help='Create combined features file')
    parser.add_argument('--normalize', '-n', choices=['none','token', 'standard', 'ratio', 'rescale'], 
                    default='token', help='Normalization method (default: token)')
    
    args = parser.parse_args()

    main(args)

    # # Created this for testing putposes:
    # import glob

    # # Test on CSV files in test folder
    # test_files = glob.glob("../data/test_cmv/*.csv")
    # for csv_file in test_files:
    #     args = argparse.Namespace(
    #         input=csv_file,
    #         output="results/test_cmv/",
    #         combined=True,
    #         normalize="standard"
    #         )
    #     #for each csv file in the test folder:
    #     main(args)


    ### Testing for different normalization methods:
    # test_files = glob.glob("../data/test/*.csv")
    # for norm in ['none', 'ratio', 'rescale']:
    #     for csv_file in test_files:
    #         args = argparse.Namespace(
    #             input=csv_file,
    #             output=f"results/tests_{norm}/",
    #             combined=True,
    #             normalize=norm
    #             )
    #         #for each csv file in the test folder:
    #         main(args)