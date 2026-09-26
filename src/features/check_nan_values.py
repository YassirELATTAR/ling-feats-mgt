import polars as pl
import os
import json
import numpy as np
from pathlib import Path

def suggest_replacement_method(df, column):
    """
    Suggest replacement method based on column characteristics.
    Returns: (method, reason)
    """
    # Get valid values (not null, not inf)
    valid_values = df.filter(
        pl.col(column).is_not_null() & 
        pl.col(column).is_finite()
    )[column]
    
    if len(valid_values) == 0:
        return ("drop_column", "All values are invalid")
    
    # Check if binary (0/1)
    unique_vals = valid_values.unique().to_list()
    if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        return ("mode", "Binary feature")
    
    # Check if all integers
    if valid_values.dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
        return ("median", "Integer feature - median preserves integer nature")
    
    # Check distribution for floats
    if valid_values.dtype in [pl.Float32, pl.Float64]:
        mean_val = valid_values.mean()
        median_val = valid_values.median()
        std_val = valid_values.std()
        
        # If std is very small, values are similar
        if std_val < 0.01:
            return ("mean", "Low variance - mean is stable")
        
        # Check skewness (simple check)
        if abs(mean_val - median_val) > std_val:
            return ("median", "Skewed distribution - median is robust")
        else:
            return ("mean", "Normal-ish distribution - mean is appropriate")
    
    return ("median", "Default safe choice")


def check_error_values(root_dir='data_features_prepared'):
    """
    Check all CSV files for NaN, Inf, and other problematic values.
    Generate detailed report sorted by most errors.
    """
    report = {}
    
    # Feature areas to check
    feature_areas = ['surface', 'pos', 'lexical_richness', 'readability', 
                     'information', 'entities', 'semantic', 'emotion',
                     'psycholinguistic', 'morphological', 'dependency', 'combined']
    
    print("Scanning for error values... \nDirectory:", root_dir)
    
    # Walk through directory
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('_features.csv'):
                # Extract feature area
                area = file.replace('_features.csv', '')
                if area not in feature_areas:
                    continue
                
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, root_dir)
                
                try:
                    # Read CSV
                    df = pl.read_csv(file_path)
                    
                    # Skip if only doc_id column
                    if len(df.columns) <= 1:
                        continue
                    
                    # Check each column (except doc_id)
                    error_features = {
                        'nan': [],
                        'inf': [],
                        'negative_inf': []
                    }
                    
                    total_errors = 0
                    total_cells = 0
                    
                    for col in df.columns:
                        if col in ['doc_id', 'id', 'comment_id']:
                            continue
                        
                        total_cells += len(df)
                        
                        # Count NaN
                        nan_count = df[col].is_null().sum()
                        if nan_count > 0:
                            error_features['nan'].append({
                                'feature': col,
                                'count': nan_count,
                                'replacement': suggest_replacement_method(df, col)
                            })
                            total_errors += nan_count
                        
                        # Count Inf (for numeric columns)
                        if df[col].dtype in [pl.Float32, pl.Float64]:
                            inf_count = df[col].is_infinite().sum()
                            if inf_count > 0:
                                # Separate positive and negative infinity
                                pos_inf = (df[col] == float('inf')).sum()
                                neg_inf = (df[col] == float('-inf')).sum()
                                
                                if pos_inf > 0:
                                    error_features['inf'].append({
                                        'feature': col,
                                        'count': pos_inf,
                                        'replacement': suggest_replacement_method(df, col)
                                    })
                                    total_errors += pos_inf
                                
                                if neg_inf > 0:
                                    error_features['negative_inf'].append({
                                        'feature': col,
                                        'count': neg_inf,
                                        'replacement': suggest_replacement_method(df, col)
                                    })
                                    total_errors += neg_inf
                    
                    # Only add to report if errors found
                    if total_errors > 0:
                        if area not in report:
                            report[area] = []
                        
                        # Compile error details
                        error_details = {
                            'file_name': os.path.basename(file_path),
                            'file_path': rel_path,
                            'total_error_values': total_errors,
                            'total_cells': total_cells,
                            'error_percentage': round((total_errors / total_cells) * 100, 2),
                            'error_types': {}
                        }
                        
                        # Add each error type
                        for error_type, features in error_features.items():
                            if features:
                                error_details['error_types'][error_type] = {
                                    'count': sum(f['count'] for f in features),
                                    'affected_features': [
                                        {
                                            'name': f['feature'],
                                            'error_count': f['count'],
                                            'replacement_method': f['replacement'][0],
                                            'replacement_reason': f['replacement'][1]
                                        }
                                        for f in features
                                    ]
                                }
                        
                        report[area].append(error_details)
                        print(f"{rel_path}: {total_errors} errors found")
                
                except Exception as e:
                    print(f"Error reading {rel_path}: {str(e)}")
    
    # Sort files within each area by error count (descending)
    for area in report:
        report[area] = sorted(report[area], 
                            key=lambda x: x['total_error_values'], 
                            reverse=True)
    
    # # Save report
    # report_path = os.path.join(root_dir, 'error_value_report.json')
    # with open(report_path, 'w') as f:
    #     json.dump(report, f, indent=2)
    
    # print(f"\nReport saved: {report_path}")
    
    # Print summary
    total_files_with_errors = sum(len(files) for files in report.values())
    total_errors_all = sum(
        f['total_error_values'] 
        for files in report.values() 
        for f in files
    )
    
    print(f"\nSUMMARY:")
    print(">>>>> Root directory:", root_dir)
    print(f"Files with errors: {total_files_with_errors}")
    print(f"Total error values: {total_errors_all}")
    print(f"\nErrors by feature area:")
    for area, files in sorted(report.items(), 
                              key=lambda x: sum(f['total_error_values'] for f in x[1]), 
                              reverse=True):
        area_errors = sum(f['total_error_values'] for f in files)
        print(f"  {area:20s}: {area_errors:8d} errors in {len(files)} files")
    
    return report




if __name__ == "__main__":  
    # print("This script helps to check for NaN and Inf values in feature CSV files.")
    # check_error_values('features/data_features_prepared')
    
    
    import argparse
    p = argparse.ArgumentParser(description='Check feature CSVs for NaN and Inf values')
    p.add_argument('--root', default='data_features_prepared')
    p.add_argument('--save-report', metavar='PATH', help='write the JSON report here')
    a = p.parse_args()
    report = check_error_values(a.root)
    if a.save_report:
        import json
        json.dump(report, open(a.save_report, 'w'), indent=2)
        print(f"Report saved: {a.save_report}")
