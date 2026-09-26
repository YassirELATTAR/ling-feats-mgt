import polars as pl
import os
import json
from collections import defaultdict

def verify_feature_consistency(root_dir='data_features_prepared'):
    """
    Verify all feature files have consistent columns across all subdirectories.
    """
    
    print("="*80)
    print("VERIFYING FEATURE CONSISTENCY")
    print("="*80)
    
    # Track features by group
    feature_groups = defaultdict(lambda: {'files': [], 'features': set(), 'all_same': True})
    
    # Feature areas to check
    areas = ['surface', 'pos', 'lexical_richness', 'readability', 
             'information', 'entities', 'semantic', 'emotion',
             'psycholinguistic', 'morphological', 'dependency', 'combined']
    
    # Collect all files by area
    print("\n[1/2] Scanning files...")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'combined_features.csv':
                area = 'combined'
            elif file.endswith('_features.csv'):
                area = file.replace('_features.csv', '')
            else:
                continue
            
            if area not in areas:
                continue
            
            file_path = os.path.join(root, file)
            df = pl.read_csv(file_path, n_rows=0)
            
            # Get features (exclude doc_id)
            features = set(df.columns) - {'doc_id'}
            
            feature_groups[area]['files'].append(file_path)
            
            # First file sets the baseline
            if not feature_groups[area]['features']:
                feature_groups[area]['features'] = features
            else:
                # Check if same as baseline
                if features != feature_groups[area]['features']:
                    feature_groups[area]['all_same'] = False
                    print(f"Mismatch in {area}: {os.path.relpath(file_path, root_dir)}")
    
    # Check consistency and prepare report
    print("\n[2/2] Checking consistency...")
    report = {}
    all_consistent = True
    
    for area in sorted(feature_groups.keys()):
        data = feature_groups[area]
        
        report[area] = {
            'features': sorted(list(data['features'])),
            'number': len(data['features']),
            'files_checked': len(data['files']),
            'all_consistent': data['all_same']
        }
        
        if data['all_same']:
            print(f"{area:20s}: {len(data['features']):3d} features - Consistent across {len(data['files'])} files")
        else:
            print(f"{area:20s}: INCONSISTENT!")
            all_consistent = False
    
    # Save report
    report_path = os.path.join(root_dir, 'feature_consistency_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved: {report_path}")
    
    if all_consistent:
        print("\nAll feature groups are consistent across all files!")
    else:
        print("\nSome feature groups have inconsistencies!")
    
    return report

if __name__ == "__main__":
    # verify_feature_consistency('data_features_prepared')

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data_features_prepared')
    report = verify_feature_consistency(p.parse_args().root)
    import sys
    sys.exit(0 if all(v['all_consistent'] for v in report.values()) else 1)
    