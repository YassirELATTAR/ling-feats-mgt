"""
Cumulative Ablation Study: Progressively remove feature groups in order of impact
"""

from ablation_study import (
    AblationTestbedClassifier,
    AblationGlobalResultsTracker,
    run_cmv_ablation,
    run_mage_ablation,
    Tee,
    load_feature_groups
)
from classifier_core import MAGETestbedConfig
import os
from datetime import datetime
import sys
import json

# =============================================================================
# CUMULATIVE ABLATION CONFIGURATION
# =============================================================================


#### These belong to the initial results without the new models (llama, glm5, gpt6.5)
# # Order feature groups from best performance (remove first) to worst (keep last)
# # ABLATION_ORDER = [
# #     'entities',
# #     'readability',
# #     'semantic',
# #     'emotion', 
# #     'morphological',
# #     'pos',
# #     'dependency',
# #     'surface',
# #     'psycholinguistic',      
# #     'information',
# #     'lexical_richness'
# # ]

# # orders by unseen domains and unseen models: testbed7
# ABLATION_ORDER = ['morphological', 'psycholinguistic', 'information', 'dependency', 'surface', 'pos', 'emotion', 'entities', 'semantic', 'readability', 'lexical_richness']


# ######## The new study when added new models:
# # Tesbed4:
# ABLATION_ORDER = [
# "readability",
# "entities",
# "emotion",
# "morphological",
# "semantic",
# "dependency",
# "surface",
# "psycholinguistic",
# "pos",
# "information",
# "lexical_richness"
# ]

# Tesbed7:
ABLATION_ORDER = [
"morphological",
"psycholinguistic",
"information",
"dependency",
"surface",
"emotion",
"entities",
"semantic",
"readability",
"pos",
"lexical_richness",
]


import yaml, pathlib
CFG = yaml.safe_load(open(pathlib.Path(__file__).parent.parent.parent / "configs/paths.yaml"))

RUN_NAME = 'cross_cmv_mage_ablation'
RESULTS_DIR = os.path.join(CFG['results_root'], RUN_NAME)
LOGS_DIR    = os.path.join(CFG['logs_root'], RUN_NAME)

TB_NUM= 7


class CumulativeAblationTracker:
    """Track cumulative ablation results"""
    
    def __init__(self, removed_groups_list, output_dir=RESULTS_DIR, tag=''):
        self.tag = tag
        self.output_dir = output_dir
        self.removed_groups = removed_groups_list  # Store at init
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []
    
    def add_result(self, testbed_name, accuracy, auroc, f1_macro):
        """Match the signature expected by run_cmv_ablation and run_mage_ablation"""
        self.results.append({
            'removed_groups': self.removed_groups,
            'num_groups_removed': len(self.removed_groups),
            'testbed': testbed_name,
            'accuracy': float(accuracy),
            'auroc': float(auroc),
            'f1_macro': float(f1_macro)
        })
    
    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        removed_str = '_'.join(self.removed_groups) if self.removed_groups else 'none'
        filepath = os.path.join(self.output_dir, f'cumulative_{self.tag}_{removed_str}_{timestamp}.json')
        
        results_with_metadata = {
            'removed_groups': self.removed_groups,
            'num_groups_removed': len(self.removed_groups),
            'timestamp': timestamp,
            'results': self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"\nCumulative ablation results saved to: {filepath}")

def run_cumulative_experiment(removed_groups_list, all_feature_groups, test_num = 4, cmv_run = False, unseen_model='gpt4'):
    """Run one cumulative ablation experiment"""
    
    # Collect all features to remove
    features_to_remove = []
    for group in removed_groups_list:
        features_to_remove.extend(all_feature_groups[group])
    
    removed_str = '_'.join(removed_groups_list)
    print(f"\nRemoving {len(features_to_remove)} features from groups: {removed_groups_list}")
    
    # Create tracker with removed_groups_list
    global_tracker = CumulativeAblationTracker(removed_groups_list,  tag=f"tb{test_num}")
    
    if cmv_run:
        # Run CMV testbeds
        print("\n" + "#"*56)
        print("# RUNNING CMV TESTBEDS (CUMULATIVE ABLATION)")
        print("#"*56)

        for ai_model in ['combined', 'gpt', 'llama', 'mistral']:
            try:
                run_cmv_ablation('combined', ai_model, removed_str, 
                                features_to_remove, global_tracker=global_tracker)
            except Exception as e:
                print(f"Error in CMV {ai_model}: {e}\n")

    # Run MAGE testbeds
    print("\n" + "#"*56)
    print("# RUNNING MAGE TESTBEDS (CUMULATIVE ABLATION)")
    print("#"*56)
    
    config_manager = MAGETestbedConfig()
    
    if test_num == 4:

        # Testbed 4: Arbitrary-domains & Arbitrary-models (representative testbed)
        print("\n" + "="*56)
        print("TESTBED 4: Arbitrary-domains & Arbitrary-models")
        print("="*56)
        
        config = config_manager.arbitrary_domains_arbitrary_models()
        try:
            run_mage_ablation(config, 'combined', removed_str,
                            features_to_remove, global_tracker=global_tracker)
        except Exception as e:
            print(f"Error: {e}\n")
    
    elif test_num == 7:
        # Testbed 7: Unseen-domains & Unseen-model (hardest testbed)
        print("\n" + "="*56)
        print("TESTBED 7: Unseen-domains & Unseen-model")
        print("="*56)
        
        config = config_manager.unseen_domains_unseen_models_combined()
        try:
            run_mage_ablation(config, 'combined', removed_str,
                            features_to_remove, global_tracker=global_tracker)
        except Exception as e:
            print(f"Error: {e}\n")
    else:
        print("Wrong test number!!")
    return global_tracker

# =============================================================================
# MAIN CUMULATIVE ABLATION
# =============================================================================

if __name__ == "__main__":
    
    print("="*56)
    print("### CUMULATIVE ABLATION STUDY ###")
    print(f">> Ablation order: {' -> '.join(ABLATION_ORDER)}")
    print(f">> Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*56)
    
    # Load feature groups
    all_feature_groups = load_feature_groups(CFG['feature_groups_file'])
    
    # Setup directories
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Master tracker for all cumulative experiments
    master_results = []
    
    # Progressive ablation: remove groups one by one
    for i in range(1, len(ABLATION_ORDER)):
        removed_groups = ABLATION_ORDER[:i]
        remaining_groups = ABLATION_ORDER[i:]
        
        print("\n" + "="*70)
        print(f"CUMULATIVE ABLATION STEP {i}/{len(ABLATION_ORDER)}")
        print(f"Removed groups ({i}): {removed_groups}")
        print(f"Remaining groups ({len(remaining_groups)}): {remaining_groups}")
        print("="*70)
        
        # Setup logging
        log_file = f"{LOGS_DIR}/step_{i:02d}_testbed7_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        sys.stdout = Tee(log_file)
        
        # Run experiment
        tracker = run_cumulative_experiment(removed_groups, all_feature_groups, test_num=TB_NUM, cmv_run=False)
        
        # Save individual step results
        tracker.save_results()
        
        # Add results to master list
        master_results.extend(tracker.results)
        
        print(f"\n{'='*70}")
        print(f"STEP {i} COMPLETE")
        print(f"{'='*70}\n")
        
        # Close log
        sys.stdout.file.close()
        sys.stdout = sys.stdout.terminal
    
    # Save master results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_filepath = f'{RESULTS_DIR}all_steps_summary_{timestamp}.json'
    with open(master_filepath, 'w') as f:
        json.dump({
            'ablation_order': ABLATION_ORDER,
            'timestamp': timestamp,
            'all_results': master_results
        }, f, indent=2)
    
    print(f"\nMaster summary saved to: {master_filepath}")
    
    print("\n" + "="*70)
    print("CUMULATIVE ABLATION STUDY COMPLETE")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")