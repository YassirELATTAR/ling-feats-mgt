"""
Ablation Study: Systematically remove feature groups to assess their impact
"""

from classifier_core import (TestbedClassifier, GlobalResultsTracker, MAGETestbedConfig, Tee)

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                            confusion_matrix, roc_auc_score, average_precision_score, f1_score)
import os
from datetime import datetime
import sys
import json
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)

import yaml, pathlib
CFG = yaml.safe_load(open(pathlib.Path(__file__).parent.parent.parent / "configs/paths.yaml"))

RUN_NAME = 'ablation'

ABLATION_RESULTS_DIR = os.path.join(CFG['results_root'], RUN_NAME)
ABLATION_LOGS_DIR    = os.path.join(CFG['logs_root'], RUN_NAME)


# =============================================================================
# LOAD FEATURE GROUPS FROM JSON
# =============================================================================

def load_feature_groups(json_path='feature_groups.json'):
    """Load feature groups from JSON file, excluding 'combined'"""
    with open(json_path, 'r') as f:
        all_groups = json.load(f)
    
    # Remove 'combined' as it's just the union of all groups
    if 'combined' in all_groups:
        del all_groups['combined']
    
    # Extract just the feature lists
    feature_groups = {group: data['features'] for group, data in all_groups.items()}
    
    return feature_groups

# =============================================================================
# ABLATION TESTBED CLASSIFIER
# =============================================================================

class AblationTestbedClassifier(TestbedClassifier):
    """Extended classifier for ablation studies"""
    
    def __init__(self, testbed_name, feature_group, ablated_group, output_dir=ABLATION_RESULTS_DIR):
        self.ablated_group = ablated_group
        # Call parent constructor with modified output directory
        super().__init__(testbed_name, feature_group, output_dir)
        
        # Override output directory to include ablation info
        if testbed_name.startswith('1_fixed_domain_model_specific_'):
            domain = testbed_name.split('_')[5]
            self.output_dir = os.path.join(output_dir, f'ablated_{ablated_group}', 
                                          'testbed1_all_models', domain, testbed_name, feature_group)
        elif testbed_name.startswith('11_fixed_domain_model_family_specific_'):
            domain = testbed_name.split('_')[6]
            self.output_dir = os.path.join(output_dir, f'ablated_{ablated_group}',
                                          'testbed_1_model_family', domain, testbed_name, feature_group)
        else:
            self.output_dir = os.path.join(output_dir, f'ablated_{ablated_group}', 
                                          testbed_name, feature_group)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def prepare_features_with_ablation(self, df, features_to_remove):
        """
        Extract features and labels, excluding features from ablated group
        
        Args:
            df: DataFrame with features
            features_to_remove: List of feature names to exclude
            
        Returns:
            X: Feature matrix
            y: Labels
            feature_names: List of feature names used
        """
        exclude_cols = ['doc_id', 'id', 'comment_id', 'post_id', 'label', 'text']
        
        # Get all available feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove features that exist in both df and features_to_remove
        available_features_to_remove = [f for f in features_to_remove if f in df.columns]
        feature_cols = [col for col in feature_cols if col not in available_features_to_remove]
        
        X = df[feature_cols].values
        y = df['label'].values
        
        print(f"Features shape after ablation: {X.shape}")
        print(f"Removed {len(available_features_to_remove)} features from '{self.ablated_group}' group")
        print(f"Labels: Human={sum(y==0)}, AI={sum(y==1)}")
        
        return X, y, feature_cols
    
    def save_results_with_ablation(self, metrics):
        """Save results with ablation information"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f'results_{timestamp}.txt')
        
        with open(filepath, 'w') as f:
            f.write(f"ABLATION STUDY\n")
            f.write(f"Ablated Feature Group: {self.ablated_group}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Testbed: {self.testbed_name}\n")
            f.write(f"Feature Group: {self.feature_group}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("="*60 + "\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*60 + "\n")
            f.write(f"Accuracy:           {metrics['accuracy']:.4f}\n")
            f.write(f"AUROC:              {metrics['auroc']:.4f}\n")
            f.write(f"Average Precision:  {metrics['avg_precision']:.4f}\n")
            f.write(f"F1 Macro:           {metrics['f1_macro']:.4f}\n")
            f.write(f"F1 Micro:           {metrics['f1_micro']:.4f}\n")
            f.write(f"F1 Weighted:        {metrics['f1_weighted']:.4f}\n")
            f.write(f"Average Recall:     {metrics['avg_recall']:.4f}\n\n")
            
            f.write("PER-CLASS METRICS\n")
            f.write("-"*60 + "\n")
            f.write("Human (Class 0):\n")
            f.write(f"  Precision: {metrics['precision_human']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall_human']:.4f}\n")
            f.write(f"  F1:        {metrics['f1_human']:.4f}\n\n")
            
            f.write("AI (Class 1):\n")
            f.write(f"  Precision: {metrics['precision_ai']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall_ai']:.4f}\n")
            f.write(f"  F1:        {metrics['f1_ai']:.4f}\n\n")
            
            f.write("CONFUSION MATRIX\n")
            f.write("-"*60 + "\n")
            cm = metrics['confusion_matrix']
            f.write(f"          Predicted\n")
            f.write(f"          Human  AI\n")
            f.write(f"True Human {cm[0,0]:5d} {cm[0,1]:5d}\n")
            f.write(f"     AI    {cm[1,0]:5d} {cm[1,1]:5d}\n")
        
        print(f"Ablation results saved to: {filepath}")


class AblationGlobalResultsTracker(GlobalResultsTracker):
    """Track ablation results across all testbeds"""
    
    def __init__(self, ablated_group, output_dir=ABLATION_RESULTS_DIR, testbed=None):
        self.ablated_group = ablated_group
        self.output_dir = os.path.join(output_dir, f'ablated_{ablated_group}')
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []
        self.testbed = testbed
    
    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.testbed is None:
            filepath = os.path.join(self.output_dir, f'global_results_{timestamp}.json')
        else:
            filepath = os.path.join(self.output_dir, f'global_results_-testbed{self.testbed}-{timestamp}.json')
        
        results_with_metadata = {
            'ablated_group': self.ablated_group,
            'timestamp': timestamp,
            'results': self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"\nAblation global results saved to: {filepath}")


# =============================================================================
# ABLATION EXECUTION FUNCTIONS
# =============================================================================

def run_cmv_ablation(feature_group, ai_model, ablated_group, features_to_remove, 
                     base_path="data_features_prepared/cmv", global_tracker=None):
    """Run CMV testbed with ablation"""
    print("\n" + "="*56)
    print(f"CMV ABLATION - {feature_group} - {ai_model}")
    print(f"Ablating: {ablated_group}")
    print("="*56)
    
    human_path = f"{base_path}/Human/comments_features/{feature_group}_features.csv"
    human_df = pd.read_csv(human_path)
    
    if ai_model == 'combined':
        ai_paths = [
            f"{base_path}/AI/gpt/gpt_features/{feature_group}_features.csv",
            f"{base_path}/AI/llama/llama_features/{feature_group}_features.csv",
            f"{base_path}/AI/mistral/mistral_features/{feature_group}_features.csv"
        ]
        ai_df = pd.concat([pd.read_csv(p) for p in ai_paths], ignore_index=True)
    else:
        ai_path = f"{base_path}/AI/{ai_model}/{ai_model}_features/{feature_group}_features.csv"
        ai_df = pd.read_csv(ai_path)
    
    if len(human_df) > len(ai_df):
        human_df = human_df.sample(n=len(ai_df), random_state=42).reset_index(drop=True)
    
    human_df['label'] = 0
    ai_df['label'] = 1
    df = pd.concat([human_df, ai_df], ignore_index=True)
    
    print(f"Total samples: {len(df)} ({sum(df['label']==0)} H, {sum(df['label']==1)} AI)")
    
    classifier = AblationTestbedClassifier(f'cmv_{ai_model}', feature_group, ablated_group)
    X, y, feature_names = classifier.prepare_features_with_ablation(df, features_to_remove)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
    )
    
    print(f"\nSplits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    model, scaler = classifier.train_svm(X_train, y_train, X_val, y_val, class_weight='balanced')
    metrics = classifier.evaluate(model, scaler, X_test, y_test)
    
    if global_tracker:
        global_tracker.add_result(f'cmv_{ai_model}', metrics['accuracy'], metrics['auroc'], metrics['f1_macro'])
    
    print(f"\nRESULTS:")
    print(f"Accuracy: {metrics['accuracy']:.4f}, AUROC: {metrics['auroc']:.4f}, F1: {metrics['f1_macro']:.4f}")
    
    classifier.save_top_features(model, feature_names, top_k=50)
    classifier.save_results_with_ablation(metrics)
    
    print("CMV ablation complete!\n")


def run_mage_ablation(config, feature_group, ablated_group, features_to_remove,config_manager=None,  global_tracker=None):
    """Run MAGE testbed with ablation"""
    print("\n" + "="*56)
    print(f"MAGE ABLATION: {config['name']} - {feature_group}")
    print(f"Ablating: {ablated_group}")
    print("="*56)

    if config_manager == None:
        config_manager = MAGETestbedConfig()
    train_df, val_df, test_df = config_manager.load_data_from_config_no_subsampling(config, feature_group)
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("Skipping - insufficient data")
        return
    
    classifier = AblationTestbedClassifier(config['name'], feature_group, ablated_group)
    
    X_train, y_train, feature_names = classifier.prepare_features_with_ablation(train_df, features_to_remove)
    X_val, y_val, _ = classifier.prepare_features_with_ablation(val_df, features_to_remove)
    X_test, y_test, _ = classifier.prepare_features_with_ablation(test_df, features_to_remove)
    
    model, scaler = classifier.train_svm(X_train, y_train, X_val, y_val, class_weight='balanced')
    metrics = classifier.evaluate(model, scaler, X_test, y_test)
    
    if global_tracker:
        global_tracker.add_result(config['name'], metrics['accuracy'], metrics['auroc'], metrics['f1_macro'])
    
    print(f"\nRESULTS:")
    print(f"Accuracy: {metrics['accuracy']:.4f}, AUROC: {metrics['auroc']:.4f}, F1: {metrics['f1_macro']:.4f}")
    
    classifier.save_top_features(model, feature_names, top_k=50)
    classifier.save_results_with_ablation(metrics)
    
    print("Ablation complete!\n")

def main_excution(testbed_number = None, cmv=False,ablated_group='information', orignal_feature_set='combined', specific_pairs = None, unseen_model= 'gpt4', features_to_remove = None):
    if features_to_remove is None:
            raise ValueError("features_to_remove must be passed explicitly")
    
    if testbed_number in (7, 74):
        tb_tag = f"{testbed_number}_{unseen_model}"
    elif testbed_number == '7c':
        tb_tag = '7c_pooled'
    else:
        tb_tag = testbed_number

    os.makedirs(ABLATION_LOGS_DIR, exist_ok=True)
    log_file = os.path.join(
        ABLATION_LOGS_DIR,
        f"ablation_{ablated_group}_tb{tb_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    prev_stdout = sys.stdout
    sys.stdout = Tee(log_file)
    try:
        print("\n" + "#"*56)
        print(f"# ABLATING GROUP: {ablated_group.upper()}  |  TESTBED {tb_tag}")
        print(f"# Removing {len(features_to_remove)} features")
        print("#"*56)

        global_tracker = AblationGlobalResultsTracker(ablated_group, testbed=tb_tag)

        if cmv:
            # Run CMV testbeds with ablation
            print("\n" + "#"*56)
            print("# RUNNING CMV TESTBEDS (ABLATION)")
            print("#"*56)
            
            for ai_model in ['combined', 'gpt', 'llama', 'mistral']:
                try:
                    run_cmv_ablation(orignal_feature_set, ai_model, ablated_group, 
                                    features_to_remove, global_tracker=global_tracker)
                except Exception as e:
                    print(f"Error in CMV {ai_model} ablation: {e}\n")
            
            # Run MAGE testbeds with ablation
            print("\n" + "#"*56)
            print("# RUNNING MAGE TESTBEDS (ABLATION)")
            print("#"*56)
        

        ### Run MAGE testbeds:
        config_manager = MAGETestbedConfig()
        
        if testbed_number == 1:
            # Testbed 1: Fixed-domain & Model-specific
            print("\n" + "="*56)
            print("TESTBED 1: Fixed-domain & Model-specific (ABLATION)")
            print("="*56)
            
            all_models = config_manager.get_all_available_models(config_manager.train_path)
            
            for model in all_models:
                for domain in config_manager.domains:
                    config = config_manager.fixed_domain_model_specific(domain, model)
                    if config['train']['ai'] is None:
                        print(f"Skipping {model} for {domain} - model not found")
                        continue
                    try:
                        run_mage_ablation(config, orignal_feature_set, ablated_group,
                                        features_to_remove, config_manager=config_manager, global_tracker=global_tracker)
                    except Exception as e:
                        print(f"Error: {e}\n")
        
        elif testbed_number == 11:  
            # Testbed 11: Fixed-domain & Model-family-specific
            print("\n" + "="*56)
            print("TESTBED 11: Fixed-domain & Model-family-specific (ABLATION)")
            print("="*56)
            
            for family in config_manager.model_families.keys():
                for domain in config_manager.domains:
                    config = config_manager.fixed_domain_model_family_specific(domain, family)
                    if not config['train']['ai']:
                        print(f"Skipping {family} for {domain} - family not found")
                        continue
                    try:
                        run_mage_ablation(config, orignal_feature_set, ablated_group,
                                        features_to_remove, config_manager=config_manager, global_tracker=global_tracker)
                    except Exception as e:
                        print(f"Error: {e}\n")
            
            
        elif testbed_number == 2:
            # Testbed 2: Arbitrary-domains & Model-specific
            print("\n" + "="*56)
            print("TESTBED 2: Arbitrary-domains & Model-specific (ABLATION)")
            print("="*56)
            for family in config_manager.model_families.keys():
                config = config_manager.arbitrary_domains_model_specific(family)
                try:
                    run_mage_ablation(config, orignal_feature_set, ablated_group,
                                    features_to_remove,  config_manager=config_manager, global_tracker=global_tracker)
                except Exception as e:
                    print(f"Error: {e}\n")
            
        
        elif testbed_number == 3:
            # Testbed 3: Fixed-domain & Arbitrary-models
            print("\n" + "="*56)
            print("TESTBED 3: Fixed-domain & Arbitrary-models (ABLATION)")
            print("="*56)
            
            for domain in config_manager.domains:
                config = config_manager.fixed_domain_arbitrary_models(domain)
                try:
                    run_mage_ablation(config, orignal_feature_set, ablated_group,
                                    features_to_remove, config_manager=config_manager, global_tracker=global_tracker)
                except Exception as e:
                    print(f"Error: {e}\n")
        
        elif testbed_number == 4:
            # Testbed 4: Arbitrary-domains & Arbitrary-models
            print("\n" + "="*56)
            print("TESTBED 4: Arbitrary-domains & Arbitrary-models (ABLATION)")
            print("="*56)
            
            config = config_manager.arbitrary_domains_arbitrary_models()
            try:
                run_mage_ablation(config, orignal_feature_set, ablated_group,
                                features_to_remove, config_manager=config_manager, global_tracker=global_tracker)
            except Exception as e:
                print(f"Error: {e}\n")

        elif testbed_number == 5:   
            # Testbed 5: Unseen Models
            print("\n" + "="*56)
            print("TESTBED 5: Unseen Models (ABLATION)")
            print("="*56)
            
            for family in config_manager.model_families.keys():
                config = config_manager.unseen_models(family)
                try:
                    run_mage_ablation(config, orignal_feature_set, ablated_group,
                                    features_to_remove, config_manager=config_manager, global_tracker=global_tracker)
                except Exception as e:
                    print(f"Error: {e}\n")
        
        elif testbed_number == 6:
            # Testbed 6: Unseen Domains
            print("\n" + "="*56)
            print("TESTBED 6: Unseen Domains (ABLATION)")
            print("="*56)
            
            for domain in config_manager.domains:
                config = config_manager.unseen_domains(domain)
                try:
                    run_mage_ablation(config, orignal_feature_set, ablated_group,
                                    features_to_remove, config_manager=config_manager, global_tracker=global_tracker)
                except Exception as e:
                    print(f"Error: {e}\n")


        elif testbed_number == 7:  
            # Testbed 7: Unseen-domains & Unseen-model
            print("\n" + "="*56)
            print("TESTBED 7: Unseen-domains & Unseen-model (ABLATION)")
            print("="*56)
            
            config = config_manager.unseen_domains_unseen_model(unseen_model=unseen_model)
            try:
                run_mage_ablation(config, orignal_feature_set, ablated_group,
                                features_to_remove, config_manager=config_manager, global_tracker=global_tracker)
            except Exception as e:
                print(f"Error: {e}\n")
        

        elif testbed_number == 74:
            # Testbed 7.4: Unseen-domain & Unseen-model - Domains separate
            print("\n" + "="*56)
            print("TESTBED 7.4: Unseen-domains & Unseen-model (per domain) - ABLATION")
            print("="*56)
            unseen_test_domains = ['cnn', 'dialogsum', 'imdb', 'pubmed']
            for test_domain in unseen_test_domains:
                config = config_manager.unseen_domains_unseen_model_separate(test_domain, unseen_model=unseen_model)
                try:
                    run_mage_ablation(config, orignal_feature_set, ablated_group,
                                features_to_remove, config_manager=config_manager, global_tracker=global_tracker)
                except Exception as e:
                    print(f"Error in testbed 7.4 for {test_domain}: {e}\n")

        # New for the ablation study, we added a new testbed 7c that combines unseen domains and pooled unseen models
        elif testbed_number == '7c':
            print("TESTBED 7c: Unseen-domains & pooled unseen models (ABLATION)")
            config = config_manager.unseen_domains_unseen_models_combined(('gpt4', 'gpt5_6_sol'))
            try:
                run_mage_ablation(config, orignal_feature_set, ablated_group,
                                features_to_remove, config_manager=config_manager, global_tracker=global_tracker)
            except Exception as e:
                print(f"Error: {e}\n")


        elif testbed_number == 8:
            print("\n" + "="*56)
            print("TESTBED 8: Unseen Domain-Model Pair (ABLATION)")

            if specific_pairs:
                print(f"Running on {len(specific_pairs)} specific pairs:")
                for pair in specific_pairs:
                    print(f"  - {pair[0]} + {pair[1]}")
                print("="*56)
                
                # Run only on specified pairs
                for domain, family in specific_pairs:
                    config = config_manager.unseen_domain_model_pair(domain, family)
                    if not config['test']['ai']:
                        print(f"Skipping {domain}-{family} pair - no test data found")
                        continue
                    try:
                        run_mage_ablation(config, orignal_feature_set, ablated_group,
                                        features_to_remove, config_manager=config_manager, global_tracker=global_tracker)
                    except Exception as e:
                        print(f"Error in testbed 8 for {domain}-{family}: {e}\n")
            else:
                print("Running on ALL domain-model pairs")
                print("="*56)
                
                # Run on all pairs
                for domain in config_manager.domains:
                    for family in config_manager.model_families.keys():
                        config = config_manager.unseen_domain_model_pair(domain, family)
                        if not config['test']['ai']:
                            print(f"Skipping {domain}-{family} pair - no test data found")
                            continue
                        try:
                            run_mage_ablation(config, orignal_feature_set, ablated_group,
                                            features_to_remove, global_tracker=global_tracker)
                        except Exception as e:
                            print(f"Error in testbed 8 for {domain}-{family}: {e}\n")

        else:
            print(">>>>> TESTBED NUMBER IS NOT CORRECT <<<<<<")


        
        # Save global results for this ablation
        global_tracker.save_results()
        
        print("\n" + "="*56)
        print(f"ABLATION OF {ablated_group.upper()} COMPLETE")
        print("="*56)
    finally:
        # Close log file
        sys.stdout.file.close()
        sys.stdout = sys.stdout.terminal

# =============================================================================
# MAIN ABLATION STUDY
# =============================================================================

if __name__ == "__main__":
    
    # Load feature groups from JSON
    print("Loading feature groups from JSON...")
    feature_groups_dict = load_feature_groups(CFG['feature_groups_file'])
    
    print(f"\nFeature groups loaded:")
    for group, features in feature_groups_dict.items():
        print(f"  - {group}: {len(features)} features")
    
    # Setup directories
    os.makedirs(ABLATION_RESULTS_DIR, exist_ok=True)
    os.makedirs(ABLATION_LOGS_DIR, exist_ok=True)
    
    # Configuration
    orignal_feature_set = 'combined'  # Use combined_features.csv features for ablation

    ## Just to finish the ablation for specific areas (In case some feature area was not yet done):
    # RUN_ONLY = ['semantic', 'surface']      # >> the two unfinished areas; use the exact keys printed at startup
    # feature_groups_dict = {g: f for g, f in feature_groups_dict.items() if g in RUN_ONLY}
    # assert len(feature_groups_dict) == len(RUN_ONLY), f"check names, got: {list(feature_groups_dict)}"
    # print(f">> Running only: {list(feature_groups_dict)}")
    
    print("\n" + "="*56)
    print("### ABLATION STUDY ###")
    print(f">> Testing feature group: {orignal_feature_set}")
    print(f">> Number of ablation experiments: {len(feature_groups_dict)}")
    print(f">> Strategy: class_weight='balanced'")
    print(f">> Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*56)

    TESTBEDS = [4] # [1,11,2,3,4,5,6,7,74,'7c',8 ] ### Default is set to TB4, but can be updated here for the number of the TB
    UNSEEN_MODELS = ['gpt5_6_sol','gpt4']

    specific_pairs_to_test = [
            ('hswag', 'llama'),
            ('hswag', 'openai'),
            ('eli5', 'llama'),
            ('eli5', 'openai'),
            ('hswag', 'flan_t5'),
            ('hswag', 'eleuther'),
            ('eli5', 'flan_t5'),
            ('eli5', 'eleuther'),
            
            ('squad', 'llama'),
            ('squad', 'openai'),
            ('xsum', 'llama'),
            ('xsum', 'openai'),
            ('squad', 'flan_t5'),
            ('squad', 'eleuther'),
            ('xsum', 'flan_t5'),
            ('xsum', 'eleuther'),
 
                ('tldr', 'llama'),
                ('wp', 'llama'),
                ('tldr', 'openai'),
                ('wp', 'openai'),
                ('tldr', 'eleuther'),
                ('wp', 'eleuther'),
                ('tldr', 'bigscience'),
                ('wp', 'bigscience'),
                ('squad', 'bigscience'),
                ('xsum', 'bigscience'),
            ]
    for testbed_num in TESTBEDS:
        models = UNSEEN_MODELS if testbed_num in (7, 74) else [None]
        for unseen_model in models:
            for ablated_group, feats in feature_groups_dict.items():
                try:
                    main_excution(testbed_number=testbed_num, cmv=False,
                                    ablated_group=ablated_group,
                                    orignal_feature_set=orignal_feature_set,
                                    specific_pairs=specific_pairs_to_test,
                                    unseen_model=unseen_model or 'gpt4',
                                    features_to_remove=feats)
                except Exception as e:
                    print(f"FATAL testbed {testbed_num} / {ablated_group}: {type(e).__name__}: {e}")
    
    
    print("\n" + "="*56)
    print("ALL ABLATION EXPERIMENTS COMPLETE")
    print("="*56)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")