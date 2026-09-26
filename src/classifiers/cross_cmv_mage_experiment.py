# cross_cmv_mage_experiment.py

import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

from classifier_core import (
    TestbedClassifier, 
    GlobalResultsTracker, 
    MAGETestbedConfig,
    Tee
)

# Set seeds
np.random.seed(42)
random.seed(42)

import yaml, pathlib
CFG = yaml.safe_load(open(pathlib.Path(__file__).parent.parent.parent / "configs/paths.yaml"))

RUN_NAME = 'cross_cmv_mage'
RESULTS_DIR = os.path.join(CFG['results_root'], RUN_NAME)
LOGS_DIR    = os.path.join(CFG['logs_root'], RUN_NAME)

# =============================================================================
# CROSS-DATASET CMV EXPERIMENTS
# =============================================================================

class CrossCMVMAGEExperiment:
    """Handle cross-dataset experiments between CMV and MAGE CMV domain"""
    
    def __init__(self, cmv_base_path=None, mage_base_path=None, output_dir=RESULTS_DIR):
        self.cmv_base_path = cmv_base_path or CFG['cmv_features']
        self.mage_base_path = mage_base_path or CFG['mage_features']
        self.output_dir = output_dir
        self.mage_config = MAGETestbedConfig(base_path=self.mage_base_path)
        
    def load_cmv_data(self, feature_group, ai_models='combined'):
        """Load CMV dataset (original)"""
        human_path = f"{self.cmv_base_path}/Human/comments_features/{feature_group}_features.csv"
        human_df = pd.read_csv(human_path)
        human_df['label'] = 0
        
        if ai_models == 'combined':
            # All three models
            ai_paths = [
                f"{self.cmv_base_path}/AI/gpt/gpt_features/{feature_group}_features.csv",
                f"{self.cmv_base_path}/AI/llama/llama_features/{feature_group}_features.csv",
                f"{self.cmv_base_path}/AI/mistral/mistral_features/{feature_group}_features.csv"
            ]
            ai_df = pd.concat([pd.read_csv(p) for p in ai_paths], ignore_index=True)
        elif ai_models == 'llama_gpt':
            # Only LLama and GPT (excluding Mistral)
            ai_paths = [
                f"{self.cmv_base_path}/AI/gpt/gpt_features/{feature_group}_features.csv",
                f"{self.cmv_base_path}/AI/llama/llama_features/{feature_group}_features.csv"
            ]
            ai_df = pd.concat([pd.read_csv(p) for p in ai_paths], ignore_index=True)
        else:
            # Single model
            ai_path = f"{self.cmv_base_path}/AI/{ai_models}/{ai_models}_features/{feature_group}_features.csv"
            ai_df = pd.read_csv(ai_path)
        
        ai_df['label'] = 1
        
        # Balance human to AI
        if len(human_df) > len(ai_df):
            human_df = human_df.sample(n=len(ai_df), random_state=42).reset_index(drop=True)
        
        return human_df, ai_df
    
    def load_mage_cmv_test_data(self, feature_group, model_filter=None):
        """Load MAGE CMV domain test data (all 30 models or filtered)"""
        domain = 'cmv'
        test_path = self.mage_config.test_path
        
        # Load human data
        human_path = f"{test_path}/Human/{domain}/samples/samples_features"
        human_feature_file = os.path.join(human_path, f"{feature_group}_features.csv")
        human_df = pd.read_csv(human_feature_file)
        human_df['label'] = 0
        
        # Load AI data
        ai_dfs = []
        domain_ai_path = f"{test_path}/AI/{domain}"
        
        if os.path.exists(domain_ai_path):
            for family in os.listdir(domain_ai_path):
                family_path = os.path.join(domain_ai_path, family)
                
                # Apply model filter if specified
                if model_filter and family not in model_filter:
                    continue
                
                if os.path.isdir(family_path):
                    for model_folder in os.listdir(family_path):
                        model_features_path = os.path.join(
                            family_path, model_folder, f"{model_folder}_features"
                        )
                        feature_file = os.path.join(
                            model_features_path, f"{feature_group}_features.csv"
                        )
                        
                        if os.path.exists(feature_file):
                            df = pd.read_csv(feature_file)
                            df['label'] = 1
                            ai_dfs.append(df)
        
        ai_df = pd.concat(ai_dfs, ignore_index=True) if ai_dfs else pd.DataFrame()
        
        return human_df, ai_df
    
    def load_mage_cmv_train_val_data(self, feature_group, model_filter=None):
        """Load MAGE CMV domain train and validation data"""
        domain = 'cmv'
        
        # Load training data
        train_path = self.mage_config.train_path
        train_human_path = f"{train_path}/Human/{domain}/samples/samples_features"
        train_human_file = os.path.join(train_human_path, f"{feature_group}_features.csv")
        train_human_df = pd.read_csv(train_human_file)
        train_human_df['label'] = 0
        
        train_ai_dfs = []
        train_domain_ai_path = f"{train_path}/AI/{domain}"
        
        if os.path.exists(train_domain_ai_path):
            for family in os.listdir(train_domain_ai_path):
                if model_filter and family not in model_filter:
                    continue
                
                family_path = os.path.join(train_domain_ai_path, family)
                if os.path.isdir(family_path):
                    for model_folder in os.listdir(family_path):
                        model_features_path = os.path.join(
                            family_path, model_folder, f"{model_folder}_features"
                        )
                        feature_file = os.path.join(
                            model_features_path, f"{feature_group}_features.csv"
                        )
                        if os.path.exists(feature_file):
                            df = pd.read_csv(feature_file)
                            df['label'] = 1
                            train_ai_dfs.append(df)
        
        train_ai_df = pd.concat(train_ai_dfs, ignore_index=True) if train_ai_dfs else pd.DataFrame()
        
        # Load validation data
        val_path = self.mage_config.val_path
        val_human_path = f"{val_path}/Human/{domain}/samples/samples_features"
        val_human_file = os.path.join(val_human_path, f"{feature_group}_features.csv")
        val_human_df = pd.read_csv(val_human_file)
        val_human_df['label'] = 0
        
        val_ai_dfs = []
        val_domain_ai_path = f"{val_path}/AI/{domain}"
        
        if os.path.exists(val_domain_ai_path):
            for family in os.listdir(val_domain_ai_path):
                if model_filter and family not in model_filter:
                    continue
                
                family_path = os.path.join(val_domain_ai_path, family)
                if os.path.isdir(family_path):
                    for model_folder in os.listdir(family_path):
                        model_features_path = os.path.join(
                            family_path, model_folder, f"{model_folder}_features"
                        )
                        feature_file = os.path.join(
                            model_features_path, f"{feature_group}_features.csv"
                        )
                        if os.path.exists(feature_file):
                            df = pd.read_csv(feature_file)
                            df['label'] = 1
                            val_ai_dfs.append(df)
        
        val_ai_df = pd.concat(val_ai_dfs, ignore_index=True) if val_ai_dfs else pd.DataFrame()
        
        return train_human_df, train_ai_df, val_human_df, val_ai_df


    def experiment_1_cmv_train_mage_cmv_test(self, feature_group, filtered_90=False):
        """
        Experiment 1: Train on CMV (3 models), Test on MAGE CMV (all 30 models)
        """
        print("\n" + "="*56)
        print("EXPERIMENT 1: CMV Train => MAGE CMV Test (All Models)")
        print(f"Feature Group: {feature_group}")
        print("="*56)
        
        # Load CMV training data (3 models: gpt, llama, mistral)
        human_df, ai_df = self.load_cmv_data(feature_group, ai_models='combined')
        cmv_df = pd.concat([human_df, ai_df], ignore_index=True)
        
        print(f"\nCMV Training Data: {len(cmv_df)} samples")
        print(f"  Human: {sum(cmv_df['label']==0)}, AI: {sum(cmv_df['label']==1)}")
        
        # Split CMV into train/val (70/10, reserve 20% but don't use for test)
        train_val_df, _ = train_test_split(
            cmv_df, test_size=0.2, random_state=42, stratify=cmv_df['label']
        )
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.125, random_state=42, stratify=train_val_df['label']
        )
        
        print(f"CMV Train: {len(train_df)}, Val: {len(val_df)}")
        
        # Load MAGE CMV test data (all 30 models)
        test_human_df, test_ai_df = self.load_mage_cmv_test_data(feature_group)
        test_df = pd.concat([test_human_df, test_ai_df], ignore_index=True)
        
        print(f"\nMAGE CMV Test Data: {len(test_df)} samples")
        print(f"  Human: {sum(test_df['label']==0)}, AI: {sum(test_df['label']==1)}")
        
        # Initialize classifier
        classifier = TestbedClassifier(
            'cross_exp1_cmv_train_mage_cmv_test', 
            feature_group,
            output_dir=self.output_dir
        )
        
        # Prepare features
        if filtered_90:
            X_train, y_train, feature_names = classifier.prepare_features_90(train_df)
            X_val, y_val, _ = classifier.prepare_features_90(val_df)
            X_test, y_test, _ = classifier.prepare_features_90(test_df)
        else:
            X_train, y_train, feature_names = classifier.prepare_features_baseline(train_df)
            X_val, y_val, _ = classifier.prepare_features_baseline(val_df)
            X_test, y_test, _ = classifier.prepare_features_baseline(test_df)
        
        # Train and evaluate
        model, scaler = classifier.train_svm(X_train, y_train, X_val, y_val, class_weight='balanced')
        metrics = classifier.evaluate(model, scaler, X_test, y_test)
        
        print(f"\nRESULTS:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"AUROC: {metrics['auroc']:.4f}")
        print(f"F1-Macro: {metrics['f1_macro']:.4f}")
        
        # Save results
        classifier.save_top_features(model, feature_names, top_k=50)
        classifier.save_results(metrics)
        
        return metrics
    
    def experiment_2_cmv_filtered_train_mage_filtered_test(self, feature_group, filtered_90=False):
        """
        Experiment 2: Train on CMV (LLama+GPT only), Test on MAGE CMV (LLama+OpenAI families)
        """
        print("\n" + "="*56)
        print("EXPERIMENT 2: CMV Train (LLama+GPT) => MAGE CMV Test (LLama+OpenAI)")
        print(f"Feature Group: {feature_group}")
        print("="*56)
        
        # Load CMV training data (only llama and gpt, excluding mistral)
        human_df, ai_df = self.load_cmv_data(feature_group, ai_models='llama_gpt')
        cmv_df = pd.concat([human_df, ai_df], ignore_index=True)
        
        print(f"\nCMV Training Data (LLama+GPT): {len(cmv_df)} samples")
        print(f"  Human: {sum(cmv_df['label']==0)}, AI: {sum(cmv_df['label']==1)}")
        
        # Split CMV into train/val
        train_val_df, _ = train_test_split(
            cmv_df, test_size=0.2, random_state=42, stratify=cmv_df['label']
        )
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.125, random_state=42, stratify=train_val_df['label']
        )
        
        print(f"CMV Train: {len(train_df)}, Val: {len(val_df)}")
        
        # Load MAGE CMV test data (only llama and openai families)
        model_filter = ['llama', 'openai']
        test_human_df, test_ai_df = self.load_mage_cmv_test_data(
            feature_group, model_filter=model_filter
        )
        test_df = pd.concat([test_human_df, test_ai_df], ignore_index=True)
        
        print(f"\nMAGE CMV Test Data (LLama+OpenAI): {len(test_df)} samples")
        print(f"  Human: {sum(test_df['label']==0)}, AI: {sum(test_df['label']==1)}")
        
        # Initialize classifier
        classifier = TestbedClassifier(
            'cross_exp2_cmv_llamagpt_train_mage_llamaopenai_test', 
            feature_group,
            output_dir=self.output_dir
        )
        
        # Prepare features
        if filtered_90:
            X_train, y_train, feature_names = classifier.prepare_features_90(train_df)
            X_val, y_val, _ = classifier.prepare_features_90(val_df)
            X_test, y_test, _ = classifier.prepare_features_90(test_df)
        else:
            X_train, y_train, feature_names = classifier.prepare_features_baseline(train_df)
            X_val, y_val, _ = classifier.prepare_features_baseline(val_df)
            X_test, y_test, _ = classifier.prepare_features_baseline(test_df)
        
        # Train and evaluate
        model, scaler = classifier.train_svm(X_train, y_train, X_val, y_val, class_weight='balanced')
        metrics = classifier.evaluate(model, scaler, X_test, y_test)
        
        print(f"\nRESULTS:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"AUROC: {metrics['auroc']:.4f}")
        print(f"F1-Macro: {metrics['f1_macro']:.4f}")
        
        # Save results
        classifier.save_top_features(model, feature_names, top_k=50)
        classifier.save_results(metrics)
        
        return metrics
    def experiment_3_mage_train_cmv_test(self, feature_group, filtered_90=False):
        """
        Experiment 3: Train on MAGE CMV (all 30 models), Test on original CMV (3 models)
        """
        print("\n" + "="*56)
        print("EXPERIMENT 3: MAGE CMV Train (All Models) => CMV Test")
        print(f"Feature Group: {feature_group}")
        print("="*56)
        
        # Load MAGE CMV train and validation data (all models)
        train_human_df, train_ai_df, val_human_df, val_ai_df = self.load_mage_cmv_train_val_data(
            feature_group
        )
        
        train_df = pd.concat([train_human_df, train_ai_df], ignore_index=True)
        val_df = pd.concat([val_human_df, val_ai_df], ignore_index=True)
        
        print(f"\nMAGE CMV Training Data: {len(train_df)} samples")
        print(f"  Human: {sum(train_df['label']==0)}, AI: {sum(train_df['label']==1)}")
        print(f"MAGE CMV Validation Data: {len(val_df)} samples")
        print(f"  Human: {sum(val_df['label']==0)}, AI: {sum(val_df['label']==1)}")
        
        # Load original CMV test data (3 models)
        test_human_df, test_ai_df = self.load_cmv_data(feature_group, ai_models='combined')
        cmv_df = pd.concat([test_human_df, test_ai_df], ignore_index=True)
        
        # Use only the test portion
        _, test_df = train_test_split(
            cmv_df, test_size=0.2, random_state=42, stratify=cmv_df['label']
        )
        
        print(f"\nOriginal CMV Test Data: {len(test_df)} samples")
        print(f"  Human: {sum(test_df['label']==0)}, AI: {sum(test_df['label']==1)}")
        
        # Initialize classifier
        classifier = TestbedClassifier(
            'cross_exp3_mage_cmv_train_cmv_test', 
            feature_group,
            output_dir=self.output_dir
        )
        
        # Prepare features
        if filtered_90:
            X_train, y_train, feature_names = classifier.prepare_features_90(train_df)
            X_val, y_val, _ = classifier.prepare_features_90(val_df)
            X_test, y_test, _ = classifier.prepare_features_90(test_df)
        else:
            X_train, y_train, feature_names = classifier.prepare_features_baseline(train_df)
            X_val, y_val, _ = classifier.prepare_features_baseline(val_df)
            X_test, y_test, _ = classifier.prepare_features_baseline(test_df)
        
        # Train and evaluate
        model, scaler = classifier.train_svm(X_train, y_train, X_val, y_val, class_weight='balanced')
        metrics = classifier.evaluate(model, scaler, X_test, y_test)
        
        print(f"\nRESULTS:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"AUROC: {metrics['auroc']:.4f}")
        print(f"F1-Macro: {metrics['f1_macro']:.4f}")
        
        # Save results
        classifier.save_top_features(model, feature_names, top_k=50)
        classifier.save_results(metrics)
        
        return metrics

    def experiment_4_mage_filtered_train_cmv_filtered_test(self, feature_group, filtered_90=False):
        """
        Experiment 4: Train on MAGE CMV (OpenAI+LLama), Test on original CMV (GPT+LLama only)
        """
        print("\n" + "="*56)
        print("EXPERIMENT 4: MAGE CMV Train (OpenAI+LLama) => CMV Test (GPT+LLama)")
        print(f"Feature Group: {feature_group}")
        print("="*56)
        
        # Load MAGE CMV train and validation data (OpenAI and LLama families)
        model_filter = ['openai', 'llama']
        train_human_df, train_ai_df, val_human_df, val_ai_df = self.load_mage_cmv_train_val_data(
            feature_group, model_filter=model_filter
        )
        
        train_df = pd.concat([train_human_df, train_ai_df], ignore_index=True)
        val_df = pd.concat([val_human_df, val_ai_df], ignore_index=True)
        
        print(f"\nMAGE CMV Training Data (OpenAI+LLama): {len(train_df)} samples")
        print(f"  Human: {sum(train_df['label']==0)}, AI: {sum(train_df['label']==1)}")
        print(f"MAGE CMV Validation Data (OpenAI+LLama): {len(val_df)} samples")
        print(f"  Human: {sum(val_df['label']==0)}, AI: {sum(val_df['label']==1)}")
        
        # Load original CMV test data (GPT+LLama only)
        test_human_df, test_ai_df = self.load_cmv_data(feature_group, ai_models='llama_gpt')
        cmv_df = pd.concat([test_human_df, test_ai_df], ignore_index=True)
        
        # Use only the test portion
        _, test_df = train_test_split(
            cmv_df, test_size=0.2, random_state=42, stratify=cmv_df['label']
        )
        
        print(f"\nOriginal CMV Test Data (GPT+LLama): {len(test_df)} samples")
        print(f"  Human: {sum(test_df['label']==0)}, AI: {sum(test_df['label']==1)}")
        
        # Initialize classifier
        classifier = TestbedClassifier(
            'cross_exp4_mage_cmv_openaillama_train_cmv_gptllama_test', 
            feature_group,
            output_dir=self.output_dir
        )
        
        # Prepare features
        if filtered_90:
            X_train, y_train, feature_names = classifier.prepare_features_90(train_df)
            X_val, y_val, _ = classifier.prepare_features_90(val_df)
            X_test, y_test, _ = classifier.prepare_features_90(test_df)
        else:
            X_train, y_train, feature_names = classifier.prepare_features_baseline(train_df)
            X_val, y_val, _ = classifier.prepare_features_baseline(val_df)
            X_test, y_test, _ = classifier.prepare_features_baseline(test_df)
        
        # Train and evaluate
        model, scaler = classifier.train_svm(X_train, y_train, X_val, y_val, class_weight='balanced')
        metrics = classifier.evaluate(model, scaler, X_test, y_test)
        
        print(f"\nRESULTS:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"AUROC: {metrics['auroc']:.4f}")
        print(f"F1-Macro: {metrics['f1_macro']:.4f}")
        
        # Save results
        classifier.save_top_features(model, feature_names, top_k=50)
        classifier.save_results(metrics)
        
        return metrics

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    # Setup logging
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    log_file = f"logs/cross_cmv_mage_experiments_v2/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    sys.stdout = Tee(log_file)
    
    # Initialize tracker
    global_tracker = GlobalResultsTracker(output_dir=RESULTS_DIR)
    
    # Configuration
    feature_groups = ['combined']  # or whatever feature groups you want
    filter_90_features = False
    
    print("="*56)
    print("### CROSS-DATASET CMV-MAGE EXPERIMENTS ###")
    print(f">> Feature Groups: {feature_groups}")
    print(f">> Filtered 90 Features: {filter_90_features}")
    print(f">> Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*56)
    
    # Initialize experiment handler
    experiment = CrossCMVMAGEExperiment()
    
    # Run experiments
    for feature_group in feature_groups:

        # Experiment 1: CMV (3 models) => MAGE CMV (30 models)
        try:
            print("\n" + "#"*56)
            print("# EXPERIMENT 1: Prompt Generalization Test")
            print("# Train: CMV (Direct Response - 3 models)")
            print("# Test: MAGE CMV (Continuation - 30 models)")
            print("#"*56)
            
            metrics1 = experiment.experiment_1_cmv_train_mage_cmv_test(
                feature_group, 
                filtered_90=filter_90_features
            )
            global_tracker.add_result(
                'cross_exp1_cmv_to_mage_cmv_all', 
                metrics1['accuracy'], 
                metrics1['auroc'], 
                metrics1['f1_macro']
            )
        except Exception as e:
            print(f"Error in Experiment 1: {e}\n")
        
        # Experiment 2: CMV (LLama+GPT) => MAGE CMV (LLama+OpenAI)
        try:
            print("\n" + "#"*56)
            print("# EXPERIMENT 2: Model Family Alignment Test")
            print("# Train: CMV (Direct Response - LLama+GPT)")
            print("# Test: MAGE CMV (Continuation - LLama+OpenAI)")
            print("#"*56)
            
            metrics2 = experiment.experiment_2_cmv_filtered_train_mage_filtered_test(
                feature_group,
                filtered_90=filter_90_features
            )
            global_tracker.add_result(
                'cross_exp2_cmv_llamagpt_to_mage_llamaopenai', 
                metrics2['accuracy'], 
                metrics2['auroc'], 
                metrics2['f1_macro']
            )
        except Exception as e:
            print(f"Error in Experiment 2: {e}\n")

            
        # Experiment 3: MAGE CMV (all models) => CMV (3 models)
        try:
            print("\n" + "#"*56)
            print("# EXPERIMENT 3: Reverse Prompt Generalization Test")
            print("# Train: MAGE CMV (Continuation - 30 models)")
            print("# Test: CMV (Direct Response - 3 models)")
            print("#"*56)
            
            metrics3 = experiment.experiment_3_mage_train_cmv_test(
                feature_group, 
                filtered_90=filter_90_features
            )
            global_tracker.add_result(
                'cross_exp3_mage_cmv_all_to_cmv', 
                metrics3['accuracy'], 
                metrics3['auroc'], 
                metrics3['f1_macro']
            )
        except Exception as e:
            print(f"Error in Experiment 3: {e}\n")

        # Experiment 4: MAGE CMV (OpenAI+LLama) => CMV (GPT+LLama)
        try:
            print("\n" + "#"*56)
            print("# EXPERIMENT 4: Reverse Model Family Alignment Test")
            print("# Train: MAGE CMV (Continuation - OpenAI+LLama)")
            print("# Test: CMV (Direct Response - GPT+LLama)")
            print("#"*56)
            
            metrics4 = experiment.experiment_4_mage_filtered_train_cmv_filtered_test(
                feature_group,
                filtered_90=filter_90_features
            )
            global_tracker.add_result(
                'cross_exp4_mage_openaillama_to_cmv_gptllama', 
                metrics4['accuracy'], 
                metrics4['auroc'], 
                metrics4['f1_macro']
            )
        except Exception as e:
            print(f"Error in Experiment 4: {e}\n")
    
    # Save global results
    global_tracker.save_results()
    
    print("\n" + "="*56)
    print("ALL CROSS-DATASET EXPERIMENTS COMPLETE")
    print("="*56)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    sys.stdout.file.close()
    sys.stdout = sys.stdout.terminal