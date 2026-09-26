from sklearnex import patch_sklearn 
patch_sklearn()
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                            confusion_matrix, roc_auc_score, average_precision_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
import warnings
import sys
import json

warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)



import yaml, json, pathlib
CFG = yaml.safe_load(open(pathlib.Path(__file__).parent.parent.parent / "configs/paths.yaml"))


# =============================================================================
# SELECTED FEATURES
# =============================================================================

SELECTED_FEATURES = {
    "surface": [
        "n_tokens"
    ],
    "pos": [
        "n_verb",
        "n_cconj"
    ],
    "emotion": [
        'avg_intensity_anticipation',
        'avg_intensity_joy',
        'avg_intensity_surprise',
        'avg_intensity_trust',
        'n_low_intensity_anger',
        'n_low_intensity_anticipation',
        'n_low_intensity_disgust',
        'n_low_intensity_fear',
        'n_low_intensity_joy',
        'n_low_intensity_sadness',
        'n_low_intensity_surprise',
        'n_low_intensity_trust',
        'n_high_intensity_anticipation',
        'n_high_intensity_disgust',
        'n_high_intensity_joy',
        'n_high_intensity_surprise',
        'n_high_intensity_trust',
        'n_positive_sentiment',
        'n_negative_sentiment'
    ],
    "psycholinguistic": [
        'n_controversial_aoa',
        'avg_sd_socialness',
        'avg_sd_iconicity',
        'n_high_Torso_sensorimotor',
        'avg_sd_Head_sensorimotor',
        'avg_concreteness'
    ],
    "readability": ["flesch_reading_ease"],
    "dependency": [
        'tree_width',
        'n_noun_chunks',
        'n_dependency_acl',
        'n_dependency_acomp',
        'n_dependency_agent',
        'n_dependency_appos',
        'n_dependency_attr',
        'n_dependency_case',
        'n_dependency_csubj',
        'n_dependency_csubjpass',
        'n_dependency_dative',
        'n_dependency_meta',
        'n_dependency_oprd',
        'n_dependency_parataxis',
        'n_dependency_pcomp',
        'n_dependency_preconj',
        'n_dependency_predet',
        'n_dependency_prt',
        'n_dependency_quantmod'
    ],
    "lexical_richness": [
        "ttr",
        "n_global_token_hapax_legomena"
    ],
    "entities": [
        'n_sym',
        'n_money',
        'n_product',
        'n_time',
        'n_percent',
        'n_work_of_art',
        'n_quantity',
        'n_norp',
        'n_loc',
        'n_event',
        'n_ordinal',
        'n_fac',
        'n_law',
        'n_language'
    ],
    "morphological": [
        'n_VERB_Tense_Past',
        'n_NOUN_Gender_Neut',
        'n_NOUN_Case_Acc',
        'n_NOUN_Case_Nom',
        'n_PRON_PronType_Art',
        'n_PRON_PronType_Dem',
        'n_PRON_PronType_Ind',
        'n_PRON_Reflex_Yes',
        'n_PRON_Gender_Neut',
        'n_PRON_Number_Plur',
        'n_PRON_Person_2',
        'n_ADJ_Degree_Cmp',
        'n_ADJ_Degree_Sup',
        'n_DET_Number_Plur',
        'n_DET_Number_Sing',
        'n_DET_Definite_Ind',
        'n_ADV_Degree_Cmp',
        'n_ADV_Degree_Pos',
        'n_ADV_Degree_Sup',
        'n_PROPN_Gender_Neut',
        'n_PROPN_Number_Plur',
        'n_PROPN_Case_Acc',
        'n_PROPN_Case_Nom',
        'n_PUNCT_PunctType_Comm',
        'n_PUNCT_PunctType_Dash'
    ],
    "semantic": ["n_hedges"]
}

# =============================================================================
# UTILITY CLASSES
# =============================================================================

class Tee:
    # Redirect stdout to both terminal and file
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, 'w')
    def __del__(self):
        self.file.close()
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.file.flush()


# =============================================================================
# TESTBED CLASSIFIER
# =============================================================================

class TestbedClassifier:
    # SVM classifier for AI text detection testbeds
    
    def __init__(self, testbed_name, feature_group, output_dir=CFG['results_root']):
        self.testbed_name = testbed_name
        self.feature_group = feature_group
        # self.output_dir = os.path.join(output_dir, testbed_name, feature_group)

        # Make sure the output for testbed1 well-organized: grouped by domain
        if testbed_name.startswith('1_fixed_domain_model_specific_'):
            domain = testbed_name.split('_')[5]
            self.output_dir = os.path.join(output_dir, 'testbed1_all_models', domain, testbed_name, feature_group)
        elif testbed_name.startswith('11_fixed_domain_model_family_specific_'):
            domain = testbed_name.split('_')[6]
            self.output_dir = os.path.join(output_dir, 'testbed_1_model_family', domain, testbed_name, feature_group)
        elif testbed_name.startswith('8_unseen_domain_model_pair_'):
            # Extract domain and family from the name: 8_unseen_domain_model_pair_{domain}_{family}
            parts = testbed_name.split('_')
            domain = parts[5]  # domain is at index 5
            family = parts[6]  # family is at index 6
            self.output_dir = os.path.join(output_dir, 'testbed8_unseen_pairs', domain, family, testbed_name, feature_group)
        
        else:
            self.output_dir = os.path.join(output_dir, testbed_name, feature_group)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def prepare_features_90(self, df):
        # Extract selected features and labels from dataframe
        selected_feature_list = []
        for group_features in SELECTED_FEATURES.values():
            selected_feature_list.extend(group_features)
        
        # Only keep selected features that exist in df
        available_features = [f for f in selected_feature_list if f in df.columns]
        # # Just confirming...
        # print("*"*56)
        # print(f"List of available features:\n{available_features}")
        # print("*"*56)

        X = df[available_features].values
        y = df['label'].values
        
        print(f"Features shape: {X.shape}")
        print(f"Labels: Human={sum(y==0)}, AI={sum(y==1)}")
        
        return X, y, available_features
    
    def prepare_features_baseline(self, df):
        """Extract features and labels from dataframe."""
        exclude_cols = ['doc_id', 'id', 'comment_id', 'post_id', 'label', 'text']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df['label'].values
        
        print(f"Features shape: {X.shape}")
        print(f"Labels: Human={sum(y==0)}, AI={sum(y==1)}")
        
        return X, y, feature_cols
    
    def train_svm(self, X_train, y_train, X_val, y_val, class_weight='balanced'):
        # Train SVM with validation monitoring
        print(f"\nTraining SVM with class_weight={class_weight}...")
        
        model = SVC(kernel='linear', C=1.0, random_state=42, class_weight=class_weight)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model.fit(X_train_scaled, y_train)
        
        val_pred = model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='macro')
        print(f"Validation — Accuracy: {val_acc:.4f} | F1-Macro: {val_f1:.4f}")
        
        return model, scaler
    
    def evaluate(self, model, scaler, X_test, y_test):
        # Evaluate model with comprehensive metrics
        print(f"\nEvaluating on test set...")
        
        X_test_scaled = scaler.transform(X_test)
        
        y_pred = model.predict(X_test_scaled)
        y_scores = model.decision_function(X_test_scaled)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auroc': roc_auc_score(y_test, y_scores),
            'avg_precision': average_precision_score(y_test, y_scores),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_micro': f1_score(y_test, y_pred, average='micro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'precision_human': precision[0],
            'precision_ai': precision[1],
            'recall_human': recall[0],
            'recall_ai': recall[1],
            'f1_human': f1[0],
            'f1_ai': f1[1],
            'avg_recall': (recall[0] + recall[1]) / 2,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def save_top_features(self, model, feature_names, top_k=50):
        # Save top k and all features with coefficients
        coefs = model.coef_[0]
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefs,
            'abs_importance': np.abs(coefs)
        })
        
        feature_importance = feature_importance.sort_values('abs_importance', ascending=False)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save top k
        top_features = feature_importance.head(top_k)
        filepath_top = os.path.join(self.output_dir, f'top_{top_k}_features_{timestamp}.csv')
        top_features[['feature', 'coefficient']].to_csv(filepath_top, index=False)
        
        # Save all features
        filepath_all = os.path.join(self.output_dir, f'all_features_{timestamp}.csv')
        feature_importance[['feature', 'coefficient']].to_csv(filepath_all, index=False)
        
        print(f"\nTop {top_k} features saved to: {filepath_top}")
        print(f"All features saved to: {filepath_all}")
        return top_features
    
    def save_results(self, metrics):
        # Save comprehensive results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f'results_{timestamp}.txt')
        
        with open(filepath, 'w') as f:
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
        
        print(f"Results saved to: {filepath}")
    
    def save_model(self, model, scaler, feature_names):
        # Save trained model, scaler, and features
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = os.path.join(self.output_dir, f'model_{timestamp}.joblib')
        scaler_path = os.path.join(self.output_dir, f'scaler_{timestamp}.joblib')
        features_path = os.path.join(self.output_dir, f'features_{timestamp}.joblib')
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(feature_names, features_path)
        
        print(f"\nModel saved to: {model_path}")


class GlobalResultsTracker:
    # Track global results across all testbeds
    def __init__(self, output_dir=CFG['results_root']):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []
    
    def add_result(self, testbed_name, accuracy, auroc, f1_macro):
        self.results.append({
            'testbed': testbed_name,
            'accuracy': float(accuracy),
            'auroc': float(auroc),
            'f1_macro': float(f1_macro)
        })
    
    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f'global_results_{timestamp}.json')
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nGlobal results saved to: {filepath}")


# =============================================================================
# MAGE TESTBED CONFIGURATION
# =============================================================================

class MAGETestbedConfig:
    # Configuration manager for all MAGE testbeds
    
    def __init__(self, base_path=None):
        base_path = base_path or CFG['mage_features']
        self.train_path = os.path.join(base_path, "train")
        self.val_path = os.path.join(base_path, "validation")
        self.test_path = os.path.join(base_path, "test")
        
        self.domains = CFG['domains']
        self.unseen_test_domains = CFG['unseen_test_domains']
        
        self.model_families = json.load(open('configs/model_families.json'))
    
    def fixed_domain_model_specific(self, domain, model='gpt_j'):
        # Testbed 1: Fixed-domain & Model-specific
        config = {
            'name': f'1_fixed_domain_model_specific_{domain}_{model}',
            'train': {
                'human': f"{self.train_path}/Human/{domain}/samples/samples_features",
                'ai': self._find_model_path(self.train_path, domain, model)
            },
            'val': {
                'human': f"{self.val_path}/Human/{domain}/samples/samples_features",
                'ai': self._find_model_path(self.val_path, domain, model)
            },
            'test': {
                'human': f"{self.test_path}/Human/{domain}/samples/samples_features",
                'ai': self._find_model_path(self.test_path, domain, model)
            }
        }
        return config
    def fixed_domain_model_family_specific(self, domain, model_family):
        #Testbed 11: Fixed-domain & Model-family-specific
        config = {
            'name': f'11_fixed_domain_model_family_specific_{domain}_{model_family}',
            'train': {
                'human': f"{self.train_path}/Human/{domain}/samples/samples_features",
                'ai': self._find_all_models_in_family_for_domain(self.train_path, domain, model_family)
            },
            'val': {
                'human': f"{self.val_path}/Human/{domain}/samples/samples_features",
                'ai': self._find_all_models_in_family_for_domain(self.val_path, domain, model_family)
            },
            'test': {
                'human': f"{self.test_path}/Human/{domain}/samples/samples_features",
                'ai': self._find_all_models_in_family_for_domain(self.test_path, domain, model_family)
            }
        }
        return config
    
    def get_all_available_models(self, split_path):
        # Get all unique model names across all domains
        models = set()
        for domain in self.domains:
            domain_path = f"{split_path}/AI/{domain}"
            if os.path.exists(domain_path):
                for family in os.listdir(domain_path):
                    family_path = os.path.join(domain_path, family)
                    if os.path.isdir(family_path):
                        for model_folder in os.listdir(family_path):
                            models.add(model_folder)
        return sorted(list(models))
    
    def arbitrary_domains_model_specific(self, model_family):
        # Testbed 2: Arbitrary-domains & Model-specific
        config = {
            'name': f'2_arbitrary_domains_model_specific_{model_family}',
            'train': {
                'human': [f"{self.train_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_models_in_family(self.train_path, model_family)
            },
            'val': {
                'human': [f"{self.val_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_models_in_family(self.val_path, model_family)
            },
            'test': {
                'human': [f"{self.test_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_models_in_family(self.test_path, model_family)
            }
        }
        return config
    
    def fixed_domain_arbitrary_models(self, domain):
        # Testbed 3: Fixed-domain & Arbitrary-models
        config = {
            'name': f'3_fixed_domain_arbitrary_models_{domain}',
            'train': {
                'human': f"{self.train_path}/Human/{domain}/samples/samples_features",
                'ai': self._find_all_models_for_domain(self.train_path, domain)
            },
            'val': {
                'human': f"{self.val_path}/Human/{domain}/samples/samples_features",
                'ai': self._find_all_models_for_domain(self.val_path, domain)
            },
            'test': {
                'human': f"{self.test_path}/Human/{domain}/samples/samples_features",
                'ai': self._find_all_models_for_domain(self.test_path, domain)
            }
        }
        return config
    
    def arbitrary_domains_arbitrary_models(self):
        # Testbed 4: Arbitrary-domains & Arbitrary-models
        config = {
            'name': '4_arbitrary_domains_arbitrary_models',
            'train': {
                'human': [f"{self.train_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_ai_data(self.train_path)
            },
            'val': {
                'human': [f"{self.val_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_ai_data(self.val_path)
            },
            'test': {
                'human': [f"{self.test_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_ai_data(self.test_path)
            }
        }
        return config
    
    def unseen_models(self, excluded_family):
        # Testbed 5: Unseen Models
        train_families = [f for f in self.model_families.keys() if f != excluded_family]
        
        config = {
            'name': f'5_unseen_models_{excluded_family}',
            'train': {
                'human': [f"{self.train_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': []
            },
            'val': {
                'human': [f"{self.val_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': []
            },
            'test': {
                'human': [f"{self.test_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_models_in_family(self.test_path, excluded_family)
            }
        }
        
        for family in train_families:
            config['train']['ai'].extend(
                self._find_all_models_in_family(self.train_path, family)
            )
            config['val']['ai'].extend(
                self._find_all_models_in_family(self.val_path, family)
            )
        
        return config
    
    def unseen_domains(self, excluded_domain):
        # Testbed 6: Unseen Domains
        train_domains = [d for d in self.domains if d != excluded_domain]
        
        config = {
            'name': f'6_unseen_domains_{excluded_domain}',
            'train': {
                'human': [f"{self.train_path}/Human/{d}/samples/samples_features" for d in train_domains],
                'ai': []
            },
            'val': {
                'human': [f"{self.val_path}/Human/{d}/samples/samples_features" for d in train_domains],
                'ai': []
            },
            'test': {
                'human': f"{self.test_path}/Human/{excluded_domain}/samples/samples_features",
                'ai': self._find_all_models_for_domain(self.test_path, excluded_domain)
            }
        }
        
        for d in train_domains:
            config['train']['ai'].extend(
                self._find_all_models_for_domain(self.train_path, d)
            )
            config['val']['ai'].extend(
                self._find_all_models_for_domain(self.val_path, d)
            )
        
        return config
    
    def unseen_domains_unseen_model(self, unseen_model = 'gpt4'):
        # Testbed 7: Unseen-domains & Unseen-model
        config = {
            'name': f'7_unseen_domains_unseen_model_{unseen_model}',
            'train': {
                'human': [f"{self.train_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_ai_data(self.train_path)
            },
            'val': {
                'human': [f"{self.val_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_ai_data(self.val_path)
            },
            'test': {
                'human': [f"{self.test_path}/unseen/Human/{d}/{d}_human/{d}_human_features" 
                         for d in ['cnn', 'dialogsum', 'imdb', 'pubmed']],
                'ai': [f"{self.test_path}/unseen/AI/{d}/{d}_{unseen_model}/{d}_{unseen_model}_features" 
                      for d in ['cnn', 'dialogsum', 'imdb', 'pubmed']]
            }
        }
        return config
    def unseen_domains_unseen_model_separate(self, test_domain, unseen_model = 'gpt4'):
        #Testbed 7.4: Unseen-domains & Unseen-model (testing on each domain separately)
        config = {
            'name': f'74_unseen_domains_unseen_model_{test_domain}_{unseen_model}',
            'train': {
                'human': [f"{self.train_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_ai_data(self.train_path)
            },
            'val': {
                'human': [f"{self.val_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_ai_data(self.val_path)
            },
            'test': {
                'human': f"{self.test_path}/unseen/Human/{test_domain}/{test_domain}_human/{test_domain}_human_features",
                'ai': f"{self.test_path}/unseen/AI/{test_domain}/{test_domain}_{unseen_model}/{test_domain}_{unseen_model}_features"
            }
        }
        return config

    def unseen_domains_unseen_models_combined(self, unseen_models=('gpt4', 'gpt5_6_sol')):
        # Testbed 7c: Unseen-domains, all unseen models pooled into one AI test set
        tag = '_'.join(unseen_models)
        config = {
            'name': f'7c_unseen_domains_models_combined_{tag}',
            'train': {
                'human': [f"{self.train_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_ai_data(self.train_path)
            },
            'val': {
                'human': [f"{self.val_path}/Human/{d}/samples/samples_features" for d in self.domains],
                'ai': self._find_all_ai_data(self.val_path)
            },
            'test': {
                'human': [f"{self.test_path}/unseen/Human/{d}/{d}_human/{d}_human_features"
                        for d in self.unseen_test_domains],
                'ai': [f"{self.test_path}/unseen/AI/{d}/{d}_{m}/{d}_{m}_features"
                    for d in self.unseen_test_domains for m in unseen_models]
            }
        }
        return config


    def unseen_domain_model_pair(self, excluded_domain, excluded_family):
        """
        Testbed 8: Unseen Domain-Model Pair (Complete Exclusion)
        
        Completely exclude both the domain AND the model family from training.
        
        Example: If excluded_domain='yelp' and excluded_family='openai':
        - Train on: All OTHER domains x All OTHER families
        - Test on: Only (yelp, openai)
        
        This means:
        - NO Yelp data at all in training (even from other models)
        - NO OpenAI data at all in training (even from other domains)
        
        With 10 domains and 7 families:
        - Training: 9 domains x 6 families = 54 combinations
        - Testing: 1 domain x 1 family = 1 combination
        """
        # Get all domains EXCEPT the excluded one
        train_domains = [d for d in self.domains if d != excluded_domain]
        
        # Get all families EXCEPT the excluded one
        train_families = [f for f in self.model_families.keys() if f != excluded_family]
        
        config = {
            'name': f'8_unseen_domain_model_pair_{excluded_domain}_{excluded_family}',
            'train': {
                'human': [f"{self.train_path}/Human/{d}/samples/samples_features" 
                        for d in train_domains],
                'ai': []
            },
            'val': {
                'human': [f"{self.val_path}/Human/{d}/samples/samples_features" 
                        for d in train_domains],
                'ai': []
            },
            'test': {
                'human': f"{self.test_path}/Human/{excluded_domain}/samples/samples_features",
                'ai': self._find_all_models_in_family_for_domain(
                    self.test_path, excluded_domain, excluded_family
                )
            }
        }
        
        # Add AI training data: ONLY from train_domains × train_families
        for domain in train_domains:
            for family in train_families:
                train_paths = self._find_all_models_in_family_for_domain(
                    self.train_path, domain, family
                )
                config['train']['ai'].extend(train_paths)
                
                val_paths = self._find_all_models_in_family_for_domain(
                    self.val_path, domain, family
                )
                config['val']['ai'].extend(val_paths)
        
        return config

    # Helper functions
    def _find_model_path(self, split_path, domain, model):
        for family, models in self.model_families.items():
            for m in models:
                if model in m or m in model:
                    path = f"{split_path}/AI/{domain}/{family}/{model}/{model}_features"
                    if os.path.exists(path):
                        return path
        return None
    def _find_all_models_in_family_for_domain(self, split_path, domain, family):
        #Find all models in a specific family for a specific domain
        paths = []
        family_path = f"{split_path}/AI/{domain}/{family}"
        if os.path.exists(family_path):
            for model_folder in os.listdir(family_path):
                model_features_path = os.path.join(family_path, model_folder, f"{model_folder}_features")
                if os.path.isdir(model_features_path):
                    paths.append(model_features_path)
        return paths
    
    def _find_all_models_in_family(self, split_path, family):
        paths = []
        for domain in self.domains:
            family_path = f"{split_path}/AI/{domain}/{family}"
            if os.path.exists(family_path):
                for model_folder in os.listdir(family_path):
                    model_features_path = os.path.join(family_path, model_folder, f"{model_folder}_features")
                    if os.path.isdir(model_features_path):
                        paths.append(model_features_path)
        return paths
    
    def _find_all_models_for_domain(self, split_path, domain):
        paths = []
        domain_path = f"{split_path}/AI/{domain}"
        if os.path.exists(domain_path):
            for family in os.listdir(domain_path):
                family_path = os.path.join(domain_path, family)
                if os.path.isdir(family_path):
                    for model_folder in os.listdir(family_path):
                        model_features_path = os.path.join(family_path, model_folder, f"{model_folder}_features")
                        if os.path.exists(model_features_path):
                            paths.append(model_features_path)
        return paths
    
    def _find_all_ai_data(self, split_path):
        paths = []
        for domain in self.domains:
            paths.extend(self._find_all_models_for_domain(split_path, domain))
        return paths

    def load_data_from_config_no_subsampling(self, config, feature_group):
        # Load train/val/test data without any balancing
        def load_from_paths(paths, label):
            dfs = []
            if isinstance(paths, str):
                paths = [paths]
            
            for path in paths:
                feature_file = os.path.join(path, f"{feature_group}_features.csv")
                if os.path.exists(feature_file):
                    df = pd.read_csv(feature_file)
                    df['label'] = 0 if 'Human'.lower() in str(path).lower() else 1
                    dfs.append(df)
            
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        
        train_human = load_from_paths(config['train']['human'], label=0)
        train_ai = load_from_paths(config['train']['ai'], label=1)
        train_df = pd.concat([train_human, train_ai], ignore_index=True)
        
        val_human = load_from_paths(config['val']['human'], label=0)
        val_ai = load_from_paths(config['val']['ai'], label=1)
        val_df = pd.concat([val_human, val_ai], ignore_index=True)
        
        test_human = load_from_paths(config['test']['human'], label=0)
        test_ai = load_from_paths(config['test']['ai'], label=1)
        test_df = pd.concat([test_human, test_ai], ignore_index=True)
        
        print(f"Train: {len(train_df)} ({sum(train_df['label']==0)} H, {sum(train_df['label']==1)} AI)")
        print(f"Val:   {len(val_df)} ({sum(val_df['label']==0)} H, {sum(val_df['label']==1)} AI)")
        print(f"Test:  {len(test_df)} ({sum(test_df['label']==0)} H, {sum(test_df['label']==1)} AI)")
        
        return train_df, val_df, test_df


# =============================================================================
# MAIN EXECUTION FUNCTIONS
# =============================================================================

def run_cmv_testbed(feature_group, ai_model='combined', base_path="data_features_prepared/cmv", global_tracker=None, filtered_90 = False):
    # Run CMV testbed with train/val/test split
    print("\n" + "="*56)
    print(f"CMV TESTBED - {feature_group} - {ai_model}")
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
    
    classifier = TestbedClassifier(f'cmv_{ai_model}', feature_group)

    if filtered_90:
        X, y, feature_names = classifier.prepare_features_90(df)
    else:
        X, y, feature_names = classifier.prepare_features_baseline(df)

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
    classifier.save_results(metrics)
    # classifier.save_model(model, scaler, feature_names)
    
    print("CMV testbed complete!\n")


def run_mage_testbed(config, feature_group, global_tracker=None, filtered_90 = False):
    # Run MAGE testbed with class_weight strategy
    print("\n" + "="*56)
    print(f"MAGE TESTBED: {config['name']} - {feature_group}")
    print("="*56)
    
    config_manager = MAGETestbedConfig()
    
    train_df, val_df, test_df = config_manager.load_data_from_config_no_subsampling(config, feature_group)
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("Skipping - insufficient data")
        return
    
    classifier = TestbedClassifier(config['name'], feature_group)
    if filtered_90:
        X_train, y_train, feature_names = classifier.prepare_features_90(train_df)
        X_val, y_val, _ = classifier.prepare_features_90(val_df)
        X_test, y_test, _ = classifier.prepare_features_90(test_df)
    else:
        X_train, y_train, feature_names = classifier.prepare_features_baseline(train_df)
        X_val, y_val, _ = classifier.prepare_features_baseline(val_df)
        X_test, y_test, _ = classifier.prepare_features_baseline(test_df)
    
    model, scaler = classifier.train_svm(X_train, y_train, X_val, y_val, class_weight='balanced')
    metrics = classifier.evaluate(model, scaler, X_test, y_test)
    
    if global_tracker:
        global_tracker.add_result(config['name'], metrics['accuracy'], metrics['auroc'], metrics['f1_macro'])
    
    print(f"\nRESULTS:")
    print(f"Accuracy: {metrics['accuracy']:.4f}, AUROC: {metrics['auroc']:.4f}, F1: {metrics['f1_macro']:.4f}")
    
    classifier.save_top_features(model, feature_names, top_k=50)
    classifier.save_results(metrics)
    # classifier.save_model(model, scaler, feature_names)
    
    print("Testbed complete!\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    RUN_NAME = 'baseline'
    
    feature_groups = ['combined'] #group of features to use

    log_dir = os.path.join(CFG['logs_root'], RUN_NAME)
    out_dir = os.path.join(CFG['results_root'], RUN_NAME)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    sys.stdout = Tee(log_file)
    
    global_tracker = GlobalResultsTracker(output_dir=out_dir)
    
    filter_90_features = False # Set to True if you want to test with the 90 features instead of the original
    
    print("="*56)
    print("### FINAL MAIN EXPERIMENT ###")
    print(f">> Feature Groups: {feature_groups}")
    print(f">> Strategy: class_weight='balanced'")
    print(f">> Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*56)
    
    # Run CMV testbeds
    print("\n" + "#"*56)
    print("# RUNNING CMV TESTBEDS")
    print("#"*56)
    
    for ai_model in ['combined', 'gpt', 'llama', 'mistral']:
        for feature_group in feature_groups:
            try:
                run_cmv_testbed(feature_group, ai_model, global_tracker=global_tracker, filtered_90=filter_90_features)
            except Exception as e:
                print(f"Error in CMV {ai_model} {feature_group}: {e}\n")
    
    # Run MAGE testbeds (1-7)
    print("\n" + "#"*56)
    print("# RUNNING MAGE TESTBEDS")
    print("#"*56)
    
    config_manager = MAGETestbedConfig()
    
    # Testbed 1: Fixed-domain & Model-specific
    print("\n" + "="*56)
    print("TESTBED 1: Fixed-domain & Model-specific")
    print("="*56)

    all_models = config_manager.get_all_available_models(config_manager.train_path)

    for model in all_models:
        for domain in config_manager.domains:
            config = config_manager.fixed_domain_model_specific(domain, model)
            # Check if model exists for this domain
            if config['train']['ai'] is None:
                print(f"Skipping {model} for {domain} - model not found")
                continue
            for feature_group in feature_groups:
                try:
                    run_mage_testbed(config, feature_group, global_tracker=global_tracker, filtered_90=filter_90_features)
                except Exception as e:
                    print(f"Error: {e}\n")
    
    # Testbed 11: Fixed-domain & Model-family-specific
    print("\n" + "="*56)
    print("TESTBED 11: Fixed-domain & Model-family-specific")
    print("="*56)

    for family in config_manager.model_families.keys():
        for domain in config_manager.domains:
            config = config_manager.fixed_domain_model_family_specific(domain, family)
            # Check if family exists for this domain
            if not config['train']['ai']:
                print(f"Skipping {family} for {domain} - family not found")
                continue
            for feature_group in feature_groups:
                try:
                    run_mage_testbed(config, feature_group, global_tracker=global_tracker, filtered_90=filter_90_features)
                except Exception as e:
                    print(f"Error: {e}\n")

    # Testbed 2: Arbitrary-domains & Model-specific
    print("\n" + "="*56)
    print("TESTBED 2: Arbitrary-domains & Model-specific")
    print("="*56)
    for family in config_manager.model_families.keys():
        config = config_manager.arbitrary_domains_model_specific(family)
        for feature_group in feature_groups:
            try:
                run_mage_testbed(config, feature_group, global_tracker=global_tracker,filtered_90=filter_90_features)
            except Exception as e:
                print(f"Error: {e}\n")
    
    # Testbed 3: Fixed-domain & Arbitrary-models
    print("\n" + "="*56)
    print("TESTBED 3: Fixed-domain & Arbitrary-models")
    print("="*56)
    for domain in config_manager.domains:
        config = config_manager.fixed_domain_arbitrary_models(domain)
        for feature_group in feature_groups:
            try:
                run_mage_testbed(config, feature_group, global_tracker=global_tracker,filtered_90=filter_90_features)
            except Exception as e:
                print(f"Error: {e}\n")
    
    # Testbed 4: Arbitrary-domains & Arbitrary-models
    print("\n" + "="*56)
    print("TESTBED 4: Arbitrary-domains & Arbitrary-models")
    print("="*56)
    config = config_manager.arbitrary_domains_arbitrary_models()
    for feature_group in feature_groups:
        try:
            run_mage_testbed(config, feature_group, global_tracker=global_tracker,filtered_90=filter_90_features)
        except Exception as e:
            print(f"Error: {e}\n")
    
    # Testbed 5: Unseen Models
    print("\n" + "="*56)
    print("TESTBED 5: Unseen Models")
    print("="*56)
    for family in config_manager.model_families.keys():
        config = config_manager.unseen_models(family)
        for feature_group in feature_groups:
            try:
                run_mage_testbed(config, feature_group, global_tracker=global_tracker,filtered_90=filter_90_features)
            except Exception as e:
                print(f"Error: {e}\n")
    
    # Testbed 6: Unseen Domains
    print("\n" + "="*56)
    print("TESTBED 6: Unseen Domains")
    print("="*56)
    for domain in config_manager.domains:
        config = config_manager.unseen_domains(domain)
        for feature_group in feature_groups:
            try:
                run_mage_testbed(config, feature_group, global_tracker=global_tracker,filtered_90=filter_90_features)
            except Exception as e:
                print(f"Error: {e}\n")
    
    UNSEEN_MODELS = ['gpt4', 'gpt5_6_sol']
    
    # Testbed 7c: Unseen-domains & Unseen-models combined
    print("\n" + "="*56)
    print("TESTBED 7c: Unseen-domains & Unseen-models combined")
    print("="*56)
    config = config_manager.unseen_domains_unseen_models_combined(('gpt4', 'gpt5_6_sol'))
    for feature_group in feature_groups:
        try:
            run_mage_testbed(config, feature_group, global_tracker=global_tracker,
                            filtered_90=filter_90_features)
        except Exception as e:
            print(f"Error in testbed 7c: {e}\n")

            
    # Testbed 7: Unseen-domains & Unseen-model
    print("\n" + "="*56)
    print("TESTBED 7: Unseen-domains & Unseen-model")
    print("="*56)
    for unseen_model in UNSEEN_MODELS:
        config = config_manager.unseen_domains_unseen_model(unseen_model)
        for feature_group in feature_groups:
            try:
                run_mage_testbed(config, feature_group, global_tracker=global_tracker,filtered_90=filter_90_features)
            except Exception as e:
                print(f"Error: {e}\n")
    
    # Testbed 7.4: Unseen-domains & Unseen-model (per domain)
    print("\n" + "="*56)
    print("TESTBED 7.4: Unseen-domains & Unseen-model (per domain)")
    print("="*56)
    
    unseen_test_domains = ['cnn', 'dialogsum', 'imdb', 'pubmed']
    for unseen_model in UNSEEN_MODELS:
        for test_domain in unseen_test_domains:
            config = config_manager.unseen_domains_unseen_model_separate(test_domain,unseen_model)
            for feature_group in feature_groups:
                try:
                    run_mage_testbed(config, feature_group, global_tracker=global_tracker, filtered_90=filter_90_features)
                except Exception as e:
                    print(f"Error in testbed 7.4 for {test_domain}: {e}\n")

    #Testbed 8: Unseen Domain-Model Pair (Complete Exclusion)
    print("\n" + "="*56)
    print("TESTBED 8: Unseen Domain-Model Pair (Complete Exclusion)")
    print("="*56)

    for domain in config_manager.domains:
        for family in config_manager.model_families.keys():
            config = config_manager.unseen_domain_model_pair(domain, family)
            
            # Check if test data exists for this specific pair
            if not config['test']['ai']:
                print(f">>>> Skipping {domain}-{family} pair - no test data found <<<<<")
                continue
            
            for feature_group in feature_groups:
                try:
                    run_mage_testbed(config, feature_group, global_tracker=global_tracker, 
                                filtered_90=filter_90_features)
                except Exception as e:
                    print(f"Error in testbed 8 for {domain}-{family}: {e}\n")
                    
    # Save global results
    global_tracker.save_results()
    
    print("\n" + "="*56)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*56)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    sys.stdout.file.close()
    sys.stdout = sys.stdout.terminal