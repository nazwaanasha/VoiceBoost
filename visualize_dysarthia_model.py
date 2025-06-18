import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from joblib import load
import warnings
warnings.filterwarnings('ignore')

class DysarthriaModelVisualizer:
    def __init__(self, model_path, data_training_path=None, data_testing_path=None):
        """
        Initialize visualizer with trained model and data paths
        
        Args:
            model_path: Path to saved model (.joblib file)
            data_training_path: Path to training data folder (optional)
            data_testing_path: Path to testing data folder (optional)
        """
        self.model_path = model_path
        self.data_training_path = data_training_path
        self.data_testing_path = data_testing_path
        
        # Load model
        self.model_data = load(model_path)
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.model_name = self.model_data.get('model_name', 'Unknown Model')
        self.accuracy = self.model_data.get('accuracy', 0)
        
        # Get actual feature names based on the training model structure
        self.feature_names = self._get_actual_feature_names()
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def _get_actual_feature_names(self):
        """Generate actual feature names based on the training model structure"""
        features = []
        
        # Formant features (from extract_formant_features)
        features.extend(['f1_mean', 'f1_std', 'f2_mean', 'f2_std', 'f3_mean', 'f3_std', 'vsa'])
        
        # Prosodic features (from extract_prosodic_features)
        features.extend([
            'pitch_mean', 'pitch_std', 'pitch_range',
            'intensity_mean', 'intensity_std', 'intensity_range'
        ])
        
        # Spectral features (from extract_spectral_features)
        features.extend([
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_bandwidth_mean', 'spectral_rolloff_mean',
            'zero_crossing_rate_mean', 'speaking_rate'
        ])
        
        # MFCC features (13 coefficients with mean and std)
        for i in range(13):
            features.extend([f'mfcc_{i}_mean', f'mfcc_{i}_std'])
        
        # Jitter and Shimmer features (from extract_jitter_shimmer)
        features.extend(['jitter_local', 'jitter_rap', 'shimmer_local', 'shimmer_db'])
        
        # Duration features (from extract_duration_features)
        features.extend([
            'utterance_duration', 'phonation_time_ratio', 'pause_count',
            'average_pause_duration', 'speech_rate'
        ])
        
        # Voice quality features (from extract_voice_quality_features)
        features.extend([
            'hnr_mean', 'breathiness', 'roughness', 'creakiness', 'spectral_flatness'
        ])
        
        # WER features (from calculate_word_error_rate)
        features.extend(['word_error_rate', 'word_count'])
        
        return features
    
    def load_actual_test_data(self, use_testing_folder=True):
        """
        Load actual test data from the same structure as training
        
        Args:
            use_testing_folder: If True, loads from data_testing folder, 
                               otherwise uses a split from training data
        
        Returns:
            X_test, y_test: Test features and labels
        """
        if use_testing_folder and self.data_testing_path and os.path.exists(self.data_testing_path):
            print(f"Loading test data from: {self.data_testing_path}")
            X_test, y_test = self._load_data_from_path(self.data_testing_path)
        elif self.data_training_path and os.path.exists(self.data_training_path):
            print(f"Loading training data and splitting for testing from: {self.data_training_path}")
            X_all, y_all = self._load_data_from_path(self.data_training_path)
            
            # Split data for testing (20% for test)
            from sklearn.model_selection import train_test_split
            _, X_test, _, y_test = train_test_split(
                X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
            )
        else:
            raise ValueError("No valid data path provided. Please set data_training_path or data_testing_path")
        
        return X_test, y_test
    
    def _load_data_from_path(self, data_path):
        """Load data from specified path using the same logic as training"""
        # Import the feature extraction class
        from train_dysarthria_model import DysarthriaClassifier
        
        # Create temporary classifier instance for feature extraction
        temp_classifier = DysarthriaClassifier(
            data_training_path=data_path,
            data_testing_path="",
            model_save_path="temp"
        )
        
        data = []
        labels = []
        
        print(f"Processing data from: {data_path}")
        
        # Process dysarthria samples
        dysarthria_path = os.path.join(data_path, 'dysarthria')
        if os.path.exists(dysarthria_path):
            print("Processing dysarthria samples...")
            dysarthria_features = self._process_category_with_classifier(temp_classifier, dysarthria_path)
            data.extend(dysarthria_features)
            labels.extend([1] * len(dysarthria_features))  # 1 for dysarthria
        
        # Process normal samples
        normal_path = os.path.join(data_path, 'normal')
        if os.path.exists(normal_path):
            print("Processing normal samples...")
            normal_features = self._process_category_with_classifier(temp_classifier, normal_path)
            data.extend(normal_features)
            labels.extend([0] * len(normal_features))  # 0 for normal
        
        print(f"Total samples loaded: {len(data)}")
        print(f"Dysarthria samples: {sum(labels)}")
        print(f"Normal samples: {len(labels) - sum(labels)}")
        
        return np.array(data), np.array(labels)
    
    def _process_category_with_classifier(self, classifier, category_path):
        """Process all audio files in a category using the classifier's feature extraction"""
        features_list = []
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(category_path):
            for file in files:
                if file.endswith('.wav'):
                    audio_path = os.path.join(root, file)
                    try:
                        features = classifier.extract_all_features(audio_path)
                        if features:
                            features_list.append(list(features.values()))
                    except Exception as e:
                        print(f"Error processing {audio_path}: {e}")
                        continue
                        
        return features_list
    
    def load_all_available_data(self):
        """Load all available data for comprehensive analysis"""
        all_X = []
        all_y = []
        
        # Load training data if available
        if self.data_training_path and os.path.exists(self.data_training_path):
            print("Loading training data...")
            X_train, y_train = self._load_data_from_path(self.data_training_path)
            all_X.append(X_train)
            all_y.append(y_train)
        
        # Load testing data if available
        if self.data_testing_path and os.path.exists(self.data_testing_path):
            print("Loading testing data...")
            X_test, y_test = self._load_data_from_path(self.data_testing_path)
            all_X.append(X_test)
            all_y.append(y_test)
        
        if all_X:
            X_combined = np.vstack(all_X)
            y_combined = np.hstack(all_y)
            return X_combined, y_combined
        else:
            raise ValueError("No data could be loaded from the specified paths")
    
    def plot_confusion_matrix(self, X_test, y_test, save_path=None):
        """Plot confusion matrix"""
        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Dysarthria'],
                   yticklabels=['Normal', 'Dysarthria'])
        plt.title(f'Confusion Matrix - {self.model_name}\nAccuracy: {self.accuracy:.3f}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, X_test, y_test, save_path=None):
        """Plot ROC curve"""
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get prediction probabilities
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = self.model.decision_function(X_test_scaled)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return roc_auc
    
    def plot_precision_recall_curve(self, X_test, y_test, save_path=None):
        """Plot Precision-Recall curve"""
        X_test_scaled = self.scaler.transform(X_test)
        
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = self.model.decision_function(X_test_scaled)
        
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.model_name}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return pr_auc
    
    def plot_feature_importance(self, X_test, y_test, save_path=None, top_n=20):
        """Plot feature importance (for tree-based models)"""
        if not hasattr(self.model, 'feature_importances_'):
            print("Feature importance not available for this model type")
            return None
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create DataFrame for easier handling
        feature_df = pd.DataFrame({
            'feature': self.feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=True).tail(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(feature_df)), feature_df['importance'])
        plt.yticks(range(len(feature_df)), feature_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances - {self.model_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_df
    
    def plot_feature_distribution(self, X, y, save_path=None, features_to_plot=None):
        """Plot distribution of selected features by class"""
        if features_to_plot is None:
            # Select some important features based on the actual model
            features_to_plot = ['f1_mean', 'f2_mean', 'pitch_mean', 'jitter_local', 
                              'word_error_rate', 'phonation_time_ratio']
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=self.feature_names[:X.shape[1]])
        df['class'] = ['Dysarthria' if label == 1 else 'Normal' for label in y]
        
        # Plot distributions
        n_features = len(features_to_plot)
        fig, axes = plt.subplots(2, (n_features + 1) // 2, figsize=(15, 8))
        axes = axes.flatten()
        
        for i, feature in enumerate(features_to_plot):
            if feature in df.columns:
                sns.boxplot(data=df, x='class', y=feature, ax=axes[i])
                axes[i].set_title(f'{feature} Distribution')
                axes[i].grid(True, alpha=0.3)
        
        # Remove unused subplots
        for i in range(len(features_to_plot), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.suptitle('Feature Distributions by Class', y=1.02, fontsize=16)
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pca_visualization(self, X, y, save_path=None):
        """Plot PCA visualization of the data"""
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red']
        labels = ['Normal', 'Dysarthria']
        
        for i in range(2):
            mask = y == i
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=colors[i], label=labels[i], alpha=0.6, s=50)
        
        plt.xlabel(f'PC1 (Variance: {pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 (Variance: {pca.explained_variance_ratio_[1]:.2%})')
        plt.title('PCA Visualization of Dysarthria vs Normal Speech')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'pca_visualization.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return pca
    
    def plot_tsne_visualization(self, X, y, save_path=None):
        """Plot t-SNE visualization of the data"""
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_scaled)
        
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red']
        labels = ['Normal', 'Dysarthria']
        
        for i in range(2):
            mask = y == i
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=colors[i], label=labels[i], alpha=0.6, s=50)
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE Visualization of Dysarthria vs Normal Speech')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'tsne_visualization.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_performance_metrics(self, X_test, y_test, save_path=None):
        """Plot comprehensive performance metrics"""
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [accuracy, precision, recall, f1]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        plt.ylim(0, 1)
        plt.ylabel('Score')
        plt.title(f'Model Performance Metrics - {self.model_name}')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def plot_prediction_probabilities(self, X_test, y_test, save_path=None):
        """Plot prediction probability distributions"""
        if not hasattr(self.model, 'predict_proba'):
            print("Prediction probabilities not available for this model type")
            return
        
        X_test_scaled = self.scaler.transform(X_test)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Probability distribution by true class
        plt.subplot(1, 2, 1)
        normal_probs = y_prob[y_test == 0]
        dysarthria_probs = y_prob[y_test == 1]
        
        plt.hist(normal_probs, bins=20, alpha=0.7, label='Normal', color='blue')
        plt.hist(dysarthria_probs, bins=20, alpha=0.7, label='Dysarthria', color='red')
        plt.xlabel('Predicted Probability (Dysarthria)')
        plt.ylabel('Frequency')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Probability vs prediction confidence
        plt.subplot(1, 2, 2)
        y_pred = self.model.predict(X_test_scaled)
        correct_predictions = (y_pred == y_test)
        
        plt.scatter(y_prob[correct_predictions], y_test[correct_predictions], 
                   alpha=0.6, label='Correct', color='green')
        plt.scatter(y_prob[~correct_predictions], y_test[~correct_predictions], 
                   alpha=0.6, label='Incorrect', color='red')
        plt.xlabel('Predicted Probability (Dysarthria)')
        plt.ylabel('True Label')
        plt.title('Prediction Confidence vs Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'prediction_probabilities.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self, save_path=None, use_testing_folder=True):
        """Generate comprehensive visualization report with actual data"""
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
        
        print("=== Generating Comprehensive Model Visualization Report ===")
        print(f"Model: {self.model_name}")
        
        # Load actual test data
        try:
            X_test, y_test = self.load_actual_test_data(use_testing_folder)
            print(f"Test samples: {len(X_test)}")
            print(f"Features: {X_test.shape[1]}")
        except Exception as e:
            print(f"Error loading test data: {e}")
            return None
        
        # 1. Confusion Matrix
        print("\n1. Plotting Confusion Matrix...")
        cm = self.plot_confusion_matrix(X_test, y_test, save_path)
        
        # 2. ROC Curve
        print("2. Plotting ROC Curve...")
        roc_auc = self.plot_roc_curve(X_test, y_test, save_path)
        
        # 3. Precision-Recall Curve
        print("3. Plotting Precision-Recall Curve...")
        pr_auc = self.plot_precision_recall_curve(X_test, y_test, save_path)
        
        # 4. Performance Metrics
        print("4. Plotting Performance Metrics...")
        metrics = self.plot_model_performance_metrics(X_test, y_test, save_path)
        
        # 5. Feature Importance (if available)
        print("5. Plotting Feature Importance...")
        self.plot_feature_importance(X_test, y_test, save_path)
        
        # 6. Feature Distributions
        print("6. Plotting Feature Distributions...")
        # Use all available data for better distribution visualization
        try:
            X_all, y_all = self.load_all_available_data()
            self.plot_feature_distribution(X_all, y_all, save_path)
        except:
            self.plot_feature_distribution(X_test, y_test, save_path)
        
        # 7. PCA Visualization
        print("7. Plotting PCA Visualization...")
        self.plot_pca_visualization(X_test, y_test, save_path)
        
        # 8. t-SNE Visualization
        print("8. Plotting t-SNE Visualization...")
        self.plot_tsne_visualization(X_test, y_test, save_path)
        
        # 9. Prediction Probabilities
        print("9. Plotting Prediction Probabilities...")
        self.plot_prediction_probabilities(X_test, y_test, save_path)
        
        # Summary
        print("\n=== Visualization Report Summary ===")
        print(f"Model Accuracy: {metrics['accuracy']:.3f}")
        print(f"ROC AUC: {roc_auc:.3f}")
        print(f"PR AUC: {pr_auc:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1-Score: {metrics['f1_score']:.3f}")
        
        if save_path:
            print(f"\nAll visualizations saved to: {save_path}")
        
        return {
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'metrics': metrics
        }

def main():
    """Example usage of the enhanced visualizer with actual data"""
    # Path to your trained model
    model_path = "models/dysarthria_classifier.joblib"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please train the model first using train_dysarthria_model.py")
        return
    
    # Paths to your actual data
    data_training_path = "data_training"
    data_testing_path = "data_testing"
    
    # Initialize visualizer with actual data paths
    visualizer = DysarthriaModelVisualizer(
        model_path=model_path,
        data_training_path=data_training_path,
        data_testing_path=data_testing_path
    )
    
    print("Initializing enhanced visualizer with actual data...")
    print(f"Model: {visualizer.model_name}")
    print(f"Expected features: {len(visualizer.feature_names)}")
    
    # Generate comprehensive report with actual data
    save_path = "visualization_output"
    results = visualizer.generate_comprehensive_report(
        save_path=save_path, 
        use_testing_folder=True  # Set to False to use split from training data
    )
    
    if results:
        print(f"\nVisualization complete! Check the '{save_path}' folder for all plots.")
    else:
        print("Visualization failed. Please check your data paths and model file.")

if __name__ == "__main__":
    main()