import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class GANValidator:
    def __init__(self, original_data_path, synthetic_data_path):
        self.original_df = pd.read_csv(original_data_path)
        self.synthetic_df = pd.read_csv(synthetic_data_path)
        
    def compare_distributions(self, feature, bins=30):
        """Compare distributions of original and synthetic data for a given feature"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(data=self.original_df, x=feature, bins=bins)
        plt.title(f'Original {feature} Distribution')
        
        plt.subplot(1, 2, 2)
        sns.histplot(data=self.synthetic_df, x=feature, bins=bins)
        plt.title(f'Synthetic {feature} Distribution')
        
        plt.tight_layout()
        plt.show()
        
    def calculate_statistics(self, feature):
        """Calculate and compare basic statistics for a feature"""
        orig_stats = self.original_df[feature].describe()
        syn_stats = self.synthetic_df[feature].describe()
        
        stats_comparison = pd.DataFrame({
            'Original': orig_stats,
            'Synthetic': syn_stats,
            'Difference %': ((syn_stats - orig_stats) / orig_stats * 100).round(2)
        })
        
        return stats_comparison
    
    def correlation_comparison(self):
        """Compare correlation matrices between original and synthetic data"""
        orig_corr = self.original_df.corr()
        syn_corr = self.synthetic_df.corr()
        
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(orig_corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Original Data Correlations')
        
        plt.subplot(1, 2, 2)
        sns.heatmap(syn_corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Synthetic Data Correlations')
        
        plt.tight_layout()
        plt.show()
        
    def ks_test(self, feature):
        """Perform Kolmogorov-Smirnov test for a feature"""
        ks_statistic, p_value = stats.ks_2samp(
            self.original_df[feature],
            self.synthetic_df[feature]
        )
        return {
            'feature': feature,
            'ks_statistic': ks_statistic,
            'p_value': p_value
        }
    
    def validate_all_features(self):
        """Run validation tests for all common features"""
        common_features = set(self.original_df.columns) & set(self.synthetic_df.columns)
        
        validation_results = {}
        for feature in common_features:
            print(f"\nValidating feature: {feature}")
            
            # Distribution comparison
            self.compare_distributions(feature)
            
            # Statistical comparison
            stats_comp = self.calculate_statistics(feature)
            print("\nStatistical Comparison:")
            print(stats_comp)
            
            # KS test
            ks_results = self.ks_test(feature)
            print("\nKolmogorov-Smirnov Test Results:")
            print(f"KS statistic: {ks_results['ks_statistic']:.4f}")
            print(f"p-value: {ks_results['p_value']:.4f}")
            
            validation_results[feature] = {
                'statistics': stats_comp,
                'ks_test': ks_results
            }
        
        # Correlation comparison
        print("\nGenerating correlation comparison...")
        self.correlation_comparison()
        
        return validation_results

# Usage example
def main():
    validator = GANValidator(
        'network_traffic.csv',
        'synthetic_network_traffic.csv'
    )
    
    validation_results = validator.validate_all_features()
    
    # Save validation results
    with open('validation_report.txt', 'w') as f:
        f.write("GAN Validation Report\n")
        f.write("====================\n\n")
        
        for feature, results in validation_results.items():
            f.write(f"Feature: {feature}\n")
            f.write("-" * (len(feature) + 9) + "\n")
            
            f.write("\nStatistical Comparison:\n")
            f.write(results['statistics'].to_string())
            
            f.write("\n\nKS Test Results:\n")
            f.write(f"KS statistic: {results['ks_test']['ks_statistic']:.4f}\n")
            f.write(f"p-value: {results['ks_test']['p_value']:.4f}\n")
            f.write("\n" + "=" * 50 + "\n\n")

if __name__ == "__main__":
    main()