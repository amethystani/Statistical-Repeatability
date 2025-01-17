"""
Comprehensive Measurement System Analysis Implementation

Implements five different measurement methods:
1. Discriminability (D_hat) - Equation 3.2
2. Fingerprint Index (F_index) - Equation 2.4
3. I2C2 (Image Intraclass Correlation Coefficient)
4. Rank Sum Statistic - Equation 2.5
5. ICC (Intraclass Correlation Coefficient)

Following methodology from the paper:
'Statistical Analysis of Data Repeatability Measures'

Author: [Your Name]
Date: December 2024
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

class ComprehensiveMSA:
    def __init__(self):
        """Initialize comprehensive measurement system analysis"""
        self.results = {}
        
    def load_data(self, filepath):
        """
        Load data from Excel file with validation
        First column is frequency, remaining columns are measurements
        """
        print(f"\nLoading data from {filepath}")
        df = pd.read_excel(filepath)
        print("Raw data shape:", df.shape)
        
        # Split frequency and measurements
        frequency = df.iloc[:, 0].values
        measurements = df.iloc[:, 1:].values
        
        n_parts = measurements.shape[0]
        n_variations = measurements.shape[1]
        
        print(f"Number of frequencies: {n_parts}")
        print(f"Number of variations per frequency: {n_variations}")
        
        return {
            'frequency': frequency,
            'measurements': measurements,
            'n_parts': n_parts,
            'n_variations': n_variations,
            'file_name': Path(filepath).stem
        }

    def calculate_discriminability(self, data):
        """
        Calculate discriminability using equation 3.2 from paper
        Uses batch processing for memory efficiency
        """
        measurements = data['measurements']
        n_parts = data['n_parts']
        n_variations = data['n_variations']
        
        discriminability_sum = 0
        total_comparisons = 0
        
        batch_size = 100
        n_batches = (n_parts + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(n_batches), desc="Calculating discriminability", unit="batch"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_parts)
            
            for i in range(start_idx, end_idx):
                for t1 in range(n_variations):
                    for t2 in range(t1 + 1, n_variations):
                        within_dist = abs(measurements[i, t1] - measurements[i, t2])
                        
                        for j in range(n_parts):
                            if i != j:
                                between_dist = abs(measurements[i, t1] - measurements[j, t2])
                                if within_dist < between_dist:
                                    discriminability_sum += 1
                                total_comparisons += 1
        
        D_hat = discriminability_sum / total_comparisons if total_comparisons > 0 else 0
        return D_hat

    def calculate_fingerprint_index(self, data):
        """
        Calculate fingerprint index using average distance comparison
        More robust implementation that better aligns with discriminability
        """
        measurements = data['measurements']
        n_parts = data['n_parts']
        n_variations = data['n_variations']
        
        correct_matches = 0
        total_comparisons = 0
        
        for i in range(n_parts):
            for t1 in range(n_variations):
                for t2 in range(n_variations):
                    if t1 != t2:
                        within_dist = abs(measurements[i, t1] - measurements[i, t2])
                        
                        # Calculate average between-distance for this comparison
                        between_distances = []
                        for j in range(n_parts):
                            if i != j:
                                between_dist = abs(measurements[i, t1] - measurements[j, t2])
                                between_distances.append(between_dist)
                        
                        # Compare within-distance to average between-distance
                        avg_between_dist = np.mean(between_distances)
                        if within_dist < avg_between_dist:
                            correct_matches += 1
                        total_comparisons += 1
        
        F_index = correct_matches / total_comparisons if total_comparisons > 0 else 0
        return F_index

    def calculate_i2c2(self, data):
        """Calculate I2C2 using trace-based method"""
        measurements = data['measurements']
        
        total_cov = np.cov(measurements.T)
        
        within_cov = np.zeros_like(total_cov)
        for i in range(data['n_parts']):
            within_cov += np.cov(measurements[i, :].reshape(-1, 1))
        within_cov /= data['n_parts']
        
        tr_total = np.trace(total_cov)
        tr_within = np.trace(within_cov)
        
        i2c2 = (tr_total - tr_within) / tr_total
        return i2c2

    def calculate_rank_sum(self, data):
        """Calculate rank sum following equation 2.5"""
        measurements = data['measurements']
        n_parts = data['n_parts']
        
        distances = squareform(pdist(measurements))
        rank_sum = 0
        total_pairs = 0
        
        for i in tqdm(range(n_parts), desc="Calculating rank sum", unit="freq"):
            for j in range(n_parts):
                if i != j:
                    rank = np.where(np.argsort(distances[i]) == j)[0][0] + 1
                    rank_sum += rank
                    total_pairs += 1
        
        # Transform to match scale of other metrics
        rank_metric = 1 - (rank_sum / (total_pairs * n_parts))
        return rank_metric

    def calculate_icc(self, data):
        """Calculate ICC using one-way ANOVA model"""
        measurements = data['measurements']
        n_parts = data['n_parts']
        n_variations = data['n_variations']
        
        grand_mean = np.mean(measurements)
        between_subject_ss = n_variations * np.sum((np.mean(measurements, axis=1) - grand_mean) ** 2)
        within_subject_ss = np.sum((measurements - np.mean(measurements, axis=1).reshape(-1, 1)) ** 2)
        
        between_ms = between_subject_ss / (n_parts - 1)
        within_ms = within_subject_ss / (n_parts * (n_variations - 1))
        
        icc = (between_ms - within_ms) / (between_ms + (n_variations - 1) * within_ms)
        return icc

    def get_icc_interpretation(self, icc_value):
        """
        Get interpretation of ICC value based on guidelines
        From Table 4.4 (Koo and Li, 2016)
        """
        if icc_value < 0.50:
            return "Poor"
        elif icc_value < 0.75:
            return "Moderate"
        elif icc_value < 0.90:
            return "Good"
        else:
            return "Excellent"

    def visualize_results(self, save_prefix='analysis'):
        """Create comprehensive visualizations following paper examples"""
        # 1. Method Comparison Plot
        plt.figure(figsize=(15, 10))
        methods = ['Discriminability', 'Fingerprint', 'I2C2', 'Rank Sum', 'ICC']
        files = list(self.results.keys())
        
        values = {
            'Discriminability': [self.results[f]['discriminability'] for f in files],
            'Fingerprint': [self.results[f]['fingerprint_index'] for f in files],
            'I2C2': [self.results[f]['i2c2'] for f in files],
            'Rank Sum': [self.results[f]['rank_sum'] for f in files],
            'ICC': [self.results[f]['icc'] for f in files]
        }
        
        # Create DataFrame for boxplot
        plt.subplot(2, 1, 1)
        data_df = pd.DataFrame(values)
        box_data = [data_df[method] for method in methods]
        plt.boxplot(box_data, labels=methods)
        plt.ylabel('Score')
        plt.title('Distribution of Measurement Method Scores')
        
        plt.subplot(2, 1, 2)
        x = np.arange(len(files))
        bar_width = 0.15
        for i, method in enumerate(methods):
            plt.bar(x + i*bar_width, values[method], bar_width, label=method)
        
        plt.xlabel('Files')
        plt.ylabel('Score')
        plt.title('Comparison Across Files')
        plt.xticks(x + bar_width*2, files, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_comparison.png')
        plt.close()
        
        # 2. ICC vs Discriminability Relationship
        plt.figure(figsize=(10, 8))
        icc_values = [self.results[f]['icc'] for f in files]
        disc_values = [self.results[f]['discriminability'] for f in files]
        
        # Plot theoretical relationship
        x = np.linspace(0, 1, 100)
        y = 0.5 + (1/np.pi) * np.arctan(x/np.sqrt((1-x)*(x+3)))
        plt.plot(x, y, 'k--', label='Theoretical')
        
        # Plot actual values
        plt.scatter(icc_values, disc_values, color='red', label='Observed')
        
        plt.xlabel('ICC')
        plt.ylabel('Discriminability')
        plt.title('ICC vs Discriminability Relationship')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_prefix}_icc_disc.png')
        plt.close()
        
        # 3. ICC Interpretation Plot
        plt.figure(figsize=(12, 6))
        colors = ['red' if v < 0.5 else 'yellow' if v < 0.75 
                 else 'lightgreen' if v < 0.9 else 'darkgreen' for v in icc_values]
        
        plt.barh(files, icc_values, color=colors)
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Poor (<0.50)')
        plt.axvline(x=0.75, color='yellow', linestyle='--', alpha=0.5, label='Moderate (<0.75)')
        plt.axvline(x=0.90, color='green', linestyle='--', alpha=0.5, label='Good (<0.90)')
        
        plt.xlabel('ICC Value')
        plt.title('ICC Values with Reliability Interpretation')
        plt.legend(title='Guidelines', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_icc_interpret.png')
        plt.close()
        
        # 4. Method Performance Plot
        plt.figure(figsize=(12, 6))
        file_indices = np.arange(1, len(files) + 1)
        
        for method in methods:
            plt.plot(file_indices, values[method], marker='o', label=method)
            
        plt.xlabel('File Index')
        plt.ylabel('Score')
        plt.title('Method Performance Comparison')
        plt.legend()
        plt.grid(True)
        plt.xticks(file_indices)
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_performance.png')
        plt.close()

    def analyze_measurement_system(self, filepath):
        """Perform comprehensive measurement system analysis"""
        try:
            data = self.load_data(filepath)
            
            print("\nCalculating all metrics...")
            results = {
                'discriminability': self.calculate_discriminability(data),
                'fingerprint_index': self.calculate_fingerprint_index(data),
                'i2c2': self.calculate_i2c2(data),
                'rank_sum': self.calculate_rank_sum(data),
                'icc': self.calculate_icc(data),
                'n_frequencies': data['n_parts'],
                'n_variations': data['n_variations']
            }
            
            self.results[data['file_name']] = results
            
            print(f"\nResults for {data['file_name']}:")
            print(f"Discriminability (D_hat): {results['discriminability']:.4f}")
            print(f"Fingerprint Index: {results['fingerprint_index']:.4f}")
            print(f"I2C2: {results['i2c2']:.4f}")
            print(f"Rank Sum: {results['rank_sum']:.4f}")
            print(f"ICC: {results['icc']:.4f}")
            print(f"ICC Interpretation: {self.get_icc_interpretation(results['icc'])}")
            
            return results
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise

def main():
    analyzer = ComprehensiveMSA()
    
    files = [
        'Project Details (1).xlsx',
        'Project Details (2).xlsx',
        'Project Details (3).xlsx',
        'Project Details.xlsx',
        'Project Surface Bonded.xlsx'
    ]
    
    print("Starting comprehensive measurement system analysis...")
    
    for file in tqdm(files, desc="Processing files"):
        try:
            analyzer.analyze_measurement_system(file)
        except Exception as e:
            print(f"\nFailed to process {file}")
            print(f"Error: {str(e)}")
            continue
    
    # Create visualization
    analyzer.visualize_results()
    print("\nAnalysis complete. Visualization saved as 'method_comparison.png'")

if __name__ == "__main__":
    main()