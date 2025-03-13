"""
Comprehensive Measurement System Analysis Implementation

Implements five different measurement methods:
1. Discriminability (D_hat) - Equation 3.2
2. Fingerprint Index (F_index) - Equation 2.4
3. I2C2 (Image Intraclass Correlation Coefficient)
4. Rank Sum Statistic - Equation 2.5
5. ICC (Intraclass Correlation Coefficient)
6. CCDM (Correlation Coefficient Deviation Metric)
7. Components of Variation Visualization
8. X-bar Charts

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
from math import sqrt
import io
import sys

warnings.filterwarnings('ignore')

class ComprehensiveMSA:
    def __init__(self):
        """Initialize comprehensive measurement system analysis"""
        self.results = {}
        self.gaussian_assumption = True  # Add this flag
        
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
        Memory-optimized version with smaller batches and better cleanup
        """
        measurements = data['measurements']
        n_parts = data['n_parts']
        n_variations = data['n_variations']
        
        discriminability_sum = 0
        total_comparisons = 0
        
        # Further reduce batch size for memory efficiency
        batch_size = 25  # Reduced from 50
        n_batches = (n_parts + batch_size - 1) // batch_size
        
        # Use a single line with counter instead of tqdm with multiple lines
        print(f"Calculating discriminability...", end="", flush=True)
        completed = 0
        
        try:
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_parts)
                batch_size_actual = end_idx - start_idx
                
                # Process each part in the batch individually to reduce memory usage
                for i in range(batch_size_actual):
                    current_part_idx = start_idx + i
                    
                    # Calculate within-distances for current part
                    for t1 in range(n_variations):
                        for t2 in range(t1 + 1, n_variations):
                            within_dist = abs(
                                measurements[current_part_idx, t1] - 
                                measurements[current_part_idx, t2]
                            )
                            
                            # Compare with other parts immediately
                            for j in range(n_parts):
                                if j != current_part_idx:
                                    between_dist = abs(
                                        measurements[current_part_idx, t1] - 
                                        measurements[j, t2]
                                    )
                                    if within_dist < between_dist:
                                        discriminability_sum += 1
                                    total_comparisons += 1
                
                # Force garbage collection after each batch
                import gc
                gc.collect()
                
                # Update progress counter
                completed += batch_size_actual
                print(f"\rCalculating discriminability... {completed}/{n_parts} parts processed ({int(100*completed/n_parts)}%)", end="", flush=True)
            
            print()  # Final newline after completion
            
        except Exception as e:
            print(f"\nError during discriminability calculation: {str(e)}")
            raise
        finally:
            # Ensure proper cleanup
            gc.collect()
        
        D_hat = discriminability_sum / total_comparisons if total_comparisons > 0 else 0
        return D_hat

    def calculate_fingerprint_index(self, data):
        """
        Calculate fingerprint index following Equation 2.3 and 2.4 from the paper
        F_index = P(δ_i,1,2 < δ_i,i',1,2; ∀ i' ≠ i)
        F̂_index = Tn/n where Tn is number of correct matches
        """
        measurements = data['measurements']
        n_parts = data['n_parts']
        
        # We'll use first two measurements for each subject as per paper definition
        correct_matches = 0
        
        for i in range(n_parts):
            # Calculate within-subject distance (δ_i,1,2)
            within_dist = abs(measurements[i, 0] - measurements[i, 1])
            
            # Check against all other subjects
            is_match = True
            for i_prime in range(n_parts):
                if i_prime != i:
                    # Calculate between-subject distance (δ_i,i',1,2)
                    between_dist = abs(measurements[i, 0] - measurements[i_prime, 1])
                    
                    # If any between-distance is smaller or equal, this is not a match
                    if within_dist >= between_dist:
                        is_match = False
                        break
            
            if is_match:
                correct_matches += 1
        
        # F̂_index = Tn/n
        F_index = correct_matches / n_parts
        return F_index

    def calculate_i2c2(self, data):
        measurements = data['measurements']
        n_parts = data['n_parts']
        n_variations = data['n_variations']
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        
        # Calculate grand mean
        grand_mean = np.mean(measurements)
        
        # Calculate between-subject covariance (Σ_μ)
        subject_means = np.mean(measurements, axis=1)
        between_ss = np.sum((subject_means - grand_mean)**2) * n_variations
        tr_between = between_ss / (n_parts - 1 + epsilon)  # Convert to variance
        
        # Calculate within-subject covariance (Σ)
        within_ss = np.sum((measurements - subject_means[:, np.newaxis])**2)
        tr_within = within_ss / (n_parts * (n_variations - 1) + epsilon)  # Convert to variance
        
        # Calculate I2C2 using paper formula with epsilon
        i2c2 = (tr_between + epsilon) / (tr_between + tr_within + 2*epsilon)
        
        return max(0, min(1, i2c2))  # Bound between 0 and 1

    def calculate_rank_sum(self, data):
        """
        Calculate rank sum following equation 2.5
        R_n = sum of ranks of within-subject distances
        Transformed to scale [0,1] where higher is better
        """
        measurements = data['measurements']
        n_parts = data['n_parts']
        
        within_distances = []
        between_distances = []
        
        # Calculate within and between distances
        for i in range(n_parts):
            # Within-subject distances
            within_dist = abs(measurements[i, 0] - measurements[i, 1])
            within_distances.append((within_dist, i))
            
            # Between-subject distances
            for j in range(n_parts):
                if i != j:
                    between_dist = abs(measurements[i, 0] - measurements[j, 1])
                    between_distances.append((between_dist, (i, j)))
        
        # Combine all distances and sort
        all_distances = within_distances + between_distances
        all_distances.sort(key=lambda x: x[0])
        
        # Calculate ranks for within-subject distances
        rank_sum = 0
        n_total = len(all_distances)
        
        for rank, (dist, idx) in enumerate(all_distances, 1):
            if isinstance(idx, int):  # This is a within-subject distance
                rank_sum += rank
        
        # Transform to [0,1] scale where higher values indicate better repeatability
        max_rank_sum = n_parts * n_total
        min_rank_sum = n_parts * (n_parts + 1) / 2
        
        normalized_rank = 1 - ((rank_sum - min_rank_sum) / 
                              (max_rank_sum - min_rank_sum))
        
        return normalized_rank

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

    def calculate_gage_rnr(self, data):
        """
        Calculate Gage R&R statistics using actual measurements from dataset
        with improved scaling and robustness
        """
        measurements = data['measurements']
        n_parts = data['n_parts']
        n_variations = data['n_variations']
        
        # Calculate Repeatability (EV) using range method
        ranges = np.ptp(measurements, axis=1)  # Range for each part
        EV = np.mean(ranges) / 2.0  # Using d2* = 2 for typical range normalization
        
        # Calculate Reproducibility (AV) using operator/variation differences
        variation_means = np.mean(measurements, axis=0)
        operator_range = np.ptp(variation_means)
        AV = operator_range / (n_variations ** 0.5)  # Scale by sqrt of variations
        
        # Calculate Gage R&R
        GRR = np.sqrt(EV**2 + AV**2)
        
        # Calculate Part-to-Part Variation (PV)
        part_means = np.mean(measurements, axis=1)
        part_range = np.ptp(part_means)
        PV = part_range / (n_parts ** 0.5)  # Scale by sqrt of parts
        
        # Calculate Total Variation (TV)
        TV = np.sqrt(GRR**2 + PV**2)
        
        # Ensure non-zero total variation to avoid division by zero
        if TV < 1e-10:
            TV = 1e-10
        
        # Calculate Percentage Contributions
        EV_percent = 100 * (EV / TV)
        AV_percent = 100 * (AV / TV)
        GRR_percent = 100 * (GRR / TV)
        PV_percent = 100 * (PV / TV)
        
        # Calculate Number of Distinct Categories (ndc)
        # Using more conservative formula
        ndc = int(np.floor(sqrt(2) * PV / GRR)) if GRR > 0 else float('inf')
        
        return {
            'EV': EV,
            'AV': AV,
            'GRR': GRR,
            'PV': PV,
            'TV': TV,
            'EV_percent': EV_percent,
            'AV_percent': AV_percent,
            'GRR_percent': GRR_percent,
            'PV_percent': PV_percent,
            'ndc': ndc
        }

    def calculate_ccdm(self, data):
        """
        Calculate Correlation Coefficient Deviation Metric (CCDM)
        Lower values indicate better measurement system correlation
        """
        measurements = data['measurements']
        n_variations = data['n_variations']
        
        total_ccdm = 0
        pairs = 0
        
        # Use first measurement as reference
        reference = measurements[:, 0]
        sigma_ref = np.std(reference)
        
        for rep in range(1, n_variations):
            current = measurements[:, rep]
            sigma_curr = np.std(current)
            
            # Skip if either standard deviation is zero (avoid division by zero)
            if sigma_ref == 0 or sigma_curr == 0:
                continue
                
            # Calculate covariance between reference and current measurement
            covariance = np.cov(reference, current, ddof=0)[0, 1]
            
            # CCDM is 1 minus the correlation coefficient
            ccdm = 1 - (covariance / (sigma_ref * sigma_curr))
            total_ccdm += ccdm
            pairs += 1
            
        # Return average CCDM across all pairs
        return total_ccdm / pairs if pairs > 0 else 0

    def calculate_components_of_variation(self, data):
        """
        Calculate variance components using ANOVA methodology
        Returns total variation and breakdown between/within subjects
        """
        measurements = data['measurements']
        n_parts = data['n_parts']
        n_variations = data['n_variations']
        
        # Calculate total variation
        total_ss = np.sum((measurements - np.mean(measurements))**2)
        
        # Calculate between-frequency variation
        part_means = np.mean(measurements, axis=1)
        between_ss = np.sum((part_means - np.mean(measurements))**2) * n_variations
        
        # Calculate within-frequency variation (residual)
        within_ss = total_ss - between_ss
        
        # Return components and percentages
        return {
            'total_variation': total_ss,
            'between_parts_variation': between_ss,
            'within_parts_variation': within_ss,
            'percent_between': (between_ss / total_ss) * 100 if total_ss > 0 else 0,
            'percent_within': (within_ss / total_ss) * 100 if total_ss > 0 else 0
        }

    def generate_xbar_chart(self, data, save_prefix='analysis'):
        """
        Generate X-bar control chart for measurement means
        Shows mean values with control limits at ±3 standard deviations
        """
        measurements = data['measurements']
        n_variations = data['n_variations']
        file_name = data['file_name']
        
        # Calculate means for each variation
        means = np.mean(measurements, axis=0)
        overall_mean = np.mean(means)
        std_dev = np.std(means, ddof=1)
        
        # Create X-bar chart
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, n_variations+1), means, 'bo-', label='Repetition Means')
        plt.axhline(overall_mean, color='r', linestyle='--', label='Overall Mean')
        plt.axhline(overall_mean + 3*std_dev, color='g', linestyle=':', label='UCL')
        plt.axhline(overall_mean - 3*std_dev, color='g', linestyle=':', label='LCL')
        
        plt.title(f'X-bar Chart - {file_name}')
        plt.xlabel('Repetition Number')
        plt.ylabel('Mean Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_prefix}_{file_name}_xbar.png')
        plt.close()
        
        return {'mean': overall_mean, 'ucl': overall_mean + 3*std_dev, 'lcl': overall_mean - 3*std_dev}

    def visualize_components(self, data, save_prefix='analysis'):
        """
        Visualize components of variation as a pie chart
        Shows breakdown between part-to-part and within-part variation
        """
        file_name = data['file_name']
        components = self.calculate_components_of_variation(data)
        
        # Create labels and sizes for pie chart
        labels = ['Between Parts', 'Within Parts']
        sizes = [components['percent_between'], components['percent_within']]
        
        # Create pie chart
        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                colors=['#ff9999','#66b3ff'])
        plt.axis('equal')  # Equal aspect ratio ensures the pie is circular
        plt.title(f'Components of Variation - {file_name}')
        plt.savefig(f'{save_prefix}_{file_name}_components.png')
        plt.close()
        
        return components

    def visualize_results(self, save_prefix='analysis'):
        """Create comprehensive visualizations following paper examples"""
        # 1. Method Comparison Plot
        plt.figure(figsize=(15, 10))
        methods = ['Discriminability', 'Fingerprint', 'I2C2', 'Rank Sum', 'ICC', 'CCDM']
        files = list(self.results.keys())
        
        values = {
            'Discriminability': [self.results[f]['discriminability'] for f in files],
            'Fingerprint': [self.results[f]['fingerprint_index'] for f in files],
            'I2C2': [self.results[f]['i2c2'] for f in files],
            'Rank Sum': [self.results[f]['rank_sum'] for f in files],
            'ICC': [self.results[f]['icc'] for f in files],
            'CCDM': [1 - self.results[f]['ccdm'] for f in files]  # Invert CCDM for consistency (higher = better)
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
        
        # 5. Components of Variation Summary
        plt.figure(figsize=(14, 8))
        between_parts = [self.results[f]['percent_between'] for f in files]
        within_parts = [self.results[f]['percent_within'] for f in files]
        
        # Create stacked bar chart
        plt.bar(files, between_parts, label='Between Parts')
        plt.bar(files, within_parts, bottom=between_parts, label='Within Parts')
        
        plt.xlabel('Files')
        plt.ylabel('Percentage of Total Variation')
        plt.title('Components of Variation Across Files')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_variance_components.png')
        plt.close()
        
        # 6. Create variance components summary table
        components_df = pd.DataFrame({
            'file': files,
            'between_parts_%': [self.results[f]['percent_between'] for f in files],
            'within_parts_%': [self.results[f]['percent_within'] for f in files],
            'GRR_%': [self.results[f]['GRR_percent'] for f in files]
        })
        
        components_df.to_csv('variance_components_table.csv', index=False)

    def analyze_measurement_system(self, filepath):
        """Perform comprehensive measurement system analysis"""
        try:
            data = self.load_data(filepath)
            
            print("\nCalculating all metrics...")
            
            # Calculate traditional metrics
            results = {
                'discriminability': self.calculate_discriminability(data),
                'fingerprint_index': self.calculate_fingerprint_index(data),
                'i2c2': self.calculate_i2c2(data),
                'rank_sum': self.calculate_rank_sum(data),
                'icc': self.calculate_icc(data),
                'n_frequencies': data['n_parts'],
                'n_variations': data['n_variations']
            }
            
            # Add Gage R&R results
            gage_results = self.calculate_gage_rnr(data)
            results.update(gage_results)
            
            # Add new metrics
            results['ccdm'] = self.calculate_ccdm(data)
            
            # Add components of variation
            components = self.calculate_components_of_variation(data)
            results.update(components)
            
            # Generate X-bar chart
            self.generate_xbar_chart(data)
            
            # Visualize components of variation
            self.visualize_components(data)
            
            self.results[data['file_name']] = results
            
            print(f"\nResults for {data['file_name']}:")
            print(f"Discriminability (D_hat): {results['discriminability']:.4f}")
            print(f"Fingerprint Index: {results['fingerprint_index']:.4f}")
            print(f"I2C2: {results['i2c2']:.4f}")
            print(f"Rank Sum: {results['rank_sum']:.4f}")
            print(f"ICC: {results['icc']:.4f}")
            print(f"ICC Interpretation: {self.get_icc_interpretation(results['icc'])}")
            print(f"CCDM: {results['ccdm']:.4f}")
            print("\nComponents of Variation:")
            print(f"Between-parts: {results['percent_between']:.1f}%")
            print(f"Within-parts: {results['percent_within']:.1f}%")
            print("\nGage R&R Results:")
            print(f"Repeatability (EV): {results['EV']:.6f} ({results['EV_percent']:.1f}%)")
            print(f"Reproducibility (AV): {results['AV']:.6f} ({results['AV_percent']:.1f}%)")
            print(f"Gage R&R: {results['GRR']:.6f} ({results['GRR_percent']:.1f}%)")
            print(f"Part-to-Part Variation: {results['PV']:.6f} ({results['PV_percent']:.1f}%)")
            print(f"Number of Distinct Categories: {results['ndc']:.1f}")
            
            return results
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise

    def __del__(self):
        """Cleanup method to ensure proper resource handling"""
        try:
            # Clear any stored results
            self.results.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception:
            pass

    def simulate_power_curves(self, n_range=(5, 40), n_iterations=1000, sigma_sq=5, sigma_mu_sq=3):
        """
        Simulate power curves for different measurement methods
        
        Parameters:
        - n_range: tuple of (min_subjects, max_subjects)
        - n_iterations: number of simulation iterations
        - sigma_sq: within-subject variance
        - sigma_mu_sq: between-subject variance
        """
        n_subjects = np.arange(n_range[0], n_range[1] + 1, 5)
        n_variations = 2  # Fixed number of variations per subject
        
        # Initialize power results dictionary
        power_results = {
            'Discriminability': np.zeros(len(n_subjects)),
            'Rank Sums': np.zeros(len(n_subjects)),
            'Fingerprint': np.zeros(len(n_subjects)),
            'I2C2': np.zeros(len(n_subjects)),  # Added I2C2
            'ICC (F-test)': np.zeros(len(n_subjects)),
            'ICC (permutation)': np.zeros(len(n_subjects)),
            'CCDM': np.zeros(len(n_subjects))  # Added CCDM
        }
        
        for i, n in enumerate(tqdm(n_subjects, desc="Simulating power curves")):
            significant_counts = {method: 0 for method in power_results.keys()}
            
            for _ in range(n_iterations):
                # Generate true subject effects
                true_effects = np.random.normal(0, np.sqrt(sigma_mu_sq), n)
                
                # Generate measurements with noise
                measurements = np.zeros((n, n_variations))
                for j in range(n):
                    # Add subject effect and measurement noise
                    measurements[j] = true_effects[j] + np.random.normal(0, np.sqrt(sigma_sq), n_variations)
                
                # For non-Gaussian case (right plot)
                if not self.gaussian_assumption:
                    measurements = np.exp(measurements)  # Log transformation
                
                # Create data dictionary
                sim_data = {
                    'measurements': measurements,
                    'n_parts': n,
                    'n_variations': n_variations
                }
                
                # Calculate test statistics - redirect stdout to suppress output during discriminability calculation
                original_stdout = sys.stdout
                sys.stdout = io.StringIO()  # Redirect to dummy stream
                try:
                    d_hat = self.calculate_discriminability(sim_data)
                finally:
                    sys.stdout = original_stdout  # Restore stdout
                
                # Calculate other metrics normally
                rank_sum = self.calculate_rank_sum(sim_data)
                f_index = self.calculate_fingerprint_index(sim_data)
                i2c2 = self.calculate_i2c2(sim_data)  # Added I2C2 calculation
                icc = self.calculate_icc(sim_data)
                ccdm = self.calculate_ccdm(sim_data)  # Added CCDM calculation
                
                # Perform significance tests (α = 0.05)
                significant_counts['Discriminability'] += (d_hat > 0.5)
                significant_counts['Rank Sums'] += (rank_sum > 0.5)
                significant_counts['Fingerprint'] += (f_index > 1/n)
                significant_counts['I2C2'] += (i2c2 > 0.5)  # Added I2C2 threshold
                significant_counts['CCDM'] += (ccdm < 0.5)  # Added CCDM threshold (lower is better)
                
                # F-test for ICC
                f_stat = (1 + (n_variations-1)*icc)/(1-icc)
                f_crit = stats.f.ppf(0.95, n-1, n*(n_variations-1))
                significant_counts['ICC (F-test)'] += (f_stat > f_crit)
                
                # Permutation test for ICC
                n_perms = 100
                perm_iccs = []
                for _ in range(n_perms):
                    perm_measurements = np.random.permutation(measurements.flatten()).reshape(n, n_variations)
                    perm_data = {**sim_data, 'measurements': perm_measurements}
                    perm_iccs.append(self.calculate_icc(perm_data))
                significant_counts['ICC (permutation)'] += (icc > np.percentile(perm_iccs, 95))
            
            # Calculate power for each method
            for method in power_results:
                power_results[method][i] = significant_counts[method] / n_iterations
        
        # Plot power curves
        plt.figure(figsize=(15, 6))
        
        # Create two subplots for Gaussian and non-Gaussian cases
        for idx, assumption in enumerate(['Gaussian', 'Non-Gaussian']):
            plt.subplot(1, 2, idx+1)
            
            for method, power in power_results.items():
                if 'ICC' in method:
                    plt.plot(n_subjects, power, '--', label=method)
                else:
                    plt.plot(n_subjects, power, '-', label=method)
            
            plt.xlabel('Number of Subjects')
            plt.ylabel('Power')
            plt.title(f'Power Analysis ({assumption} Assumption)')
            plt.grid(True)
            if idx == 1:
                plt.legend(title='Type of Test', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('power_analysis.png', bbox_inches='tight', dpi=300)
        plt.close()

def main():
    analyzer = ComprehensiveMSA()
    
    # Regular analysis
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
    
    # Create visualizations
    analyzer.visualize_results()
    
    # Generate power curves for both Gaussian and non-Gaussian cases
    print("\nGenerating power analysis curves...")
    print("Using subject range from actual data...")
    
    analyzer.gaussian_assumption = True
    analyzer.simulate_power_curves()
    
    analyzer.gaussian_assumption = False
    analyzer.simulate_power_curves()
    
    print("\nAnalysis complete. Visualizations saved.")

if __name__ == "__main__":
    main()