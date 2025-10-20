# %%
import numpy as np
import pandas as pd
import time
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegressionCV
from neuroHarmonize import harmonizationLearn, harmonizationApply

import matplotlib.pyplot as plt
from tqdm import tqdm
from prettyharmonize import PrettYharmonizeClassifier

import warnings

warnings.filterwarnings("ignore")


class SyntheticDataGenerator:
    """Generate synthetic data with controlled parameters"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)

    def generate_data(self, n_samples=500, n_features=100, n_classes=2, n_sites=4):
        """
        Generate balanced synthetic data across sites and classes
        """
        # Generate base classification data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, min(10, n_features // 10)),
            n_redundant=max(0, min(5, n_features // 20)),
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=self.random_state,
        )

        # Assign sites in a balanced way
        sites = np.zeros(n_samples, dtype=int)
        samples_per_site = n_samples // n_sites

        for site_id in range(n_sites):
            start_idx = site_id * samples_per_site
            end_idx = (
                (site_id + 1) * samples_per_site if site_id < n_sites - 1 else n_samples
            )
            sites[start_idx:end_idx] = site_id

        # Ensure class balance within each site
        for site_id in range(n_sites):
            site_indices = np.where(sites == site_id)[0]
            if len(site_indices) > 0:
                site_y = y[site_indices]
                unique_classes, class_counts = np.unique(site_y, return_counts=True)

                # If imbalance detected, resample
                if len(unique_classes) > 1 and np.std(class_counts) > 0.2 * np.mean(
                    class_counts
                ):
                    min_count = np.min(class_counts)
                    balanced_indices = []

                    for class_id in unique_classes:
                        class_indices = site_indices[y[site_indices] == class_id]
                        if len(class_indices) > min_count:
                            class_indices = np.random.choice(
                                class_indices, min_count, replace=False
                            )
                        balanced_indices.extend(class_indices)

                    # Update the site assignment
                    if len(balanced_indices) < len(site_indices):
                        # Need to add more samples to maintain site size
                        additional_needed = len(site_indices) - len(balanced_indices)
                        additional_indices = np.random.choice(
                            site_indices, additional_needed, replace=True
                        )
                        balanced_indices.extend(additional_indices)

                    # Reassign the site
                    sites[site_indices] = -1  # Mark for reassignment
                    sites[balanced_indices] = site_id

        return X, y, sites


class BenchmarkExperiment:
    """Run benchmarking experiments comparing models"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.data_generator = SyntheticDataGenerator(random_state)
        self.results = []

    def create_models(self):
        """Create the models to compare"""
        baseline_model = LogisticRegressionCV(
            penalty="l1",
            solver="liblinear",
            verbose=False,
            n_jobs=-1,
            random_state=self.random_state,
        )

        # Assuming PrettYharmonizeClassifier is implemented
        proposed_model = PrettYharmonizeClassifier(
            pred_model=LogisticRegressionCV(
                penalty="l1",
                solver="liblinear",
                verbose=False,
                n_jobs=-1,
                random_state=self.random_state,
            ),
            stack_model="logit",
            random_state=self.random_state,
            n_splits=5,
        )

        return baseline_model, proposed_model

    def run_experiment(self, X, y, sites, n_repeats=3):
        """Run timing experiment for both models"""
        baseline_model, proposed_model = self.create_models()

        # Benchmark baseline model
        baseline_times = []
        covars = pd.DataFrame(sites, columns=["SITE"])
        covars["Target"] = y.ravel()

        for _ in range(n_repeats):
            start_time = time.time()
            harm_model, _ = harmonizationLearn(X, covars)
            data_cheat = harmonizationApply(X, covars, harm_model)
            baseline_model.fit(data_cheat, y)
            end_time = time.time()
            baseline_times.append(end_time - start_time)

        # Benchmark proposed model
        proposed_times = []
        for _ in range(n_repeats):
            start_time = time.time()
            proposed_model.fit(X, y, sites=sites)
            end_time = time.time()
            proposed_times.append(end_time - start_time)

        return {
            "baseline_time": np.mean(baseline_times),
            "proposed_time": np.mean(proposed_times),
            "speedup": np.mean(baseline_times) / np.mean(proposed_times),
            "baseline_std": np.std(baseline_times),
            "proposed_std": np.std(proposed_times),
        }

    def vary_parameter(self, param_name, values, base_config):
        """Vary one parameter while keeping others constant"""
        results = []

        for value in tqdm(values, desc=f"Varying {param_name}"):
            config = base_config.copy()
            config[param_name] = value

            # Generate data
            X, y, sites = self.data_generator.generate_data(**config)

            # Run experiment
            try:
                result = self.run_experiment(X, y, sites)
                result.update(config)
                results.append(result)
            except Exception as e:
                print(f"Error with {param_name}={value}: {e}")
                continue

        return results

    def run_comprehensive_experiment(self):
        """Run the complete benchmarking experiment"""
        # Base configuration
        base_config = {
            "n_samples": 500,
            "n_features": 100,
            "n_classes": 2,
            "n_sites": 4,
        }

        # Parameter ranges
        n_samples_range = np.logspace(np.log10(20), np.log10(2000), 10).astype(int)
        n_features_range = np.logspace(np.log10(2), np.log10(1000), 10).astype(int)
        n_classes_range = np.linspace(2, 100, 10).astype(int)
        n_sites_range = np.linspace(2, 20, 10).astype(int)

        # Vary each parameter
        all_results = []

        # Vary number of samples
        print("Varying number of samples...")
        results_n = self.vary_parameter("n_samples", n_samples_range, base_config)
        all_results.extend(results_n)

        # Vary number of features
        print("Varying number of features...")
        results_f = self.vary_parameter("n_features", n_features_range, base_config)
        all_results.extend(results_f)

        # Vary number of classes
        print("Varying number of classes...")
        results_c = self.vary_parameter("n_classes", n_classes_range, base_config)
        all_results.extend(results_c)

        # Vary number of sites
        print("Varying number of sites...")
        results_s = self.vary_parameter("n_sites", n_sites_range, base_config)
        all_results.extend(results_s)

        self.results = all_results
        return all_results

    def plot_results(self):
        """Plot the benchmarking results"""
        if not self.results:
            print("No results to plot. Run experiment first.")
            return

        df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        # Plot for varying samples
        sample_data = df[
            df["n_samples"].isin(
                np.logspace(np.log10(20), np.log10(2000), 10).astype(int)
            )
        ]
        if not sample_data.empty:
            axes[0].plot(
                sample_data["n_samples"],
                sample_data["baseline_time"],
                "o-",
                label="Baseline",
            )
            axes[0].plot(
                sample_data["n_samples"],
                sample_data["proposed_time"],
                "s-",
                label="Proposed",
            )
            axes[0].set_xscale("log")
            axes[0].set_yscale("log")
            axes[0].set_xlabel("Number of Samples")
            axes[0].set_ylabel("Time (seconds)")
            axes[0].set_title("Computational Time vs Number of Samples")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        # Plot for varying features
        feature_data = df[
            df["n_features"].isin(
                np.logspace(np.log10(2), np.log10(2000), 10).astype(int)
            )
        ]
        if not feature_data.empty:
            axes[1].plot(
                feature_data["n_features"],
                feature_data["baseline_time"],
                "o-",
                label="Baseline",
            )
            axes[1].plot(
                feature_data["n_features"],
                feature_data["proposed_time"],
                "s-",
                label="Proposed",
            )
            axes[1].set_xscale("log")
            axes[1].set_yscale("log")
            axes[1].set_xlabel("Number of Features")
            axes[1].set_ylabel("Time (seconds)")
            axes[1].set_title("Computational Time vs Number of Features")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        # Plot for varying classes
        class_data = df[df["n_classes"].isin(np.linspace(2, 100, 10).astype(int))]
        if not class_data.empty:
            axes[2].plot(
                class_data["n_classes"],
                class_data["baseline_time"],
                "o-",
                label="Baseline",
            )
            axes[2].plot(
                class_data["n_classes"],
                class_data["proposed_time"],
                "s-",
                label="Proposed",
            )
            axes[2].set_yscale("log")
            axes[2].set_xlabel("Number of Classes")
            axes[2].set_ylabel("Time (seconds)")
            axes[2].set_title("Computational Time vs Number of Classes")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        # Plot for varying sites
        site_data = df[df["n_sites"].isin(np.linspace(2, 20, 10).astype(int))]
        if not site_data.empty:
            axes[3].plot(
                site_data["n_sites"], site_data["baseline_time"], "o-", label="Baseline"
            )
            axes[3].plot(
                site_data["n_sites"], site_data["proposed_time"], "s-", label="Proposed"
            )
            axes[3].set_yscale("log")
            axes[3].set_xlabel("Number of Sites")
            axes[3].set_ylabel("Time (seconds)")
            axes[3].set_title("Computational Time vs Number of Sites")
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Plot speedup factor
        plt.figure(figsize=(10, 6))
        for param in ["n_samples", "n_features", "n_classes", "n_sites"]:
            param_data = df.dropna(subset=[param, "speedup"])
            if not param_data.empty:
                plt.plot(param_data[param], param_data["speedup"], "o-", label=param)

        plt.xlabel("Parameter Value")
        plt.ylabel("Speedup Factor (Baseline/Proposed)")
        plt.title("Speedup Factor Across Different Parameters")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize experiment
    experiment = BenchmarkExperiment(random_state=42)

    # Run comprehensive experiment
    results = experiment.run_comprehensive_experiment()

    # Plot results
    experiment.plot_results()

    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv("benchmark_results.csv", index=False)
    print("Results saved to benchmark_results.csv")

    # Print summary
    print("\n=== Benchmark Summary ===")
    print(f"Total experiments: {len(results)}")
    print(f"Average speedup: {df_results['speedup'].mean():.2f}x")
    print(f"Maximum speedup: {df_results['speedup'].max():.2f}x")
    print(f"Minimum speedup: {df_results['speedup'].min():.2f}x")
# %%
