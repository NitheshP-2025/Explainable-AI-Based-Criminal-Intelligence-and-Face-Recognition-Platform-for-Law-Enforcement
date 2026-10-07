# src/unsupervised/tsne_visualization.py
"""
t-SNE Visualization Module
Unit IV Concept: t-SNE
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE as SklearnTSNE
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class TSNEVisualization:
    """
    t-SNE Implementation for Visualization
    
    Implements:
    1. t-SNE for High-Dimensional Data Visualization
    2. 2D and 3D Visualization
    3. Perplexity Tuning
    """
    
    def __init__(self, n_components=2, perplexity=30, learning_rate=200, random_state=42):
        """
        Initialize t-SNE
        
        Args:
            n_components: Number of components (2 or 3)
            perplexity: Perplexity parameter (5-50)
            learning_rate: Learning rate
            random_state: Random seed
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.embedding = None
        self.scaler = StandardScaler()
        
        logger.info(f"TSNEVisualization initialized with {n_components} components")
    
    def fit_transform(self, X):
        """
        Fit t-SNE and transform data
        
        Args:
            X: Feature matrix
            
        Returns:
            t-SNE embedding
        """
        logger.info(f"Running t-SNE with perplexity={self.perplexity}...")
        
        # Scale data first
        X_scaled = self.scaler.fit_transform(X)
        
        # Use subset if too large (t-SNE is slow for large datasets)
        if len(X_scaled) > 5000:
            logger.warning(f"Dataset too large ({len(X_scaled)} samples). Using subset of 5000.")
            indices = np.random.choice(len(X_scaled), 5000, replace=False)
            X_scaled = X_scaled[indices]
        
        self.model = SklearnTSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_iter=1000,
            verbose=0
        )
        
        self.embedding = self.model.fit_transform(X_scaled)
        
        logger.info(f"✅ t-SNE complete. Shape: {self.embedding.shape}")
        
        return self.embedding
    
    def plot_2d(self, labels=None, save_path=None):
        """
        Plot t-SNE in 2D
        
        Args:
            labels: Optional labels for coloring
            save_path: Optional save path
        """
        if self.embedding is None:
            raise ValueError("Run fit_transform() first.")
        
        if self.n_components != 2:
            print("n_components must be 2 for 2D plot")
            return
        
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            classes = np.unique(labels)
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
            
            for i, class_label in enumerate(classes):
                mask = labels == class_label
                plt.scatter(self.embedding[mask, 0], self.embedding[mask, 1],
                           color=colors[i % len(colors)],
                           label=f'Class {class_label}', alpha=0.6)
            plt.legend()
        else:
            plt.scatter(self.embedding[:, 0], self.embedding[:, 1], alpha=0.6)
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE Visualization (2D)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ t-SNE plot saved to: {save_path}")
        
        plt.show()
    
    def plot_3d(self, labels=None, save_path=None):
        """
        Plot t-SNE in 3D
        
        Args:
            labels: Optional labels for coloring
            save_path: Optional save path
        """
        if self.embedding is None:
            raise ValueError("Run fit_transform() first.")
        
        if self.n_components != 3:
            print("n_components must be 3 for 3D plot")
            return
        
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            if labels is not None:
                classes = np.unique(labels)
                colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
                
                for i, class_label in enumerate(classes):
                    mask = labels == class_label
                    ax.scatter(self.embedding[mask, 0], 
                              self.embedding[mask, 1],
                              self.embedding[mask, 2],
                              color=colors[i % len(colors)],
                              label=f'Class {class_label}', alpha=0.6)
                ax.legend()
            else:
                ax.scatter(self.embedding[:, 0], 
                          self.embedding[:, 1],
                          self.embedding[:, 2], alpha=0.6)
            
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.set_zlabel('t-SNE Component 3')
            ax.set_title('t-SNE Visualization (3D)')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"✅ t-SNE 3D plot saved to: {save_path}")
            
            plt.show()
            
        except ImportError:
            print("matplotlib 3D plotting not available")
    
    def find_optimal_perplexity(self, X, perplexities=[5, 10, 20, 30, 40, 50]):
        """
        Find optimal perplexity value
        
        Args:
            X: Feature matrix
            perplexities: List of perplexity values to test
            
        Returns:
            Dictionary of results
        """
        print("\n" + "="*60)
        print("🔍 OPTIMAL PERPLEXITY SEARCH")
        print("="*60)
        
        results = {}
        
        # Use small subset for testing
        if len(X) > 1000:
            indices = np.random.choice(len(X), 1000, replace=False)
            X_subset = X[indices]
        else:
            X_subset = X
        
        for perp in perplexities:
            print(f"\nTesting perplexity={perp}...")
            tsne = SklearnTSNE(
                n_components=2,
                perplexity=perp,
                learning_rate=200,
                random_state=self.random_state,
                n_iter=500
            )
            embedding = tsne.fit_transform(X_subset)
            
            # Calculate quality metric (KL divergence approximation)
            results[perp] = {
                'embedding': embedding,
                'kl_divergence': tsne.kl_divergence_
            }
            print(f"   KL Divergence: {tsne.kl_divergence_:.4f}")
        
        # Find best perplexity
        best_perp = min(results.keys(), key=lambda k: results[k]['kl_divergence'])
        print("\n" + "="*60)
        print(f"🏆 Best Perplexity: {best_perp}")
        print(f"   KL Divergence: {results[best_perp]['kl_divergence']:.4f}")
        
        return results, best_perp
    
    def plot_comparison(self, X, perplexities=[10, 30, 50], labels=None, save_path=None):
        """
        Plot t-SNE with different perplexity values
        
        Args:
            X: Feature matrix
            perplexities: List of perplexity values
            labels: Optional labels
            save_path: Optional save path
        """
        # Use small subset
        if len(X) > 2000:
            indices = np.random.choice(len(X), 2000, replace=False)
            X_subset = X[indices]
            if labels is not None:
                labels_subset = labels[indices]
            else:
                labels_subset = None
        else:
            X_subset = X
            labels_subset = labels
        
        fig, axes = plt.subplots(1, len(perplexities), figsize=(15, 5))
        if len(perplexities) == 1:
            axes = [axes]
        
        for idx, perp in enumerate(perplexities):
            tsne = SklearnTSNE(
                n_components=2,
                perplexity=perp,
                learning_rate=200,
                random_state=self.random_state,
                n_iter=500
            )
            embedding = tsne.fit_transform(X_subset)
            
            if labels_subset is not None:
                classes = np.unique(labels_subset)
                colors = ['blue', 'red', 'green', 'orange', 'purple']
                for i, class_label in enumerate(classes):
                    mask = labels_subset == class_label
                    axes[idx].scatter(embedding[mask, 0], embedding[mask, 1],
                                     color=colors[i % len(colors)],
                                     label=f'Class {class_label}', alpha=0.6)
            else:
                axes[idx].scatter(embedding[:, 0], embedding[:, 1], alpha=0.6)
            
            axes[idx].set_title(f'Perplexity={perp}')
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('t-SNE Comparison: Different Perplexity Values')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ t-SNE comparison saved to: {save_path}")
        
        plt.show()
    
    def get_summary(self):
        """Get t-SNE summary"""
        if self.embedding is None:
            return {"status": "Not fitted"}
        
        summary = {
            'n_components': self.n_components,
            'perplexity': self.perplexity,
            'learning_rate': self.learning_rate,
            'embedding_shape': self.embedding.shape
        }
        
        print("\n" + "="*60)
        print("📊 t-SNE SUMMARY")
        print("="*60)
        print(f"Components: {summary['n_components']}")
        print(f"Perplexity: {summary['perplexity']}")
        print(f"Learning Rate: {summary['learning_rate']}")
        print(f"Embedding Shape: {summary['embedding_shape']}")
        
        return summary
