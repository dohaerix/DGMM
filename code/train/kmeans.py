import pandas as pd
import numpy as np
import seaborn as sns
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.stats import skew
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import joblib
from collections import Counter

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def aggregate_sequence(features):
    mean = np.mean(features, axis=1)
    iqr = np.percentile(features, 75, axis=1) - np.percentile(features, 25, axis=1)
    aggregated = np.concatenate([mean, iqr], axis=1)
    return aggregated


def get_clustering_metrics(encoded_features, cluster_labels):
    silhouette = silhouette_score(encoded_features, cluster_labels)
    davies_bouldin = davies_bouldin_score(encoded_features, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(encoded_features, cluster_labels)
    print(f'Silhouette Score: {silhouette:.3f}')
    print(f'Davies-Bouldin Score: {davies_bouldin:.3f}')
    print(f'Calinski-Harabasz Score: {calinski_harabasz:.3f}')
    return silhouette, davies_bouldin, calinski_harabasz


def visualize_clusters(X_train, train_labels, X_test, test_labels):
    # We must combine data for t-SNE so train and test share the same coordinate space
    n_train = len(X_train)
    combined_data = np.vstack((X_train, X_test))
    
    # Define a consistent colormap for both plots (assuming labels are 0-9)
    # This ensures "Gesture 0" is the same color in Train and Test
    unique_labels = np.unique(np.concatenate((train_labels, test_labels)))
    colors = plt.cm.tab10(np.linspace(0, 1, 10)) # limits to 10 distinct colors

    # ==========================================
    # 2D t-SNE & Visualization
    # ==========================================
    print("Running 2D t-SNE on combined data...")
    tsne_2d = TSNE(n_components=2, random_state=42) # Fixed seed for reproducibility
    combined_2d = tsne_2d.fit_transform(combined_data)
    
    # Split back into train and test
    train_2d = combined_2d[:n_train]
    test_2d = combined_2d[n_train:]
    
    # 1. Train 2D
    plt.figure(figsize=(12, 9))
    sns.scatterplot(x=train_2d[:, 0], y=train_2d[:, 1], hue=train_labels, palette='tab10', legend='full', s=200, alpha=0.7)
    plt.xlabel('t-SNE Component 1', fontsize=18, fontweight='bold')
    plt.ylabel('t-SNE Component 2', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    legend = plt.legend(title='Gesture', title_fontsize=16, fontsize=14, loc='best', frameon=True, shadow=True)

    legend.get_title().set_fontweight('bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.title('Train Data (2D)', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/cluster_train_2d_kmeans.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Test 2D
    plt.figure(figsize=(12, 9))
    sns.scatterplot(x=test_2d[:, 0], y=test_2d[:, 1], 
                   hue=test_labels, palette='tab10', legend='full', s=200, alpha=0.7)
    plt.xlabel('t-SNE Component 1', fontsize=18, fontweight='bold')
    plt.ylabel('t-SNE Component 2', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    legend = plt.legend(title='Gesture', title_fontsize=16, fontsize=14, 
                       loc='best', frameon=True, shadow=True)
    legend.get_title().set_fontweight('bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.title('Test Data (2D)', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/cluster_test_2d_kmeans.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ==========================================
    # 3D t-SNE & Visualization
    # ==========================================
    print("Running 3D t-SNE on combined data...")
    tsne_3d = TSNE(n_components=3, random_state=42)
    combined_3d = tsne_3d.fit_transform(combined_data)
    
    # Split back into train and test
    train_3d = combined_3d[:n_train]
    test_3d = combined_3d[n_train:]
    
    # Helper function to prevent code duplication for 3D plots
    def plot_3d(data, labels, filename):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set the camera angle so Z is visible
        ax.view_init(elev=30, azim=135) 
        
        for i, cluster in enumerate(np.unique(labels)):
            mask = labels == cluster
            # Ensure we pick the color corresponding to the specific integer label
            # This prevents color mismatch if Test is missing a class
            color_idx = int(cluster) % 10 
            ax.scatter(data[mask, 0], data[mask, 1], data[mask, 2], c=[colors[color_idx]], label=f'Gesture {cluster}', s=200, alpha=0.7)
        
        ax.set_xlabel('t-SNE Component 1', fontsize=15, fontweight='bold', labelpad=10)
        ax.set_ylabel('t-SNE Component 2', fontsize=15, fontweight='bold', labelpad=10)
        ax.set_zlabel('t-SNE Component 3', fontsize=15, fontweight='bold', labelpad=10)
        
        # Set tight axis limits to reduce blank space
        margin = 0.05
        ax.set_xlim([data[:, 0].min() - margin, data[:, 0].max() + margin])
        ax.set_ylim([data[:, 1].min() - margin, data[:, 1].max() + margin])
        ax.set_zlim([data[:, 2].min() - margin, data[:, 2].max() + margin])
        
        ax.tick_params(labelsize=12)
        legend = ax.legend(title='Gesture', title_fontsize=14, fontsize=12, loc='best')
        legend.get_title().set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300) 
        plt.close()

    # 3. Train 3D
    plot_3d(train_3d, train_labels, 'results/cluster_train_3d_kmeans.png')
    
    # 4. Test 3D
    plot_3d(test_3d, test_labels, 'results/cluster_test_3d_kmeans.png')
    print("Visualization complete.")

def main():
    # Load dataset
    features = np.load('features.npy')
    labels = np.load('labels.npy')

    features = aggregate_sequence(features)
    print(f"Feature set shape = {features.shape}")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=labels)

    # Normalize features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # KMeans Clustering
    km = KMeans(n_clusters=8, random_state=RANDOM_SEED, max_iter=1000)
    km.fit(X_train)
    joblib.dump(km, 'results/kmeans_model.pkl')

    train_labels = km.predict(X_train)
    test_labels = km.predict(X_test)

    print("Training set cluster distribution:", Counter(train_labels))
    print("Testing set cluster distribution:", Counter(test_labels))

    print("\n=== Training Set Metrics ===")
    train_silhouette, train_db, train_ch = get_clustering_metrics(X_train, train_labels)
    
    print("\n=== Test Set Metrics ===")
    test_silhouette, test_db, test_ch = get_clustering_metrics(X_test, test_labels)

    # Save all metrics to CSV
    scores_df = pd.DataFrame({
        'Dataset': ['Train', 'Test'],
        'Silhouette_Score': [train_silhouette, test_silhouette],
        'Davies_Bouldin_Score': [train_db, test_db],
        'Calinski_Harabasz_Score': [train_ch, test_ch]
    })
    scores_df.to_csv('results/clustering_metrics_kmeans.csv', index=False)
    visualize_clusters(X_train, train_labels, X_test, test_labels)

if __name__ == "__main__":
    main()

