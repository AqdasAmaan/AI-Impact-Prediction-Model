import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Function to perform all processing and modeling
def load_and_process_data(data_path, n_clusters=2):
    # Load dataset
    data = pd.read_csv(data_path)
    
    # -------------------- Random Forest Regression -------------------- #
    # # Prepare data for Random Forest
    # X = data[['Tasks', 'AI Models', 'AI Workload Ratio']]
    # y = data['AI Impact']
    
    # # Split data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # # Initialize and train the Random Forest Regressor
    # rf_model = RandomForestRegressor(random_state=42)
    # rf_model.fit(X_train, y_train)

    # One-hot encoding for Task Type and Model Sophistication 
    encoders = {
        'Task Type': OneHotEncoder(),
        'Model Sophistication': OneHotEncoder()
    }
    
    for column, encoder in encoders.items():
        encoded = encoder.fit_transform(data[[column]]).toarray()
        encoded_cols = [f"{column}_{cat}" for cat in encoder.categories_[0]]
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=data.index)
        data = pd.concat([data, encoded_df], axis=1)
    
    # One-hot encoding for Task Type and Model Sophistication
    task_type_encoded = encoders['Task Type'].transform(data[['Task Type']]).toarray()
    model_sophistication_encoded = encoders['Model Sophistication'].transform(data[['Model Sophistication']]).toarray()

    # Combine encoded features
    encoded_columns = list(encoders['Task Type'].categories_[0]) + list(encoders['Model Sophistication'].categories_[0])
    encoded_features = np.hstack([task_type_encoded, model_sophistication_encoded])
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns, index=data.index)

    # Combine all features for training
    X = pd.concat([data[['Tasks', 'AI Models', 'AI Workload Ratio']], encoded_df], axis=1)
    y = data['AI Impact']

    # Train Random Forest model
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X, y)

    
    # -------------------- K-Means Clustering -------------------- #
    # Select features for clustering
    features = data[['AI Impact', 'Tasks', 'AI Models', 'AI Workload Ratio']]
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    from sklearn.ensemble import IsolationForest

    iso_forest = IsolationForest(contamination=0.5, random_state=0)  # Adjust contamination as needed
    outliers = iso_forest.fit_predict(features_scaled) == -1  # -1 labels anomalies

# Filter out anomalies
    features_filtered = features_scaled[~outliers]
    data_filtered = data[~outliers].copy()

    # Use Elbow Method for optimal clusters
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(features_filtered)
        inertia.append(kmeans.inertia_)
    
    # Perform KMeans clustering with specified n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    data_filtered['Cluster'] = kmeans.fit_predict(features_filtered)
    
    # Get cluster centers in original scale
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Store clustering results
    clustering_results = {
        "data_with_clusters": data_filtered,
        "cluster_centers": cluster_centers,
        "inertia_values": inertia
    }
    
    # Return combined data dictionary
    return {
        "clustering_results": clustering_results
    }
