import pandas as pd
import dash
from dash import dcc, html, Input, Output, callback
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from model.utils import load_and_process_data

dash.register_page(__name__, path='/')

app = dash.get_app()

dataset_path = r'data\processed_dataset.csv'

# Load all processed data for cluster information
processed_data = load_and_process_data(dataset_path)
cluster_centers = processed_data["clustering_results"]["cluster_centers"]

# Load and process data for predictions
def load_and_process_prediction_data():
    # Load dataset
    data = pd.read_csv(dataset_path)
    
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
    
    # Prepare KMeans model
    features = data[['AI Impact', 'Tasks', 'AI Models', 'AI Workload Ratio']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    iso_forest = IsolationForest(contamination=0.5, random_state=0)
    outliers = iso_forest.fit_predict(features_scaled) == -1

    # Filter out anomalies
    features_filtered = features_scaled[~outliers]
    data_filtered = data[~outliers]

    kmeans = KMeans(n_clusters=2, random_state=0)
    data_filtered['Cluster'] = kmeans.fit_predict(features_filtered)

    return rf_model, kmeans, scaler, data_filtered, encoders

rf_model, kmeans, scaler, data, encoders = load_and_process_prediction_data()

# Cluster centers as insight cards
def cluster_center_cards(cluster_centers):
    cards = []
    for i, center in enumerate(cluster_centers):
        if i == 0:
            card = html.Div(className="card", children=[
                html.Div(className="card-content", children=[
                    html.H3(f"Cluster {i+1}: AI-Augmented Roles", className="card-title"),
                    html.P([html.Strong('AI Impact: '), f'{center[0]:.1%} (Moderate influence)'], className="card-metric"),
                    html.P([html.Strong('Tasks: '), f'{center[1]:.0f} (Fewer tasks per role)'], className="card-metric"),
                    html.P([html.Strong('AI Models: '), f'{center[2]:.0f} (Lower AI model usage)'], className="card-metric"),
                    html.P([html.Strong('AI Workload: '), f'{center[3]:.1%} (Smaller AI workload share)'], className="card-metric"),
                    html.P([html.Strong('Insight: '), 'These roles use AI as a supportive tool, maintaining human-driven workflows with AI enhancement.'], className="card-metric"),
                ])
            ])
        
        if i == 1:
            card = html.Div(className="card", children=[
                html.Div(className="card-content", children=[
                    html.H3(f"Cluster {i+1}: AI-Integrated Roles", className="card-title"),
                    html.P([html.Strong('AI Impact: '), f'{center[0]:.1%} (Lower direct impact)'], className="card-metric"),
                    html.P([html.Strong('Tasks: '), f'{center[1]:.0f} (More tasks per role)'], className="card-metric"),
                    html.P([html.Strong('AI Models: '), f'{center[2]:.0f} (Higher AI model reliance)'], className="card-metric"),
                    html.P([html.Strong('AI Workload: '), f'{center[3]:.1%} (Larger AI workload proportion)'], className="card-metric"),
                    html.P([html.Strong('Insight: '), 'These roles have substantial AI integration for task execution, indicating higher automation potential.'], className="card-metric"),
                ])
            ])

        cards.append(card)
    return html.Div(className="card-section", children=cards)

# Dashboard layout
layout = html.Div([
    html.H1("AI Impact Dashboard"),
    
    # AI Impact Prediction Section
    html.Div(className="predictive-form", children=[
        html.H2("AI Impact Prediction"),
        html.P("Adjust the parameters below to predict how AI might impact a specific job role:", className="form-description"),
        
        # Sliders for numerical inputs
        html.Div(className='sliders', children=[
            html.Div(className='slider', children=[
                html.Label("Number of Tasks"),
                dcc.Slider(id="tasks-slider", min=1, max=1200, marks=None, value=100, 
                   tooltip={"placement": "bottom", 'always_visible': True})
            ]),
            html.Div(className='slider', children=[
                html.Label("AI Models Used"),
                dcc.Slider(id="ai-models-slider", min=1, max=3000, marks=None, value=1000, 
                   tooltip={"placement": "bottom", 'always_visible': True}),
            ]),
            html.Div(className='slider', children=[
                html.Label("AI Workload Ratio"),
                dcc.Slider(id="ai-workload-slider", min=0, max=0.75, marks=None, value=0.25, 
                   tooltip={"placement": "bottom", 'always_visible': True})
            ]),
        ]),
        
        # Dropdown for categorical inputs
        html.Div(className='dropdowns', children=[
            html.Div(className='dropdown', children=[
                html.Label("Task Type"),
                dcc.Dropdown(id="task-type-dropdown", 
                     options=[{'label': cat, 'value': cat} for cat in encoders['Task Type'].categories_[0]],
                     value=encoders['Task Type'].categories_[0][0])
            ]),
            html.Div(className='dropdown', children=[
                html.Label("Model Sophistication"),
                dcc.Dropdown(id="model-sophistication-dropdown", 
                     options=[{'label': cat, 'value': cat} for cat in encoders['Model Sophistication'].categories_[0]],
                     value=encoders['Model Sophistication'].categories_[0][0]),
            ]),
        ]),
        
        # Prediction Results
        html.H3("Prediction Results", style={'margin-top': '2rem', 'margin-bottom': '1rem'}),
        html.Div(className="card-section", children=[
            html.Div(className="rcard", id='predicted-ai-impact-card', children=[
                html.Div(className="card-content", children=[
                    html.H3("Predicted AI Impact", className="card-title"),
                    html.P(id="predicted-ai-impact", className="card-metric")
                ])
            ]),
            html.Div(className="rcard", id='assigned-cluster-card', children=[
                html.Div(className="card-content", children=[
                    html.H3("Cluster Assignment", className="card-title"),
                    html.P(id="cluster-assignment", className="card-metric")
                ])
            ]),
        ]),
    ]),
    
    # Cluster Analysis Section
    html.H2("Cluster Analysis", style={'margin-top': '3rem', 'margin-bottom': '1rem'}),
    html.P("Understanding the two main categories of AI impact on jobs:", className="section-description"),
    cluster_center_cards(cluster_centers),
    
    # Similar Job Titles Section
    html.H2("Similar Job Titles", style={'margin-top': '3rem', 'margin-bottom': '1rem'}),
    html.P("Jobs with similar AI impact characteristics:", className="section-description"),
    html.Div(id="job-title-list", className="job-title-list"),

    # AI Role Details Section
    html.H2("AI Role Analysis", style={'margin-top': '3rem', 'margin-bottom': '1rem'}),
    html.Div(id="ai-role-details", className="ai-role-details")
])

@app.callback(
    [Output("predicted-ai-impact", "children"),
     Output("cluster-assignment", "children"),
     Output("job-title-list", "children"),
     Output("ai-role-details", "children"),
     Output('predicted-ai-impact', 'className'),
    ],
    [Input("tasks-slider", "value"),
     Input("ai-models-slider", "value"),
     Input("ai-workload-slider", "value"),
     Input("task-type-dropdown", "value"),
     Input("model-sophistication-dropdown", "value")
    ]
)
def update_predictions(tasks, ai_models, ai_workload, task_type, model_sophistication):
    # One-hot encode dropdown inputs
    task_type_encoded = encoders['Task Type'].transform([[task_type]]).toarray()[0]
    model_sophistication_encoded = encoders['Model Sophistication'].transform([[model_sophistication]]).toarray()[0]

    # Combine inputs
    features = np.concatenate([[tasks, ai_models, ai_workload], task_type_encoded, model_sophistication_encoded])
    
    # Check shape compatibility
    if features.shape[0] != rf_model.n_features_in_:
        raise ValueError(f"Expected {rf_model.n_features_in_} features, but got {features.shape[0]}.")

    # Predict AI Impact
    predicted_impact = rf_model.predict([features])[0]

    # Cluster Assignment
    cluster_features_scaled = scaler.transform([[predicted_impact, tasks, ai_models, ai_workload]])
    cluster_label = kmeans.predict(cluster_features_scaled)[0]

    # Find Similar Job Titles
    cluster_data = data[data['Cluster'] == cluster_label]
    similar_jobs = cluster_data.sample(n=min(5, len(cluster_data)))['Job Titles'].tolist()

    # AI Role Details based on Cluster
    if cluster_label == 0:
        ai_role_info_text = f"AI is likely to augment this job by {predicted_impact*100:.1f}%. This role will benefit from AI assistance while maintaining human oversight."
    else:
        ai_role_info_text = f"This role has a {predicted_impact*100:.1f}% likelihood of significant AI integration. Consider upskilling in AI collaboration."

    # Class Assignment based on predicted impact score and cluster assignment
    class_name = 'card-metric'
    if predicted_impact > 0.5:
        class_name += ' score-high'
    elif predicted_impact > 0.3:
        class_name += ' score-medium'
    else:
        class_name += ' score-low'

    # Format Output
    predicted_impact_str = f"{predicted_impact*100:.1f}%"
    assigned_cluster = f"Cluster {cluster_label + 1}"
    similar_job_titles = [html.Div(job, className="job-title-item") for job in similar_jobs]
    ai_role_info = html.P(ai_role_info_text, className="ai-role-description")
    
    return predicted_impact_str, assigned_cluster, similar_job_titles, ai_role_info, class_name


