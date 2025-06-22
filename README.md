# AI Nexus Dashboard 🤖

A comprehensive data science dashboard for analyzing AI impact on job markets, built with Python Dash and featuring interactive visualizations, predictive modeling, and a clean minimalist interface.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Visit-blue?style=for-the-badge&logo=rocket)](https://ai-nexus-116f.onrender.com)

## 🎯 Project Goals

The AI Nexus Dashboard was developed to:

- **Demonstrate data science workflows** from raw data processing to interactive visualization
- **Explore AI impact on employment** through statistical analysis and machine learning
- **Showcase modern dashboard development** using Python Dash framework
- **Provide educational insights** into job market trends and automation risks
- **Create an intuitive interface** for non-technical users to explore complex data


### Data Processing & Modifications

#### 1. Data Cleaning (`data_processing.ipynb`)

# Key cleaning operations performed:
- Removed duplicate entries 
- Standardized job titles and industry categories
- Handled missing values in salary and location fields
- Normalized skill requirements formatting
- Converted categorical variables to numerical encodings


## 🛠️ Technical Implementation

### Architecture
```
AI Nexus Dashboard/
├── app.py                         # Main application entry point
├── assets/                        # Static assets and styling
│   ├── style.css                  # Global styles
│   ├── sidebar.css                # Navigation styling
│   ├── dashboard.css              # Dashboard-specific styles
│   └── icons/                     # UI icons and graphics
├── pages/                         # Multi-page application structure
│   ├── dashboard.py               # Main dashboard page
│   ├── role_overview.py           # Role analysis page
│   └── about.py                   # About page
├── model/                         # Machine learning components
│   └── utils.py                   # Model utilities and predictions
├── data/                          # Data files and notebooks
│   ├── Dataset_1.csv              # Original dataset
│   ├── processed_dataset.csv      # Cleaned data
│   ├── data_processing.ipynb      # Data cleaning notebook
│   └── model_training.ipynb       # ML model development
└── README.md                      # This file
```

### Key Features
- **Real-time Predictions**: ML model integration for job risk assessment
- **Responsive Design**: Mobile-friendly minimalist interface
- **Multi-page Navigation**: Organized content structure

### Technology Stack
- **Backend**: Python 3.9+, Dash 2.14+
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Plotly, Dash Bootstrap Components
- **Styling**: Custom CSS with minimalist black/white theme
- **ML Pipeline**: Random Forest, Feature Engineering, Cross-validation

## 🚀 Installation & Usage

### Prerequisites
Python 3.9 or higher
pip package manager


### Installation Steps

# Clone the repository
git clone https://github.com/AqdasAmaan/AI-Impact-Prediction-Model.git <br>
cd AI-Impact-Prediction-Model

# Install required packages
pip install -r requirements.txt

# Run the application
python app.py


### Access the Dashboard
- Open your browser to `http://localhost:8050`
- Navigate through different pages using the sidebar
- Explore job predictions and AI impact analysis

## 📈 Dashboard Pages

### 1. Dashboard (Main)
- **Individual Analysis**: Enter job details for personalized risk assessment
- **Prediction Engine**: Real-time ML predictions with confidence scores
- **Comparison Tools**: Compare multiple roles side-by-side
- **Quick Insights**: Key findings and recommendations

### 3. About
- **Project Information**: Goals, methodology, and limitations
- **Data Sources**: Dataset details and processing information
- **Technical Details**: Model performance and validation results
- **Contact Information**: Developer and project links

## ⚠️ Important Disclaimers

### Data Limitations
- **Arbitrary Dataset**: Based on synthetic Kaggle data for educational purposes
- **No Real-World Correlation**: Predictions do not reflect actual job market conditions
- **Educational Use Only**: Not suitable for real career or business decisions
- **Regional Limitations**: Data may not represent all geographic markets

### Model Limitations
- **Training Constraints**: Limited to available dataset features
- **Temporal Scope**: Snapshot analysis, not longitudinal trends
- **Bias Considerations**: May reflect dataset biases and limitations
- **Accuracy Bounds**: ~60% accuracy with inherent uncertainty ranges
