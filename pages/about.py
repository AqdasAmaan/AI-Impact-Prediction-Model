import dash
from dash import html, dcc, Input, Output, callback, dash_table
import pandas as pd

dash.register_page(__name__, path='/about')

app = dash.get_app()

df1 = pd.read_csv('data/processed_dataset.csv')

tech_section = html.Div(className="tech-stack-list", children=[
    html.Div(className="tech-item", children=[
        html.Img(src="/assets/icons/python.png", className="tech-icon"),
        html.Span("Python")
    ]),
    html.Div(className="tech-item", children=[
        html.Img(src="https://plotly.github.io/images/dash.png", className="tech-icon"),
        html.Span("Dash")
    ]),
    html.Div(className="tech-item", children=[
        html.Img(src="https://avatars.githubusercontent.com/u/5997976?s=280&v=4", className="tech-icon"), 
        html.Span("Plotly")
    ]),
        html.Div(className="tech-item", children=[
        html.Img(src="/assets/icons/html.png", className="tech-icon"),
        html.Span("HTML")
    ]),
    html.Div(className="tech-item", children=[
        html.Img(src="/assets/icons/css.png", className="tech-icon"),
        html.Span("CSS")
    ]),
])

profile_section = html.Div(className="social-links-section", children=[
        html.Div(className="social-link-item", children=[
            html.Img(src="/assets/icons/linkedin.png", className="social-icon"),
            html.A("LinkedIn", href="https://www.linkedin.com/in/aqdas-amaan", target="_blank", className="social-link")
        ]),
        html.Div(className="social-link-item", children=[
            html.Img(src="/assets/icons/github.png", className="social-icon"),
            html.A("GitHub", href="https://github.com/AqdasAmaan", target="_blank", className="social-link")
        ]),
        html.Div(className="social-link-item", children=[
            html.Img(src="/assets/icons/gmail.png", className="social-icon"),
            html.A("Gmail", href="mailto:aqdasamaan88@gmail.com", className="social-link")
        ])
    ])

# How It Works Section
how_it_works_section = html.Div(className="how-it-works-section", children=[
    html.H3("How It Works?", className="sub-heading"),

    # Overview
    html.Div(className="explanation-card", children=[
        html.H4("🤖 AI Impact Prediction Engine", className="explanation-title"),
        html.P([
            "This tool predicts how likely a job role is to be impacted by AI technologies. ",
            "Using a Random Forest model trained on curated data, it outputs an AI impact score and assigns the role to a cluster representing overall vulnerability or augmentation potential."
        ], className="explanation-text"),
    ]),

    # Model Inputs
    html.Div(className="explanation-card", children=[
        html.H4("📊 What Inputs Does the Model Use?", className="explanation-title"),
        html.Div(className="features-grid", children=[
            html.Div(className="feature-item", children=[
                html.Div("🔢", className="feature-icon"),
                html.Div([
                    html.Strong("Tasks"),
                    html.P("Number of distinct responsibilities or work activities.")
                ], className="feature-content")
            ]),
            html.Div(className="feature-item", children=[
                html.Div("📘", className="feature-icon"),
                html.Div([
                    html.Strong("AI Models"),
                    html.P("Count of AI models associated with the job.")
                ], className="feature-content")
            ]),
            html.Div(className="feature-item", children=[
                html.Div("📈", className="feature-icon"),
                html.Div([
                    html.Strong("AI Workload Ratio"),
                    html.P("Proportion of job workload likely to be taken over by AI.")
                ], className="feature-content")
            ]),
            html.Div(className="feature-item", children=[
                html.Div("🧠", className="feature-icon"),
                html.Div([
                    html.Strong("Task Type"),
                    html.P("The nature of work — e.g., routine or complex")
                ], className="feature-content")
            ]),
            html.Div(className="feature-item", children=[
                html.Div("⚙️", className="feature-icon"),
                html.Div([
                    html.Strong("Model Sophistication"),
                    html.P("Level of AI complexity required to automate tasks in this role.")
                ], className="feature-content")
            ]),
        ])
    ]),

    # Process Flow
    html.Div(className="explanation-card", children=[
        html.H4("🔄 Behind the Scenes", className="explanation-title"),
        html.Div(className="process-steps", children=[
            html.Div(className="step", children=[
                html.Div("1", className="step-number"),
                html.Div([
                    html.Strong("Feature Encoding"),
                    html.P("Categorical inputs like 'Task Type' and 'Model Sophistication' are one-hot encoded.")
                ], className="step-content")
            ]),
            html.Div("→", className="arrow"),
            html.Div(className="step", children=[
                html.Div("2", className="step-number"),
                html.Div([
                    html.Strong("Prediction"),
                    html.P("The Random Forest model estimates AI impact score based on combined inputs.")
                ], className="step-content")
            ]),
            html.Div("→", className="arrow"),
            html.Div(className="step", children=[
                html.Div("3", className="step-number"),
                html.Div([
                    html.Strong("Clustering"),
                    html.P("KMeans groups the role into clusters based on AI impact and complexity.")
                ], className="step-content")
            ]),
            html.Div("→", className="arrow"),
            html.Div(className="step", children=[
                html.Div("4", className="step-number"),
                html.Div([
                    html.Strong("Job Matching"),
                    html.P("System recommends 5 similar roles within the same cluster.")
                ], className="step-content")
            ]),
        ])
    ]),

    # Example
    html.Div(className="explanation-card example-card", children=[
        html.H4("💡 Example Prediction", className="explanation-title"),
        html.Div(className="example-content", children=[
            html.Div(className="example-input", children=[
                html.H5("Input:", className="example-subtitle"),
                html.Ul([
                    html.Li("Tasks: 100"),
                    html.Li("AI Models: 1000"),
                    html.Li("AI Workload Ratio: 0.15"),
                    html.Li("Task Type: Routine"),
                    html.Li("Model Sophistication: Basic")
                ])
            ]),
            html.Div("⬇️", className="example-arrow"),
            html.Div(className="example-analysis", children=[
                html.H5("Model Analysis:", className="example-subtitle"),
                html.P([
                    "• Tasks are on the lower end, suggesting less variety ✱", html.Br(),
                    "• Uses a moderate number of AI models ✱", html.Br(),
                    "• Low AI workload ratio — most work is still manual ✓", html.Br(),
                    "• Task type is routine, but impact remains moderate ✱", html.Br(),
                    "• Model sophistication is basic, implying simpler AI applications ✓"
                ])
            ]),
            html.Div("⬇️", className="example-arrow"),
            html.Div(className="example-result", children=[
                html.H5("Result:", className="example-subtitle"),
                html.Div(className="result-box", children=[
                    html.Div("AI Impact Score: 41.48%", className="score-medium"),
                    html.Div("Cluster Assignment: Cluster 1 (More likely to augment)", className="risk-medium"),
                    html.P("This role has a high potential for AI automation due to its structured nature and lower task complexity.")
                ])
            ])
        ])
    ]),

    # Note
    html.Div(className="explanation-card note-card", children=[
        html.H4("⚠️ Please Note", className="explanation-title"),
        html.Ul([
            html.Li("This tool is built for educational purposes using a sample dataset."),
            html.Li("Predictions are approximations, not industry-certified results."),
            html.Li("A high AI impact does not mean job loss — it may mean role evolution."),
        ], className="note-list")
    ])
])


layout = html.Div(className='about-container', children=[

    html.Div(className="header-section", children=[
        html.H1("About the Project", className="main-heading"),
    ]),
    
    # 3D Cards
    html.Div(className='cards-section-3d', children=[
        html.Div(className='card-link', children=[
            html.Div(id='c1', className='cardx text-card', children=[
                html.Div(className='wrapper', children=[
                    html.Div(className='cover-text', children=['Developed By'])
                ]),
                html.Div(id='title-name', className='title-text', children=['Aqdas Amaan']),
                html.Div(id='social-links', className='title-text', children=[profile_section])
            ]),
        ]),

        html.Div(className='card-link', children=[
            html.Div(id='c2', className='cardx text-card', children=[
                html.Div(className='wrapper', children=[
                    html.Div(className='cover-text', children=['Tech Stack'])
                ]),
                html.Div(className='title-text', children=[tech_section]),
                html.Div(className='description', children=[
                ])
            ]),
        ]),
    ]),

    how_it_works_section,

    # Datasets Table 
    html.Div(className="datasets-section", children=[
        html.H3("Dataset Used", className="sub-heading"),
        html.Div(className="dataset-table", children=[
            dash_table.DataTable(
                df1.to_dict('records'),
                [{"name": i, "id": i} for i in df1.columns],
                page_size=20,
                style_table={'overflowX': 'scroll'},
                 style_header={
                    'backgroundColor': 'rgb(30, 30, 30)',
                    'color': 'white',
                    'textAlign': 'left'
                },
                style_data={
                    'backgroundColor': 'rgb(250, 250, 250)',
                    'color': 'black',
                    'textAlign': 'left'
                }
            )
        ]),
    ]),

    # References Section
    html.Div(className="references-section", children=[
        html.H3("References", className="sub-heading"),
        html.Ul(children=[
            html.Li(html.A("McKinsey AI Insights", href="https://www.mckinsey.com/capabilities/quantumblack/our-insights", target="_blank")),
            html.Li(html.A("Kaggle Dataset", href="https://www.kaggle.com/datasets/manavgupta92/from-data-entry-to-ceo-the-ai-job-threat-index", target="_blank")),
            html.Li(html.A("World Economic Forum AI in the Workforce", href="https://www.weforum.org/stories/jobs-and-the-future-of-work/", target="_blank"))
        ], className="reference-list")
    ])
])
