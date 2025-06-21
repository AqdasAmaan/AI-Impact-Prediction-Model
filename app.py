import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css",
    "https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,300,0,0&icon_names=accessibility,analytics,arrow_back_ios,bar_chart,business,business_center,close,dashboard,domain,engineering,home,info,menu,page_info,paid,school,settings,storage,trending_up,tune,work_outline&display=block",
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    ], 
    use_pages=True, 
    suppress_callback_exceptions=True)


app.layout = html.Div(className="container", children=[
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='sidebar-state', storage_type='session', data={'collapsed': True}),

    dcc.Store(id='disclaimer-shown', storage_type='local', data=False),

    # Disclaimer Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Important Disclaimer", className="text-warning")),
        dbc.ModalBody([
            html.P([
                "⚠️ ",
                html.Strong("Data Disclaimer: "),
                "The prediction results and AI impact analysis presented in this dashboard are based on an arbitrary dataset sourced from Kaggle for educational and demonstration purposes only."
            ], className="mb-3"),
            html.P([
                "📊 ",
                html.Strong("No Real-World Correlation: "),
                "These predictions do not reflect actual job market trends, real AI impact assessments, or genuine employment forecasts. The data has been processed and modified for research and learning objectives."
            ], className="mb-3"),
            html.P([
                "🎓 ",
                html.Strong("Educational Purpose: "),
                "This tool is designed for educational exploration of data science concepts, machine learning techniques, and dashboard development. It should not be used for making real career or business decisions."
            ], className="mb-3"),
            html.P([
                "💡 ",
                html.Strong("Recommendation: "),
                "For actual career guidance or AI impact analysis, please consult official labor statistics, industry reports, and professional career advisors."
            ], className="mb-0")
        ]),
        dbc.ModalFooter([
            dbc.Button("I Understand", id="disclaimer-close", className="btn-primary", n_clicks=0)
        ])
    ], id="disclaimer-modal", is_open=False, backdrop="static", keyboard=False),

    # Sidebar
    html.Aside(id="sidebar", className="sidebar", children=[
        html.Button(
            html.Span('menu', className="material-symbols-rounded", id="toggle-btn-icon"),
            id="toggle-btn", className="toggle-btn"),
        html.Nav(className="nav-menu", children=[
            html.A(href="/", id='d-link', className="nav-link", children=[
                html.Span('dashboard', className="material-symbols-rounded"),
                html.Span('Dashboard', className="nav-text")
            ]),
            html.A(href="/about", id='a-link', className="nav-link", children=[
                html.Span('info', className="material-symbols-rounded"),
                html.Span("About", className="nav-text")
            ]),
        ])
    ]),
    
    # Main content
    html.Div(className="main-content", children=[
        html.Header(className="header", children=[
            html.H1("AI Nexus", className="header-title"),
        ]),
        
        html.Section(className="content", children=[
            dash.page_container
        ])
    ]),
])

# Disclaimer modal callback
@app.callback(
    [Output("disclaimer-modal", "is_open"),
     Output("disclaimer-shown", "data")],
    [Input("disclaimer-close", "n_clicks")],
    [State("disclaimer-shown", "data")]
)
def handle_disclaimer(close_clicks, disclaimer_shown):
    if disclaimer_shown:
        return False, True  # Show modal if not shown before
    if close_clicks:
        return False, True  # Close modal and mark as shown
    return True, False


# Callback to toggle the 'collapsed' class on the sidebar and store its state
@app.callback(
    [Output('sidebar', 'className'),
     Output('toggle-btn-icon', 'children'),
     Output("sidebar-state", "data")],
    [Input('toggle-btn', 'n_clicks')],
    [Input("sidebar-state", "data")]
)
def toggle_sidebar(n_clicks, sidebar_state):
    is_collapsed = sidebar_state.get("collapsed", False)
    
    if n_clicks:
        is_collapsed = not is_collapsed
    sidebar_class = 'sidebar collapsed' if is_collapsed else 'sidebar'
    icon = 'menu' if is_collapsed else 'close'
    
    return sidebar_class, icon, {"collapsed": is_collapsed}

# Callback to add 'active-link' class for the active page
@app.callback([
    Output('d-link', 'className'),
    Output('a-link', 'className')],
    [Input('url', 'pathname')])
def update_active_link(pathname):
    base_class = 'nav-link'
    active_class = 'nav-link active-link'
    return [
        active_class if pathname == '/' else base_class,
        active_class if pathname == '/about' else base_class
    ]

if __name__ == '__main__':
    app.run_server(debug=True)
