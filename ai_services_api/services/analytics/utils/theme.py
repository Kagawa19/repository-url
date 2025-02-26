import streamlit as st

def toggle_theme():
    """
    Toggle between light and dark theme.
    
    This function is called when the user clicks the theme toggle button in the sidebar. It checks the current theme
    state stored in the Streamlit session state. If the current theme is 'light', it updates the theme to 'dark', and
    vice versa. This allows the user to switch between light and dark themes dynamically.
    """
    if st.session_state.theme == 'light':
        st.session_state.theme = 'dark'
    else:
        st.session_state.theme = 'light'
    
    # Apply the updated theme configuration
    apply_theme()

def apply_theme():
    """
    Apply comprehensive theme configuration based on the current theme state.
    
    This function is responsible for applying the appropriate CSS styles and plot configurations based on the current
    theme state. It uses the Streamlit markdown function to inject CSS styles into the page.
    
    If the current theme is 'dark', it applies a dark theme configuration, which includes:
    - Setting the main app background color to a dark color (#0E1117)
    - Changing the text color to white (#FAFAFA)
    - Adjusting the sidebar background color to a darker shade (#262730)
    - Modifying the styles of metric cards, inputs, controls, and other UI elements to match the dark theme
    - Updating the plot template to use dark background colors and white text
    
    If the current theme is 'light', it applies a light theme configuration, which is the default Streamlit styling.
    It sets the main app background color to white (#FFFFFF) and the text color to black (#000000).
    """
    if st.session_state.theme == 'dark':
        # Dark theme configuration
        st.markdown("""
            <style>
                /* Main app background and text */
                .stApp {
                    background-color: #0E1117;
                    color: #FAFAFA;
                }
                
                /* Sidebar */
                .css-1d391kg, [data-testid="stSidebar"] {
                    background-color: #262730;
                }
                
                /* Metric cards */
                [data-testid="stMetricValue"] {
                    background-color: #262730;
                    color: #FFFFFF !important;
                }
                
                /* Inputs and controls */
                .stSelectbox, .stSlider, .stDateInput {
                    background-color: #262730;
                    color: #FFFFFF;
                }
                
                /* DataFrames */
                .stDataFrame {
                    background-color: #262730;
                }
                .dataframe {
                    background-color: #262730;
                    color: #FFFFFF;
                }
                .dataframe th {
                    background-color: #404040;
                    color: #FFFFFF;
                }
                .dataframe td {
                    background-color: #262730;
                    color: #FFFFFF;
                }
                
                /* Headers and text */
                h1, h2, h3, h4, h5, h6, .css-10trblm {
                    color: #FFFFFF !important;
                }
                .css-145kmo2 {
                    color: #FFFFFF;
                }
                
                /* Buttons */
                .stButton > button {
                    background-color: #262730;
                    color: #FFFFFF;
                    border: 1px solid #4F4F4F;
                }
                .stButton > button:hover {
                    background-color: #404040;
                    color: #FFFFFF;
                    border: 1px solid #4F4F4F;
                }
                
                /* Warning messages */
                .stAlert {
                    background-color: #262730;
                    color: #FFFFFF;
                }
                
                /* Tooltips */
                .tooltip {
                    background-color: #262730 !important;
                    color: #FFFFFF !important;
                }
                
                /* Plotly figure backgrounds */
                .js-plotly-plot .plotly {
                    background-color: #262730 !important;
                }
                
                /* Custom styling for selection boxes */
                .SelectBox {
                    background-color: #262730 !important;
                    color: #FFFFFF !important;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Update plot templates for dark theme
        plot_template = {
            'layout': {
                'paper_bgcolor': '#262730',
                'plot_bgcolor': '#262730',
                'font': {'color': '#FFFFFF'},
                'xaxis': {
                    'gridcolor': '#4F4F4F',
                    'linecolor': '#4F4F4F',
                    'zerolinecolor': '#4F4F4F',
                    'tickfont': {'color': '#FFFFFF'}
                },
                'yaxis': {
                    'gridcolor': '#4F4F4F',
                    'linecolor': '#4F4F4F',
                    'zerolinecolor': '#4F4F4F',
                    'tickfont': {'color': '#FFFFFF'}
                },
                'legend': {'font': {'color': '#FFFFFF'}}
            }
        }
    else:
        # Light theme configuration (default Streamlit)
        st.markdown("""
            <style>
                .stApp {
                    background-color: #FFFFFF;
                    color: #000000;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Update plot template for light theme
        plot_template = {
            'layout': {
                'paper_bgcolor': '#FFFFFF',
                'plot_bgcolor': '#FFFFFF',
                'font': {'color': '#000000'}
            }
        }
    
    # Set the updated plot template as an attribute of the class
    st.session_state.plot_template = plot_template

def update_plot_theme(fig):
    """
    Update the theme of a Plotly figure based on the current theme setting.
    
    This function takes a Plotly figure object as input and updates its theme based on the current theme state. If the
    current theme is 'dark', it applies dark theme colors to the plot background, font, and axis lines. If the current
    theme is 'light', it uses the default light theme colors.
    
    Parameters:
    - fig (plotly.graph_objects.Figure): The Plotly figure object to be updated.
    
    Returns:
    - fig (plotly.graph_objects.Figure): The updated Plotly figure object with the applied theme.
    """
    if st.session_state.theme == 'dark':
        fig.update_layout(
            paper_bgcolor='#262730',
            plot_bgcolor='#262730',
            font={'color': '#FFFFFF'},
            xaxis=dict(
                gridcolor='#4F4F4F',
                linecolor='#4F4F4F',
                zerolinecolor='#4F4F4F'
            ),
            yaxis=dict(
                gridcolor='#4F4F4F',
                linecolor='#4F4F4F',
                zerolinecolor='#4F4F4F'
            )
        )
    return fig
