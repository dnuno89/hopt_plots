import streamlit as st
import json
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from os import walk
import joblib

def save_dict_to_joblib(data_dict, file_path):
    joblib.dump(data_dict, file_path) # , compress=('xz', 6)
    
def load_dict_from_joblib(file_path):
    return joblib.load(file_path)


def load_data(basefolder, only_one=False):
    iter_index = []
    configs = []
    sum_losses = []
    sum_profits = []
    num_trades = []
    other_sells = []
    dimensions = None
    categorical_dims = set()
    raw_data = []

    # Walk through the folders and load files
    for folder_main, _, files in walk(basefolder, topdown=True):
        files.sort()
        for idxf1, fname in enumerate(files):
            st.write(f"Processing file {fname}")
            data = [json.loads(line) for line in open(f'{folder_main}/{fname}', 'r')]
            # raw_data.append(data)
            for idxf2, bt in enumerate(data):
                if dimensions is None:
                    # Initialize dimensions and configs based on the first JSON entry
                    dimensions = list(bt["params_dict"].keys())
                    configs = [[] for _ in range(len(dimensions))]

                for indVar, dimension in enumerate(dimensions):
                    value = bt["params_dict"][dimension]
                    configs[indVar].append(value)

                    # Determine if the dimension is categorical by checking data type
                    if isinstance(value, str):
                        categorical_dims.add(dimension)

                iter_index.append((idxf1, idxf2))
                # Handle exit reasons and calculate roi sells
                exit_summary = bt['results_metrics']['exit_reason_summary']
                roi_sells = next((sr['trades'] for sr in exit_summary if sr['key'] == 'roi'), 0)         
                # Capture trades and calculate profit/loss summary
                trades = bt['results_metrics']['trades']
                num_trades.append(len(trades))
                other_sells.append(len(trades) - roi_sells)

                total_profit = 0
                total_loss = 0
                for trade in trades:
                    profit = trade['profit_abs']
                    if profit > 0:
                        total_profit += profit
                    else:
                        total_loss += profit
                sum_profits.append(total_profit)
                sum_losses.append(np.min([total_loss, -4]))
            if only_one:
                break

    profits = np.array(sum_profits)
    losses = np.array(sum_losses)
    pen_profit = profits + 2 * losses
    profit_factor = -profits / losses
    only_profit = profits + losses

    num_trades = np.array(num_trades)
    other_sells = np.array(other_sells)
    params = [np.array(cfg) for cfg in configs]
    
    return {
        'params': params,
        'profits': profits,
        'losses': losses,
        'Num Trades': num_trades,
        'other_sells': other_sells,
        'PenProfit': pen_profit,
        'Profit Factor': profit_factor,
        'Only Profit': only_profit,
        'dimensions': dimensions,
        'categorical_dims': list(categorical_dims),
        'raw_data': raw_data
    }

def create_hyperopt_3dscatters():
    data = st.session_state['data']
    dimensions = data['dimensions']
    categorical_dims = data['categorical_dims']
    metrics_to_plot = ['Only Profit', 'PenProfit', 'Profit Factor', 'Num Trades']
    
    # User selects 3 parameters to plot
    param_choices = st.multiselect("Select 3 parameters to plot:", dimensions, default=[dim for dim in dimensions if dim not in categorical_dims][:3])
    filter_params = [dim for dim in dimensions if dim not in param_choices]

    if len(param_choices) == 3:
        filters = {}
        
        menu_col1, menu_col2 = st.columns(2)
        with menu_col1:        
            with st.expander("Filter by params"):
                all_params = st.checkbox("Filter by all params", value=False)
                selected_params = dimensions if all_params else filter_params
                for param in selected_params:
                    param_idx = dimensions.index(param)
                    param_values = data['params'][param_idx]
                    
                    if param in categorical_dims:
                        unique_values = list(set(param_values))
                        selected_values = st.multiselect(f"Filter for {param} (categorical)", unique_values, default=unique_values)
                        filters[param] = [val in selected_values for val in param_values]
                    else:
                        min_val, max_val = np.min(param_values), np.max(param_values)
                        selected_range = st.slider(f"Filter for {param} (numeric)", min_val, max_val, (min_val, max_val))
                        filters[param] = (param_values >= selected_range[0]) & (param_values <= selected_range[1])
                
        with menu_col2:
            with st.expander("Plot Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    dot_size = st.slider("Plot dot size:", min_value=1, max_value=12, value=4)
                with col2:
                    layout_height = st.slider("Sub-Plot Height:", min_value=600, max_value=2400, value=1600)
        
        with st.expander("Filter by metric:"):
            for metric in metrics_to_plot:
                min_val, max_val = np.min(data[metric]), np.max(data[metric])
                selected_range = st.slider(f"Filter for {metric}", min_val, max_val, (min_val, max_val))
                filters[metric] = (data[metric] >= selected_range[0]) & (data[metric] <= selected_range[1])
                

        # Apply filters to the data
        filter_mask = np.ones(len(data['profits']), dtype=bool)
        for param, mask in filters.items():
            filter_mask &= np.array(mask)
        x_data = data['params'][dimensions.index(param_choices[0])][filter_mask]
        y_data = data['params'][dimensions.index(param_choices[1])][filter_mask]
        z_data = data['params'][dimensions.index(param_choices[2])][filter_mask]

        # Plotting the 2x2 grid of 3D scatter plots
        fig = make_subplots(
            rows=2, cols=2,
            # subplot_titles=("Profit (In-Out)", "Penalized Profit (In-2*Out)", "Profit Factor", "Number of Trades"),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}], [{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            horizontal_spacing=0.05, vertical_spacing=0.05
        )

        fig.add_trace(go.Scatter3d(
            x=x_data, y=y_data, z=z_data, mode='markers',
            marker=dict(size=dot_size, colorscale='Viridis', color=data['Only Profit'][filter_mask], showscale=True, colorbar=dict(title="Profit", x=0.4, y=0.75, len=0.35, thickness=15)),
            name="Profit (In-Out)",
            hovertemplate='Profit: %{marker.color}<br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>'
        ), row=1, col=1)

        fig.add_trace(go.Scatter3d(
            x=x_data, y=y_data, z=z_data, mode='markers',
            marker=dict(size=dot_size, colorscale='Viridis', color=data['PenProfit'][filter_mask], showscale=True, colorbar=dict(title="PenProfit", x=0.93, y=0.75, len=0.35, thickness=15)),
            name="Penalized Profit (In-2*Out)",
            hovertemplate='Penalized Profit: %{marker.color}<br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>'
        ), row=1, col=2)

        fig.add_trace(go.Scatter3d(
            x=x_data, y=y_data, z=z_data, mode='markers',
            marker=dict(size=dot_size, colorscale='Viridis', color=data['Profit Factor'][filter_mask], showscale=True, colorbar=dict(title="Profit Factor", x=0.4, y=0.25, len=0.35, thickness=15)),
            name="Profit Factor",
            hovertemplate='Profit Factor: %{marker.color}<br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>'
        ), row=2, col=1)

        fig.add_trace(go.Scatter3d(
            x=x_data, y=y_data, z=z_data, mode='markers',
            marker=dict(size=dot_size, colorscale='Viridis', color=data['Num Trades'][filter_mask], showscale=True, colorbar=dict(title="Num Trades", x=0.93, y=0.25, len=0.35, thickness=15)),
            name="Number of Trades",
            hovertemplate='Number of Trades: %{marker.color}<br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>'
        ), row=2, col=2)
        
        # Add axis labels to each subplot
        axis_titles = dict(
            xaxis_title=param_choices[0],
            yaxis_title=param_choices[1],
            zaxis_title=param_choices[2]
        )
        for row, col in [(1, 1), (1, 2), (2, 1), (2, 2)]:
            fig.update_scenes(axis_titles, row=row, col=col)
        # Adjust the position of subplot titles
        fig.update_layout(
            height=layout_height,
            showlegend=False,
            margin=dict(t=50),
            annotations=[
                dict(
                    text="Profit (In-Out)", x=0.225, y=1.025, showarrow=False, xref="paper", yref="paper", font=dict(size=14)
                ),
                dict(
                    text="Penalized Profit (In-2*Out)", x=0.775, y=1.025, showarrow=False, xref="paper", yref="paper", font=dict(size=14)
                ),
                dict(
                    text="Profit Factor", x=0.225, y=0.475, showarrow=False, xref="paper", yref="paper", font=dict(size=14)
                ),
                dict(
                    text="Number of Trades", x=0.775, y=0.475, showarrow=False, xref="paper", yref="paper", font=dict(size=14)
                )
            ],
            scene=dict(camera=dict(up=dict(x=0, y=0, z=1))),
            scene2=dict(camera=dict(up=dict(x=0, y=0, z=1))),
            scene3=dict(camera=dict(up=dict(x=0, y=0, z=1))),
            scene4=dict(camera=dict(up=dict(x=0, y=0, z=1)))
        )
        with st.expander("Plots"):
            st.plotly_chart(fig, height=layout_height, use_container_width=True, key='3d_scatter_plots')
        
        with st.expander("Params & Metrics Histograms:"):
            st.write("Params histograms:")
            hist_cols = st.columns(len(dimensions))
            for idx, param in enumerate(dimensions):
                with hist_cols[idx]:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=data['params'][dimensions.index(param)],
                        marker_color=np.where(filter_mask, 'orange', 'brown'),
                        opacity=0.5,
                        name='Filtered'  # This name will be used for the legend
                    ))
                    fig.update_layout(
                        barmode='overlay',
                        title=param,
                        yaxis=dict(automargin=True)
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"paramHist_{param}")


                    # st.plotly_chart(
                    #     go.Figure(data=[
                    #         go.Histogram(x=data['params'][dimensions.index(param)][~filter_mask], name='Rest', opacity=0.5, marker=dict(color='brown')),
                    #         go.Histogram(x=data['params'][dimensions.index(param)][filter_mask], name='Filtered', opacity=0.5, marker=dict(color='orange'))
                    #     ]).update_layout(
                    #         barmode='overlay',
                    #         title=param,
                    #         yaxis=dict(automargin=True)
                    #     ),
                    #     use_container_width=True,
                    #     key=f"paramHist_{param}"
                    # )
            
            st.write("Metric histograms:")
            hist_cols = st.columns(len(metrics_to_plot))
            for idx, metric in enumerate(metrics_to_plot):
                with hist_cols[idx]:
                    # st.write(metric)
                    st.plotly_chart(
                        go.Figure(data=[
                            go.Histogram(x=data[metric][~filter_mask], name='Rest', opacity=0.5, marker=dict(color='brown')),
                            go.Histogram(x=data[metric][filter_mask], name='Filtered', opacity=0.5, marker=dict(color='orange'))
                        ]).update_layout(
                            barmode='overlay',
                            title=metric,
                            yaxis=dict(automargin=True)
                        ),
                        use_container_width=True,
                        key=f"metricHist_{metric}"
                    )
        

                    # Streamlit app code

st.set_page_config(layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Loading", "3D Scatter Plots"])

if page == "Data Loading":
    st.title("Data Loading")
    basefolder = st.text_input("Enter base folder path for JSON files:", "C:/Users/danin/src/myFtMod/user_data/hyperopt_results/maxBreak/m5")
    col1, col2, col3 = st.columns(3)
    with col1:
        load_button = st.button("Load Data")
    with col2:
        only_load_one = st.checkbox("Load only one file", value=False)
    with col3:
        save_hdf5 = st.checkbox("Save loaded data to .PKL", value=False)

    uploaded_file = st.file_uploader("Upload a file", type=["pkl"])

    if load_button:
        if uploaded_file:
            data = load_dict_from_joblib(uploaded_file)
            st.success("Data Loaded Successfully")
        elif basefolder:
            if only_load_one:
                data = load_data(basefolder, only_one=True)
            else:
                data = load_data(basefolder)
            st.success("Data Loaded Successfully")
            if save_hdf5:
                save_dict_to_joblib(data, './hyperopt_data.pkl')
        
        st.session_state['data'] = data

elif page == "3D Scatter Plots":
    st.title("3D Scatter Plots")
    if 'data' not in st.session_state:
        st.warning("Please load the data in the 'Data Loading' section first.")
    else:
        create_hyperopt_3dscatters()
        
        
