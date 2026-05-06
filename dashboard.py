import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback_context, no_update
import glob
import os
import json

# --- CONFIGURATION ---
CSV_FILES = glob.glob('benchmarks/*.csv') 
KERAS_JSON = 'keras_structures.json'     # <--- Changed
TORCH_JSON = 'torch_structures.json'     # <--- Added
TFLITE_JSON = 'tflite_structures.json'

def load_json(filepath):
    try:
        with open(filepath, 'r') as f: return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ {filepath} not found!")
        return {}

# 1. Load Keras and Torch separately
keras_structures = load_json(KERAS_JSON)
torch_structures = load_json(TORCH_JSON)

# 2. Merge them into one unified dictionary for the inspector callback
model_structures = {**keras_structures, **torch_structures}

# 3. Load TFLite
tflite_structures = load_json(TFLITE_JSON)


# ==========================================
# 1. DATA PROCESSING (Framework Parsing)
# ==========================================
def load_and_combine_data(filepaths):
    if not filepaths: raise ValueError("No CSV files found in 'benchmarks/' folder!")
        
    all_dfs = []
    for i, file in enumerate(filepaths):
        df = pd.read_csv(file)
        df['Run_ID'] = f"Run_{i+1}"
        all_dfs.append(df)
        
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Intelligently split Keras vs Torch
    def parse_meta(row):
        name = row['Model'].replace('.tflite', '')
        parts = name.split('_')
        
        if parts[0] == 'torch':
            fw, arch, n, seq, bn, q = 'PyTorch', parts[1].upper(), int(parts[2]), parts[3], parts[4], parts[5]
            base_name = f"torch_{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}"
        else:
            fw, arch, n, seq, bn, q = 'Keras', parts[0].upper(), int(parts[1]), parts[2], parts[3], parts[4]
            base_name = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"
            
        return fw, arch, n, bn, q, base_name
    
    merged_df[['Framework', 'Arch', 'Layers', 'BN', 'Quantization', 'Base_Name']] = merged_df.apply(lambda r: pd.Series(parse_meta(r)), axis=1)
    merged_df['Status'] = merged_df['Inference_Time_ms'].apply(lambda x: 'Failed' if 'ERROR' in str(x) else 'Success')

    run_status = merged_df.groupby(['Model', 'Framework', 'Arch', 'Layers', 'Seq_Length', 'BN', 'Quantization', 'Run_ID'])['Status'].apply(
        lambda x: 'Success' if (x == 'Success').any() else x.iloc[0]
    ).reset_index()
    
    stability_df = run_status.groupby(['Model', 'Framework', 'Arch', 'Layers', 'Seq_Length', 'BN', 'Quantization']).agg(
        Total_Runs=('Run_ID', 'count'),
        Successes=('Status', lambda x: (x == 'Success').sum()),
        Failures=('Status', lambda x: (x != 'Success').sum()),
        Failed_In_Runs=('Run_ID', lambda x: ", ".join(x[run_status.loc[x.index, 'Status'] != 'Success']))
    ).reset_index()
    
    def assign_category(row):
        if row['Failures'] == 0: return "✅ Stable"
        if row['Successes'] == 0: return "❌ Always Fails"
        return "⚠️ FLAKY"
        
    stability_df['Category'] = stability_df.apply(assign_category, axis=1)
    return merged_df, stability_df

merged_df, df_stability = load_and_combine_data(CSV_FILES)

# ==========================================
# 2. DASH APP LAYOUT
# ==========================================
app = Dash(__name__)
app.title = "ESP32 ML Benchmark"

GRAPH_CONFIG = {'toImageButtonOptions': {'format': 'png', 'filename': 'export', 'height': 600, 'width': 1000, 'scale': 2}, 'displaylogo': False}

app.layout = html.Div(style={'fontFamily': 'Arial', 'margin': '10px'}, children=[
    html.H1("🚀 ESP32 TFLite Benchmark (Keras vs PyTorch)", style={'textAlign': 'center'}),
    html.Div(style={'display': 'flex', 'flexDirection': 'row', 'height': '90vh'}, children=[
        
        # --- LEFT SIDEBAR (Filters) ---
        html.Div(style={'width': '15%', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}, children=[
            html.H3("⚙️ Filters"),
            html.Label("1. Framework:", style={'fontWeight': 'bold'}),
            dcc.Checklist(id='fw-checklist', options=[{'label': f' {f}', 'value': f} for f in ['Keras', 'PyTorch']], value=['Keras', 'PyTorch'], style={'marginBottom': '15px'}),
            
            html.Label("2. Report:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='run-dropdown', options=[{'label': 'Grand Average', 'value': 'ALL'}] + [{'label': r, 'value': r} for r in sorted(merged_df['Run_ID'].unique())], value='ALL', clearable=False, style={'marginBottom': '15px'}),

            html.Label("3. Depth (Layers):", style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='layer-dropdown', options=[{'label': f'{i} Layers', 'value': i} for i in sorted(merged_df['Layers'].unique())], value=1, clearable=False, style={'marginBottom': '15px'}),
            
            html.Label("4. Batch Norm:", style={'fontWeight': 'bold'}),
            dcc.Checklist(id='bn-checklist', options=[{'label': f' {b}', 'value': b} for b in ['ON', 'OFF']], value=['OFF'], style={'marginBottom': '15px'}),
            
            html.Label("5. Quantization:", style={'fontWeight': 'bold'}),
            dcc.Checklist(id='quant-checklist', options=[{'label': f' {q}', 'value': q} for q in ['int8', 'float32', 'hybrid']], value=['int8', 'float32'])
        ]),
        
        # --- CENTER CONTENT ---
        html.Div(style={'width': '55%', 'padding': '0 20px', 'overflowY': 'auto'}, children=[
            dcc.Tabs([
                dcc.Tab(label='📊 Performance Graphs', children=[
                    dcc.Graph(id='time-graph', config=GRAPH_CONFIG),
                    dcc.Graph(id='memory-graph', config=GRAPH_CONFIG),
                    dcc.Graph(id='mse-graph', config=GRAPH_CONFIG),
                ]),
                dcc.Tab(label='⚠️ Stability Analysis', children=[
                    html.H3("Flaky Models (Inconsistent)"),
                    dash_table.DataTable(id='flaky-table', style_table={'overflowX': 'auto', 'border': '1px solid #ccc'}, style_header={'backgroundColor': 'rgb(255, 200, 200)'}, style_data={'cursor': 'pointer'}),
                    html.H3("Models that ALWAYS Fail"),
                    dash_table.DataTable(id='fail-table', style_table={'overflowX': 'auto', 'border': '1px solid #ccc'}, style_header={'backgroundColor': 'rgb(230, 230, 230)'}, style_data={'cursor': 'pointer'})
                ])
            ])
        ]),

        # --- RIGHT SIDEBAR (Inspector) ---
        html.Div(style={'width': '30%', 'padding': '15px', 'backgroundColor': '#e9ecef', 'borderRadius': '10px', 'overflowY': 'auto'}, children=[
            html.H3("🔎 Model Inspector"),
            html.Div(id='model-inspector-content')
        ])
    ])
])

# ==========================================
# 3. INTERACTIVITY
# ==========================================
@app.callback(
    [Output('time-graph', 'figure'), Output('memory-graph', 'figure'), Output('mse-graph', 'figure'),
     Output('flaky-table', 'data'), Output('flaky-table', 'columns'), Output('fail-table', 'data'), Output('fail-table', 'columns')],
    [Input('fw-checklist', 'value'), Input('run-dropdown', 'value'), Input('layer-dropdown', 'value'), Input('bn-checklist', 'value'), Input('quant-checklist', 'value')]
)
def update_graphs_and_tables(selected_fw, selected_run, selected_layer, selected_bn, selected_quant):
    df = merged_df[(merged_df['Framework'].isin(selected_fw)) & (merged_df['Layers'] == selected_layer) & 
                   (merged_df['BN'].isin(selected_bn)) & (merged_df['Quantization'].isin(selected_quant)) & (merged_df['Status'] == 'Success')].copy()

    if selected_run != 'ALL': df = df[df['Run_ID'] == selected_run]

    df['Inference_Time_ms'] = pd.to_numeric(df['Inference_Time_ms'], errors='coerce')
    df['Arena_Memory_Bytes'] = pd.to_numeric(df['Arena_Memory_Bytes'], errors='coerce')
    df['MSE'] = pd.to_numeric(df['MSE'], errors='coerce')

    df_agg = df.groupby(['Framework', 'Arch', 'Seq_Length', 'BN', 'Quantization', 'Base_Name', 'Model']).agg({
        'Inference_Time_ms': 'mean', 'Arena_Memory_Bytes': 'max', 'MSE': 'mean'
    }).reset_index()

    if not df_agg.empty:
        df_agg['Seq_Length'] = df_agg['Seq_Length'].astype(str)
        # Add Framework to Legend string!
        df_agg['Model_Config'] = "[" + df_agg['Framework'] + "] " + df_agg['Arch'] + " | BN:" + df_agg['BN'] + " | " + df_agg['Quantization']

        fig_time = px.bar(df_agg, x='Seq_Length', y='Inference_Time_ms', color='Model_Config', barmode='group', custom_data=['Base_Name', 'Model'])
        fig_mem = px.bar(df_agg, x='Seq_Length', y='Arena_Memory_Bytes', color='Model_Config', barmode='group', custom_data=['Base_Name', 'Model'])
        fig_mse = px.bar(df_agg, x='Seq_Length', y='MSE', color='Model_Config', barmode='group', custom_data=['Base_Name', 'Model'])
        fig_mse.update_layout(yaxis=dict(range=[df_agg['MSE'].min() * 0.95, df_agg['MSE'].max() * 1.05]))
    else:
        empty_fig = px.bar(title="No successful models match the selected filters.")
        fig_time, fig_mem, fig_mse = empty_fig, empty_fig, empty_fig

    # Update Tables
    filtered_stab = df_stability[(df_stability['Framework'].isin(selected_fw)) & (df_stability['Layers'] == selected_layer) & 
                                 (df_stability['BN'].isin(selected_bn)) & (df_stability['Quantization'].isin(selected_quant))]
    
    flaky_data = filtered_stab[filtered_stab['Category'] == "⚠️ FLAKY"][['Framework', 'Model', 'Seq_Length', 'Failed_In_Runs']].to_dict('records')
    fail_data = filtered_stab[filtered_stab['Category'] == "❌ Always Fails"][['Framework', 'Model', 'Seq_Length', 'Total_Runs']].to_dict('records')

    if not flaky_data: flaky_data = [{"Model": "No flaky models detected! 🎉"}]
    if not fail_data: fail_data = [{"Model": "No permanent failures! 🎉"}]

    return fig_time, fig_mem, fig_mse, flaky_data, [{"name": i, "id": i} for i in flaky_data[0].keys()], fail_data, [{"name": i, "id": i} for i in fail_data[0].keys()]

@app.callback(
    Output('model-inspector-content', 'children'),
    [Input('time-graph', 'clickData'), Input('memory-graph', 'clickData'), Input('mse-graph', 'clickData'),
     Input('flaky-table', 'active_cell'), Input('fail-table', 'active_cell')],
    [State('flaky-table', 'data'), State('fail-table', 'data')]
)
def display_model_info(clk_time, clk_mem, clk_mse, active_flaky, active_fail, flaky_data, fail_data):
    ctx = callback_context
    if not ctx.triggered: return html.P("Click a bar or table row to inspect.")
        
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    base_name, tflite_name = None, None

    if trigger_id in ['time-graph', 'memory-graph', 'mse-graph']:
        click_data = clk_time if trigger_id == 'time-graph' else (clk_mem if trigger_id == 'memory-graph' else clk_mse)
        if click_data is None: return no_update
        base_name, tflite_name = click_data['points'][0]['customdata'][0], click_data['points'][0]['customdata'][1]
    else:
        active = active_flaky if trigger_id == 'flaky-table' else active_fail
        data = flaky_data if trigger_id == 'flaky-table' else fail_data
        if active is None or not data: return no_update
        tflite_name = data[active['row']].get('Model', '')
        if "No " in tflite_name: return no_update
        
        parts = tflite_name.replace('.tflite', '').split('_')
        base_name = f"torch_{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}" if parts[0] == 'torch' else f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"

    keras_content = model_structures.get(base_name, {})
    tflite_content = tflite_structures.get(tflite_name, "⚠️ TFLite execution details not found. Run extract_tflite_info.py.")

    return html.Div([
        html.H4(f"Model: {tflite_name}", style={'wordWrap': 'break-word'}),
        html.P(f"Total Params: {keras_content.get('params', 'N/A'):,}", style={'fontWeight': 'bold', 'color': '#007BFF'}),
        dcc.Tabs([
            dcc.Tab(label='Source Architecture', children=[
                html.Pre(keras_content.get('summary', '⚠️ Source summary not found.'), style={'backgroundColor': '#f1f3f5', 'padding': '10px', 'fontSize': '11px', 'overflowX': 'auto'})
            ]),
            dcc.Tab(label='TFLite Graph', children=[
                html.Pre(tflite_content, style={'backgroundColor': '#282c34', 'color': '#abb2bf', 'padding': '10px', 'fontSize': '11px', 'overflowX': 'auto'})
            ])
        ])
    ])

if __name__ == '__main__':
    app.run(debug=True)