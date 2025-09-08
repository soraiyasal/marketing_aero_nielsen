import dash
from dash import dcc, html, Input, Output, callback, dash_table, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import re
import io
import base64
from datetime import datetime

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "CMO/CDMO Dashboard"

# Design tokens (Steve Jobs style)
COLORS = {
    'accent': '#2F80ED',
    'bg': '#FFFFFF',
    'text': '#0B0B0C',
    'muted': '#6B7280',
    'success': '#0E9F6E',
    'danger': '#E11D48',
    'card': '#F9FAFB',
    'border': '#E5E7EB'
}

# Global variables for data
df = None
dimensions = []
measures = []
price_cols = []

# Month mapping for period parsing
MONTHS = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
}

def create_period_key(s):
    """Convert period string to YYYY-MM format (fixed version)"""
    s = str(s)
    # Handle formats like "001_May 2023" -> "2023-05"
    part = s.split('_')[-1] if '_' in s else s
    m = re.search(r'([A-Za-z]{3,})\s+(\d{4})', part)
    if m:
        month = MONTHS.get(m.group(1)[:3].upper(), 1)
        year = int(m.group(2))
        return f"{year}-{month:02d}"
    return s  # fallback

def yoy_series(ser, periods=12):
    """Calculate YoY growth with 12-period lag"""
    return ser / ser.shift(periods) - 1

def load_and_process_data():
    """Load and process CSV data with all fixes applied"""
    global df, dimensions, measures, price_cols
    
    try:
        # Load CSV (replace with your file path)
        df = pd.read_csv('DEO-pivot.csv')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Fix 1: Use ORDERED PERIOD correctly
        preferred = [c for c in df.columns if c.strip().upper() == 'ORDERED PERIOD']
        fallback = [c for c in df.columns if c.strip().upper() == 'PERIOD']
        period_cols = preferred or fallback
        
        if period_cols:
            df['period_key'] = df[period_cols[0]].apply(create_period_key)
        else:
            df['period_key'] = '2023-01'  # fallback
        
        # Fix 6: Define price_cols globally early
        price_cols = [c for c in df.columns if 'price' in c.lower()]
        
        # Auto-detect dimensions and measures
        dimensions = []
        measures = []
        
        for col in df.columns:
            if df[col].dtype in ['object', 'string']:
                dimensions.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                measures.append(col)
        
        # Remove rows with all null measures
        measure_cols = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['units', 'volume', 'value', 'price'])]
        df = df.dropna(subset=measure_cols, how='all')
        
        # Sort by period for proper YoY calculations
        df = df.sort_values('period_key')
        
        return True
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

def calculate_yoy_growth(data, value_col, group_cols=None):
    """Calculate true YoY growth (12-month lag) - Fixed version"""
    if group_cols is None:
        group_cols = ['BRAND'] if 'BRAND' in data.columns else []
    
    growth_data = []
    
    if not group_cols:
        # Overall YoY
        monthly_data = data.groupby('period_key')[value_col].sum().sort_index()
        if len(monthly_data) >= 13:  # Need at least 13 periods for YoY
            yoy_growth = yoy_series(monthly_data, 12) * 100
            latest_period = monthly_data.index[-1]
            if not pd.isna(yoy_growth.iloc[-1]):
                growth_data.append({
                    'Entity': 'Total Market',
                    'Latest_Value': monthly_data.iloc[-1],
                    'YoY_Growth': yoy_growth.iloc[-1],
                    'Period': latest_period
                })
    else:
        # Group-wise YoY
        for entity in data[group_cols[0]].unique():
            entity_data = data[data[group_cols[0]] == entity]
            monthly_data = entity_data.groupby('period_key')[value_col].sum().sort_index()
            
            if len(monthly_data) >= 13:
                yoy_growth = yoy_series(monthly_data, 12) * 100
                latest_period = monthly_data.index[-1]
                if not pd.isna(yoy_growth.iloc[-1]):
                    growth_data.append({
                        'Entity': entity,
                        'Latest_Value': monthly_data.iloc[-1],
                        'YoY_Growth': yoy_growth.iloc[-1],
                        'Period': latest_period
                    })
    
    return pd.DataFrame(growth_data)

def calculate_promo_uplift(data, value_col):
    """Calculate promo uplift with exact string matching - Fixed version"""
    if 'PROMO/NO PROMO' not in data.columns:
        return pd.DataFrame()
    
    # Fix 3: Exact string matching for promo
    promo_norm = data['PROMO/NO PROMO'].astype(str).str.strip().str.upper()
    is_promo = promo_norm.eq('PROMO')
    is_no_promo = promo_norm.eq('NO PROMO')
    
    uplift_data = []
    
    if 'BRAND' in data.columns:
        for brand in data['BRAND'].unique():
            brand_data = data[data['BRAND'] == brand]
            brand_promo_norm = brand_data['PROMO/NO PROMO'].astype(str).str.strip().str.upper()
            
            promo_sales = brand_data.loc[brand_promo_norm.eq('PROMO'), value_col].mean()
            no_promo_sales = brand_data.loc[brand_promo_norm.eq('NO PROMO'), value_col].mean()
            
            if pd.notna(promo_sales) and pd.notna(no_promo_sales) and no_promo_sales > 0:
                uplift = ((promo_sales - no_promo_sales) / no_promo_sales) * 100
                uplift_data.append({
                    'Brand': brand,
                    'Promo_Avg': promo_sales,
                    'No_Promo_Avg': no_promo_sales,
                    'Uplift_Percent': uplift
                })
    
    return pd.DataFrame(uplift_data)

def create_kpi_card(title, value, change=None, units="", icon="📊"):
    """Create Apple-style KPI card"""
    change_component = html.Div()
    if change is not None and not pd.isna(change):
        color = COLORS['success'] if change >= 0 else COLORS['danger']
        symbol = "▲" if change >= 0 else "▼"
        change_component = html.Div([
            html.Span(symbol, style={'marginRight': '4px'}),
            html.Span(f"{abs(change):.1f}%")
        ], style={
            'color': color,
            'fontSize': '14px',
            'fontWeight': '500',
            'marginTop': '8px'
        })
    
    return html.Div([
        html.Div([
            html.Span(icon, style={'fontSize': '24px', 'marginBottom': '8px'}),
            html.Div(f"{value:,.0f}" if isinstance(value, (int, float)) else str(value), 
                    style={
                        'fontSize': '32px',
                        'fontWeight': '700',
                        'color': COLORS['text'],
                        'lineHeight': '1'
                    }),
            html.Div(f"{title} {units}", style={
                'fontSize': '14px',
                'color': COLORS['muted'],
                'marginTop': '4px',
                'fontWeight': '500'
            }),
            change_component
        ])
    ], style={
        'backgroundColor': COLORS['card'],
        'padding': '24px',
        'borderRadius': '12px',
        'border': f'1px solid {COLORS["border"]}',
        'textAlign': 'center',
        'minHeight': '140px',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center'
    })

def create_help_modal():
    """Create help modal explaining how to read the dashboard"""
    return html.Div([
        dcc.Store(id='help-modal-open', data=False),
        html.Div(id='help-modal-backdrop', children=[
            html.Div([
                html.Div([
                    html.H2("How to Read This Dashboard", style={
                        'color': COLORS['text'],
                        'marginBottom': '24px',
                        'fontSize': '24px',
                        'fontWeight': '600'
                    }),
                    html.Div([
                        html.H3("📊 Data Model", style={'color': COLORS['accent'], 'fontSize': '18px'}),
                        html.P("Each row represents: In month X, variant Y (Brand Z, Manufacturer M) sold Units/Volume/Value with specific attributes (Promo, Pack/Form, Target User, Weight)."),
                        
                        html.H3("🎯 Key Insights for CMO/CDMO", style={'color': COLORS['accent'], 'fontSize': '18px', 'marginTop': '20px'}),
                        html.Ul([
                            html.Li("Pipeline Scouting: Fastest-growing variants/brands likely to outsource"),
                            html.Li("Competitive Structure: Top-5 vs long tail; fragmented = more opportunity"),
                            html.Li("Capacity Alignment: Spiky SKUs (high volatility) = short contract runs"),
                            html.Li("Pricing Power: ↑Price & ↑Sales = strongest brand leverage"),
                            html.Li("Promo Dependence: Growth only under PROMO = margin risk"),
                            html.Li("Innovation: New variants gaining share quickly"),
                            html.Li("Private Label: Retailer PL share momentum")
                        ]),
                        
                        html.H3("🔢 How Numbers Are Calculated", style={'color': COLORS['accent'], 'fontSize': '18px', 'marginTop': '20px'}),
                        html.Ul([
                            html.Li("YoY Growth: Compare vs same month last year (12-period lag)"),
                            html.Li("Promo Uplift: (Promo Sales - No Promo Sales) / No Promo Sales"),
                            html.Li("Volatility: Coefficient of Variation (std/mean) of monthly demand"),
                            html.Li("Units: Already in thousands (as labeled in source data)")
                        ])
                    ], style={'color': COLORS['text'], 'lineHeight': '1.6'})
                ], style={
                    'backgroundColor': 'white',
                    'padding': '32px',
                    'borderRadius': '16px',
                    'maxWidth': '600px',
                    'maxHeight': '80vh',
                    'overflowY': 'auto'
                }),
                html.Button("✕", id='close-help-modal', style={
                    'position': 'absolute',
                    'top': '16px',
                    'right': '16px',
                    'background': 'none',
                    'border': 'none',
                    'fontSize': '24px',
                    'cursor': 'pointer',
                    'color': COLORS['muted']
                })
            ], style={
                'position': 'fixed',
                'top': '50%',
                'left': '50%',
                'transform': 'translate(-50%, -50%)',
                'zIndex': '1001'
            })
        ], style={
            'display': 'none',
            'position': 'fixed',
            'top': '0',
            'left': '0',
            'width': '100%',
            'height': '100%',
            'backgroundColor': 'rgba(0, 0, 0, 0.5)',
            'zIndex': '1000'
        })
    ])

# App layout with Apple-grade design
app.layout = html.Div([
    create_help_modal(),
    
    # Header
    html.Div([
        html.H1("CMO/CDMO Dashboard", style={
            'color': COLORS['text'],
            'fontSize': '28px',
            'fontWeight': '700',
            'margin': '0',
            'letterSpacing': '-0.02em'
        }),
        html.Button("ⓘ Help", id='open-help-modal', style={
            'background': 'none',
            'border': f'1px solid {COLORS["border"]}',
            'borderRadius': '8px',
            'padding': '8px 16px',
            'color': COLORS['muted'],
            'cursor': 'pointer',
            'fontSize': '14px'
        })
    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'padding': '24px 32px',
        'backgroundColor': 'white',
        'borderBottom': f'1px solid {COLORS["border"]}'
    }),
    
    # Main container
    html.Div([
        # Sidebar
        html.Div([
            html.H3("Filters", style={
                'color': COLORS['text'],
                'fontSize': '18px',
                'fontWeight': '600',
                'marginBottom': '24px'
            }),
            
            html.Div([
                html.Label("Period Range", style={'fontSize': '14px', 'fontWeight': '500', 'color': COLORS['text']}),
                dcc.RangeSlider(
                    id='period-slider',
                    marks={},
                    value=[],
                    tooltip={'placement': 'bottom', 'always_visible': True}
                )
            ], style={'marginBottom': '24px'}),
            
            html.Div([
                html.Label("Market", style={'fontSize': '14px', 'fontWeight': '500', 'color': COLORS['text']}),
                dcc.Dropdown(
                    id='market-filter',
                    multi=True,
                    placeholder="Select markets...",
                    style={'marginTop': '8px'}
                )
            ], style={'marginBottom': '24px'}),
            
            html.Div([
                html.Label("Brand", style={'fontSize': '14px', 'fontWeight': '500', 'color': COLORS['text']}),
                dcc.Dropdown(
                    id='brand-filter',
                    multi=True,
                    placeholder="Select brands...",
                    style={'marginTop': '8px'}
                )
            ], style={'marginBottom': '24px'}),
            
            html.Div([
                html.Label("Manufacturer", style={'fontSize': '14px', 'fontWeight': '500', 'color': COLORS['text']}),
                dcc.Dropdown(
                    id='manufacturer-filter',
                    multi=True,
                    placeholder="Select manufacturers...",
                    style={'marginTop': '8px'}
                )
            ], style={'marginBottom': '24px'}),
            
            html.Div([
                html.Label("Promo Status", style={'fontSize': '14px', 'fontWeight': '500', 'color': COLORS['text']}),
                dcc.Dropdown(
                    id='promo-filter',
                    multi=True,
                    placeholder="Select promo status...",
                    style={'marginTop': '8px'}
                )
            ], style={'marginBottom': '24px'}),
            
            html.Div([
                html.Label("Target User", style={'fontSize': '14px', 'fontWeight': '500', 'color': COLORS['text']}),
                dcc.Dropdown(
                    id='target-filter',
                    multi=True,
                    placeholder="Select target users...",
                    style={'marginTop': '8px'}
                )
            ], style={'marginBottom': '24px'})
            
        ], style={
            'width': '320px',
            'padding': '24px',
            'backgroundColor': COLORS['card'],
            'borderRight': f'1px solid {COLORS["border"]}',
            'height': 'calc(100vh - 89px)',
            'overflowY': 'auto'
        }),
        
        # Main content
        html.Div([
            dcc.Tabs(id='main-tabs', value='overview', children=[
                dcc.Tab(label='Overview', value='overview'),
                dcc.Tab(label='Growth Hotspots', value='growth'),
                dcc.Tab(label='Promo Effectiveness', value='promo'),
                dcc.Tab(label='Pack & Form Factor', value='pack'),
                dcc.Tab(label='Pricing Power', value='pricing'),
                dcc.Tab(label='Volatility', value='volatility'),
                dcc.Tab(label='Private Label', value='privatelabel')
            ], style={
                'borderBottom': f'1px solid {COLORS["border"]}',
                'marginBottom': '24px'
            }),
            
            html.Div(id='tab-content')
            
        ], style={
            'flex': '1',
            'padding': '24px 32px',
            'backgroundColor': 'white',
            'overflowY': 'auto'
        })
        
    ], style={
        'display': 'flex',
        'height': 'calc(100vh - 89px)'
    })
    
], style={
    'fontFamily': '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif',
    'margin': '0',
    'backgroundColor': COLORS['bg']
})

# Callbacks for help modal
@app.callback(
    Output('help-modal-backdrop', 'style'),
    [Input('open-help-modal', 'n_clicks'),
     Input('close-help-modal', 'n_clicks')],
    prevent_initial_call=True
)
def toggle_help_modal(open_clicks, close_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return {'display': 'none'}
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'open-help-modal':
        return {
            'display': 'block',
            'position': 'fixed',
            'top': '0',
            'left': '0',
            'width': '100%',
            'height': '100%',
            'backgroundColor': 'rgba(0, 0, 0, 0.5)',
            'zIndex': '1000'
        }
    else:
        return {'display': 'none'}

# Initialize filters callback
@app.callback(
    [Output('period-slider', 'min'),
     Output('period-slider', 'max'),
     Output('period-slider', 'marks'),
     Output('period-slider', 'value'),
     Output('market-filter', 'options'),
     Output('brand-filter', 'options'),
     Output('manufacturer-filter', 'options'),
     Output('promo-filter', 'options'),
     Output('target-filter', 'options')],
    [Input('main-tabs', 'value')]  # Trigger on app load
)
def initialize_filters(_):
    if df is None:
        return [], [], {}, [], [], [], [], [], []
    
    # Period slider
    periods = sorted(df['period_key'].unique())
    period_marks = {i: periods[i] for i in range(0, len(periods), max(1, len(periods)//6))}
    if len(periods) > 0:
        period_marks[len(periods)-1] = periods[-1]  # Always show last period
    
    # Filter options
    market_options = [{'label': m, 'value': m} for m in sorted(df['MARKET'].unique())] if 'MARKET' in df.columns else []
    brand_options = [{'label': b, 'value': b} for b in sorted(df['BRAND'].unique())] if 'BRAND' in df.columns else []
    manufacturer_options = [{'label': m, 'value': m} for m in sorted(df['MANUFACTURER'].unique())] if 'MANUFACTURER' in df.columns else []
    promo_options = [{'label': p, 'value': p} for p in sorted(df['PROMO/NO PROMO'].unique())] if 'PROMO/NO PROMO' in df.columns else []
    target_options = [{'label': t, 'value': t} for t in sorted(df['TARGET USER'].unique())] if 'TARGET USER' in df.columns else []
    
    return (0, len(periods)-1, period_marks, [0, len(periods)-1], 
            market_options, brand_options, manufacturer_options, promo_options, target_options)

# Main content callback
@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'value'),
     Input('period-slider', 'value'),
     Input('market-filter', 'value'),
     Input('brand-filter', 'value'),
     Input('manufacturer-filter', 'value'),
     Input('promo-filter', 'value'),
     Input('target-filter', 'value')]
)
def update_tab_content(active_tab, period_range, markets, brands, manufacturers, promos, targets):
    if df is None:
        return html.Div("Please ensure DEO-pivot.csv is in the app directory", 
                       style={'padding': '40px', 'textAlign': 'center', 'color': COLORS['muted']})
    
    # Apply filters
    filtered_df = df.copy()
    
    # Period filter
    if period_range:
        periods = sorted(df['period_key'].unique())
        if len(periods) > period_range[0]:
            start_period = periods[period_range[0]]
            end_period = periods[min(period_range[1], len(periods)-1)]
            filtered_df = filtered_df[(filtered_df['period_key'] >= start_period) & 
                                    (filtered_df['period_key'] <= end_period)]
    
    # Other filters
    if markets:
        filtered_df = filtered_df[filtered_df['MARKET'].isin(markets)]
    if brands:
        filtered_df = filtered_df[filtered_df['BRAND'].isin(brands)]
    if manufacturers:
        filtered_df = filtered_df[filtered_df['MANUFACTURER'].isin(manufacturers)]
    if promos:
        filtered_df = filtered_df[filtered_df['PROMO/NO PROMO'].isin(promos)]
    if targets:
        filtered_df = filtered_df[filtered_df['TARGET USER'].isin(targets)]
    
    if len(filtered_df) == 0:
        return html.Div("No data available with current filters. Try expanding your selection.", 
                       style={'padding': '40px', 'textAlign': 'center', 'color': COLORS['muted']})
    
    # Get measure columns
    value_cols = [col for col in filtered_df.columns if 'value' in col.lower() and 'in 1000' in col.lower()]
    volume_cols = [col for col in filtered_df.columns if 'volume' in col.lower() and 'in 1000' in col.lower()]
    units_cols = [col for col in filtered_df.columns if 'units' in col.lower() and 'in 1000' in col.lower()]
    
    if active_tab == 'overview':
        return create_overview_tab(filtered_df, value_cols, volume_cols, units_cols)
    elif active_tab == 'growth':
        return create_growth_tab(filtered_df, value_cols, volume_cols)
    elif active_tab == 'promo':
        return create_promo_tab(filtered_df, value_cols, units_cols)
    elif active_tab == 'pack':
        return create_pack_tab(filtered_df, value_cols)
    elif active_tab == 'pricing':
        return create_pricing_tab(filtered_df, value_cols)
    elif active_tab == 'volatility':
        return create_volatility_tab(filtered_df, units_cols, volume_cols)
    elif active_tab == 'privatelabel':
        return create_privatelabel_tab(filtered_df, value_cols)
    
    return html.Div()

def create_overview_tab(filtered_df, value_cols, volume_cols, units_cols):
    """Create overview tab with KPIs and brand analysis"""
    
    # KPI calculations
    kpis = []
    
    if value_cols:
        total_value = filtered_df[value_cols[0]].sum()
        value_growth = calculate_yoy_growth(filtered_df, value_cols[0])
        value_yoy = value_growth['YoY_Growth'].iloc[0] if not value_growth.empty else None
        kpis.append(create_kpi_card("Total Value", total_value, value_yoy, "(in 1,000 RUR)", "💰"))
    
    if volume_cols:
        total_volume = filtered_df[volume_cols[0]].sum()
        volume_growth = calculate_yoy_growth(filtered_df, volume_cols[0])
        volume_yoy = volume_growth['YoY_Growth'].iloc[0] if not volume_growth.empty else None
        kpis.append(create_kpi_card("Total Volume", total_volume, volume_yoy, "(in 1,000 LTR)", "📊"))
    
    if units_cols:
        total_units = filtered_df[units_cols[0]].sum()
        units_growth = calculate_yoy_growth(filtered_df, units_cols[0])
        units_yoy = units_growth['YoY_Growth'].iloc[0] if not units_growth.empty else None
        kpis.append(create_kpi_card("Total Units", total_units, units_yoy, "(in 1,000 PACKS)", "📦"))
    
    active_brands = filtered_df['BRAND'].nunique() if 'BRAND' in filtered_df.columns else 0
    kpis.append(create_kpi_card("Active Brands", active_brands, None, "", "🏷️"))
    
    # Brand share trend
    brand_chart = html.Div("Brand analysis not available", style={'color': COLORS['muted'], 'textAlign': 'center', 'padding': '40px'})
    
    if 'BRAND' in filtered_df.columns and value_cols:
        brand_trend = filtered_df.groupby(['period_key', 'BRAND'])[value_cols[0]].sum().reset_index()
        brand_trend['total_by_period'] = brand_trend.groupby('period_key')[value_cols[0]].transform('sum')
        brand_trend['share'] = (brand_trend[value_cols[0]] / brand_trend['total_by_period']) * 100
        
        # Top 5 vs Others
        top_brands = filtered_df.groupby('BRAND')[value_cols[0]].sum().nlargest(5).index
        brand_trend['brand_group'] = brand_trend['BRAND'].apply(
            lambda x: x if x in top_brands else 'Others'
        )
        
        trend_summary = brand_trend.groupby(['period_key', 'brand_group'])['share'].sum().reset_index()
        
        fig = px.line(trend_summary, x='period_key', y='share', color='brand_group',
                     title="Market Share: Top 5 Brands vs Long Tail")
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_family='-apple-system, BlinkMacSystemFont, SF Pro Display, Segoe UI, sans-serif',
            title_font_size=18,
            title_font_color=COLORS['text']
        )
        brand_chart = dcc.Graph(figure=fig)
    
    # Growth table
    growth_table = html.Div("Growth analysis not available", style={'color': COLORS['muted'], 'textAlign': 'center', 'padding': '20px'})
    
    if value_cols:
        growth_df = calculate_yoy_growth(filtered_df, value_cols[0], ['BRAND'])
        if not growth_df.empty:
            growth_df = growth_df.sort_values('YoY_Growth', ascending=False).head(10)
            growth_table = dash_table.DataTable(
                data=growth_df.to_dict('records'),
                columns=[
                    {'name': 'Brand', 'id': 'Entity'},
                    {'name': 'YoY Growth (%)', 'id': 'YoY_Growth', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                    {'name': 'Latest Value', 'id': 'Latest_Value', 'type': 'numeric', 'format': {'specifier': ',.0f'}}
                ],
                style_cell={'textAlign': 'left', 'fontFamily': '-apple-system, BlinkMacSystemFont, SF Pro Display, Segoe UI, sans-serif'},
                style_header={'backgroundColor': COLORS['accent'], 'color': 'white', 'fontWeight': '600'},
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{YoY_Growth} >= 0'},
                        'backgroundColor': '#f0f9ff',
                        'color': COLORS['success']
                    },
                    {
                        'if': {'filter_query': '{YoY_Growth} < 0'},
                        'backgroundColor': '#fef2f2',
                        'color': COLORS['danger']
                    }
                ]
            )
    
    return html.Div([
        # KPI Grid
        html.Div(kpis, style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))',
            'gap': '20px',
            'marginBottom': '32px'
        }),
        
        # Brand Share Chart
        html.Div([
            brand_chart
        ], style={
            'backgroundColor': 'white',
            'borderRadius': '12px',
            'border': f'1px solid {COLORS["border"]}',
            'marginBottom': '24px',
            'padding': '20px'
        }),
        
        # Growth Table
        html.Div([
            html.H3("Fastest Growing Brands (YoY)", style={
                'fontSize': '18px',
                'fontWeight': '600',
                'color': COLORS['text'],
                'marginBottom': '16px'
            }),
            growth_table
        ], style={
            'backgroundColor': 'white',
            'borderRadius': '12px',
            'border': f'1px solid {COLORS["border"]}',
            'padding': '20px'
        })
    ])

def create_growth_tab(filtered_df, value_cols, volume_cols):
    """Create growth hotspots tab"""
    
    # Top growing variants
    variant_chart = html.Div("Variant analysis not available", 
                            style={'color': COLORS['muted'], 'textAlign': 'center', 'padding': '40px'})
    
    if 'VARIANT (SUB-BRAND)' in filtered_df.columns and value_cols:
        variant_growth = calculate_yoy_growth(filtered_df, value_cols[0], ['VARIANT (SUB-BRAND)'])
        if not variant_growth.empty:
            top_variants = variant_growth.nlargest(10, 'YoY_Growth')
            
            fig = px.bar(top_variants, x='YoY_Growth', y='Entity', orientation='h',
                        title="Top 10 Fastest Growing Variants (YoY Value Growth)")
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_family='-apple-system, BlinkMacSystemFont, SF Pro Display, Segoe UI, sans-serif',
                title_font_size=18,
                title_font_color=COLORS['text'],
                yaxis={'categoryorder': 'total ascending'}
            )
            variant_chart = dcc.Graph(figure=fig)
    
    # Brand momentum sparklines
    momentum_chart = html.Div("Brand momentum not available", 
                             style={'color': COLORS['muted'], 'textAlign': 'center', 'padding': '40px'})
    
    if 'BRAND' in filtered_df.columns and value_cols:
        brand_momentum = filtered_df.groupby(['BRAND', 'period_key'])[value_cols[0]].sum().reset_index()
        brands = brand_momentum['BRAND'].unique()[:12]  # Limit to 12 brands
        
        if len(brands) > 0:
            fig = make_subplots(
                rows=3, cols=4,
                subplot_titles=brands[:12],
                vertical_spacing=0.1
            )
            
            for i, brand in enumerate(brands):
                row = (i // 4) + 1
                col = (i % 4) + 1
                
                brand_data = brand_momentum[brand_momentum['BRAND'] == brand].sort_values('period_key')
                fig.add_trace(
                    go.Scatter(
                        x=brand_data['period_key'],
                        y=brand_data[value_cols[0]],
                        mode='lines+markers',
                        name=brand,
                        showlegend=False,
                        line=dict(width=2, color=COLORS['accent']),
                        marker=dict(size=4)
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                height=600,
                title_text="Brand Performance Sparklines",
                font_family='-apple-system, BlinkMacSystemFont, SF Pro Display, Segoe UI, sans-serif',
                title_font_size=18,
                title_font_color=COLORS['text'],
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            momentum_chart = dcc.Graph(figure=fig)
    
    # New entrants
    new_entrants_table = html.Div("New entrants analysis not available", 
                                 style={'color': COLORS['muted'], 'textAlign': 'center', 'padding': '20px'})
    
    if 'VARIANT (SUB-BRAND)' in filtered_df.columns and 'period_key' in filtered_df.columns:
        periods = sorted(filtered_df['period_key'].unique())
        recent_periods = periods[-6:] if len(periods) >= 6 else periods
        
        new_entrants = []
        for variant in filtered_df['VARIANT (SUB-BRAND)'].unique():
            variant_periods = filtered_df[filtered_df['VARIANT (SUB-BRAND)'] == variant]['period_key'].unique()
            first_appearance = min(variant_periods)
            if first_appearance in recent_periods:
                brand = filtered_df[filtered_df['VARIANT (SUB-BRAND)'] == variant]['BRAND'].iloc[0] if 'BRAND' in filtered_df.columns else 'N/A'
                new_entrants.append({
                    'Variant': variant,
                    'First_Appeared': first_appearance,
                    'Brand': brand
                })
        
        if new_entrants:
            new_entrants_df = pd.DataFrame(new_entrants)
            new_entrants_table = dash_table.DataTable(
                data=new_entrants_df.to_dict('records'),
                columns=[
                    {'name': 'Variant', 'id': 'Variant'},
                    {'name': 'Brand', 'id': 'Brand'},
                    {'name': 'First Appeared', 'id': 'First_Appeared'}
                ],
                style_cell={'textAlign': 'left', 'fontFamily': '-apple-system, BlinkMacSystemFont, SF Pro Display, Segoe UI, sans-serif'},
                style_header={'backgroundColor': COLORS['accent'], 'color': 'white', 'fontWeight': '600'}
            )
    
    return html.Div([
        # Variant Growth Chart
        html.Div([
            variant_chart
        ], style={
            'backgroundColor': 'white',
            'borderRadius': '12px',
            'border': f'1px solid {COLORS["border"]}',
            'marginBottom': '24px',
            'padding': '20px'
        }),
        
        # Brand Momentum
        html.Div([
            momentum_chart
        ], style={
            'backgroundColor': 'white',
            'borderRadius': '12px',
            'border': f'1px solid {COLORS["border"]}',
            'marginBottom': '24px',
            'padding': '20px'
        }),
        
        # New Entrants
        html.Div([
            html.H3("New Market Entrants (Last 6 Months)", style={
                'fontSize': '18px',
                'fontWeight': '600',
                'color': COLORS['text'],
                'marginBottom': '16px'
            }),
            new_entrants_table
        ], style={
            'backgroundColor': 'white',
            'borderRadius': '12px',
            'border': f'1px solid {COLORS["border"]}',
            'padding': '20px'
        })
    ])

def create_promo_tab(filtered_df, value_cols, units_cols):
    """Create promo effectiveness tab"""
    
    if 'PROMO/NO PROMO' not in filtered_df.columns:
        return html.Div("Promo/No Promo column not found in data", 
                       style={'color': COLORS['muted'], 'textAlign': 'center', 'padding': '40px'})
    
    # Promo trends
    charts = []
    
    if units_cols:
        promo_units = filtered_df.groupby(['period_key', 'PROMO/NO PROMO'])[units_cols[0]].sum().reset_index()
        fig_units = px.line(promo_units, x='period_key', y=units_cols[0], color='PROMO/NO PROMO',
                           title="Units Sales: Promo vs No-Promo Trend")
        fig_units.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_family='-apple-system, BlinkMacSystemFont, SF Pro Display, Segoe UI, sans-serif',
            title_font_size=18,
            title_font_color=COLORS['text']
        )
        charts.append(dcc.Graph(figure=fig_units))
    
    if value_cols:
        promo_value = filtered_df.groupby(['period_key', 'PROMO/NO PROMO'])[value_cols[0]].sum().reset_index()
        fig_value = px.line(promo_value, x='period_key', y=value_cols[0], color='PROMO/NO PROMO',
                           title="Value Sales: Promo vs No-Promo Trend")
        fig_value.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_family='-apple-system, BlinkMacSystemFont, SF Pro Display, Segoe UI, sans-serif',
            title_font_size=18,
            title_font_color=COLORS['text']
        )
        charts.append(dcc.Graph(figure=fig_value))
    
    # Promo uplift analysis
    uplift_table = html.Div("Uplift analysis not available", 
                           style={'color': COLORS['muted'], 'textAlign': 'center', 'padding': '20px'})
    
    if value_cols:
        promo_df = calculate_promo_uplift(filtered_df, value_cols[0])
        if not promo_df.empty:
            uplift_table = dash_table.DataTable(
                data=promo_df.round(2).to_dict('records'),
                columns=[
                    {'name': 'Brand', 'id': 'Brand'},
                    {'name': 'Promo Avg', 'id': 'Promo_Avg', 'type': 'numeric', 'format': {'specifier': ',.1f'}},
                    {'name': 'No Promo Avg', 'id': 'No_Promo_Avg', 'type': 'numeric', 'format': {'specifier': ',.1f'}},
                    {'name': 'Uplift (%)', 'id': 'Uplift_Percent', 'type': 'numeric', 'format': {'specifier': '.1f'}}
                ],
                style_cell={'textAlign': 'left', 'fontFamily': '-apple-system, BlinkMacSystemFont, SF Pro Display, Segoe UI, sans-serif'},
                style_header={'backgroundColor': COLORS['accent'], 'color': 'white', 'fontWeight': '600'},
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{Uplift_Percent} >= 0'},
                        'backgroundColor': '#f0f9ff',
                        'color': COLORS['success']
                    },
                    {
                        'if': {'filter_query': '{Uplift_Percent} < 0'},
                        'backgroundColor': '#fef2f2',
                        'color': COLORS['danger']
                    }
                ]
            )
    
    # Price distribution
    price_chart = html.Div("Price analysis not available", 
                          style={'color': COLORS['muted'], 'textAlign': 'center', 'padding': '40px'})
    
    if price_cols:
        fig_price = px.box(filtered_df, x='PROMO/NO PROMO', y=price_cols[0],
                          title=f"Price Distribution by Promo Status")
        fig_price.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_family='-apple-system, BlinkMacSystemFont, SF Pro Display, Segoe UI, sans-serif',
            title_font_size=18,
            title_font_color=COLORS['text']
        )
        price_chart = dcc.Graph(figure=fig_price)
    
    return html.Div([
        # Trend Charts
        html.Div(charts, style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(auto-fit, minmax(400px, 1fr))',
            'gap': '20px',
            'marginBottom': '24px'
        }),
        
        # Uplift Table
        html.Div([
            html.H3("Promo Uplift Analysis", style={
                'fontSize': '18px',
                'fontWeight': '600',
                'color': COLORS['text'],
                'marginBottom': '16px'
            }),
            uplift_table
        ], style={
            'backgroundColor': 'white',
            'borderRadius': '12px',
            'border': f'1px solid {COLORS["border"]}',
            'marginBottom': '24px',
            'padding': '20px'
        }),
        
        # Price Distribution
        html.Div([
            price_chart
        ], style={
            'backgroundColor': 'white',
            'borderRadius': '12px',
            'border': f'1px solid {COLORS["border"]}',
            'padding': '20px'
        })
    ])

def create_pack_tab(filtered_df, value_cols):
    """Create pack & form factor tab"""
    
    charts = []
    
    # Sales mix by weight
    if 'WEIGHT' in filtered_df.columns and value_cols:
        weight_mix = filtered_df.groupby(['period_key', 'WEIGHT'])[value_cols[0]].sum().reset_index()
        fig_weight = px.bar(weight_mix, x='period_key', y=value_cols[0], color='WEIGHT',
                           title="Sales Mix by Weight Over Time")
        fig_weight.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_family='-apple-system, BlinkMacSystemFont, SF Pro Display, Segoe UI, sans-serif',
            title_font_size=18,
            title_font_color=COLORS['text']
        )
        charts.append(dcc.Graph(figure=fig_weight))
    
    # Form/Pack share
    form_col = 'DEODORANT FORM/PACK.'
    if form_col in filtered_df.columns and value_cols:
        form_share = filtered_df.groupby(form_col)[value_cols[0]].sum().reset_index()
        fig_form = px.pie(form_share, names=form_col, values=value_cols[0],
                         title="Market Share by Form/Pack Type")
        fig_form.update_layout(
            font_family='-apple-system, BlinkMacSystemFont, SF Pro Display, Segoe UI, sans-serif',
            title_font_size=18,
            title_font_color=COLORS['text']
        )
        charts.append(dcc.Graph(figure=fig_form))
    
    # Price vs Weight scatter
    if price_cols and 'WEIGHT' in filtered_df.columns and value_cols:
        scatter_data = filtered_df.groupby('WEIGHT').agg({
            price_cols[0]: 'mean',
            value_cols[0]: 'sum'
        }).reset_index()
        
        fig_scatter = px.scatter(scatter_data, x='WEIGHT', y=price_cols[0], size=value_cols[0],
                               title="Price per Unit vs Weight (Bubble size = Total Value)")
        fig_scatter.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_family='-apple-system, BlinkMacSystemFont, SF Pro Display, Segoe UI, sans-serif',
            title_font_size=18,
            title_font_color=COLORS['text']
        )
        charts.append(dcc.Graph(figure=fig_scatter))
    
    if not charts:
        return html.Div("Pack & form factor analysis not available with current data", 
                       style={'color': COLORS['muted'], 'textAlign': 'center', 'padding': '40px'})
    
    return html.Div([
        html.Div(charts, style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(auto-fit, minmax(400px, 1fr))',
            'gap': '20px'
        })
    ])

def create_pricing_tab(filtered_df, value_cols):
    """Create pricing power tab"""
    
    if not price_cols or not value_cols or 'BRAND' not in filtered_df.columns:
        return html.Div("Pricing analysis requires price, value, and brand columns", 
                       style={'color': COLORS['muted'], 'textAlign': 'center', 'padding': '40px'})
    
    # Calculate price and sales growth for quadrant analysis
    pricing_analysis = []
    
    for brand in filtered_df['BRAND'].unique():
        brand_data = filtered_df[filtered_df['BRAND'] == brand]
        periods = sorted(brand_data['period_key'].unique())
        
        if len(periods) >= 13:  # Need 13+ periods for YoY
            # Get data for latest and 12 months ago
            latest = periods[-1]
            prev_yoy = periods[-13]
            
            latest_price = brand_data[brand_data['period_key'] == latest][price_cols[0]].mean()
            prev_price = brand_data[brand_data['period_key'] == prev_yoy][price_cols[0]].mean()
            
            latest_sales = brand_data[brand_data['period_key'] == latest][value_cols[0]].sum()
            prev_sales = brand_data[brand_data['period_key'] == prev_yoy][value_cols[0]].sum()
            
            if pd.notna(latest_price) and pd.notna(prev_price) and prev_price > 0 and prev_sales > 0:
                price_growth = ((latest_price - prev_price) / prev_price) * 100
                sales_growth = ((latest_sales - prev_sales) / prev_sales) * 100
                
                pricing_analysis.append({
                    'Brand': brand,
                    'Price_Growth': price_growth,
                    'Sales_Growth': sales_growth,
                    'Latest_Sales': latest_sales
                })
    
    if not pricing_analysis:
        return html.Div("Insufficient data for pricing power analysis (need 13+ periods)", 
                       style={'color': COLORS['muted'], 'textAlign': 'center', 'padding': '40px'})
    
    pricing_df = pd.DataFrame(pricing_analysis)
    
    # Quadrant chart
    fig = px.scatter(pricing_df, x='Price_Growth', y='Sales_Growth', size='Latest_Sales',
                     hover_name='Brand', title="Pricing Power Quadrant Analysis")
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant labels
    fig.add_annotation(x=max(pricing_df['Price_Growth'])*0.7, y=max(pricing_df['Sales_Growth'])*0.7, 
                      text="Premium Growth<br>(↑Price, ↑Sales)", showarrow=False, 
                      bgcolor="rgba(14, 159, 110, 0.1)", bordercolor=COLORS['success'])
    fig.add_annotation(x=min(pricing_df['Price_Growth'])*0.7, y=max(pricing_df['Sales_Growth'])*0.7, 
                      text="Volume Growth<br>(↓Price, ↑Sales)", showarrow=False, 
                      bgcolor="rgba(47, 128, 237, 0.1)", bordercolor=COLORS['accent'])
    fig.add_annotation(x=max(pricing_df['Price_Growth'])*0.7, y=min(pricing_df['Sales_Growth'])*0.7, 
                      text="Margin Focus<br>(↑Price, ↓Sales)", showarrow=False, 
                      bgcolor="rgba(245, 158, 11, 0.1)", bordercolor="#F59E0B")
    fig.add_annotation(x=min(pricing_df['Price_Growth'])*0.7, y=min(pricing_df['Sales_Growth'])*0.7, 
                      text="Under Pressure<br>(↓Price, ↓Sales)", showarrow=False, 
                      bgcolor="rgba(225, 29, 72, 0.1)", bordercolor=COLORS['danger'])
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_family='-apple-system, BlinkMacSystemFont, SF Pro Display, Segoe UI, sans-serif',
        title_font_size=18,
        title_font_color=COLORS['text'],
        xaxis_title="Price Growth (%)",
        yaxis_title="Sales Growth (%)"
    )
    
    # Premiumization opportunities
    premium_brands = pricing_df[
        (pricing_df['Price_Growth'] > 0) & (pricing_df['Sales_Growth'] > 0)
    ].sort_values('Sales_Growth', ascending=False)
    
    premium_table = html.Div("No premiumization opportunities identified", 
                            style={'color': COLORS['muted'], 'textAlign': 'center', 'padding': '20px'})
    
    if not premium_brands.empty:
        premium_table = dash_table.DataTable(
            data=premium_brands[['Brand', 'Price_Growth', 'Sales_Growth']].round(2).to_dict('records'),
            columns=[
                {'name': 'Brand', 'id': 'Brand'},
                {'name': 'Price Growth (%)', 'id': 'Price_Growth', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                {'name': 'Sales Growth (%)', 'id': 'Sales_Growth', 'type': 'numeric', 'format': {'specifier': '.1f'}}
            ],
            style_cell={'textAlign': 'left', 'fontFamily': '-apple-system, BlinkMacSystemFont, SF Pro Display, Segoe UI, sans-serif'},
            style_header={'backgroundColor': COLORS['success'], 'color': 'white', 'fontWeight': '600'}
        )
    
    return html.Div([
        # Quadrant Chart
        html.Div([
            dcc.Graph(figure=fig)
        ], style={
            'backgroundColor': 'white',
            'borderRadius': '12px',
            'border': f'1px solid {COLORS["border"]}',
            'marginBottom': '24px',
            'padding': '20px'
        }),
        
        # Premiumization Table
        html.Div([
            html.H3("Premiumization Opportunities", style={
                'fontSize': '18px',
                'fontWeight': '600',
                'color': COLORS['text'],
                'marginBottom': '16px'
            }),
            premium_table
        ], style={
            'backgroundColor': 'white',
            'borderRadius': '12px',
            'border': f'1px solid {COLORS["border"]}',
            'padding': '20px'
        })
    ])

def create_volatility_tab(filtered_df, units_cols, volume_cols):
    """Create volatility analysis tab"""
    
    if not units_cols or 'VARIANT (SUB-BRAND)' not in filtered_df.columns:
        return html.Div("Volatility analysis requires units and variant columns", 
                       style={'color': COLORS['muted'], 'textAlign': 'center', 'padding': '40px'})
    
    # Calculate coefficient of variation for each variant
    volatility_analysis = []
    
    for variant in filtered_df['VARIANT (SUB-BRAND)'].unique():
        variant_data = filtered_df[filtered_df['VARIANT (SUB-BRAND)'] == variant]
        units_series = variant_data[units_cols[0]].dropna()
        
        if len(units_series) > 3:  # Need at least 4 data points
            cv = (units_series.std() / units_series.mean()) * 100 if units_series.mean() > 0 else 0
            brand = variant_data['BRAND'].iloc[0] if 'BRAND' in variant_data.columns else 'N/A'
            volatility_analysis.append({
                'Variant': variant,
                'Brand': brand,
                'Mean_Units': units_series.mean(),
                'CV_Percent': cv
            })
    
    if not volatility_analysis:
        return html.Div("Insufficient data for volatility analysis", 
                       style={'color': COLORS['muted'], 'textAlign': 'center', 'padding': '40px'})
    
    volatility_df = pd.DataFrame(volatility_analysis)
    top_volatile = volatility_df.nlargest(15, 'CV_Percent')
    
    # Volatility chart
    fig = px.bar(top_volatile, x='CV_Percent', y='Variant', orientation='h', color='Brand',
                title="Most Volatile SKUs - Potential Outsourcing Opportunities")
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_family='-apple-system, BlinkMacSystemFont, SF Pro Display, Segoe UI, sans-serif',
        title_font_size=18,
        title_font_color=COLORS['text'],
        yaxis={'categoryorder': 'total ascending'}
    )
    
    # Volatility table
    volatility_table = dash_table.DataTable(
        data=top_volatile[['Variant', 'Brand', 'CV_Percent', 'Mean_Units']].round(2).to_dict('records'),
        columns=[
            {'name': 'Variant', 'id': 'Variant'},
            {'name': 'Brand', 'id': 'Brand'},
            {'name': 'Volatility (CV %)', 'id': 'CV_Percent', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'Avg Units', 'id': 'Mean_Units', 'type': 'numeric', 'format': {'specifier': ',.1f'}}
        ],
        style_cell={'textAlign': 'left', 'fontFamily': '-apple-system, BlinkMacSystemFont, SF Pro Display, Segoe UI, sans-serif'},
        style_header={'backgroundColor': COLORS['accent'], 'color': 'white', 'fontWeight': '600'}
    )
    
    return html.Div([
        # Volatility Chart
        html.Div([
            dcc.Graph(figure=fig)
        ], style={
            'backgroundColor': 'white',
            'borderRadius': '12px',
            'border': f'1px solid {COLORS["border"]}',
            'marginBottom': '24px',
            'padding': '20px'
        }),
        
        # Volatility Table
        html.Div([
            html.H3("Volatility Rankings", style={
                'fontSize': '18px',
                'fontWeight': '600',
                'color': COLORS['text'],
                'marginBottom': '16px'
            }),
            volatility_table
        ], style={
            'backgroundColor': 'white',
            'borderRadius': '12px',
            'border': f'1px solid {COLORS["border"]}',
            'padding': '20px'
        })
    ])

def create_privatelabel_tab(filtered_df, value_cols):
    """Create private label analysis tab"""
    
    # Detect private label products
    if 'MANUFACTURER' not in filtered_df.columns or 'BRAND' not in filtered_df.columns:
        return html.Div("Private label analysis requires manufacturer and brand columns", 
                       style={'color': COLORS['muted'], 'textAlign': 'center', 'padding': '40px'})
    
    filtered_df = filtered_df.copy()
    filtered_df['is_private_label'] = (
        filtered_df['MANUFACTURER'].str.contains('Private Label', case=False, na=False) |
        filtered_df['BRAND'].str.contains('PL|Private', case=False, na=False)
    )
    
    if not filtered_df['is_private_label'].any():
        return html.Div([
            html.H3("No Private Label Products Detected", style={
                'fontSize': '18px',
                'fontWeight': '600',
                'color': COLORS['text'],
                'textAlign': 'center',
                'marginBottom': '16px'
            }),
            html.P([
                "Detection criteria: Manufacturer contains 'Private Label' or Brand contains 'PL'/'Private'. ",
                "If your data uses different naming conventions, the detection logic may need adjustment."
            ], style={'color': COLORS['muted'], 'textAlign': 'center'})
        ], style={'padding': '40px'})
    
    # PL vs Branded trend
    if value_cols:
        pl_trend = filtered_df.groupby(['period_key', 'is_private_label'])[value_cols[0]].sum().reset_index()
        pl_trend['label_type'] = pl_trend['is_private_label'].map({True: 'Private Label', False: 'Branded'})
        
        # Calculate market share
        pl_trend['total_by_period'] = pl_trend.groupby('period_key')[value_cols[0]].transform('sum')
        pl_trend['market_share'] = (pl_trend[value_cols[0]] / pl_trend['total_by_period']) * 100
        
        fig = px.area(pl_trend, x='period_key', y='market_share', color='label_type',
                     title="Private Label vs Branded Market Share Evolution")
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_family='-apple-system, BlinkMacSystemFont, SF Pro Display, Segoe UI, sans-serif',
            title_font_size=18,
            title_font_color=COLORS['text'],
            xaxis_title="Period",
            yaxis_title="Market Share (%)",
            yaxis=dict(range=[0, 100])
        )
        
        # PL analysis summary
        pl_summary = filtered_df.groupby('is_private_label').agg({
            value_cols[0]: ['sum', 'mean'],
            'BRAND': 'nunique'
        }).round(2)
        pl_summary.index = ['Branded', 'Private Label']
        
        # Convert to format suitable for dash table
        pl_summary_flat = []
        for idx in pl_summary.index:
            pl_summary_flat.append({
                'Type': idx,
                'Total_Value': pl_summary.loc[idx, (value_cols[0], 'sum')],
                'Avg_Value': pl_summary.loc[idx, (value_cols[0], 'mean')],
                'Unique_Brands': pl_summary.loc[idx, ('BRAND', 'nunique')]
            })
        
        pl_table = dash_table.DataTable(
            data=pl_summary_flat,
            columns=[
                {'name': 'Type', 'id': 'Type'},
                {'name': 'Total Value', 'id': 'Total_Value', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                {'name': 'Avg Value', 'id': 'Avg_Value', 'type': 'numeric', 'format': {'specifier': ',.1f'}},
                {'name': 'Unique Brands', 'id': 'Unique_Brands', 'type': 'numeric', 'format': {'specifier': '.0f'}}
            ],
            style_cell={'textAlign': 'left', 'fontFamily': '-apple-system, BlinkMacSystemFont, SF Pro Display, Segoe UI, sans-serif'},
            style_header={'backgroundColor': COLORS['accent'], 'color': 'white', 'fontWeight': '600'}
        )
        
        return html.Div([
            # PL Trend Chart
            html.Div([
                dcc.Graph(figure=fig)
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '12px',
                'border': f'1px solid {COLORS["border"]}',
                'marginBottom': '24px',
                'padding': '20px'
            }),
            
            # PL Summary Table
            html.Div([
                html.H3("Private Label vs Branded Summary", style={
                    'fontSize': '18px',
                    'fontWeight': '600',
                    'color': COLORS['text'],
                    'marginBottom': '16px'
                }),
                pl_table
            ], style={
                'backgroundColor': 'white',
                'borderRadius': '12px',
                'border': f'1px solid {COLORS["border"]}',
                'padding': '20px'
            })
        ])

# Load data on app start
if __name__ == '__main__':
    print("Loading data...")
    if load_and_process_data():
        print("✅ Data loaded successfully")
        print(f"📊 Schema detected: {len(dimensions)} dimensions, {len(measures)} measures")
        if 'period_key' in df.columns:
            periods = sorted(df['period_key'].unique())
            print(f"📅 Period range: {periods[0]} to {periods[-1]} ({len(periods)} periods)")
        print(f"🔍 Price columns: {price_cols}")
        print("🚀 Starting server on http://0.0.0.0:3000")
        app.run_server(host='0.0.0.0', port=3000, debug=False)
    else:
        print("❌ Failed to load data. Please ensure DEO-pivot.csv is in the app directory.")

# Custom CSS for Apple-grade styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif;
                margin: 0;
                background-color: #ffffff;
                color: #0B0B0C;
            }
            
            .dash-dropdown .Select-control {
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                background-color: white;
            }
            
            .dash-dropdown .Select-control:hover {
                border-color: #2F80ED;
            }
            
            .dash-dropdown .Select--multi .Select-value {
                background-color: #2F80ED;
                color: white;
                border-radius: 4px;
            }
            
            .tab-content {
                border: none !important;
            }
            
            .tab-selected {
                border-bottom: 3px solid #2F80ED !important;
                color: #2F80ED !important;
                background-color: rgba(47, 128, 237, 0.05) !important;
            }
            
            .tab-parent {
                border-bottom: 1px solid #E5E7EB !important;
            }
            
            .dash-table-container {
                border-radius: 8px;
                overflow: hidden;
                border: 1px solid #E5E7EB;
            }
            
            .dash-spreadsheet-container .dash-spreadsheet-inner {
                border-radius: 8px;
            }
            
            .rc-slider-track {
                background-color: #2F80ED;
            }
            
            .rc-slider-handle {
                border-color: #2F80ED;
            }
            
            .rc-slider-handle:active {
                border-color: #2F80ED;
                box-shadow: 0 0 5px #2F80ED;
            }
            
            /* Loading states */
            .dash-loading {
                opacity: 0.7;
            }
            
            /* Smooth transitions */
            .dash-graph {
                transition: all 0.3s ease;
            }
            
            /* Error states */
            .dash-error {
                background-color: #FEF2F2;
                border: 1px solid #FECACA;
                border-radius: 8px;
                padding: 16px;
                color: #DC2626;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''