import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime, timedelta
import io
import base64

# Page configuration
st.set_page_config(
    page_title="CMO/CDMO Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        text-align: center;
    }
    .kpi-container {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    .kpi-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
    }
    .kpi-change {
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data(uploaded_file):
    """Load and process the uploaded CSV file"""
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Auto-detect dimensions and measures
        dimensions = []
        measures = []
        
        for col in df.columns:
            if df[col].dtype in ['object', 'string']:
                dimensions.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                measures.append(col)
        
        # Create period_key from PERIOD or ORDERED PERIOD
        period_cols = [col for col in df.columns if 'PERIOD' in col.upper()]
        if period_cols:
            df['period_key'] = df[period_cols[0]].apply(create_period_key)
        
        # Remove rows with all null measures
        measure_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['units', 'volume', 'value', 'price'])]
        df = df.dropna(subset=measure_cols, how='all')
        
        return df, dimensions, measures
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def create_period_key(period_str):
    """Convert period string to YYYY-MM format"""
    if pd.isna(period_str):
        return None
    
    try:
        # Handle formats like "001_May 2023" or "May 2023"
        match = re.search(r'(\w+)\s+(\d{4})', str(period_str))
        if match:
            month_name = match.group(1)
            year = match.group(2)
            
            month_map = {
                'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
            }
            
            month_num = month_map.get(month_name[:3], '01')
            return f"{year}-{month_num}"
    except:
        pass
    
    return str(period_str)

def apply_filters(df, filters):
    """Apply selected filters to dataframe"""
    filtered_df = df.copy()
    
    for column, values in filters.items():
        if values and 'All' not in values:
            filtered_df = filtered_df[filtered_df[column].isin(values)]
    
    return filtered_df

def calculate_yoy_growth(df, value_col, period_col):
    """Calculate year-over-year growth"""
    if period_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    
    # Sort by period
    df_sorted = df.sort_values(period_col)
    
    # Group by relevant dimensions and calculate YoY
    growth_data = []
    
    for brand in df['BRAND'].unique() if 'BRAND' in df.columns else [None]:
        brand_data = df_sorted[df_sorted['BRAND'] == brand] if brand else df_sorted
        
        # Get latest and previous year data
        periods = sorted(brand_data[period_col].unique())
        if len(periods) >= 2:
            latest_period = periods[-1]
            prev_period = periods[-2]
            
            latest_value = brand_data[brand_data[period_col] == latest_period][value_col].sum()
            prev_value = brand_data[brand_data[period_col] == prev_period][value_col].sum()
            
            if prev_value > 0:
                growth = ((latest_value - prev_value) / prev_value) * 100
                growth_data.append({
                    'Brand': brand,
                    'Latest_Value': latest_value,
                    'Previous_Value': prev_value,
                    'YoY_Growth': growth
                })
    
    return pd.DataFrame(growth_data)

def create_kpi_card(title, value, change=None, format_func=None):
    """Create a KPI card"""
    if format_func:
        formatted_value = format_func(value)
    else:
        formatted_value = f"{value:,.0f}" if isinstance(value, (int, float)) else str(value)
    
    change_html = ""
    if change is not None:
        color = "green" if change >= 0 else "red"
        symbol = "↑" if change >= 0 else "↓"
        change_html = f'<div class="kpi-change" style="color: {color};">{symbol} {abs(change):.1f}%</div>'
    
    st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-value">{formatted_value}</div>
            <div class="kpi-label">{title}</div>
            {change_html}
        </div>
    """, unsafe_allow_html=True)

def download_csv(df, filename):
    """Create download link for CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

def detect_private_label(df):
    """Detect private label products"""
    if 'MANUFACTURER' not in df.columns or 'BRAND' not in df.columns:
        return df
    
    df = df.copy()
    df['is_private_label'] = (
        df['MANUFACTURER'].str.contains('Private Label', case=False, na=False) |
        df['BRAND'].str.contains('PL|Private', case=False, na=False)
    )
    return df

def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🏭 CMO/CDMO Dashboard - Contract Manufacturing & Marketing Analytics</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV data file", type=['csv'])
    
    if uploaded_file is None:
        st.info("👆 Please upload a CSV file to begin analysis")
        st.markdown("""
        ### Expected Data Format:
        - **Dimensions**: Market, Manufacturer, Brand, Variant, Target User, Form/Pack, Promo/No Promo, Weight, etc.
        - **Measures**: Units, Volume, Value, Distribution metrics, Pricing metrics
        - **Time**: Period column (e.g., "May 2023", "001_May 2023")
        """)
        return
    
    # Load data
    df, dimensions, measures = load_and_process_data(uploaded_file)
    
    if df is None:
        return
    
    # Display schema info
    with st.expander("📋 Data Schema Detected", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Rows**: {len(df):,}")
        with col2:
            st.write(f"**Dimensions**: {len(dimensions)}")
        with col3:
            st.write(f"**Measures**: {len(measures)}")
        
        st.write("**Dimensions detected:**", ", ".join(dimensions[:10]))
        st.write("**Measures detected:**", ", ".join(measures[:10]))
        
        if 'period_key' in df.columns:
            periods = sorted(df['period_key'].dropna().unique())
            st.write(f"**Period range**: {periods[0] if periods else 'N/A'} to {periods[-1] if periods else 'N/A'}")
    
    # Sidebar filters
    st.sidebar.header("🎛️ Filters")
    
    filters = {}
    
    # Period filter
    if 'period_key' in df.columns:
        periods = sorted(df['period_key'].dropna().unique())
        period_range = st.sidebar.select_slider(
            "Period Range",
            options=periods,
            value=(periods[0], periods[-1]) if len(periods) > 1 else (periods[0], periods[0])
        )
        df = df[(df['period_key'] >= period_range[0]) & (df['period_key'] <= period_range[1])]
    
    # Other filters
    filter_columns = ['MARKET', 'BRAND', 'MANUFACTURER', 'PROMO/NO PROMO', 'TARGET USER']
    for col in filter_columns:
        if col in df.columns:
            unique_values = ['All'] + sorted(df[col].dropna().unique().tolist())
            selected = st.sidebar.multiselect(
                col.replace('/', ' / ').title(),
                unique_values,
                default=['All']
            )
            if 'All' not in selected:
                filters[col] = selected
    
    # Apply filters
    filtered_df = apply_filters(df, filters)
    
    if len(filtered_df) == 0:
        st.warning("No data available with current filters")
        return
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Overview", 
        "🚀 Growth Hotspots", 
        "🎯 Promo Effectiveness",
        "📦 Pack & Form Factor",
        "💰 Pricing Power",
        "📊 Volatility",
        "🏪 Private Label"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("Key Performance Indicators")
        
        # KPI Cards
        kpi_cols = st.columns(4)
        
        value_cols = [col for col in filtered_df.columns if 'value' in col.lower()]
        volume_cols = [col for col in filtered_df.columns if 'volume' in col.lower()]
        units_cols = [col for col in filtered_df.columns if 'units' in col.lower()]
        
        with kpi_cols[0]:
            if value_cols:
                total_value = filtered_df[value_cols[0]].sum()
                create_kpi_card("Total Value", total_value, format_func=lambda x: f"₽{x:,.0f}K")
        
        with kpi_cols[1]:
            if volume_cols:
                total_volume = filtered_df[volume_cols[0]].sum()
                create_kpi_card("Total Volume", total_volume, format_func=lambda x: f"{x:,.0f}K L")
        
        with kpi_cols[2]:
            if units_cols:
                total_units = filtered_df[units_cols[0]].sum()
                create_kpi_card("Total Units", total_units, format_func=lambda x: f"{x:,.0f}K")
        
        with kpi_cols[3]:
            unique_brands = filtered_df['BRAND'].nunique() if 'BRAND' in filtered_df.columns else 0
            create_kpi_card("Active Brands", unique_brands)
        
        # Brand share trend
        if 'BRAND' in filtered_df.columns and value_cols:
            st.subheader("Brand Share Trend")
            
            brand_trend = filtered_df.groupby(['period_key', 'BRAND'])[value_cols[0]].sum().reset_index()
            brand_trend['total_by_period'] = brand_trend.groupby('period_key')[value_cols[0]].transform('sum')
            brand_trend['share'] = (brand_trend[value_cols[0]] / brand_trend['total_by_period']) * 100
            
            # Top 5 vs Others
            top_brands = filtered_df.groupby('BRAND')[value_cols[0]].sum().nlargest(5).index
            brand_trend['brand_group'] = brand_trend['BRAND'].apply(
                lambda x: x if x in top_brands else 'Others'
            )
            
            trend_summary = brand_trend.groupby(['period_key', 'brand_group'])['share'].sum().reset_index()
            
            fig = px.line(
                trend_summary, 
                x='period_key', 
                y='share', 
                color='brand_group',
                title="Top 5 Brands vs Long Tail Market Share"
            )
            fig.update_layout(xaxis_title="Period", yaxis_title="Market Share (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Growth table
        if value_cols and 'period_key' in filtered_df.columns:
            st.subheader("Fastest Growing Brands (YoY)")
            
            growth_df = calculate_yoy_growth(filtered_df, value_cols[0], 'period_key')
            if not growth_df.empty:
                growth_df = growth_df.sort_values('YoY_Growth', ascending=False).head(10)
                
                st.dataframe(
                    growth_df[['Brand', 'YoY_Growth', 'Latest_Value']].round(2),
                    column_config={
                        'YoY_Growth': st.column_config.NumberColumn("YoY Growth (%)", format="%.1f%%"),
                        'Latest_Value': st.column_config.NumberColumn("Latest Value", format="%.0f")
                    }
                )
                
                st.markdown(download_csv(growth_df, "fastest_growing_brands.csv"), unsafe_allow_html=True)
    
    # Tab 2: Growth Hotspots
    with tab2:
        st.header("Growth Hotspots Analysis")
        
        if 'VARIANT (SUB-BRAND)' in filtered_df.columns and value_cols:
            # Top growing variants
            st.subheader("Top 10 Fastest Growing Variants")
            
            variant_growth = calculate_yoy_growth(filtered_df, value_cols[0], 'period_key')
            if not variant_growth.empty:
                top_variants = variant_growth.nlargest(10, 'YoY_Growth')
                
                fig = px.bar(
                    top_variants,
                    x='YoY_Growth',
                    y='Brand',
                    orientation='h',
                    title="YoY Value Growth by Variant"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Brand momentum sparklines
        if 'BRAND' in filtered_df.columns and value_cols:
            st.subheader("Brand Momentum")
            
            brand_momentum = filtered_df.groupby(['BRAND', 'period_key'])[value_cols[0]].sum().reset_index()
            
            # Create small multiples
            brands = brand_momentum['BRAND'].unique()[:12]  # Limit to 12 for readability
            
            fig = make_subplots(
                rows=3, cols=4,
                subplot_titles=brands,
                vertical_spacing=0.1
            )
            
            for i, brand in enumerate(brands):
                row = (i // 4) + 1
                col = (i % 4) + 1
                
                brand_data = brand_momentum[brand_momentum['BRAND'] == brand]
                fig.add_trace(
                    go.Scatter(
                        x=brand_data['period_key'],
                        y=brand_data[value_cols[0]],
                        mode='lines',
                        name=brand,
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(height=600, title_text="Brand Performance Sparklines")
            st.plotly_chart(fig, use_container_width=True)
        
        # New entrants
        if 'period_key' in filtered_df.columns:
            st.subheader("New Market Entrants (Last 6 Months)")
            
            periods = sorted(filtered_df['period_key'].unique())
            recent_periods = periods[-6:] if len(periods) >= 6 else periods
            
            new_entrants = []
            for variant in filtered_df['VARIANT (SUB-BRAND)'].unique() if 'VARIANT (SUB-BRAND)' in filtered_df.columns else []:
                variant_periods = filtered_df[filtered_df['VARIANT (SUB-BRAND)'] == variant]['period_key'].unique()
                first_appearance = min(variant_periods)
                if first_appearance in recent_periods:
                    new_entrants.append({
                        'Variant': variant,
                        'First_Appeared': first_appearance,
                        'Brand': filtered_df[filtered_df['VARIANT (SUB-BRAND)'] == variant]['BRAND'].iloc[0] if 'BRAND' in filtered_df.columns else 'N/A'
                    })
            
            if new_entrants:
                new_entrants_df = pd.DataFrame(new_entrants)
                st.dataframe(new_entrants_df)
                st.markdown(download_csv(new_entrants_df, "new_entrants.csv"), unsafe_allow_html=True)
            else:
                st.info("No new entrants detected in recent periods")
    
    # Tab 3: Promo Effectiveness
    with tab3:
        st.header("Promotional Effectiveness Analysis")
        
        promo_col = 'PROMO/NO PROMO'
        if promo_col in filtered_df.columns:
            # Promo vs No-Promo trends
            col1, col2 = st.columns(2)
            
            with col1:
                if units_cols:
                    promo_units = filtered_df.groupby(['period_key', promo_col])[units_cols[0]].sum().reset_index()
                    fig = px.line(
                        promo_units,
                        x='period_key',
                        y=units_cols[0],
                        color=promo_col,
                        title="Units Sales: Promo vs No-Promo"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if value_cols:
                    promo_value = filtered_df.groupby(['period_key', promo_col])[value_cols[0]].sum().reset_index()
                    fig = px.line(
                        promo_value,
                        x='period_key',
                        y=value_cols[0],
                        color=promo_col,
                        title="Value Sales: Promo vs No-Promo"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Promo uplift calculation
            st.subheader("Promo Uplift Analysis")
            
            promo_analysis = []
            for brand in filtered_df['BRAND'].unique() if 'BRAND' in filtered_df.columns else [None]:
                brand_data = filtered_df[filtered_df['BRAND'] == brand] if brand else filtered_df
                
                promo_sales = brand_data[brand_data[promo_col].str.contains('PROMO', na=False)][value_cols[0]].mean() if value_cols else 0
                no_promo_sales = brand_data[brand_data[promo_col].str.contains('NO PROMO', na=False)][value_cols[0]].mean() if value_cols else 0
                
                if no_promo_sales > 0:
                    uplift = ((promo_sales - no_promo_sales) / no_promo_sales) * 100
                    promo_analysis.append({
                        'Brand': brand,
                        'Promo_Avg': promo_sales,
                        'No_Promo_Avg': no_promo_sales,
                        'Uplift_Percent': uplift
                    })
            
            if promo_analysis:
                promo_df = pd.DataFrame(promo_analysis)
                st.dataframe(
                    promo_df.round(2),
                    column_config={
                        'Uplift_Percent': st.column_config.NumberColumn("Uplift (%)", format="%.1f%%")
                    }
                )
                st.markdown(download_csv(promo_df, "promo_uplift.csv"), unsafe_allow_html=True)
            
            # Price distribution by promo status
            price_cols = [col for col in filtered_df.columns if 'price' in col.lower()]
            if price_cols:
                st.subheader("Price Distribution by Promo Status")
                
                fig = px.box(
                    filtered_df,
                    x=promo_col,
                    y=price_cols[0],
                    title=f"Price Distribution: {price_cols[0]}"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Promo/No Promo column not found in data")
    
    # Tab 4: Pack & Form Factor
    with tab4:
        st.header("Pack & Form Factor Analysis")
        
        weight_col = 'WEIGHT'
        form_col = 'DEODORANT FORM/PACK.'
        
        if weight_col in filtered_df.columns and value_cols:
            # Sales mix by weight
            st.subheader("Sales Mix by Weight")
            
            weight_mix = filtered_df.groupby(['period_key', weight_col])[value_cols[0]].sum().reset_index()
            fig = px.bar(
                weight_mix,
                x='period_key',
                y=value_cols[0],
                color=weight_col,
                title="Sales Value by Weight Category Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if form_col in filtered_df.columns and value_cols:
            # Form/Pack share
            st.subheader("Market Share by Form/Pack Type")
            
            form_share = filtered_df.groupby(form_col)[value_cols[0]].sum().reset_index()
            fig = px.pie(
                form_share,
                names=form_col,
                values=value_cols[0],
                title="Market Share by Deodorant Form/Pack"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Price vs Weight scatter
        price_cols = [col for col in filtered_df.columns if 'price per unit' in col.lower()]
        if price_cols and weight_col in filtered_df.columns and value_cols:
            st.subheader("Price vs Weight Analysis")
            
            # Create bubble chart
            scatter_data = filtered_df.groupby([weight_col]).agg({
                price_cols[0]: 'mean',
                value_cols[0]: 'sum'
            }).reset_index()
            
            fig = px.scatter(
                scatter_data,
                x=weight_col,
                y=price_cols[0],
                size=value_cols[0],
                title="Price per Unit vs Weight (Bubble size = Total Value)",
                hover_data=[value_cols[0]]
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Pricing Power
    with tab5:
        st.header("Pricing Power Analysis")
        
        if price_cols and value_cols and 'period_key' in filtered_df.columns:
            # Calculate price and sales growth
            pricing_analysis = []
            
            for brand in filtered_df['BRAND'].unique() if 'BRAND' in filtered_df.columns else [None]:
                brand_data = filtered_df[filtered_df['BRAND'] == brand] if brand else filtered_df
                
                periods = sorted(brand_data['period_key'].unique())
                if len(periods) >= 2:
                    latest = periods[-1]
                    previous = periods[-2]
                    
                    latest_price = brand_data[brand_data['period_key'] == latest][price_cols[0]].mean()
                    prev_price = brand_data[brand_data['period_key'] == previous][price_cols[0]].mean()
                    
                    latest_sales = brand_data[brand_data['period_key'] == latest][value_cols[0]].sum()
                    prev_sales = brand_data[brand_data['period_key'] == previous][value_cols[0]].sum()
                    
                    if prev_price > 0 and prev_sales > 0:
                        price_growth = ((latest_price - prev_price) / prev_price) * 100
                        sales_growth = ((latest_sales - prev_sales) / prev_sales) * 100
                        
                        pricing_analysis.append({
                            'Brand': brand,
                            'Price_Growth': price_growth,
                            'Sales_Growth': sales_growth,
                            'Latest_Sales': latest_sales
                        })
            
            if pricing_analysis:
                pricing_df = pd.DataFrame(pricing_analysis)
                
                # Quadrant analysis
                st.subheader("Price vs Sales Growth Quadrant Analysis")
                
                fig = px.scatter(
                    pricing_df,
                    x='Price_Growth',
                    y='Sales_Growth',
                    size='Latest_Sales',
                    hover_name='Brand',
                    title="Pricing Power Quadrant (Size = Latest Sales Volume)"
                )
                
                # Add quadrant lines
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                
                # Add quadrant labels
                fig.add_annotation(x=5, y=5, text="Premium Growth<br>(↑Price, ↑Sales)", showarrow=False, bgcolor="lightgreen")
                fig.add_annotation(x=-5, y=5, text="Volume Growth<br>(↓Price, ↑Sales)", showarrow=False, bgcolor="lightblue")
                fig.add_annotation(x=5, y=-5, text="Margin Focus<br>(↑Price, ↓Sales)", showarrow=False, bgcolor="lightyellow")
                fig.add_annotation(x=-5, y=-5, text="Under Pressure<br>(↓Price, ↓Sales)", showarrow=False, bgcolor="lightcoral")
                
                fig.update_layout(
                    xaxis_title="Price Growth (%)",
                    yaxis_title="Sales Growth (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Premiumization opportunities
                st.subheader("Premiumization Opportunities")
                premium_brands = pricing_df[
                    (pricing_df['Price_Growth'] > 0) & (pricing_df['Sales_Growth'] > 0)
                ].sort_values('Sales_Growth', ascending=False)
                
                if not premium_brands.empty:
                    st.dataframe(
                        premium_brands[['Brand', 'Price_Growth', 'Sales_Growth']].round(2),
                        column_config={
                            'Price_Growth': st.column_config.NumberColumn("Price Growth (%)", format="%.1f%%"),
                            'Sales_Growth': st.column_config.NumberColumn("Sales Growth (%)", format="%.1f%%")
                        }
                    )
                    st.markdown(download_csv(premium_brands, "premiumization_opportunities.csv"), unsafe_allow_html=True)
                else:
                    st.info("No clear premiumization opportunities identified")
    
    # Tab 6: Volatility Analysis
    with tab6:
        st.header("Demand Volatility Analysis")
        
        if units_cols and 'VARIANT (SUB-BRAND)' in filtered_df.columns:
            # Calculate coefficient of variation for each variant
            volatility_analysis = []
            
            for variant in filtered_df['VARIANT (SUB-BRAND)'].unique():
                variant_data = filtered_df[filtered_df['VARIANT (SUB-BRAND)'] == variant]
                units_series = variant_data[units_cols[0]].dropna()
                
                if len(units_series) > 3:  # Need at least 4 data points
                    cv = (units_series.std() / units_series.mean()) * 100 if units_series.mean() > 0 else 0
                    volatility_analysis.append({
                        'Variant': variant,
                        'Mean_Units': units_series.mean(),
                        'Std_Units': units_series.std(),
                        'CV_Percent': cv,
                        'Brand': variant_data['BRAND'].iloc[0] if 'BRAND' in variant_data.columns else 'N/A'
                    })
            
            if volatility_analysis:
                volatility_df = pd.DataFrame(volatility_analysis)
                
                # Top 15 most volatile SKUs
                st.subheader("Top 15 Most Volatile SKUs (Coefficient of Variation)")
                
                top_volatile = volatility_df.nlargest(15, 'CV_Percent')
                
                fig = px.bar(
                    top_volatile,
                    x='CV_Percent',
                    y='Variant',
                    orientation='h',
                    color='Brand',
                    title="Most Volatile SKUs - Potential Outsourcing Opportunities"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Volatility table
                st.dataframe(
                    top_volatile[['Variant', 'Brand', 'CV_Percent', 'Mean_Units']].round(2),
                    column_config={
                        'CV_Percent': st.column_config.NumberColumn("Volatility (CV %)", format="%.1f%%"),
                        'Mean_Units': st.column_config.NumberColumn("Avg Units", format="%.1f")
                    }
                )
                st.markdown(download_csv(top_volatile, "volatility_analysis.csv"), unsafe_allow_html=True)
                
                # Volatility sparklines
                st.subheader("Demand Pattern Sparklines (Top 12 Volatile SKUs)")
                
                top_12_variants = top_volatile.head(12)['Variant'].tolist()
                
                fig = make_subplots(
                    rows=3, cols=4,
                    subplot_titles=top_12_variants,
                    vertical_spacing=0.15
                )
                
                for i, variant in enumerate(top_12_variants):
                    row = (i // 4) + 1
                    col = (i % 4) + 1
                    
                    variant_data = filtered_df[filtered_df['VARIANT (SUB-BRAND)'] == variant].sort_values('period_key')
                    
                    fig.add_trace(
                        go.Scatter(
                            x=variant_data['period_key'],
                            y=variant_data[units_cols[0]],
                            mode='lines+markers',
                            name=variant,
                            showlegend=False,
                            line=dict(width=2),
                            marker=dict(size=4)
                        ),
                        row=row, col=col
                    )
                
                fig.update_layout(height=700, title_text="Demand Volatility Patterns")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for volatility analysis")
    
    # Tab 7: Private Label Analysis
    with tab7:
        st.header("Private Label vs Branded Analysis")
        
        # Detect private label products
        df_with_pl = detect_private_label(filtered_df)
        
        if df_with_pl['is_private_label'].any():
            # PL vs Branded trend
            if value_cols:
                st.subheader("Private Label vs Branded Market Share Trend")
                
                pl_trend = df_with_pl.groupby(['period_key', 'is_private_label'])[value_cols[0]].sum().reset_index()
                pl_trend['label_type'] = pl_trend['is_private_label'].map({True: 'Private Label', False: 'Branded'})
                
                # Calculate market share
                pl_trend['total_by_period'] = pl_trend.groupby('period_key')[value_cols[0]].transform('sum')
                pl_trend['market_share'] = (pl_trend[value_cols[0]] / pl_trend['total_by_period']) * 100
                
                fig = px.area(
                    pl_trend,
                    x='period_key',
                    y='market_share',
                    color='label_type',
                    title="Private Label vs Branded Market Share Evolution"
                )
                fig.update_layout(
                    xaxis_title="Period",
                    yaxis_title="Market Share (%)",
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Private label analysis table
            st.subheader("Private Label Performance Analysis")
            
            pl_analysis = []
            
            # Overall PL vs Branded comparison
            pl_summary = df_with_pl.groupby('is_private_label').agg({
                value_cols[0]: ['sum', 'mean'] if value_cols else ['count'],
                units_cols[0]: ['sum', 'mean'] if units_cols else ['count'],
                'BRAND': 'nunique' if 'BRAND' in df_with_pl.columns else 'count'
            }).round(2) if value_cols or units_cols else pd.DataFrame()
            
            if not pl_summary.empty:
                pl_summary.index = ['Branded', 'Private Label']
                st.dataframe(pl_summary)
            
            # Top private label manufacturers
            if 'MANUFACTURER' in df_with_pl.columns:
                st.subheader("Top Private Label Manufacturers")
                
                pl_manufacturers = df_with_pl[df_with_pl['is_private_label']].groupby('MANUFACTURER').agg({
                    value_cols[0]: 'sum' if value_cols else 'count',
                    'BRAND': 'nunique' if 'BRAND' in df_with_pl.columns else 'count'
                }).sort_values(value_cols[0] if value_cols else 'BRAND', ascending=False).head(10)
                
                if not pl_manufacturers.empty:
                    st.dataframe(pl_manufacturers)
                    st.markdown(download_csv(pl_manufacturers.reset_index(), "private_label_manufacturers.csv"), unsafe_allow_html=True)
            
            # PL penetration by category
            if 'TYPE-BEAUTY/COSMETIC' in df_with_pl.columns:
                st.subheader("Private Label Penetration by Category")
                
                category_pl = df_with_pl.groupby(['TYPE-BEAUTY/COSMETIC', 'is_private_label'])[value_cols[0]].sum().reset_index() if value_cols else pd.DataFrame()
                
                if not category_pl.empty:
                    category_pl['label_type'] = category_pl['is_private_label'].map({True: 'Private Label', False: 'Branded'})
                    
                    fig = px.bar(
                        category_pl,
                        x='TYPE-BEAUTY/COSMETIC',
                        y=value_cols[0],
                        color='label_type',
                        title="Private Label vs Branded Sales by Category",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No private label products detected in the current dataset")
            st.markdown("""
            **Private Label Detection Criteria:**
            - Manufacturer contains "Private Label"
            - Brand contains "PL" or "Private"
            
            If you have private label products with different naming conventions, 
            please adjust the detection logic in the code.
            """)
    
    # Footer with additional insights
    st.markdown("---")
    st.markdown("""
    ### 💡 Dashboard Insights Summary
    
    **Key Actions for CMO/CDMO Teams:**
    1. **Growth Opportunities**: Focus on fastest-growing variants and new market entrants
    2. **Promo Strategy**: Optimize promotional effectiveness based on uplift analysis
    3. **Capacity Planning**: Monitor volatile SKUs for potential outsourcing opportunities
    4. **Pricing Power**: Identify premiumization opportunities in the pricing quadrant
    5. **Private Label**: Track PL vs branded dynamics for competitive positioning
    
    **Data Quality Notes:**
    - All calculations handle missing values gracefully
    - YoY comparisons require at least 2 periods of data
    - Volatility analysis needs minimum 4 data points per variant
    """)

if __name__ == "__main__":
    main()