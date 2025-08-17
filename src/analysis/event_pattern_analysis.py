"""
Event Pattern Analysis for Demand Forecasting
- Define exact event periods based on demand spikes
- Identify common characteristics across event periods
- Analyze patterns for 2023-2024 prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta

# Set matplotlib to use Agg backend for saving plots
plt.switch_backend('Agg')

DATA_DIR = Path('C:/projects/smartphone-supplychain/data')

def load_data():
    """Load all required data"""
    # Load demand data
    with sqlite3.connect(DATA_DIR / 'demand_train.db') as conn:
        demand = pd.read_sql_query('SELECT * FROM demand_train', conn)
    
    demand['date'] = pd.to_datetime(demand['date'])
    
    # City-to-country mapping
    country_map = {
        'Washington_DC': 'USA', 'New_York': 'USA', 'Chicago': 'USA', 'Dallas': 'USA',
        'Berlin': 'DEU', 'Munich': 'DEU', 'Frankfurt': 'DEU', 'Hamburg': 'DEU',
        'Paris': 'FRA', 'Lyon': 'FRA', 'Marseille': 'FRA', 'Toulouse': 'FRA',
        'Seoul': 'KOR', 'Busan': 'KOR', 'Incheon': 'KOR', 'Gwangju': 'KOR',
        'Tokyo': 'JPN', 'Osaka': 'JPN', 'Nagoya': 'JPN', 'Fukuoka': 'JPN',
        'Manchester': 'GBR', 'London': 'GBR', 'Birmingham': 'GBR', 'Glasgow': 'GBR',
        'Ottawa': 'CAN', 'Toronto': 'CAN', 'Vancouver': 'CAN', 'Montreal': 'CAN',
        'Canberra': 'AUS', 'Sydney': 'AUS', 'Melbourne': 'AUS', 'Brisbane': 'AUS',
        'Brasilia': 'BRA', 'Sao_Paulo': 'BRA', 'Rio_de_Janeiro': 'BRA', 'Salvador': 'BRA',
        'Pretoria': 'ZAF', 'Johannesburg': 'ZAF', 'Cape_Town': 'ZAF', 'Durban': 'ZAF'
    }
    
    demand['country'] = demand['city'].map(country_map)
    
    # Aggregate daily demand per country
    country_daily = (
        demand.groupby(['date', 'country'])['demand']
        .sum()
        .reset_index()
    )
    
    # Load external factors
    consumer_conf = pd.read_csv(DATA_DIR / 'consumer_confidence_processed.csv', parse_dates=['date'])
    oil_price = pd.read_csv(DATA_DIR / 'oil_price_processed.csv', parse_dates=['date'])
    currency = pd.read_csv(DATA_DIR / 'currency_processed.csv', parse_dates=['date'])
    marketing = pd.read_csv(DATA_DIR / 'marketing_spend.csv', parse_dates=['date'])
    weather = pd.read_csv(DATA_DIR / 'weather.csv', parse_dates=['date'])
    
    return country_daily, consumer_conf, oil_price, currency, marketing, weather

def detect_event_periods(country_daily, threshold_multiplier=1.2, min_duration_days=30, max_duration_days=90):
    """
    Detect exact event periods based on demand spikes
    
    Args:
        country_daily: Daily demand data by country
        threshold_multiplier: Demand must be this many times higher than baseline
        min_duration_days: Minimum event duration in days
        max_duration_days: Maximum event duration in days
    
    Returns:
        List of detected event periods with start/end dates
    """
    event_periods = []
    
    # Known event countries and years from EDA
    event_candidates = [
        ('KOR', 2018), ('JPN', 2019), ('USA', 2020), ('USA', 2021), ('KOR', 2022)
    ]
    
    for country, year in event_candidates:
        # Filter data for specific country and year
        mask = (country_daily['country'] == country) & (country_daily['date'].dt.year == year)
        year_data = country_daily[mask].sort_values('date')
        
        if year_data.empty:
            continue
            
        # Calculate baseline demand (median of non-peak periods)
        demand_values = year_data['demand'].values
        baseline = np.median(demand_values)
        threshold = baseline * threshold_multiplier
        
        # Find periods where demand exceeds threshold
        peak_mask = year_data['demand'] > threshold
        peak_dates = year_data[peak_mask]['date'].tolist()
        
        if not peak_dates:
            # If no peaks found, try with lower threshold
            threshold = baseline * 1.3
            peak_mask = year_data['demand'] > threshold
            peak_dates = year_data[peak_mask]['date'].tolist()
            
        if not peak_dates:
            # If still no peaks, use top 20% of demand days
            top_20_percentile = year_data['demand'].quantile(0.8)
            peak_mask = year_data['demand'] > top_20_percentile
            peak_dates = year_data[peak_mask]['date'].tolist()
        
        if not peak_dates:
            continue
            
        # Group consecutive peak dates into periods
        periods = []
        current_period = [peak_dates[0]]
        
        for i in range(1, len(peak_dates)):
            if (peak_dates[i] - peak_dates[i-1]).days <= 7:  # Allow 7-day gaps
                current_period.append(peak_dates[i])
            else:
                if len(current_period) >= min_duration_days // 7:  # Rough estimate
                    periods.append(current_period)
                current_period = [peak_dates[i]]
        
        if len(current_period) >= min_duration_days // 7:
            periods.append(current_period)
        
        # Select the longest period within max_duration_days
        for period in periods:
            if min_duration_days <= len(period) <= max_duration_days:
                start_date = min(period)
                end_date = max(period)
                
                # Calculate event characteristics
                event_data = year_data[
                    (year_data['date'] >= start_date) & 
                    (year_data['date'] <= end_date)
                ]
                
                avg_demand = event_data['demand'].mean()
                peak_demand = event_data['demand'].max()
                demand_increase = (avg_demand / baseline - 1) * 100
                
                event_periods.append({
                    'country': country,
                    'year': year,
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration_days': (end_date - start_date).days + 1,
                    'baseline_demand': baseline,
                    'avg_event_demand': avg_demand,
                    'peak_demand': peak_demand,
                    'demand_increase_pct': demand_increase
                })
                break
    
    return event_periods

def analyze_event_characteristics(event_periods, country_daily, consumer_conf, oil_price, currency, marketing, weather):
    """
    Analyze common characteristics across event periods
    """
    characteristics = []
    
    # Calculate global statistics for scaling
    if not consumer_conf.empty:
        if 'country' in consumer_conf.columns:
            global_conf_mean = consumer_conf['confidence_index'].mean()
            global_conf_std = consumer_conf['confidence_index'].std()
        else:
            global_conf_mean = consumer_conf['confidence_index'].mean()
            global_conf_std = consumer_conf['confidence_index'].std()
        print(f"Global consumer confidence - Mean: {global_conf_mean:.2f}, Std: {global_conf_std:.2f}")
    
    for event in event_periods:
        country = event['country']
        start_date = event['start_date']
        end_date = event['end_date']
        year = event['year']
        
        # Define comparison periods (3 months before and after event)
        pre_start = start_date - timedelta(days=90)
        pre_end = start_date - timedelta(days=1)
        post_start = end_date + timedelta(days=1)
        post_end = end_date + timedelta(days=90)
        
        # Non-event period: 3 months before + 3 months after
        non_event_before = (
            (country_daily['country'] == country) & 
            (country_daily['date'] >= pre_start) & 
            (country_daily['date'] <= pre_end)
        )
        
        non_event_after = (
            (country_daily['country'] == country) & 
            (country_daily['date'] >= post_start) & 
            (country_daily['date'] <= post_end)
        )
        
        # Combine before and after non-event periods
        non_event_mask = non_event_before | non_event_after
        non_event_data = country_daily[non_event_mask]
        
        # Event period data
        event_mask = (
            (country_daily['country'] == country) & 
            (country_daily['date'] >= start_date) & 
            (country_daily['date'] <= end_date)
        )
        event_data = country_daily[event_mask]
        
        # External factors analysis
        char = {
            'country': country,
            'year': event['year'],
            'start_date': start_date,
            'end_date': end_date,
            'duration_days': event['duration_days'],
            'demand_increase_pct': event['demand_increase_pct']
        }
        
        # Consumer Confidence (with scaling)
        if not consumer_conf.empty:
            # Check if consumer_conf has country column
            if 'country' in consumer_conf.columns:
                event_conf_data = consumer_conf[
                    (consumer_conf['country'] == country) & 
                    (consumer_conf['date'] >= start_date) & 
                    (consumer_conf['date'] <= end_date)
                ]['confidence_index']
                
                non_event_conf_data = consumer_conf[
                    (consumer_conf['country'] == country) & 
                    ((consumer_conf['date'] >= pre_start) & (consumer_conf['date'] <= pre_end) |
                     (consumer_conf['date'] >= post_start) & (consumer_conf['date'] <= post_end))
                ]['confidence_index']
            else:
                # If no country column, use global data
                event_conf_data = consumer_conf[
                    (consumer_conf['date'] >= start_date) & 
                    (consumer_conf['date'] <= end_date)
                ]['confidence_index']
                
                non_event_conf_data = consumer_conf[
                    ((consumer_conf['date'] >= pre_start) & (consumer_conf['date'] <= pre_end) |
                     (consumer_conf['date'] >= post_start) & (consumer_conf['date'] <= post_end))
                ]['confidence_index']
            
            # Raw values
            event_conf_raw = event_conf_data.mean()
            non_event_conf_raw = non_event_conf_data.mean()
            
            # Scaled values (z-score)
            event_conf_scaled = (event_conf_raw - global_conf_mean) / global_conf_std if global_conf_std > 0 else 0
            non_event_conf_scaled = (non_event_conf_raw - global_conf_mean) / global_conf_std if global_conf_std > 0 else 0
            
            # Percentage change
            conf_change_pct = ((event_conf_raw - non_event_conf_raw) / non_event_conf_raw * 100) if non_event_conf_raw != 0 else np.nan
            
            char['consumer_conf_event_raw'] = event_conf_raw
            char['consumer_conf_non_event_raw'] = non_event_conf_raw
            char['consumer_conf_event_scaled'] = event_conf_scaled
            char['consumer_conf_non_event_scaled'] = non_event_conf_scaled
            char['consumer_conf_change_raw'] = event_conf_raw - non_event_conf_raw if not pd.isna(event_conf_raw) and not pd.isna(non_event_conf_raw) else np.nan
            char['consumer_conf_change_scaled'] = event_conf_scaled - non_event_conf_scaled if not pd.isna(event_conf_scaled) and not pd.isna(non_event_conf_scaled) else np.nan
            char['consumer_conf_change_pct'] = conf_change_pct
        
        # Oil Price
        if not oil_price.empty:
            event_oil = oil_price[
                (oil_price['date'] >= start_date) & 
                (oil_price['date'] <= end_date)
            ]['brent_usd'].mean()
            
            non_event_oil = oil_price[
                ((oil_price['date'] >= pre_start) & (oil_price['date'] <= pre_end) |
                 (oil_price['date'] >= post_start) & (oil_price['date'] <= post_end))
            ]['brent_usd'].mean()
            
            char['oil_price_event'] = event_oil
            char['oil_price_non_event'] = non_event_oil
            char['oil_price_change'] = event_oil - non_event_oil if not pd.isna(event_oil) and not pd.isna(non_event_oil) else np.nan
        
        # Currency
        if not currency.empty:
            # Map countries to their currency columns
            currency_columns = {
                'DEU': 'EUR=X', 'FRA': 'EUR=X',
                'KOR': 'KRW=X', 'JPN': 'JPY=X', 
                'GBR': 'GBP=X', 'CAN': 'CAD=X',
                'AUS': 'AUD=X', 'BRA': 'BRL=X', 
                'ZAF': 'ZAR=X'
            }
            
            if country == 'USA':
                # USD is the base currency, so it's always 1.0
                char['currency_event'] = 1.0
                char['currency_non_event'] = 1.0
                char['currency_change'] = 0.0
            else:
                currency_col = currency_columns.get(country, 'EUR=X')
                
                if currency_col in currency.columns:
                    event_currency = currency[
                        (currency['date'] >= start_date) & 
                        (currency['date'] <= end_date)
                    ][currency_col].mean()
                    
                    non_event_currency = currency[
                        ((currency['date'] >= pre_start) & (currency['date'] <= pre_end) |
                         (currency['date'] >= post_start) & (currency['date'] <= post_end))
                    ][currency_col].mean()
                    
                    char['currency_event'] = event_currency
                    char['currency_non_event'] = non_event_currency
                    char['currency_change'] = event_currency - non_event_currency if not pd.isna(event_currency) and not pd.isna(non_event_currency) else np.nan
                else:
                    char['currency_event'] = np.nan
                    char['currency_non_event'] = np.nan
                    char['currency_change'] = np.nan
        
        # Marketing Spend
        if not marketing.empty:
            # Check if marketing has country column
            if 'country' in marketing.columns:
                event_marketing = marketing[
                    (marketing['country'] == country) & 
                    (marketing['date'] >= start_date) & 
                    (marketing['date'] <= end_date)
                ]['spend_usd'].sum()
                
                non_event_marketing = marketing[
                    (marketing['country'] == country) & 
                    ((marketing['date'] >= pre_start) & (marketing['date'] <= pre_end) |
                     (marketing['date'] >= post_start) & (marketing['date'] <= post_end))
                ]['spend_usd'].sum()
            else:
                # If no country column, use global data
                event_marketing = marketing[
                    (marketing['date'] >= start_date) & 
                    (marketing['date'] <= end_date)
                ]['spend_usd'].sum()
                
                non_event_marketing = marketing[
                    ((marketing['date'] >= pre_start) & (marketing['date'] <= pre_end) |
                     (marketing['date'] >= post_start) & (marketing['date'] <= post_end))
                ]['spend_usd'].sum()
            
            char['marketing_event'] = event_marketing
            char['marketing_non_event'] = non_event_marketing
            char['marketing_change'] = event_marketing - non_event_marketing if not pd.isna(event_marketing) and not pd.isna(non_event_marketing) else np.nan
        
        # Weather (temperature)
        if not weather.empty:
            # Check if weather has country column
            if 'country' in weather.columns:
                event_weather = weather[
                    (weather['country'] == country) & 
                    (weather['date'] >= start_date) & 
                    (weather['date'] <= end_date)
                ]['avg_temp'].mean()
                
                non_event_weather = weather[
                    (weather['country'] == country) & 
                    ((weather['date'] >= pre_start) & (weather['date'] <= pre_end) |
                     (weather['date'] >= post_start) & (weather['date'] <= post_end))
                ]['avg_temp'].mean()
            else:
                # If no country column, use global data
                event_weather = weather[
                    (weather['date'] >= start_date) & 
                    (weather['date'] <= end_date)
                ]['avg_temp'].mean()
                
                non_event_weather = weather[
                    ((weather['date'] >= pre_start) & (weather['date'] <= pre_end) |
                     (weather['date'] >= post_start) & (weather['date'] <= post_end))
                ]['avg_temp'].mean()
            
            char['weather_event'] = event_weather
            char['weather_non_event'] = non_event_weather
            char['weather_change'] = event_weather - non_event_weather if not pd.isna(event_weather) and not pd.isna(non_event_weather) else np.nan
        
        characteristics.append(char)
    
    return characteristics

def visualize_event_analysis(event_periods, characteristics):
    """Create visualizations for event analysis"""
    
    print("Creating event periods overview...")
    
    # 1. Event periods overview
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Event timeline
    for event in event_periods:
        ax1.barh(f"{event['country']} {event['year']}", 
                event['duration_days'], 
                left=event['start_date'], 
                height=0.6, 
                alpha=0.7,
                label=f"Demand +{event['demand_increase_pct']:.1f}%")
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Event Periods')
    ax1.set_title('Detected Event Periods Timeline')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Demand increase comparison
    countries = [f"{e['country']} {e['year']}" for e in event_periods]
    increases = [e['demand_increase_pct'] for e in event_periods]
    
    bars = ax2.bar(countries, increases, color='skyblue', alpha=0.7)
    ax2.set_ylabel('Demand Increase (%)')
    ax2.set_title('Demand Increase During Event Periods')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, increase in zip(bars, increases):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{increase:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(DATA_DIR / 'event_periods_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Creating external factors comparison...")
    
    # 2. External factors comparison
    if characteristics:
        char_df = pd.DataFrame(characteristics)
        print(f"Characteristics DataFrame for visualization: {char_df.shape}")
        print(f"Available columns: {char_df.columns.tolist()}")
        
        # Create comparison plots for external factors
        factors = ['consumer_conf', 'oil_price', 'currency', 'marketing', 'weather']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, factor in enumerate(factors):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            print(f"Checking factor: {factor}")
            
            if factor == 'consumer_conf':
                # Use scaled values for consumer confidence
                non_event_col = f'{factor}_non_event_scaled'
                event_col = f'{factor}_event_scaled'
                change_col = f'{factor}_change_scaled'
                
                print(f"  - {non_event_col} in columns: {non_event_col in char_df.columns}")
                print(f"  - {event_col} in columns: {event_col in char_df.columns}")
                
                if non_event_col in char_df.columns and event_col in char_df.columns:
                    # Non-event vs Event comparison
                    x = np.arange(len(char_df))
                    width = 0.35
                    
                    non_event_values = char_df[non_event_col].fillna(0)
                    event_values = char_df[event_col].fillna(0)
                    
                    print(f"  - Non-event values: {non_event_values.values}")
                    print(f"  - Event values: {event_values.values}")
                    
                    ax.bar(x - width/2, non_event_values, width, label='Non-Event Period', alpha=0.7)
                    ax.bar(x + width/2, event_values, width, label='Event Period', alpha=0.7)
                    
                    ax.set_xlabel('Event Periods')
                    ax.set_ylabel('Consumer Confidence (Scaled)')
                    ax.set_title('Consumer Confidence Comparison (Scaled)')
                    ax.set_xticks(x)
                    ax.set_xticklabels([f"{row['country']} {row['year']}" for _, row in char_df.iterrows()], rotation=45)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    print(f"  - Missing columns for {factor}")
                    ax.text(0.5, 0.5, f'No data for {factor}', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{factor.replace("_", " ").title()} - No Data')
            else:
                non_event_col = f'{factor}_non_event'
                event_col = f'{factor}_event'
                change_col = f'{factor}_change'
                
                print(f"  - {non_event_col} in columns: {non_event_col in char_df.columns}")
                print(f"  - {event_col} in columns: {event_col in char_df.columns}")
                
                if non_event_col in char_df.columns and event_col in char_df.columns:
                    # Non-event vs Event comparison
                    x = np.arange(len(char_df))
                    width = 0.35
                    
                    non_event_values = char_df[non_event_col].fillna(0)
                    event_values = char_df[event_col].fillna(0)
                    
                    print(f"  - Non-event values: {non_event_values.values}")
                    print(f"  - Event values: {event_values.values}")
                    
                    ax.bar(x - width/2, non_event_values, width, label='Non-Event Period', alpha=0.7)
                    ax.bar(x + width/2, event_values, width, label='Event Period', alpha=0.7)
                    
                    ax.set_xlabel('Event Periods')
                    ax.set_ylabel(factor.replace('_', ' ').title())
                    ax.set_title(f'{factor.replace("_", " ").title()} Comparison')
                    ax.set_xticks(x)
                    ax.set_xticklabels([f"{row['country']} {row['year']}" for _, row in char_df.iterrows()], rotation=45)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    print(f"  - Missing columns for {factor}")
                    ax.text(0.5, 0.5, f'No data for {factor}', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{factor.replace("_", " ").title()} - No Data')
        
        # Remove empty subplots
        for i in range(len(factors), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(DATA_DIR / 'event_external_factors.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations completed!")
    else:
        print("No characteristics data available for visualization")

def visualize_consumer_confidence_by_year(consumer_conf, event_periods):
    """Create yearly consumer confidence visualizations with event periods marked"""
    
    print("Creating consumer confidence by year visualizations...")
    
    if consumer_conf.empty:
        print("No consumer confidence data available")
        return
    
    # Calculate global statistics for scaling
    if 'country' in consumer_conf.columns:
        global_conf_mean = consumer_conf['confidence_index'].mean()
        global_conf_std = consumer_conf['confidence_index'].std()
    else:
        global_conf_mean = consumer_conf['confidence_index'].mean()
        global_conf_std = consumer_conf['confidence_index'].std()
    
    print(f"Global consumer confidence - Mean: {global_conf_mean:.2f}, Std: {global_conf_std:.2f}")
    
    # Get unique years from event periods
    event_years = sorted(list(set([event['year'] for event in event_periods])))
    
    # Create one subplot per year
    fig, axes = plt.subplots(len(event_years), 1, figsize=(16, 4 * len(event_years)))
    if len(event_years) == 1:
        axes = [axes]
    
    for i, year in enumerate(event_years):
        ax = axes[i]
        
        # Filter data for this year
        year_mask = consumer_conf['date'].dt.year == year
        year_data = consumer_conf[year_mask].copy()
        
        if year_data.empty:
            ax.text(0.5, 0.5, f'No data for {year}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Consumer Confidence {year} - No Data')
            continue
        
        # Get unique countries
        if 'country' in year_data.columns:
            countries = sorted(year_data['country'].unique())
        else:
            # If no country column, treat as global data
            countries = ['Global']
            year_data['country'] = 'Global'
        
        # Plot each country's consumer confidence
        for country in countries:
            country_data = year_data[year_data['country'] == country].sort_values('date')
            
            if not country_data.empty:
                # Scale the confidence index
                scaled_conf = (country_data['confidence_index'] - global_conf_mean) / global_conf_std
                
                # Plot the line
                ax.plot(country_data['date'], scaled_conf, 
                       label=country, linewidth=2, alpha=0.8)
        
        # Mark event periods for this year
        year_events = [event for event in event_periods if event['year'] == year]
        
        for event in year_events:
            # Add shaded region for event period
            ax.axvspan(event['start_date'], event['end_date'], 
                      alpha=0.3, color='red', 
                      label=f"Event: {event['country']} (+{event['demand_increase_pct']:.1f}%)")
            
            # Add vertical lines at start and end
            ax.axvline(event['start_date'], color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax.axvline(event['end_date'], color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            # Add annotation
            ax.annotate(f"{event['country']}\n+{event['demand_increase_pct']:.1f}%", 
                       xy=(event['start_date'], ax.get_ylim()[1] * 0.9),
                       xytext=(10, 0), textcoords='offset points',
                       fontsize=10, color='red', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_title(f'Consumer Confidence {year} (Scaled)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Consumer Confidence (Z-score)')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Format x-axis
        ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
        
        # Set x-axis limits to the year
        ax.set_xlim(pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year}-12-31'))
    
    plt.tight_layout()
    plt.savefig(DATA_DIR / 'consumer_confidence_by_year.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Consumer confidence by year visualization completed!")

def visualize_marketing_spend_by_year(marketing, event_periods):
    """Create yearly marketing spend visualizations with event periods marked"""
    
    print("Creating marketing spend by year visualizations...")
    
    if marketing.empty:
        print("No marketing data available")
        return
    
    # Check if category column exists
    has_category = 'category' in marketing.columns
    print(f"Marketing data has category column: {has_category}")
    
    # Get unique years from event periods
    event_years = sorted(list(set([event['year'] for event in event_periods])))
    
    # Create one subplot per year
    fig, axes = plt.subplots(len(event_years), 1, figsize=(16, 4 * len(event_years)))
    if len(event_years) == 1:
        axes = [axes]
    
    for i, year in enumerate(event_years):
        ax = axes[i]
        
        # Filter data for this year
        year_mask = marketing['date'].dt.year == year
        year_data = marketing[year_mask].copy()
        
        if year_data.empty:
            ax.text(0.5, 0.5, f'No data for {year}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Marketing Spend {year} - No Data')
            continue
        
        # Get unique countries (show all available countries, not just event countries)
        if 'country' in year_data.columns:
            countries = sorted(year_data['country'].unique())
        else:
            # If no country column, treat as global data
            countries = ['Global']
            year_data['country'] = 'Global'
        
        print(f"Countries to plot for {year}: {countries}")
        
        # Plot each country's marketing spend (total across all categories)
        for country in countries:
            country_data = year_data[year_data['country'] == country]
            
            if not country_data.empty:
                # Aggregate by date (sum all categories for each date)
                daily_spend = country_data.groupby('date')['spend_usd'].sum().reset_index()
                
                # Plot the line
                ax.plot(daily_spend['date'], daily_spend['spend_usd'], 
                       label=country, linewidth=2, alpha=0.8, marker='o', markersize=3)
        
        # Mark event periods for this year
        year_events = [event for event in event_periods if event['year'] == year]
        
        for event in year_events:
            # Add shaded region for event period
            ax.axvspan(event['start_date'], event['end_date'], 
                      alpha=0.3, color='red', 
                      label=f"Event: {event['country']} (+{event['demand_increase_pct']:.1f}%)")
            
            # Add vertical lines at start and end
            ax.axvline(event['start_date'], color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax.axvline(event['end_date'], color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            # Add annotation
            ax.annotate(f"{event['country']}\n+{event['demand_increase_pct']:.1f}%", 
                       xy=(event['start_date'], ax.get_ylim()[1] * 0.9),
                       xytext=(10, 0), textcoords='offset points',
                       fontsize=10, color='red', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_title(f'Marketing Spend {year}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Marketing Spend (USD)')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Format x-axis
        ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
        
        # Set x-axis limits to the year
        ax.set_xlim(pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year}-12-31'))
    
    plt.tight_layout()
    plt.savefig(DATA_DIR / 'marketing_spend_by_year.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Marketing spend by year visualization completed!")

def visualize_currency_by_year(currency, event_periods):
    """Create yearly currency exchange rate visualizations with event periods marked"""
    
    print("Creating currency exchange rates by year visualizations...")
    
    if currency.empty:
        print("No currency data available")
        return
    
    # Currency columns (excluding date)
    currency_columns = ['EUR=X', 'KRW=X', 'JPY=X', 'GBP=X', 'CAD=X', 'AUD=X', 'BRL=X', 'ZAR=X']
    
    # Calculate scaling statistics for each currency
    currency_stats = {}
    for col in currency_columns:
        if col in currency.columns:
            currency_stats[col] = {
                'mean': currency[col].mean(),
                'std': currency[col].std()
            }
            print(f"{col} - Mean: {currency_stats[col]['mean']:.2f}, Std: {currency_stats[col]['std']:.2f}")
    
    # Get unique years from event periods
    event_years = sorted(list(set([event['year'] for event in event_periods])))
    
    # Create one subplot per year
    fig, axes = plt.subplots(len(event_years), 1, figsize=(16, 4 * len(event_years)))
    if len(event_years) == 1:
        axes = [axes]
    
    for i, year in enumerate(event_years):
        ax = axes[i]
        
        # Filter data for this year
        year_mask = currency['date'].dt.year == year
        year_data = currency[year_mask].copy()
        
        if year_data.empty:
            ax.text(0.5, 0.5, f'No data for {year}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Currency Exchange Rates {year} - No Data')
            continue
        
        # Plot each currency's exchange rate
        for col in currency_columns:
            if col in year_data.columns and col in currency_stats:
                # Get the data for this currency
                currency_data = year_data[['date', col]].dropna().sort_values('date')
                
                if not currency_data.empty:
                    # Scale the exchange rate
                    scaled_rate = (currency_data[col] - currency_stats[col]['mean']) / currency_stats[col]['std']
                    
                    # Plot the line
                    ax.plot(currency_data['date'], scaled_rate, 
                           label=col.replace('=X', ''), linewidth=2, alpha=0.8)
        
        # Mark event periods for this year
        year_events = [event for event in event_periods if event['year'] == year]
        
        for event in year_events:
            # Add shaded region for event period
            ax.axvspan(event['start_date'], event['end_date'], 
                      alpha=0.3, color='red', 
                      label=f"Event: {event['country']} (+{event['demand_increase_pct']:.1f}%)")
            
            # Add vertical lines at start and end
            ax.axvline(event['start_date'], color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax.axvline(event['end_date'], color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            # Add annotation
            ax.annotate(f"{event['country']}\n+{event['demand_increase_pct']:.1f}%", 
                       xy=(event['start_date'], ax.get_ylim()[1] * 0.9),
                       xytext=(10, 0), textcoords='offset points',
                       fontsize=10, color='red', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_title(f'Currency Exchange Rates {year} (Scaled)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Exchange Rate (Z-score)')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Format x-axis
        ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
        
        # Set x-axis limits to the year
        ax.set_xlim(pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year}-12-31'))
    
    plt.tight_layout()
    plt.savefig(DATA_DIR / 'currency_exchange_rates_by_year.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Currency exchange rates by year visualization completed!")

def visualize_consumer_confidence_by_year_country_scaled(consumer_conf, event_periods):
    """Create yearly consumer confidence visualizations with country-specific scaling"""
    
    print("Creating consumer confidence by year with country-specific scaling...")
    
    if consumer_conf.empty:
        print("No consumer confidence data available")
        return
    
    # Get unique years from event periods
    event_years = sorted(list(set([event['year'] for event in event_periods])))
    
    # Create one subplot per year
    fig, axes = plt.subplots(len(event_years), 1, figsize=(16, 4 * len(event_years)))
    if len(event_years) == 1:
        axes = [axes]
    
    for i, year in enumerate(event_years):
        ax = axes[i]
        
        # Filter data for this year
        year_mask = consumer_conf['date'].dt.year == year
        year_data = consumer_conf[year_mask].copy()
        
        if year_data.empty:
            ax.text(0.5, 0.5, f'No data for {year}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Consumer Confidence {year} - No Data')
            continue
        
        # Get unique countries
        if 'country' in year_data.columns:
            countries = sorted(year_data['country'].unique())
        else:
            # If no country column, treat as global data
            countries = ['Global']
            year_data['country'] = 'Global'
        
        print(f"Countries to plot for {year}: {countries}")
        
        # Plot each country's consumer confidence with country-specific scaling
        for country in countries:
            country_data = year_data[year_data['country'] == country].sort_values('date')
            
            if not country_data.empty:
                # Calculate country-specific statistics
                country_mean = country_data['confidence_index'].mean()
                country_std = country_data['confidence_index'].std()
                
                # Scale the confidence index (country-specific)
                scaled_conf = (country_data['confidence_index'] - country_mean) / country_std if country_std > 0 else 0
                
                # Plot the line
                ax.plot(country_data['date'], scaled_conf, 
                       label=f"{country} (μ={country_mean:.1f}, σ={country_std:.1f})", 
                       linewidth=2, alpha=0.8)
        
        # Mark event periods for this year
        year_events = [event for event in event_periods if event['year'] == year]
        
        for event in year_events:
            # Add shaded region for event period
            ax.axvspan(event['start_date'], event['end_date'], 
                      alpha=0.3, color='red', 
                      label=f"Event: {event['country']} (+{event['demand_increase_pct']:.1f}%)")
            
            # Add vertical lines at start and end
            ax.axvline(event['start_date'], color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax.axvline(event['end_date'], color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            # Add annotation
            ax.annotate(f"{event['country']}\n+{event['demand_increase_pct']:.1f}%", 
                       xy=(event['start_date'], ax.get_ylim()[1] * 0.9),
                       xytext=(10, 0), textcoords='offset points',
                       fontsize=10, color='red', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_title(f'Consumer Confidence {year} (Country-Specific Scaled)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Consumer Confidence (Z-score)')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Format x-axis
        ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
        
        # Set x-axis limits to the year
        ax.set_xlim(pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year}-12-31'))
    
    plt.tight_layout()
    plt.savefig(DATA_DIR / 'consumer_confidence_by_year_country_scaled.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Consumer confidence by year with country-specific scaling completed!")

def main():
    """Main analysis function"""
    print("Loading data...")
    country_daily, consumer_conf, oil_price, currency, marketing, weather = load_data()
    
    # Debug: Check data shapes
    print(f"Country daily data shape: {country_daily.shape}")
    print(f"Consumer confidence data shape: {consumer_conf.shape}")
    print(f"Oil price data shape: {oil_price.shape}")
    print(f"Currency data shape: {currency.shape}")
    print(f"Marketing data shape: {marketing.shape}")
    print(f"Weather data shape: {weather.shape}")
    
    # Debug: Check column names
    print(f"\nConsumer confidence columns: {consumer_conf.columns.tolist()}")
    print(f"Oil price columns: {oil_price.columns.tolist()}")
    print(f"Currency columns: {currency.columns.tolist()}")
    print(f"Marketing columns: {marketing.columns.tolist()}")
    print(f"Weather columns: {weather.columns.tolist()}")
    
    print("\nDetecting event periods...")
    event_periods = detect_event_periods(country_daily)
    
    print(f"Detected {len(event_periods)} event periods:")
    for event in event_periods:
        print(f"  {event['country']} {event['year']}: {event['start_date'].strftime('%Y-%m-%d')} to {event['end_date'].strftime('%Y-%m-%d')} "
              f"(Duration: {event['duration_days']} days, Demand +{event['demand_increase_pct']:.1f}%)")
    
    if not event_periods:
        print("No event periods detected! Check the detection logic.")
        return
    
    print("\nAnalyzing event characteristics...")
    characteristics = analyze_event_characteristics(
        event_periods, country_daily, consumer_conf, oil_price, currency, marketing, weather
    )
    
    print(f"Generated {len(characteristics)} characteristic records")
    
    # Debug: Check characteristics data
    if characteristics:
        char_df = pd.DataFrame(characteristics)
        print(f"Characteristics DataFrame shape: {char_df.shape}")
        print(f"Characteristics columns: {char_df.columns.tolist()}")
        
        # Check for non-null values in key columns
        for factor in ['consumer_conf', 'oil_price', 'currency', 'marketing', 'weather']:
            if factor == 'consumer_conf':
                # Check both raw and scaled values
                event_col_raw = f'{factor}_event_raw'
                non_event_col_raw = f'{factor}_non_event_raw'
                event_col_scaled = f'{factor}_event_scaled'
                non_event_col_scaled = f'{factor}_non_event_scaled'
                change_col_scaled = f'{factor}_change_scaled'
                change_col_pct = f'{factor}_change_pct'
                
                if event_col_raw in char_df.columns:
                    non_null_count = char_df[event_col_raw].notna().sum()
                    print(f"{event_col_raw}: {non_null_count}/{len(char_df)} non-null values")
                if event_col_scaled in char_df.columns:
                    non_null_count = char_df[event_col_scaled].notna().sum()
                    print(f"{event_col_scaled}: {non_null_count}/{len(char_df)} non-null values")
            else:
                event_col = f'{factor}_event'
                non_event_col = f'{factor}_non_event'
                change_col = f'{factor}_change'
                
                if event_col in char_df.columns:
                    non_null_count = char_df[event_col].notna().sum()
                    print(f"{event_col}: {non_null_count}/{len(char_df)} non-null values")
                if non_event_col in char_df.columns:
                    non_null_count = char_df[non_event_col].notna().sum()
                    print(f"{non_event_col}: {non_null_count}/{len(char_df)} non-null values")
    
    print("\nCreating visualizations...")
    visualize_event_analysis(event_periods, characteristics)
    visualize_consumer_confidence_by_year(consumer_conf, event_periods)
    visualize_marketing_spend_by_year(marketing, event_periods)
    visualize_currency_by_year(currency, event_periods)
    visualize_consumer_confidence_by_year_country_scaled(consumer_conf, event_periods)
    
    # Save detailed results
    event_df = pd.DataFrame(event_periods)
    char_df = pd.DataFrame(characteristics)
    
    event_df.to_csv(DATA_DIR / 'detected_event_periods.csv', index=False)
    char_df.to_csv(DATA_DIR / 'event_characteristics.csv', index=False)
    
    print(f"\nResults saved:")
    print(f"  - {DATA_DIR / 'detected_event_periods.csv'}")
    print(f"  - {DATA_DIR / 'event_characteristics.csv'}")
    print(f"  - {DATA_DIR / 'event_periods_analysis.png'}")
    print(f"  - {DATA_DIR / 'event_external_factors.png'}")
    print(f"  - {DATA_DIR / 'consumer_confidence_by_year.png'}")
    print(f"  - {DATA_DIR / 'marketing_spend_by_year.png'}")
    print(f"  - {DATA_DIR / 'currency_exchange_rates_by_year.png'}")
    print(f"  - {DATA_DIR / 'consumer_confidence_by_year_country_scaled.png'}")
    
    # Print summary statistics
    if not char_df.empty:
        print("\n=== Event Characteristics Summary ===")
        for factor in ['consumer_conf', 'oil_price', 'currency', 'marketing', 'weather']:
            if factor == 'consumer_conf':
                change_col_scaled = f'{factor}_change_scaled'
                change_col_pct = f'{factor}_change_pct'
                if change_col_scaled in char_df.columns:
                    changes = char_df[change_col_scaled].dropna()
                    if not changes.empty:
                        print(f"{factor.replace('_', ' ').title()} (Scaled):")
                        print(f"  Average change: {changes.mean():.2f}")
                        print(f"  Range: {changes.min():.2f} to {changes.max():.2f}")
                        print(f"  Positive changes: {(changes > 0).sum()}/{len(changes)}")
                    else:
                        print(f"{factor.replace('_', ' ').title()} (Scaled): No valid changes found")
                if change_col_pct in char_df.columns:
                    changes = char_df[change_col_pct].dropna()
                    if not changes.empty:
                        print(f"{factor.replace('_', ' ').title()} (Percentage):")
                        print(f"  Average change: {changes.mean():.2f}")
                        print(f"  Range: {changes.min():.2f} to {changes.max():.2f}")
                        print(f"  Positive changes: {(changes > 0).sum()}/{len(changes)}")
                    else:
                        print(f"{factor.replace('_', ' ').title()} (Percentage): No valid changes found")
            else:
                change_col = f'{factor}_change'
                if change_col in char_df.columns:
                    changes = char_df[change_col].dropna()
                    if not changes.empty:
                        print(f"{factor.replace('_', ' ').title()}:")
                        print(f"  Average change: {changes.mean():.2f}")
                        print(f"  Range: {changes.min():.2f} to {changes.max():.2f}")
                        print(f"  Positive changes: {(changes > 0).sum()}/{len(changes)}")
                    else:
                        print(f"{factor.replace('_', ' ').title()}: No valid changes found")
                else:
                    print(f"{factor.replace('_', ' ').title()}: Column not found")

if __name__ == "__main__":
    main() 