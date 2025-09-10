#!/usr/bin/env python3
"""
COVID-19 Data Analytics Project
Analysis of Johns Hopkins COVID-19 dataset (Feb 9, 2020)

This script performs comprehensive analysis including:
- Data inspection and cleaning
- Bar charts for top affected regions
- Scatter plots for relationships between metrics
- Pie charts for regional distribution
- Analysis of China's dominance in early phase
- Regional mortality and recovery rate insights
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import os
from datetime import datetime

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

def download_covid_data():
    """Download COVID-19 data for February 9, 2020"""
    
    # URLs for Johns Hopkins COVID-19 data as of Feb 9, 2020
    base_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/"
    date_file = "02-09-2020.csv"
    
    url = base_url + date_file
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    data_path = "data/covid_02_09_2020.csv"
    
    try:
        print(f"Downloading COVID-19 data for February 9, 2020...")
        response = requests.get(url)
        response.raise_for_status()
        
        # Save to file
        with open(data_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Data successfully downloaded to {data_path}")
        return data_path
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        
        # Create sample data if download fails
        print("Creating sample data for demonstration...")
        sample_data = {
            'Province/State': ['Hubei', 'Guangdong', 'Zhejiang', 'Henan', 'Hunan', 
                             'Anhui', 'Jiangxi', 'Jiangsu', 'Chongqing', 'Shandong',
                             'Taiwan', 'Singapore', 'Thailand', 'South Korea', 'Japan',
                             'Malaysia', 'Vietnam', 'Australia', 'Cambodia', 'Philippines'],
            'Country/Region': ['China', 'China', 'China', 'China', 'China', 
                             'China', 'China', 'China', 'China', 'China',
                             'Taiwan', 'Singapore', 'Thailand', 'South Korea', 'Japan',
                             'Malaysia', 'Vietnam', 'Australia', 'Cambodia', 'Philippines'],
            'Confirmed': [29631, 1241, 1145, 1212, 1001, 950, 913, 593, 525, 506,
                         18, 40, 32, 27, 26, 18, 15, 15, 1, 3],
            'Deaths': [871, 2, 0, 19, 4, 6, 1, 0, 5, 1,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            'Recovered': [1795, 52, 296, 32, 69, 34, 22, 18, 15, 6,
                         1, 2, 10, 3, 1, 1, 6, 5, 0, 1]
        }
        
        df_sample = pd.DataFrame(sample_data)
        df_sample.to_csv(data_path, index=False)
        return data_path

def load_and_inspect_data(file_path):
    """Load and perform initial inspection of COVID-19 data"""
    
    print("=" * 60)
    print("COVID-19 DATA INSPECTION (February 9, 2020)")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(file_path)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Display first few rows
    print("\nFirst 10 rows:")
    print(df.head(10))
    
    # Basic statistics
    print("\nBasic statistics:")
    print(df.describe())
    
    # Check for missing values
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    return df

def clean_data(df):
    """Clean and prepare data for analysis"""
    
    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)
    
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    # Fill missing values
    numeric_cols = ['Confirmed', 'Deaths', 'Recovered']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    
    # Handle missing geographic data
    df_clean['Province/State'].fillna('', inplace=True)
    
    # Standardize country names
    df_clean['Country/Region'] = df_clean['Country/Region'].replace('Mainland China', 'China')
    
    # Create combined location column
    df_clean['Location'] = df_clean.apply(
        lambda row: f"{row['Province/State']}, {row['Country/Region']}" 
        if row['Province/State'] else row['Country/Region'], 
        axis=1
    )
    
    # Calculate recovery and mortality rates
    df_clean['Mortality_Rate'] = np.where(
        df_clean['Confirmed'] > 0, 
        (df_clean['Deaths'] / df_clean['Confirmed']) * 100, 
        0
    )
    
    df_clean['Recovery_Rate'] = np.where(
        df_clean['Confirmed'] > 0, 
        (df_clean['Recovered'] / df_clean['Confirmed']) * 100, 
        0
    )
    
    print(f"Cleaned dataset shape: {df_clean.shape}")
    print(f"Total confirmed cases globally: {df_clean['Confirmed'].sum():,}")
    print(f"Total deaths globally: {df_clean['Deaths'].sum():,}")
    print(f"Total recovered globally: {df_clean['Recovered'].sum():,}")
    
    return df_clean

def create_bar_charts(df):
    """Create bar charts for top affected regions"""
    
    print("\n" + "=" * 60)
    print("CREATING BAR CHARTS")
    print("=" * 60)
    
    # Top 10 regions by confirmed cases
    top_confirmed = df.nlargest(10, 'Confirmed')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('COVID-19 Analysis - Top Affected Regions (Feb 9, 2020)', fontsize=16, fontweight='bold')
    
    # Bar chart 1: Top 10 confirmed cases
    bars1 = ax1.bar(range(len(top_confirmed)), top_confirmed['Confirmed'], 
                   color='steelblue', alpha=0.7)
    ax1.set_title('Top 10 Regions by Confirmed Cases', fontweight='bold')
    ax1.set_xlabel('Region')
    ax1.set_ylabel('Confirmed Cases')
    ax1.set_xticks(range(len(top_confirmed)))
    ax1.set_xticklabels(top_confirmed['Location'], rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    
    # Bar chart 2: Top 10 deaths
    top_deaths = df.nlargest(10, 'Deaths')
    bars2 = ax2.bar(range(len(top_deaths)), top_deaths['Deaths'], 
                   color='crimson', alpha=0.7)
    ax2.set_title('Top 10 Regions by Deaths', fontweight='bold')
    ax2.set_xlabel('Region')
    ax2.set_ylabel('Deaths')
    ax2.set_xticks(range(len(top_deaths)))
    ax2.set_xticklabels(top_deaths['Location'], rotation=45, ha='right')
    
    # Add value labels
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # Bar chart 3: Top 10 recovered
    top_recovered = df.nlargest(10, 'Recovered')
    bars3 = ax3.bar(range(len(top_recovered)), top_recovered['Recovered'], 
                   color='forestgreen', alpha=0.7)
    ax3.set_title('Top 10 Regions by Recovered Cases', fontweight='bold')
    ax3.set_xlabel('Region')
    ax3.set_ylabel('Recovered Cases')
    ax3.set_xticks(range(len(top_recovered)))
    ax3.set_xticklabels(top_recovered['Location'], rotation=45, ha='right')
    
    # Add value labels
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # Bar chart 4: Country-level summary
    country_summary = df.groupby('Country/Region').agg({
        'Confirmed': 'sum',
        'Deaths': 'sum', 
        'Recovered': 'sum'
    }).reset_index()
    top_countries = country_summary.nlargest(8, 'Confirmed')
    
    x = np.arange(len(top_countries))
    width = 0.25
    
    ax4.bar(x - width, top_countries['Confirmed'], width, label='Confirmed', color='steelblue', alpha=0.7)
    ax4.bar(x, top_countries['Deaths'], width, label='Deaths', color='crimson', alpha=0.7)
    ax4.bar(x + width, top_countries['Recovered'], width, label='Recovered', color='forestgreen', alpha=0.7)
    
    ax4.set_title('Top Countries - Confirmed, Deaths, Recovered', fontweight='bold')
    ax4.set_xlabel('Country')
    ax4.set_ylabel('Cases')
    ax4.set_xticks(x)
    ax4.set_xticklabels(top_countries['Country/Region'], rotation=45, ha='right')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('covid19_bar_charts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return top_confirmed, country_summary

def create_scatter_plots(df):
    """Create scatter plots to explore relationships between variables"""
    
    print("\n" + "=" * 60)
    print("CREATING SCATTER PLOTS")
    print("=" * 60)
    
    # Filter out zero values for better visualization
    df_scatter = df[df['Confirmed'] > 0].copy()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('COVID-19 Correlation Analysis (Feb 9, 2020)', fontsize=16, fontweight='bold')
    
    # Scatter plot 1: Confirmed vs Deaths
    scatter1 = ax1.scatter(df_scatter['Confirmed'], df_scatter['Deaths'], 
                          alpha=0.6, color='red', s=60)
    ax1.set_xlabel('Confirmed Cases')
    ax1.set_ylabel('Deaths')
    ax1.set_title('Deaths vs Confirmed Cases', fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Add correlation coefficient
    if len(df_scatter) > 1:
        corr_cd = np.corrcoef(df_scatter['Confirmed'], df_scatter['Deaths'])[0, 1]
        ax1.text(0.05, 0.95, f'Correlation: {corr_cd:.3f}', 
                transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # Scatter plot 2: Confirmed vs Recovered
    ax2.scatter(df_scatter['Confirmed'], df_scatter['Recovered'], 
               alpha=0.6, color='green', s=60)
    ax2.set_xlabel('Confirmed Cases')
    ax2.set_ylabel('Recovered Cases')
    ax2.set_title('Recovered vs Confirmed Cases', fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # Add correlation coefficient
    if len(df_scatter) > 1:
        corr_cr = np.corrcoef(df_scatter['Confirmed'], df_scatter['Recovered'])[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {corr_cr:.3f}', 
                transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # Scatter plot 3: Mortality Rate vs Confirmed Cases
    ax3.scatter(df_scatter['Confirmed'], df_scatter['Mortality_Rate'], 
               alpha=0.6, color='orange', s=60)
    ax3.set_xlabel('Confirmed Cases')
    ax3.set_ylabel('Mortality Rate (%)')
    ax3.set_title('Mortality Rate vs Confirmed Cases', fontweight='bold')
    ax3.set_xscale('log')
    
    # Scatter plot 4: Recovery Rate vs Confirmed Cases
    ax4.scatter(df_scatter['Confirmed'], df_scatter['Recovery_Rate'], 
               alpha=0.6, color='purple', s=60)
    ax4.set_xlabel('Confirmed Cases')
    ax4.set_ylabel('Recovery Rate (%)')
    ax4.set_title('Recovery Rate vs Confirmed Cases', fontweight='bold')
    ax4.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('covid19_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Scatter plots analysis completed.")

def create_pie_charts(df):
    """Create pie charts for regional distribution analysis"""
    
    print("\n" + "=" * 60)
    print("CREATING PIE CHARTS")
    print("=" * 60)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('COVID-19 Regional Distribution (Feb 9, 2020)', fontsize=16, fontweight='bold')
    
    # Pie chart 1: Top countries by confirmed cases
    country_summary = df.groupby('Country/Region')['Confirmed'].sum().sort_values(ascending=False)
    top_countries = country_summary.head(6)
    others = country_summary.tail(-6).sum()
    
    if others > 0:
        pie_data = list(top_countries) + [others]
        pie_labels = list(top_countries.index) + ['Others']
    else:
        pie_data = list(top_countries)
        pie_labels = list(top_countries.index)
    
    colors1 = plt.cm.Set3(np.linspace(0, 1, len(pie_data)))
    wedges1, texts1, autotexts1 = ax1.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', 
                                          colors=colors1, startangle=90)
    ax1.set_title('Confirmed Cases by Country', fontweight='bold')
    
    # Pie chart 2: China's provinces (if China dominates)
    china_data = df[df['Country/Region'] == 'China']
    if len(china_data) > 1:
        china_provinces = china_data.groupby('Province/State')['Confirmed'].sum().sort_values(ascending=False)
        top_provinces = china_provinces.head(5)
        others_china = china_provinces.tail(-5).sum()
        
        if others_china > 0:
            pie_data2 = list(top_provinces) + [others_china]
            pie_labels2 = list(top_provinces.index) + ['Other Provinces']
        else:
            pie_data2 = list(top_provinces)
            pie_labels2 = list(top_provinces.index)
            
        colors2 = plt.cm.Reds(np.linspace(0.3, 1, len(pie_data2)))
        ax2.pie(pie_data2, labels=pie_labels2, autopct='%1.1f%%', 
               colors=colors2, startangle=90)
        ax2.set_title('China: Confirmed Cases by Province', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Insufficient Chinese\nProvincial Data', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('China: Provincial Data', fontweight='bold')
    
    # Pie chart 3: Deaths by country
    country_deaths = df.groupby('Country/Region')['Deaths'].sum().sort_values(ascending=False)
    country_deaths = country_deaths[country_deaths > 0]  # Only countries with deaths
    
    if len(country_deaths) > 0:
        if len(country_deaths) > 5:
            top_deaths = country_deaths.head(5)
            others_deaths = country_deaths.tail(-5).sum()
            pie_data3 = list(top_deaths) + [others_deaths]
            pie_labels3 = list(top_deaths.index) + ['Others']
        else:
            pie_data3 = list(country_deaths)
            pie_labels3 = list(country_deaths.index)
            
        colors3 = plt.cm.Reds(np.linspace(0.4, 1, len(pie_data3)))
        ax3.pie(pie_data3, labels=pie_labels3, autopct='%1.1f%%', 
               colors=colors3, startangle=90)
        ax3.set_title('Deaths by Country', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No Death Data\nAvailable', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Deaths by Country', fontweight='bold')
    
    # Pie chart 4: Recovery by country
    country_recovered = df.groupby('Country/Region')['Recovered'].sum().sort_values(ascending=False)
    country_recovered = country_recovered[country_recovered > 0]  # Only countries with recoveries
    
    if len(country_recovered) > 0:
        if len(country_recovered) > 5:
            top_recovered = country_recovered.head(5)
            others_recovered = country_recovered.tail(-5).sum()
            pie_data4 = list(top_recovered) + [others_recovered]
            pie_labels4 = list(top_recovered.index) + ['Others']
        else:
            pie_data4 = list(country_recovered)
            pie_labels4 = list(country_recovered.index)
            
        colors4 = plt.cm.Greens(np.linspace(0.4, 1, len(pie_data4)))
        ax4.pie(pie_data4, labels=pie_labels4, autopct='%1.1f%%', 
               colors=colors4, startangle=90)
        ax4.set_title('Recovered Cases by Country', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No Recovery Data\nAvailable', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Recovered Cases by Country', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('covid19_pie_charts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return country_summary

def analyze_key_insights(df, country_summary):
    """Analyze and report key insights from the data"""
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS AND ANALYSIS")
    print("=" * 60)
    
    total_confirmed = df['Confirmed'].sum()
    total_deaths = df['Deaths'].sum()
    total_recovered = df['Recovered'].sum()
    
    print(f"\n📊 GLOBAL SNAPSHOT (February 9, 2020)")
    print(f"   Total Confirmed Cases: {total_confirmed:,}")
    print(f"   Total Deaths: {total_deaths:,}")
    print(f"   Total Recovered: {total_recovered:,}")
    print(f"   Global Mortality Rate: {(total_deaths/total_confirmed)*100:.2f}%")
    print(f"   Global Recovery Rate: {(total_recovered/total_confirmed)*100:.2f}%")
    
    # China's dominance analysis
    china_cases = country_summary[country_summary['Country/Region'] == 'China']['Confirmed'].sum()
    china_percentage = (china_cases / total_confirmed) * 100
    
    print(f"\n🇨🇳 CHINA'S DOMINANCE IN EARLY PHASE")
    print(f"   China's Cases: {china_cases:,}")
    print(f"   Percentage of Global Cases: {china_percentage:.1f}%")
    print(f"   China clearly dominated the early pandemic phase with {china_percentage:.1f}% of all cases")
    
    # Top affected regions
    top_regions = df.nlargest(5, 'Confirmed')
    print(f"\n🔥 TOP 5 MOST AFFECTED REGIONS:")
    for i, region in enumerate(top_regions.itertuples(), 1):
        print(f"   {i}. {region.Location}: {region.Confirmed:,} cases")
    
    # Regional differences in mortality and recovery
    print(f"\n📈 REGIONAL MORTALITY AND RECOVERY PATTERNS:")
    
    # Calculate rates by region with significant cases
    significant_regions = df[df['Confirmed'] >= 10].copy()  # Only regions with 10+ cases
    
    if len(significant_regions) > 0:
        print(f"   Regions with highest mortality rates (≥10 cases):")
        high_mortality = significant_regions.nlargest(3, 'Mortality_Rate')
        for region in high_mortality.itertuples():
            if region.Mortality_Rate > 0:
                print(f"     • {region.Location}: {region.Mortality_Rate:.1f}% ({region.Deaths}/{region.Confirmed})")
        
        print(f"\n   Regions with highest recovery rates (≥10 cases):")
        high_recovery = significant_regions.nlargest(3, 'Recovery_Rate')
        for region in high_recovery.itertuples():
            if region.Recovery_Rate > 0:
                print(f"     • {region.Location}: {region.Recovery_Rate:.1f}% ({region.Recovered}/{region.Confirmed})")
    
    # Reporting patterns
    print(f"\n📋 REPORTING PATTERNS:")
    total_regions = len(df)
    regions_with_deaths = len(df[df['Deaths'] > 0])
    regions_with_recoveries = len(df[df['Recovered'] > 0])
    
    print(f"   Total Reporting Regions: {total_regions}")
    print(f"   Regions Reporting Deaths: {regions_with_deaths} ({(regions_with_deaths/total_regions)*100:.1f}%)")
    print(f"   Regions Reporting Recoveries: {regions_with_recoveries} ({(regions_with_recoveries/total_regions)*100:.1f}%)")
    
    # Country-level insights
    print(f"\n🌍 COUNTRY-LEVEL INSIGHTS:")
    affected_countries = len(country_summary[country_summary['Confirmed'] > 0])
    print(f"   Countries with confirmed cases: {affected_countries}")
    
    non_china_cases = total_confirmed - china_cases
    print(f"   Cases outside China: {non_china_cases:,} ({((non_china_cases/total_confirmed)*100):.1f}%)")
    
    if len(country_summary) > 1:
        second_most = country_summary.nlargest(2, 'Confirmed').iloc[1]
        print(f"   Second most affected country: {second_most['Country/Region']} ({second_most['Confirmed']:,} cases)")

def main():
    """Main function to run the complete COVID-19 analysis"""
    
    print("🦠 COVID-19 DATA ANALYTICS PROJECT")
    print("📅 Analysis Date: February 9, 2020")
    print("🏥 Data Source: Johns Hopkins University")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    os.chdir('output') if os.path.exists('output') else None
    
    # Step 1: Download data
    data_file = download_covid_data()
    
    # Step 2: Load and inspect data
    df_raw = load_and_inspect_data(data_file)
    
    # Step 3: Clean data
    df_clean = clean_data(df_raw)
    
    # Step 4: Create visualizations
    top_confirmed, country_summary = create_bar_charts(df_clean)
    create_scatter_plots(df_clean)
    pie_summary = create_pie_charts(df_clean)
    
    # Step 5: Generate insights
    analyze_key_insights(df_clean, country_summary)
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print("📊 Generated visualizations:")
    print("   • covid19_bar_charts.png - Top affected regions analysis")
    print("   • covid19_scatter_plots.png - Correlation analysis")
    print("   • covid19_pie_charts.png - Regional distribution analysis")
    print("=" * 60)

if __name__ == "__main__":
    main()