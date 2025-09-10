# COVID-19 Data Analytics Project

## 📊 Overview
Comprehensive analysis of Johns Hopkins COVID-19 dataset from February 9, 2020, providing insights into the early phase of the pandemic. This project includes data inspection, cleaning, and visualization with multiple chart types to understand global and regional trends in confirmed cases, deaths, and recoveries.

## 🎯 Project Objectives
- **Data Inspection**: Examine the structure and quality of COVID-19 data
- **Data Cleaning**: Handle missing values and standardize geographic information
- **Visualization**: Create informative charts showing pandemic patterns
- **Analysis**: Identify key insights about the early pandemic phase
- **Regional Patterns**: Explore differences in mortality and recovery rates

## 📈 Key Features
- **Bar Charts**: Top affected regions by confirmed cases, deaths, and recoveries
- **Scatter Plots**: Correlation analysis between different metrics
- **Pie Charts**: Regional distribution and proportional analysis
- **Statistical Analysis**: Mortality rates, recovery rates, and global statistics
- **China Analysis**: Detailed examination of China's dominance in early phase

## 🔍 Key Insights Discovered
- **China's Dominance**: 99.1% of global cases were in China on Feb 9, 2020
- **Regional Concentration**: Hubei province was the primary epicenter with 29,631 cases
- **Global Mortality Rate**: 2.26% overall mortality rate
- **Recovery Patterns**: 8.08% overall recovery rate with significant regional variations
- **International Spread**: Early signs of global spread with 28 countries reporting cases

## 📁 Project Structure
```
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── covid19_analysis.py           # Main analysis script
├── COVID19_Analysis.ipynb        # Jupyter notebook version
├── data/                         # COVID-19 dataset
│   └── covid_02_09_2020.csv     # Raw data from Johns Hopkins
├── covid19_bar_charts.png        # Bar chart visualizations
├── covid19_scatter_plots.png     # Scatter plot analysis
└── covid19_pie_charts.png        # Pie chart distributions
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Required packages (see requirements.txt)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd DATA_-ANALYTICS_PRJECT
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis
You can run the analysis in two ways:

#### Option 1: Python Script
```bash
python covid19_analysis.py
```

#### Option 2: Jupyter Notebook
```bash
jupyter notebook COVID19_Analysis.ipynb
```

## 📊 Generated Visualizations

### 1. Bar Charts Analysis
- **Top 10 regions** by confirmed cases, deaths, and recoveries
- **Country-level comparison** showing confirmed, deaths, and recovered cases side-by-side
- **Regional insights** highlighting the most affected areas

### 2. Scatter Plot Analysis
- **Deaths vs Confirmed Cases**: Shows correlation between case numbers and fatalities
- **Recovered vs Confirmed Cases**: Illustrates recovery patterns
- **Mortality Rate vs Confirmed Cases**: Reveals mortality patterns across regions
- **Recovery Rate vs Confirmed Cases**: Shows recovery efficiency by region size

### 3. Pie Chart Distribution
- **Global case distribution** by country
- **China provincial breakdown** showing internal distribution
- **Deaths by country** proportional analysis
- **Recoveries by country** showing recovery patterns

## 🔍 Data Analysis Highlights

### Global Snapshot (February 9, 2020)
- **Total Confirmed Cases**: 40,135
- **Total Deaths**: 908
- **Total Recovered**: 3,244
- **Global Mortality Rate**: 2.26%
- **Global Recovery Rate**: 8.08%

### Top Affected Regions
1. **Hubei, China**: 29,631 cases (73.8% of global total)
2. **Guangdong, China**: 1,131 cases
3. **Zhejiang, China**: 1,075 cases
4. **Henan, China**: 1,033 cases
5. **Hunan, China**: 838 cases

### Regional Patterns
- **Highest Mortality Rates**: Hong Kong (3.4%), Hubei (2.9%), Gansu (2.4%)
- **Highest Recovery Rates**: Thailand (34.4%), Ningxia (28.9%), Hunan (22.2%)
- **Reporting Coverage**: 108 regions across 28 countries

## 🛠️ Technical Implementation

### Data Sources
- **Primary**: Johns Hopkins University COVID-19 dataset
- **Date**: February 9, 2020 daily report
- **Format**: CSV with province/state, country, confirmed, deaths, recovered data

### Technologies Used
- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **NumPy**: Numerical computing
- **Requests**: HTTP library for data download

### Data Processing Pipeline
1. **Data Download**: Fetch from Johns Hopkins GitHub repository
2. **Data Cleaning**: Handle missing values, standardize country names
3. **Feature Engineering**: Calculate mortality and recovery rates
4. **Visualization**: Generate multiple chart types
5. **Analysis**: Extract key insights and patterns

## 📋 Data Quality Notes
- **Missing Values**: 28 missing province/state entries (handled by filling with empty strings)
- **Geographic Standardization**: "Mainland China" converted to "China" for consistency
- **Rate Calculations**: Mortality and recovery rates calculated only for regions with confirmed cases

## 🎓 Educational Value
This project demonstrates:
- **Data Science Workflow**: End-to-end analysis process
- **Visualization Techniques**: Multiple chart types for different insights
- **Statistical Analysis**: Rate calculations and correlation analysis
- **Real-world Application**: Analysis of actual pandemic data
- **Documentation**: Comprehensive project documentation and code comments

## 📊 Future Enhancements
- Time series analysis with multiple dates
- Geographic mapping with coordinates
- Predictive modeling for case projections
- Interactive dashboard development
- Comparative analysis with other diseases

## 📝 License
This project is for educational and research purposes. Data courtesy of Johns Hopkins University.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

## 📞 Contact
For questions or suggestions, please open an issue in this repository.
