import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
file_path = 'FDI_data.csv'  # Replace with your file path
fdi_data = pd.read_csv(file_path)

# Calculate total FDI for each sector over the entire period
fdi_data['Total_FDI'] = fdi_data.iloc[:, 1:].sum(axis=1)

# Sort sectors by total FDI received
sector_total_fdi = fdi_data[['Sector', 'Total_FDI']].sort_values(by='Total_FDI', ascending=False)

# Select the top 5 sectors for visualization purposes
top_5_sectors = sector_total_fdi['Sector'].head(5)
fdi_top_5 = fdi_data[fdi_data['Sector'].isin(top_5_sectors)].set_index('Sector')

# Transpose data for year-over-year analysis
fdi_top_5_transposed = fdi_top_5.iloc[:, :-1].T  # Exclude the Total_FDI column

# Plotting the FDI inflows over the years for the top 5 sectors
plt.figure(figsize=(14, 8))
for sector in top_5_sectors:
    plt.plot(fdi_top_5_transposed.index, fdi_top_5_transposed[sector], marker='o', label=sector)

plt.title('FDI Inflows Over Time for Top 5 Sectors (2000-01 to 2016-17)')
plt.xlabel('Financial Year')
plt.ylabel('FDI Inflows (in Millions USD)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()

# Calculate year-over-year growth rates for the top 5 sectors
fdi_growth_top_5 = fdi_top_5.iloc[:, :-1].pct_change(axis=1) * 100

# Plot the year-over-Year FDI Growth Rates for Top 5 Sectors
plt.figure(figsize=(14, 8))
for sector in top_5_sectors:
    plt.plot(fdi_growth_top_5.columns, fdi_growth_top_5.loc[sector], marker='o', label=sector)

plt.title('Year-over-Year FDI Growth Rates for Top 5 Sectors (2000-01 to 2016-17)')
plt.xlabel('Financial Year')
plt.ylabel('Growth Rate (%)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()

# Calculate sectoral contribution to total FDI each year
yearly_totals = fdi_data.iloc[:, 1:-1].sum(axis=0)
sectoral_contributions = fdi_top_5.iloc[:, :-1].div(yearly_totals, axis=1) * 100

# Plot sectoral contributions over the years
plt.figure(figsize=(14, 8))
sectoral_contributions.T.plot(kind='bar', stacked=True, figsize=(14, 8))

plt.title('Sectoral Contribution to Total FDI (2000-01 to 2016-17)')
plt.xlabel('Financial Year')
plt.ylabel('Contribution to Total FDI (%)')
plt.legend(title="Sector", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Key Metrics Calculation
fdi_numeric_data = fdi_data.iloc[:, 1:-4]

# Recalculate average annual FDI growth rate (excluding NaN growth)
fdi_data['Avg_Annual_Growth'] = fdi_numeric_data.pct_change(axis=1).mean(axis=1, skipna=True) * 100

# Recalculate the volatility (standard deviation) of FDI inflows for each sector
fdi_data['FDI_Volatility'] = fdi_numeric_data.std(axis=1, ddof=0)

# Corrected top sectors by average annual growth rate
top_sectors_growth_corrected = fdi_data[['Sector', 'Avg_Annual_Growth']].sort_values(by='Avg_Annual_Growth', ascending=False).head(10)

# Corrected top sectors by FDI volatility
top_sectors_volatility_corrected = fdi_data[['Sector', 'FDI_Volatility']].sort_values(by='FDI_Volatility', ascending=False).head(10)

# Correlation Analysis
correlation_matrix = fdi_data.iloc[:, 1:-4].corr()

# Visualization of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix of FDI Inflows Between Sectors')
plt.show()

# Export the processed data to a CSV file for use in Tableau
export_file_path = "FDI_processed_data_for_Tableau.csv"
fdi_data.to_csv(export_file_path, index=False)
