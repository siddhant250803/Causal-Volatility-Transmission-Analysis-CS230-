# Data Directory

Place your data file here:

- `HF_Returns_Stocks.csv` - High-frequency stock returns data (5-minute intervals)

Alternatively, you can place the CSV file in the project root directory. The scripts will automatically find it in either location.

## Data Format

The CSV should contain:
- Time series data with timestamps
- 5-minute stock returns for multiple stocks
- Columns: timestamp + stock ticker symbols
- Missing values handled automatically by the framework

