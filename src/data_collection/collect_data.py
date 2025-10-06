"""
Data Collection Script
Fetches all required data from FRED and other free sources
"""

import os
import pandas as pd
import numpy as np
from fredapi import Fred
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

class DataCollector:
    def __init__(self, fred_api_key=None):
        """Initialize data collector with FRED API"""
        self.api_key = fred_api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise ValueError("FRED API key required. Get one at https://fred.stlouisfed.org/docs/api/api_key.html")
        
        self.fred = Fred(api_key=self.api_key)
        self.start_date = '1990-01-01'
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        
    def collect_macro_data(self):
        """Collect macroeconomic indicators for regime identification"""
        print("Collecting macroeconomic data...")
        
        macro_series = {
            'gdp_growth': 'A191RL1Q225SBEA',  # Real GDP growth rate (quarterly)
            'unemployment': 'UNRATE',          # Unemployment rate
            'cpi_yoy': 'CPIAUCSL',            # CPI (we'll calculate YoY)
            'core_pce': 'PCEPILFE',           # Core PCE inflation
            'ism_manufacturing': 'MANEMP',     # ISM Manufacturing
            'industrial_production': 'INDPRO', # Industrial production
            'fed_funds': 'FEDFUNDS',          # Federal funds rate
            'vix': 'VIXCLS'                   # VIX volatility index
        }
        
        data = {}
        for name, series_id in macro_series.items():
            try:
                series = self.fred.get_series(series_id, 
                                             observation_start=self.start_date,
                                             observation_end=self.end_date)
                data[name] = series
                print(f"  ✓ {name}: {len(series)} observations")
            except Exception as e:
                print(f"  ✗ Failed to fetch {name}: {e}")
        
        # Convert to DataFrame and align to monthly frequency
        df = pd.DataFrame(data)
        
        # Calculate YoY inflation from CPI
        if 'cpi_yoy' in df.columns:
            df['inflation'] = df['cpi_yoy'].pct_change(12) * 100
        
        # Resample quarterly GDP to monthly (forward fill)
        if 'gdp_growth' in df.columns:
            df['gdp_growth'] = df['gdp_growth'].fillna(method='ffill')
        
        return df
    
    def collect_treasury_yields(self):
        """Collect Treasury yield curve data"""
        print("\nCollecting Treasury yield curve data...")
        
        treasury_series = {
            'DGS3MO': '3M',
            'DGS6MO': '6M',
            'DGS1': '1Y',
            'DGS2': '2Y',
            'DGS3': '3Y',
            'DGS5': '5Y',
            'DGS7': '7Y',
            'DGS10': '10Y',
            'DGS20': '20Y',
            'DGS30': '30Y'
        }
        
        yields = {}
        for series_id, maturity in treasury_series.items():
            try:
                series = self.fred.get_series(series_id,
                                             observation_start=self.start_date,
                                             observation_end=self.end_date)
                yields[maturity] = series
                print(f"  ✓ {maturity} Treasury: {len(series)} observations")
            except Exception as e:
                print(f"  ✗ Failed to fetch {maturity}: {e}")
        
        df = pd.DataFrame(yields)
        return df
    
    def collect_treasury_returns(self):
        """
        Collect or calculate Treasury total returns
        Using simplified approach with yield changes and duration
        """
        print("\nCalculecting Treasury returns from yields...")
        
        # Get yields
        yields = self.collect_treasury_yields()
        
        # Approximate durations for each maturity
        durations = {
            '3M': 0.25,
            '6M': 0.5,
            '1Y': 1.0,
            '2Y': 1.9,
            '3Y': 2.8,
            '5Y': 4.5,
            '7Y': 6.3,
            '10Y': 8.5,
            '20Y': 14.5,
            '30Y': 18.0
        }
        
        returns = pd.DataFrame(index=yields.index)
        
        for maturity in yields.columns:
            if maturity in durations:
                # Calculate approximate total return
                # Return ≈ Yield/12 - Duration × ΔYield
                y = yields[maturity] / 100  # Convert to decimal
                duration = durations[maturity]
                
                # Monthly return approximation
                carry = y / 12
                price_change = -duration * y.diff()
                
                returns[maturity] = carry + price_change
        
        print(f"  ✓ Calculated returns for {len(returns.columns)} maturities")
        return returns
    
    def collect_credit_spreads(self):
        """Collect credit spread data"""
        print("\nCollecting credit spread data...")
        
        spread_series = {
            'BAA10Y': 'BAA-10Y Spread',      # BAA corporate spread
            'AAA10Y': 'AAA-10Y Spread',      # AAA corporate spread
            'BAMLH0A0HYM2': 'HY Spread',     # High yield spread
        }
        
        spreads = {}
        for series_id, name in spread_series.items():
            try:
                series = self.fred.get_series(series_id,
                                             observation_start=self.start_date,
                                             observation_end=self.end_date)
                spreads[name] = series
                print(f"  ✓ {name}: {len(series)} observations")
            except Exception as e:
                print(f"  ✗ Failed to fetch {name}: {e}")
        
        df = pd.DataFrame(spreads)
        return df
    
    def save_all_data(self, output_dir='data/raw'):
        """Collect and save all data"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("STARTING DATA COLLECTION")
        print(f"{'='*60}")
        
        # Collect all datasets
        macro = self.collect_macro_data()
        yields = self.collect_treasury_yields()
        returns = self.collect_treasury_returns()
        spreads = self.collect_credit_spreads()
        
        # Save to CSV
        print(f"\n{'='*60}")
        print("SAVING DATA")
        print(f"{'='*60}")
        
        datasets = {
            'macro_data.csv': macro,
            'treasury_yields.csv': yields,
            'treasury_returns.csv': returns,
            'credit_spreads.csv': spreads
        }
        
        for filename, df in datasets.items():
            filepath = Path(output_dir) / filename
            df.to_csv(filepath)
            print(f"  ✓ Saved {filename}: {df.shape}")
        
        print(f"\n{'='*60}")
        print("DATA COLLECTION COMPLETE")
        print(f"{'='*60}")
        print(f"\nData saved to: {output_dir}")
        print(f"Date range: {self.start_date} to {self.end_date}")
        
        return datasets

def main():
    """Main execution function"""
    try:
        collector = DataCollector()
        datasets = collector.save_all_data()
        
        print("\n✓ SUCCESS! All data collected and saved.")
        print("\nNext step: Run regime model estimation")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        print("\nMake sure you have:")
        print("1. Created .env file with FRED_API_KEY=your_key")
        print("2. Installed requirements: pip install -r requirements.txt")
        raise

if __name__ == "__main__":
    main()
