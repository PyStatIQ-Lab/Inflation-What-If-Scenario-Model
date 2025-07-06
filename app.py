import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime, timedelta
import ipywidgets as widgets
from IPython.display import display
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
sns.set_palette("husl")

## 1. Data Collection Functions

def fetch_current_prices(assets):
    """Fetch current prices for all assets in portfolios"""
    ticker_map = {
        # Stocks
        "Infosys": "INFY.NS",
        "TCS": "TCS.NS",
        "Reliance": "RELIANCE.NS",
        "HDFC Bank": "HDFCBANK.NS",
        
        # ETFs
        "Sensex ETF": "^BSESN",  # Using BSE Sensex as proxy
        "NIFTY ETF": "^NSEI",   # Using Nifty 50 as proxy
    }
    
    # Mutual funds don't have direct tickers - we'll use category averages
    mf_category_map = {
        "ICICI Pru Equity Fund": "large_cap",
        "Axis Midcap Fund": "mid_cap",
        "HDFC Balanced Fund": "balanced",
        "SBI Bluechip Fund": "large_cap",
        "Mirae Emerging Bluechip": "mid_cap"
    }
    
    # Fetch equity prices
    equity_tickers = [ticker_map[a] for a in assets if a in ticker_map]
    if equity_tickers:
        try:
            equity_data = yf.download(equity_tickers, period="1d")['Adj Close']
        except:
            equity_data = pd.DataFrame()
    else:
        equity_data = pd.DataFrame()
    
    # For mutual funds, we'll use category returns (simplified)
    mf_returns = {
        "large_cap": 0.10,  # 10% annualized
        "mid_cap": 0.12,
        "balanced": 0.08
    }
    
    current_prices = {}
    
    # Process equity prices
    for asset in assets:
        if asset in ticker_map:
            ticker = ticker_map[asset]
            if not equity_data.empty and ticker in equity_data.columns:
                current_prices[asset] = equity_data[ticker].iloc[-1]
            else:
                # Fallback to last known price if today's data unavailable
                try:
                    hist_data = yf.Ticker(ticker).history(period="1mo")
                    if not hist_data.empty:
                        current_prices[asset] = hist_data['Adj Close'].iloc[-1]
                    else:
                        current_prices[asset] = np.nan
                except:
                    current_prices[asset] = np.nan
        
        elif asset in mf_category_map:
            category = mf_category_map[asset]
            current_prices[asset] = 1000 * (1 + mf_returns[category]/365)  # NAV approximation
    
    return current_prices

def fetch_historical_inflation():
    """Fetch historical inflation data (mock - in practice use RBI/World Bank API)"""
    dates = pd.date_range(start="2023-05-01", end="2025-06-01", freq='MS')
    values = [4.4, 5.6, 7.5, 6.9, 5.0, 4.9, 5.6, 5.7, 5.1, 5.1, 
              4.9, 5.2, 5.2, 5.2, 5.1, 5.1, 5.1, 5.0, 5.0, 4.9,
              4.9, 4.9, 4.8, 4.8, 4.8, 4.7]
    return pd.Series(values, index=dates, name='Inflation')

## 2. Portfolio Analysis Class

class PortfolioAnalyzer:
    def __init__(self, client_data):
        self.client_data = client_data
        self.portfolio = self._parse_portfolio()
        self.current_prices = None
        self.historical_inflation = fetch_historical_inflation()
        self.current_inflation = self.historical_inflation.iloc[-1]
        
    def _parse_portfolio(self):
        """Parse the client's portfolio into a structured format"""
        try:
            equity_holdings = [x.strip() for x in self.client_data["Equity Portfolio (Stocks, ETFs)"].split(",")]
            mf_holdings = [x.strip() for x in self.client_data["Mutual Fund Holdings"].split(",")]
        except:
            equity_holdings = []
            mf_holdings = []
        
        # For demo purposes, assume equal allocation between equity and MF
        portfolio = {}
        n_equity = max(len(equity_holdings), 1)  # Avoid division by zero
        n_mf = max(len(mf_holdings), 1)
        
        for asset in equity_holdings:
            portfolio[asset] = {'type': 'equity', 'allocation': 0.5/n_equity}
            
        for asset in mf_holdings:
            portfolio[asset] = {'type': 'mf', 'allocation': 0.5/n_mf}
            
        return portfolio
    
    def fetch_current_values(self):
        """Fetch current market values for all holdings"""
        assets = list(self.portfolio.keys())
        self.current_prices = fetch_current_prices(assets)
        
        # Calculate current values
        try:
            total_value = float(self.client_data["Total Portfolio Size (in lakhs)"]) * 100000  # Convert to rupees
        except:
            total_value = 0
        
        for asset, data in self.portfolio.items():
            if asset in self.current_prices and not np.isnan(self.current_prices[asset]):
                data['current_price'] = self.current_prices[asset]
                data['units'] = (total_value * data['allocation']) / data['current_price']
            else:
                data['current_price'] = np.nan
                data['units'] = np.nan
    
    def get_inflation_sensitivity(self, asset, horizon="short_term"):
        """Get inflation sensitivity for an asset"""
        # Base sensitivities
        sensitivities = {
            # Stocks
            "Infosys": {"short_term": -0.5, "long_term": 0.3},
            "TCS": {"short_term": -0.4, "long_term": 0.4},
            "Reliance": {"short_term": -0.3, "long_term": 0.5},
            "HDFC Bank": {"short_term": -0.6, "long_term": 0.2},
            
            # ETFs
            "Sensex ETF": {"short_term": -0.4, "long_term": 0.35},
            "NIFTY ETF": {"short_term": -0.45, "long_term": 0.4},
            
            # Mutual Fund Categories
            "large_cap": {"short_term": -0.4, "long_term": 0.3},
            "mid_cap": {"short_term": -0.6, "long_term": 0.5},
            "balanced": {"short_term": -0.3, "long_term": 0.25}
        }
        
        mf_category_map = {
            "ICICI Pru Equity Fund": "large_cap",
            "Axis Midcap Fund": "mid_cap",
            "HDFC Balanced Fund": "balanced",
            "SBI Bluechip Fund": "large_cap",
            "Mirae Emerging Bluechip": "mid_cap"
        }
        
        if asset in sensitivities:
            return sensitivities[asset][horizon]
        elif asset in mf_category_map:
            return sensitivities[mf_category_map[asset]][horizon]
        else:
            return 0  # Default if asset not found
    
    def generate_inflation_scenarios(self, periods=12):
        """Generate various inflation scenarios"""
        scenarios = {
            "Baseline": [self.current_inflation] * periods,
            "Mild Increase (0.1% monthly)": [self.current_inflation + 0.1*i for i in range(periods)],
            "Sharp Increase (0.3% monthly)": [self.current_inflation + 0.3*i for i in range(periods)],
            "Mild Decrease (0.1% monthly)": [self.current_inflation - 0.1*i for i in range(periods)],
            "Sharp Decrease (0.3% monthly)": [self.current_inflation - 0.3*i for i in range(periods)],
            "Volatile (±0.5%)": [self.current_inflation + (0.5 if i%2 else -0.3) for i in range(periods)],
            "Historical Average (5.1%)": [5.1] * periods,
            "High Inflation (7%+)": [7.0 + 0.1*i for i in range(periods)]
        }
        return scenarios
    
    def calculate_portfolio_impact(self, inflation_scenario, horizon="short_term"):
        """Calculate portfolio impact for a given inflation scenario"""
        try:
            avg_inflation_change = np.mean(inflation_scenario) - self.current_inflation
        except:
            avg_inflation_change = 0
            
        total_impact = 0
        breakdown = {}
        
        for asset, data in self.portfolio.items():
            if 'allocation' in data:
                sensitivity = self.get_inflation_sensitivity(asset, horizon)
                impact = sensitivity * avg_inflation_change * data['allocation']
                total_impact += impact
                breakdown[asset] = {
                    'impact': impact,
                    'allocation': data['allocation'],
                    'sensitivity': sensitivity
                }
        
        return {
            'total_impact': total_impact,
            'expected_return_change': total_impact,
            'breakdown': breakdown,
            'scenario': inflation_scenario
        }
    
    def analyze_all_scenarios(self, horizon="short_term"):
        """Run analysis for all scenarios"""
        scenarios = self.generate_inflation_scenarios()
        results = {}
        
        for name, scenario in scenarios.items():
            results[name] = self.calculate_portfolio_impact(scenario, horizon)
        
        return results
    
    def visualize_results(self, analysis_results):
        """Create visualizations of the analysis"""
        # Convert results to DataFrame for easier plotting
        results_df = pd.DataFrame({
            'Scenario': analysis_results.keys(),
            'Impact (%)': [r['total_impact']*100 for r in analysis_results.values()]
        })
        
        # Impact bar chart
        plt.figure(figsize=(12, 6))
        sns.barplot(data=results_df, x='Impact (%)', y='Scenario', palette="viridis")
        plt.title(f"Portfolio Impact Under Different Inflation Scenarios\n{self.client_data['Client Name']}")
        plt.xlabel("Expected Return Impact (%)")
        plt.tight_layout()
        plt.show()
        
        # Detailed breakdown for first scenario
        first_scenario = list(analysis_results.keys())[0]
        breakdown = analysis_results[first_scenario]['breakdown']
        breakdown_df = pd.DataFrame.from_dict(breakdown, orient='index')
        breakdown_df['impact_abs'] = breakdown_df['impact'].abs()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=breakdown_df.reset_index(), 
                    x='impact', y='index', 
                    hue='impact_abs', palette="rocket", dodge=False)
        plt.title(f"Asset-level Impact: {first_scenario} Scenario")
        plt.xlabel("Impact on Returns")
        plt.ylabel("Asset")
        plt.legend().remove()
        plt.tight_layout()
        plt.show()
        
        # Inflation scenario paths
        plt.figure(figsize=(12, 6))
        for name, result in analysis_results.items():
            plt.plot(result['scenario'], label=name)
        
        plt.axhline(self.current_inflation, color='black', linestyle='--', label='Current Inflation')
        plt.title("Inflation Scenario Paths")
        plt.ylabel("Inflation Rate (%)")
        plt.xlabel("Months")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive report"""
        self.fetch_current_values()
        results = self.analyze_all_scenarios()
        
        print(f"=== Portfolio Analysis Report for {self.client_data['Client Name']} ===")
        print(f"Current Portfolio Value: ₹{self.client_data['Total Portfolio Size (in lakhs)']:.2f} lakhs")
        print(f"Risk Category: {self.client_data['Risk Category']}")
        print(f"Current Inflation Rate: {self.current_inflation}%")
        print("\nAsset Allocation:")
        for asset, data in self.portfolio.items():
            print(f"- {asset}: {data['allocation']*100:.1f}%")
        
        print("\nScenario Analysis Summary:")
        scenario_df = pd.DataFrame({
            'Scenario': results.keys(),
            'Expected Impact (%)': [r['total_impact']*100 for r in results.values()]
        })
        
        # Use pandas to_string() instead of to_markdown() to avoid tabulate dependency
        print(scenario_df.to_string(index=False))
        
        self.visualize_results(results)

## 3. News Sentiment Analysis Integration

def fetch_financial_news():
    """Fetch recent financial news (using mock data - replace with actual API)"""
    try:
        # This is a placeholder - replace with your actual news API call
        url = "https://service.upstox.com/content/open/v5/news/sub-category/news/list//market-news/stocks?page=1&pageSize=500"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Mock processing - in practice you'd parse real articles
        articles = data.get('data', [])[:5]  # Get top 5 articles
        return [{
            'title': f"Sample Article {i}",
            'summary': f"This is a sample summary about market conditions {i}",
            'sentiment': np.random.choice(['positive', 'neutral', 'negative'])
        } for i in range(5)]
    
    except Exception as e:
        print(f"Error fetching news: {e}")
        return [{
            'title': "Error Loading News",
            'summary': "Could not fetch real news data",
            'sentiment': 'neutral'
        }]

def adjust_sensitivities_based_on_news(sensitivities, news_articles):
    """Adjust asset sensitivities based on news sentiment"""
    sentiment_scores = {
        'positive': 0.1,
        'neutral': 0,
        'negative': -0.1
    }
    
    # Count sentiment in news mentioning inflation
    inflation_mentions = [a for a in news_articles if 'inflation' in a['summary'].lower()]
    if inflation_mentions:
        avg_sentiment = np.mean([sentiment_scores[a['sentiment']] for a in inflation_mentions])
        
        # Adjust sensitivities based on news sentiment
        for asset in sensitivities:
            sensitivities[asset]['short_term'] *= (1 + avg_sentiment)
            sensitivities[asset]['long_term'] *= (1 + avg_sentiment/2)  # Less impact on long-term
    
    return sensitivities

## 4. Interactive Dashboard

def create_interactive_dashboard(clients):
    """Create an interactive dashboard for scenario analysis"""
    client_names = [c['Client Name'] for c in clients]
    
    # Create widgets
    client_dropdown = widgets.Dropdown(options=client_names, description='Client:')
    horizon_radio = widgets.RadioButtons(options=['short_term', 'long_term'], description='Horizon:')
    scenario_dropdown = widgets.Dropdown(
        options=['Mild Increase', 'Sharp Increase', 'Mild Decrease', 'Sharp Decrease', 'Volatile'],
        description='Scenario:'
    )
    
    # Output widget
    out = widgets.Output()
    
    def update_analysis(change):
        """Update analysis based on widget changes"""
        with out:
            out.clear_output()
            
            # Get selected client
            selected_client = next(c for c in clients if c['Client Name'] == client_dropdown.value)
            analyzer = PortfolioAnalyzer(selected_client)
            analyzer.fetch_current_values()
            
            # Get news and adjust sensitivities
            news = fetch_financial_news()
            print("\nLatest Financial News Headlines:")
            for article in news[:3]:  # Show top 3
                print(f"- {article['title']} ({article['sentiment']})")
            
            # Run analysis
            scenario_name = f"{scenario_dropdown.value} (0.3% monthly)" if "Increase" in scenario_dropdown.value or "Decrease" in scenario_dropdown.value else scenario_dropdown.value
            scenario = analyzer.generate_inflation_scenarios()[scenario_name]
            result = analyzer.calculate_portfolio_impact(scenario, horizon_radio.value)
            
            # Display results
            print(f"\nAnalysis for {selected_client['Client Name']}")
            print(f"Risk Profile: {selected_client['Risk Category']}")
            print(f"Portfolio Value: ₹{selected_client['Total Portfolio Size (in lakhs)']:.2f} lakhs")
            print(f"\nScenario: {scenario_dropdown.value}")
            print(f"Expected Return Impact: {result['total_impact']*100:.2f}%")
            
            # Show breakdown
            print("\nAsset-level Impact:")
            breakdown_df = pd.DataFrame.from_dict(result['breakdown'], orient='index')
            print(breakdown_df[['allocation', 'sensitivity', 'impact']].sort_values('impact').to_string())
            
            # Plot scenario path
            plt.figure(figsize=(10, 4))
            plt.plot(scenario, marker='o')
            plt.title(f"Inflation Scenario: {scenario_dropdown.value}")
            plt.ylabel("Inflation Rate (%)")
            plt.xlabel("Months")
            plt.grid(True)
            plt.show()
    
    # Set up observers
    for widget in [client_dropdown, horizon_radio, scenario_dropdown]:
        widget.observe(update_analysis, names='value')
    
    # Initial update
    update_analysis(None)
    
    # Display widgets
    display(widgets.VBox([client_dropdown, horizon_radio, scenario_dropdown, out]))

## 5. Example Usage

if __name__ == "__main__":
    # Example client data
    example_client = {
      "Client Name": "Sneha Sharma",
      "Email ID": "sneha.sharma@example.com",
      "Phone Number": "+917078421792",
      "Equity Portfolio (Stocks, ETFs)": "Infosys, Sensex ETF",
      "Mutual Fund Holdings": "ICICI Pru Equity Fund, Axis Midcap Fund",
      "Total Portfolio Size (in lakhs)": 39.19,
      "Risk Profile Score": 3,
      "Risk Category": "Conservative"
    }
    
    # Create analyzer and run full analysis
    analyzer = PortfolioAnalyzer(example_client)
    analyzer.generate_report()
    
    # For interactive dashboard with multiple clients
    # Uncomment below and pass your full clients list
    # clients_data = [...]  # Your full list of clients
    # create_interactive_dashboard(clients_data)
