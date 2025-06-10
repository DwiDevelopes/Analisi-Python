import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot
import yfinance as yf 

plt.style.use('seaborn')
plt.style.use('seaborn-v0_8') 
sns.set_theme() 

class StockAnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Professional Stock Analysis Tool")
        self.geometry("1200x800")
        self.iconbitmap('analis.ico')
        
        # Data variables
        self.df = None
        self.current_column = None
        self.ticker_symbol = None
        
        # Create UI
        self.create_widgets()
        
        # Configure styles
        self.style = ttk.Style()
        self.configure_styles()
        
    def configure_styles(self):
        """Configure ttk styles"""
        self.style.configure('TFrame', background='#f5f5f5')
        self.style.configure('TLabel', background='#f5f5f5', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10), padding=5)
        self.style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        self.style.configure('Error.TLabel', foreground='red', font=('Arial', 10, 'bold'))
        self.style.configure('TCombobox', padding=5)
        
    def create_widgets(self):
        """Create all widgets"""
        # Main container
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File selection
        self.file_label = ttk.Label(self.control_frame, text="No file selected")
        self.file_label.grid(row=0, column=0, padx=5, sticky=tk.W)
        
        self.browse_button = ttk.Button(self.control_frame, text="Load CSV", command=self.load_csv)
        self.browse_button.grid(row=0, column=1, padx=5)
        
        # Ticker symbol entry
        self.ticker_label = ttk.Label(self.control_frame, text="Yahoo Finance Ticker:")
        self.ticker_label.grid(row=0, column=2, padx=5)
        
        self.ticker_entry = ttk.Entry(self.control_frame, width=10)
        self.ticker_entry.grid(row=0, column=3, padx=5)
        
        self.fetch_button = ttk.Button(self.control_frame, text="Fetch Data", command=self.fetch_yfinance_data)
        self.fetch_button.grid(row=0, column=4, padx=5)
        
        # Column selection
        self.column_label = ttk.Label(self.control_frame, text="Select Column:")
        self.column_label.grid(row=0, column=5, padx=5)
        
        self.column_combo = ttk.Combobox(self.control_frame, state="readonly", width=20)
        self.column_combo.grid(row=0, column=6, padx=5)
        self.column_combo.bind("<<ComboboxSelected>>", self.update_column)
        
        # Analysis button
        self.analyze_button = ttk.Button(self.control_frame, text="Analyze Data", command=self.analyze_data, state=tk.DISABLED)
        self.analyze_button.grid(row=0, column=7, padx=5)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
    def load_csv(self):
        """Load CSV file"""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path, parse_dates=True, index_col=0)
                self.file_label.config(text=f"File: {file_path.split('/')[-1]}")
                self.update_column_combo()
                self.analyze_button.config(state=tk.NORMAL)
                self.ticker_symbol = None
            except Exception as e:
                self.show_error_message(f"Failed to load file: {str(e)}")
    
    def fetch_yfinance_data(self):
        """Fetch data from Yahoo Finance"""
        ticker = self.ticker_entry.get().strip()
        if not ticker:
            messagebox.showwarning("Warning", "Please enter a ticker symbol")
            return
            
        try:
            self.file_label.config(text=f"Fetching data for {ticker}...")
            self.update()
            
            # Fetch historical data
            data = yf.download(ticker, period="5y", auto_adjust=True)
            if data.empty:
                raise ValueError("No data returned for this ticker")
                
            self.df = data
            self.ticker_symbol = ticker
            self.file_label.config(text=f"Data for {ticker} (Yahoo Finance)")
            self.update_column_combo()
            self.analyze_button.config(state=tk.NORMAL)
            
        except Exception as e:
            self.show_error_message(f"Error fetching data: {str(e)}")
            self.file_label.config(text="No file selected")
    
    def update_column_combo(self):
        """Update column combo box with dataframe columns"""
        self.column_combo['values'] = []
        if self.df is not None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.column_combo['values'] = numeric_cols
            if numeric_cols:
                self.column_combo.current(0)
                self.current_column = numeric_cols[0]
    
    def update_column(self, event=None):
        """Update current column"""
        if self.df is not None and self.column_combo.get():
            self.current_column = self.column_combo.get()
    
    def analyze_data(self):
        """Perform data analysis and generate visualizations"""
        if self.df is None or self.current_column is None:
            return
            
        # Clear previous tabs
        for child in self.notebook.winfo_children():
            child.destroy()
        
        try:
            # Basic Statistics Tab
            self.create_statistics_tab()
            
            # Time Series Tab
            self.create_time_series_tab()
            
            # Distribution Tab
            self.create_distribution_tab()
            
            # Correlation Tab
            self.create_correlation_tab()
            
            # Decomposition Tab
            self.create_decomposition_tab()
            
            # Moving Average Tab
            self.create_moving_avg_tab()
            
            # Volatility Tab
            self.create_volatility_tab()
            
            # Stationarity Test Tab
            self.create_stationarity_tab()
            
            # ACF/PACF Tab
            self.create_acf_pacf_tab()
            
            # Technical Indicators Tab
            self.create_technical_indicators_tab()
            
            # Monte Carlo Simulation Tab (for stocks)
            if self.ticker_symbol:
                self.create_monte_carlo_tab()
            
        except Exception as e:
            self.show_error_message(f"Analysis error: {str(e)}")
    
    def create_tab_with_scroll(self, title):
        """Create a tab with scrollable canvas"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=title)
        
        # Create canvas with scrollbar
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        return scrollable_frame
    
    def create_statistics_tab(self):
        """Create tab with basic statistics"""
        frame = self.create_tab_with_scroll("Basic Statistics")
        
        # Calculate statistics
        stats = self.df[self.current_column].describe()
        stats_df = pd.DataFrame(stats)
        stats_df.columns = ['Value']
        
        # Add additional statistics
        stats_df.loc['Skewness'] = self.df[self.current_column].skew()
        stats_df.loc['Kurtosis'] = self.df[self.current_column].kurtosis()
        stats_df.loc['Median'] = self.df[self.current_column].median()
        stats_df.loc['IQR'] = stats_df.loc['75%'] - stats_df.loc['25%']
        
        # Create table visualization
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.axis('off')
        ax1.table(cellText=stats_df.values,
                 colLabels=stats_df.columns,
                 rowLabels=stats_df.index,
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.5]*len(stats_df.columns))
        ax1.set_title('Descriptive Statistics')
        
        # Add to canvas
        canvas1 = FigureCanvasTkAgg(fig1, master=frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Add pie chart of value composition
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        top_values = self.df[self.current_column].nlargest(5)
        top_values.plot.pie(autopct='%1.1f%%', ax=ax2)
        ax2.set_title('Top 5 Value Composition')
        
        canvas2 = FigureCanvasTkAgg(fig2, master=frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Add returns statistics if this is stock data
        if self.ticker_symbol and 'Close' in self.df.columns:
            returns = self.df['Close'].pct_change().dropna()
            returns_stats = returns.describe()
            returns_stats_df = pd.DataFrame(returns_stats)
            returns_stats_df.columns = ['Daily Returns']
            
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.axis('off')
            ax3.table(cellText=returns_stats_df.values,
                     colLabels=returns_stats_df.columns,
                     rowLabels=returns_stats_df.index,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.5]*len(returns_stats_df.columns))
            ax3.set_title('Daily Returns Statistics')
            
            canvas3 = FigureCanvasTkAgg(fig3, master=frame)
            canvas3.draw()
            canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
    
    def create_time_series_tab(self):
        """Create time series visualization tab"""
        frame = self.create_tab_with_scroll("Time Series")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        self.df[self.current_column].plot(ax=ax)
        ax.set_title(f'Time Series - {self.current_column}')
        ax.set_ylabel('Value')
        ax.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_distribution_tab(self):
        """Create distribution visualization tab"""
        frame = self.create_tab_with_scroll("Distribution")
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Histogram
        sns.histplot(self.df[self.current_column], kde=True, ax=ax1)
        ax1.set_title(f'Distribution - {self.current_column}')
        
        # Boxplot
        sns.boxplot(y=self.df[self.current_column], ax=ax2)
        ax2.set_title(f'Boxplot - {self.current_column}')
        
        # QQ plot
        from scipy import stats
        stats.probplot(self.df[self.current_column].dropna(), plot=ax3)
        ax3.set_title(f'Q-Q Plot - {self.current_column}')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_correlation_tab(self):
        """Create correlation visualization tab"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return
            
        frame = self.create_tab_with_scroll("Correlation")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = self.df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix')
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Pairplot for top correlated features
        top_corr = corr_matrix[self.current_column].abs().sort_values(ascending=False).index[1:4]
        if len(top_corr) >= 2:
            fig2 = plt.figure(figsize=(12, 10))
            sns.pairplot(self.df[top_corr.tolist() + [self.current_column]])
            
            canvas2 = FigureCanvasTkAgg(fig2, master=frame)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
    
    def create_decomposition_tab(self):
        """Create time series decomposition tab"""
        frame = self.create_tab_with_scroll("Decomposition")
        
        try:
            # Ensure datetime index
            if not isinstance(self.df.index, pd.DatetimeIndex):
                self.df.index = pd.to_datetime(self.df.index)
            
            # Handle missing values
            if self.df[self.current_column].isnull().any():
                self.df[self.current_column] = self.df[self.current_column].interpolate()
            
            # Determine period based on data frequency
            freq = pd.infer_freq(self.df.index)
            if freq == 'D':
                period = 30  # Monthly seasonality for daily data
            elif freq == 'M':
                period = 12  # Yearly seasonality for monthly data
            else:
                period = 7  # Default weekly seasonality
                
            decomposition = seasonal_decompose(self.df[self.current_column], 
                                             model='additive', 
                                             period=period,
                                             extrapolate_trend='freq')
            
            fig = decomposition.plot()
            fig.set_size_inches(12, 8)
            fig.suptitle(f'Decomposition - {self.current_column}')
            
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            self.show_error_message(f"Decomposition error: {str(e)}")
    
    def create_moving_avg_tab(self):
        """Create moving average visualization tab"""
        frame = self.create_tab_with_scroll("Moving Averages")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot original
        self.df[self.current_column].plot(ax=ax, label='Original', alpha=0.5)
        
        # Calculate and plot moving averages
        windows = [7, 20, 50, 200] if len(self.df) > 300 else [7, 14, 30, 90]
        for window in windows:
            if window < len(self.df):
                ma = self.df[self.current_column].rolling(window=window).mean()
                ma.plot(ax=ax, label=f'MA {window}')
        
        ax.set_title(f'Moving Averages - {self.current_column}')
        ax.legend()
        ax.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_volatility_tab(self):
        """Create volatility visualization tab"""
        frame = self.create_tab_with_scroll("Volatility")
        
        # Calculate daily returns (if Close column exists, use that)
        col = 'Close' if 'Close' in self.df.columns else self.current_column
        returns = self.df[col].pct_change().dropna()
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Returns plot
        returns.plot(ax=ax1)
        ax1.set_title(f'Daily Returns - {col}')
        ax1.grid(True)
        
        # Volatility (rolling std)
        windows = [10, 30, 90] if len(returns) > 100 else [5, 10, 20]
        for window in windows:
            if window < len(returns):
                volatility = returns.rolling(window=window).std()
                volatility.plot(ax=ax2, label=f'{window}-day rolling volatility')
        ax2.set_title(f'Volatility (Rolling Std Dev) - {col}')
        ax2.legend()
        ax2.grid(True)
        
        # Cumulative returns
        cumulative_returns = (1 + returns).cumprod() - 1
        cumulative_returns.plot(ax=ax3)
        ax3.set_title(f'Cumulative Returns - {col}')
        ax3.grid(True)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_stationarity_tab(self):
        """Create stationarity test tab"""
        frame = self.create_tab_with_scroll("Stationarity")
        
        # Perform Augmented Dickey-Fuller test
        result = adfuller(self.df[self.current_column].dropna())
        adf_statistic = result[0]
        p_value = result[1]
        critical_values = result[4]
        
        # Create results table
        test_results = pd.DataFrame({
            'Metric': ['ADF Statistic', 'p-value', 'Critical Values'],
            'Value': [adf_statistic, p_value, str(critical_values)]
        })
        
        fig1, ax1 = plt.subplots(figsize=(10, 2))
        ax1.axis('off')
        ax1.table(cellText=test_results.values,
                 colLabels=test_results.columns,
                 cellLoc='center',
                 loc='center')
        ax1.set_title('Augmented Dickey-Fuller Test Results')
        
        canvas1 = FigureCanvasTkAgg(fig1, master=frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Plot rolling statistics
        fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Rolling mean and std
        rolling_window = min(30, len(self.df) // 4)
        rolling_mean = self.df[self.current_column].rolling(window=rolling_window).mean()
        rolling_std = self.df[self.current_column].rolling(window=rolling_window).std()
        
        self.df[self.current_column].plot(ax=ax2, label='Original')
        rolling_mean.plot(ax=ax2, label='Rolling Mean')
        ax2.set_title('Rolling Mean & Standard Deviation')
        ax2.legend()
        ax2.grid(True)
        
        rolling_std.plot(ax=ax3, label='Rolling Std', color='red')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        
        canvas2 = FigureCanvasTkAgg(fig2, master=frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
    
    def create_acf_pacf_tab(self):
        """Create ACF and PACF plots tab"""
        frame = self.create_tab_with_scroll("ACF/PACF")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # ACF plot
        plot_acf(self.df[self.current_column].dropna(), lags=40, ax=ax1)
        ax1.set_title('Autocorrelation Function (ACF)')
        
        # PACF plot
        plot_pacf(self.df[self.current_column].dropna(), lags=40, ax=ax2)
        ax2.set_title('Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_technical_indicators_tab(self):
        """Create tab with technical indicators"""
        if not self.ticker_symbol or 'Close' not in self.df.columns:
            return
            
        frame = self.create_tab_with_scroll("Technical Indicators")
        
        # Calculate indicators
        close_prices = self.df['Close']
        
        # RSI
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        exp12 = close_prices.ewm(span=12, adjust=False).mean()
        exp26 = close_prices.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        sma20 = close_prices.rolling(window=20).mean()
        std20 = close_prices.rolling(window=20).std()
        upper_band = sma20 + (std20 * 2)
        lower_band = sma20 - (std20 * 2)
        
        # Create plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Price with Bollinger Bands
        close_prices.plot(ax=ax1, label='Close Price')
        sma20.plot(ax=ax1, label='20-day SMA')
        upper_band.plot(ax=ax1, label='Upper Band', linestyle='--')
        lower_band.plot(ax=ax1, label='Lower Band', linestyle='--')
        ax1.fill_between(close_prices.index, lower_band, upper_band, alpha=0.1)
        ax1.set_title('Bollinger Bands')
        ax1.legend()
        ax1.grid(True)
        
        # RSI
        rsi.plot(ax=ax2)
        ax2.axhline(70, color='r', linestyle='--')
        ax2.axhline(30, color='g', linestyle='--')
        ax2.set_title('Relative Strength Index (RSI)')
        ax2.set_ylim(0, 100)
        ax2.grid(True)
        
        # MACD
        macd.plot(ax=ax3, label='MACD')
        signal.plot(ax=ax3, label='Signal Line')
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title('Moving Average Convergence Divergence (MACD)')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_monte_carlo_tab(self):
        """Create Monte Carlo simulation tab for stock prices"""
        if not self.ticker_symbol or 'Close' not in self.df.columns:
            return
            
        frame = self.create_tab_with_scroll("Monte Carlo Simulation")
        
        # Calculate daily returns and volatility
        returns = self.df['Close'].pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std()
        
        # Simulation parameters
        days = 252  # 1 year of trading days
        num_simulations = 100
        last_price = self.df['Close'].iloc[-1]
        
        # Run simulations
        np.random.seed(42)
        price_paths = np.zeros((days, num_simulations))
        price_paths[0] = last_price
        
        for t in range(1, days):
            shock = np.random.normal(mu, sigma, num_simulations)
            price_paths[t] = price_paths[t-1] * (1 + shock)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot all simulations
        for i in range(num_simulations):
            ax.plot(price_paths[:, i], linewidth=1, alpha=0.05, color='blue')
        
        # Plot mean path
        mean_path = price_paths.mean(axis=1)
        ax.plot(mean_path, linewidth=2, color='red', label='Mean Path')
        
        # Plot confidence intervals
        upper_ci = np.percentile(price_paths, 95, axis=1)
        lower_ci = np.percentile(price_paths, 5, axis=1)
        ax.plot(upper_ci, linestyle='--', color='green', label='95% CI')
        ax.plot(lower_ci, linestyle='--', color='green', label='5% CI')
        
        ax.set_title(f'Monte Carlo Simulation ({num_simulations} paths)')
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Add statistics
        final_prices = price_paths[-1]
        mean_final_price = final_prices.mean()
        median_final_price = np.median(final_prices)
        prob_profit = (final_prices > last_price).mean() * 100
        max_loss = (final_prices.min() - last_price) / last_price * 100
        max_gain = (final_prices.max() - last_price) / last_price * 100
        
        stats_text = (
            f"Simulation Results:\n\n"
            f"Initial Price: {last_price:.2f}\n"
            f"Mean Final Price: {mean_final_price:.2f}\n"
            f"Median Final Price: {median_final_price:.2f}\n"
            f"Probability of Profit: {prob_profit:.1f}%\n"
            f"Maximum Simulated Loss: {max_loss:.1f}%\n"
            f"Maximum Simulated Gain: {max_gain:.1f}%"
        )
        
        stats_label = ttk.Label(frame, text=stats_text, font=('Arial', 10))
        stats_label.pack(pady=10)
    
    def show_error_message(self, message):
        """Show error message"""
        messagebox.showerror("Error", message)

if __name__ == "__main__":
    app = StockAnalysisApp()
    app.mainloop()