import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="QUANTEDGE - Advanced Quant Analysis",
    page_icon="📈",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3B82F6;
        padding-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.4rem;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin: 0.5rem 0;
    }
    .stButton button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
    }
    .success-box {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 应用标题
st.markdown('<h1 class="main-header">📊 QUANTEDGE By LabGen25</h1>', unsafe_allow_html=True)
st.markdown("### Advanced Quantitative Analysis Platform for Global Indices")

# 侧边栏配置
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # 日期范围选择
    st.markdown("### 📅 Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", 
                                  value=datetime(2020, 1, 1),
                                  min_value=datetime(2000, 1, 1))
    with col2:
        end_date = st.date_input("End Date", 
                                value=datetime.now(),
                                min_value=datetime(2000, 1, 1))
    
    # 分析参数
    st.markdown("### 📊 Analysis Parameters")
    
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100
    
    # 布林带参数
    st.markdown("#### Bollinger Bands Parameters")
    bb_window = st.slider("Window Size (days)", 10, 100, 20)
    bb_std = st.slider("Standard Deviations", 1.0, 3.0, 2.0, 0.1)
    
    # 指数选择 - 使用正确的Yahoo Finance代码
    st.markdown("### 🌐 Select Indices")
    
    # 使用正确的Yahoo Finance代码
    all_indices = {
        '^HSI': 'Hang Seng',
        '^IXIC': 'Nasdaq',
        '^GSPC': 'S&P 500',
        '^RUT': 'US Russell 2000',
        '^BVSP': 'Bovespa',
        'URTH': "MSCI World",  # iShares MSCI World ETF代替
        '^N225': "Nikkei 225",
        '^STOXX': 'STOXX 600',
        '^GDAXI': 'DAX',
        '^FTSE': 'FTSE 100',
        '^FCHI': 'CAC 40',
        '^STOXX50E': 'EURO STOXX 50',
        '^SSMI': 'SWISS SMI',
        'XU100.IS': 'BIST 100',
        '^AXJO': 'S&P/ASX 200',
        '000001.SS': 'Shanghai Composite',
        '399001.SZ': 'SZSE Component',
        'CNYA': 'China A50 ETF',  # iShares China A ETF代替
        '^KS11': 'KOSPI',
        '^TWII': 'Taiwan Weighted',
        '^NSEI': 'NIFTY 50'
    }
    
    # 让用户选择要分析的指数
    selected_indices = {}
    for ticker, name in all_indices.items():
        if st.checkbox(f"{name} ({ticker})", value=True):
            selected_indices[ticker] = name
    
    if not selected_indices:
        st.warning("⚠️ Please select at least one index")
    
    # 分析按钮
    st.markdown("---")
    analyze_button = st.button("🚀 Run Quantitative Analysis", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    ### 📚 About QUANTEDGE
    **Advanced quantitative analysis platform** 
    for global financial indices.
    
    Features:
    • Individual index analysis
    • Bollinger Bands on log returns
    • Comprehensive performance metrics
    • Risk-adjusted returns analysis
    
    **Developed by LabGen25**
    """)

# 数据处理函数
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_index_data(tickers_dict, start_date, end_date):
    """获取多个指数的历史数据"""
    all_data = pd.DataFrame()
    failed_tickers = []
    
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    total_tickers = len(tickers_dict)
    
    for i, (ticker, name) in enumerate(tickers_dict.items()):
        try:
            status_text.text(f"📥 Downloading {name}...")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if not data.empty and len(data) > 10:
                # 使用调整后的收盘价
                if 'Adj Close' in data.columns:
                    price_series = data['Adj Close']
                else:
                    price_series = data['Close']
                
                price_series.name = name
                all_data = pd.concat([all_data, price_series], axis=1)
            else:
                failed_tickers.append((ticker, "Insufficient data"))
        except Exception as e:
            failed_tickers.append((ticker, str(e)[:50]))
        
        progress_bar.progress((i + 1) / total_tickers)
    
    progress_bar.empty()
    status_text.empty()
    
    if failed_tickers:
        with st.sidebar.expander("⚠️ Failed Downloads", expanded=False):
            for ticker, error in failed_tickers:
                st.write(f"{ticker}: {error}")
    
    return all_data

# 计算对数收益率
def calculate_log_returns(prices):
    """计算对数收益率"""
    return np.log(prices / prices.shift(1)).dropna()

# 计算布林带
def calculate_bollinger_bands(series, window=20, num_std=2):
    """计算布林带"""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    return rolling_mean, upper_band, lower_band

# 手动计算绩效指标（不使用QuantStats以避免API问题）
def calculate_performance_metrics(prices, risk_free_rate=0.02):
    """手动计算绩效指标"""
    metrics_dict = {}
    
    for idx_name in prices.columns:
        try:
            # 获取该指数的价格和收益率
            idx_prices = prices[idx_name].dropna()
            
            if len(idx_prices) < 50:  # 数据太少
                continue
            
            # 计算收益率
            returns = idx_prices.pct_change().dropna()
            log_returns = np.log(idx_prices / idx_prices.shift(1)).dropna()
            
            # 基本参数
            trading_days = 252
            years = len(idx_prices) / trading_days
            
            # 总收益率
            total_return = (idx_prices.iloc[-1] / idx_prices.iloc[0] - 1) * 100
            
            # 年化收益率 (CAGR)
            cagr = ((1 + total_return/100) ** (1/years) - 1) * 100
            
            # 年化波动率
            annual_volatility = returns.std() * np.sqrt(trading_days) * 100
            
            # 夏普比率
            if annual_volatility > 0:
                sharpe_ratio = (cagr - risk_free_rate * 100) / annual_volatility
            else:
                sharpe_ratio = np.nan
            
            # 最大回撤
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative / running_max - 1) * 100
            max_drawdown = drawdown.min()
            
            # 索提诺比率
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                downside_std = negative_returns.std() * np.sqrt(trading_days) * 100
                if downside_std > 0:
                    sortino_ratio = (cagr - risk_free_rate * 100) / downside_std
                else:
                    sortino_ratio = np.nan
            else:
                sortino_ratio = np.nan
            
            # Calmar比率
            if max_drawdown != 0:
                calmar_ratio = (cagr - risk_free_rate * 100) / abs(max_drawdown)
            else:
                calmar_ratio = np.nan
            
            # Omega比率 (近似计算)
            threshold = risk_free_rate / trading_days  # 日无风险利率
            excess_returns = returns - threshold
            positive_excess = excess_returns[excess_returns > 0].sum()
            negative_excess = abs(excess_returns[excess_returns < 0].sum())
            omega_ratio = positive_excess / negative_excess if negative_excess > 0 else np.nan
            
            # 偏度
            skewness = returns.skew()
            
            # 峰度
            kurtosis = returns.kurtosis()
            
            # VaR (95%)
            var_95 = np.percentile(returns, 5) * 100
            
            # CVaR (95%)
            cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
            
            # 赢率
            win_rate = (returns > 0).mean() * 100
            
            # 平均收益/平均损失
            avg_win = returns[returns > 0].mean() * 100 if len(returns[returns > 0]) > 0 else 0
            avg_loss = returns[returns < 0].mean() * 100 if len(returns[returns < 0]) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
            
            metrics_dict[idx_name] = {
                'Total Return (%)': total_return,
                'CAGR (%)': cagr,
                'Volatility (%)': annual_volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Sortino Ratio': sortino_ratio,
                'Calmar Ratio': calmar_ratio,
                'Max Drawdown (%)': max_drawdown,
                'Omega Ratio': omega_ratio,
                'Skewness': skewness,
                'Kurtosis': kurtosis,
                'VaR 95% (%)': var_95,
                'CVaR 95% (%)': cvar_95,
                'Win Rate (%)': win_rate,
                'Profit Factor': profit_factor,
                'Avg Win (%)': avg_win,
                'Avg Loss (%)': avg_loss
            }
            
        except Exception as e:
            st.warning(f"Error calculating metrics for {idx_name}: {str(e)[:100]}")
            continue
    
    return pd.DataFrame(metrics_dict).T

# 绘制单个指数的标准化价格图表
def plot_normalized_price_single(index_name, prices, ax):
    """绘制单个指数的标准化价格图表"""
    if index_name not in prices.columns:
        return
    
    normalized_prices = prices[index_name] / prices[index_name].iloc[0] * 100
    
    ax.plot(normalized_prices.index, normalized_prices.values, 
            linewidth=2, color='#3B82F6', label='Normalized Price')
    
    ax.set_title(f'{index_name} - Normalized Price', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Price (Start=100)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend()
    
    # 添加统计信息
    total_return = (normalized_prices.iloc[-1] - 100) / 100 * 100
    days_held = (normalized_prices.index[-1] - normalized_prices.index[0]).days
    if days_held > 0:
        annualized_return = ((1 + total_return/100) ** (365/days_held) - 1) * 100
        stats_text = f'Total Return: {total_return:.2f}%\nAnnualized: {annualized_return:.2f}%'
    else:
        stats_text = f'Total Return: {total_return:.2f}%'
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 绘制单个指数的布林带对数收益率图表
def plot_bollinger_bands_log_returns_single(index_name, prices, ax, window=20, num_std=2):
    """绘制单个指数的布林带对数收益率图表"""
    if index_name not in prices.columns:
        return
    
    # 计算对数收益率
    log_returns = calculate_log_returns(prices[index_name].dropna())
    
    # 计算布林带
    rolling_mean, upper_band, lower_band = calculate_bollinger_bands(
        log_returns, window=window, num_std=num_std
    )
    
    # 绘制图表
    ax.plot(log_returns.index, log_returns.values * 100, 
            linewidth=1, color='#666666', alpha=0.7, label='Log Returns')
    ax.plot(rolling_mean.index, rolling_mean.values * 100, 
            linewidth=2, color='#3B82F6', label=f'{window}-day MA')
    ax.plot(upper_band.index, upper_band.values * 100, 
            linewidth=1.5, color='#EF4444', linestyle='--', label=f'Upper Band (+{num_std}σ)')
    ax.plot(lower_band.index, lower_band.values * 100, 
            linewidth=1.5, color='#10B981', linestyle='--', label=f'Lower Band (-{num_std}σ)')
    
    # 填充布林带区域
    ax.fill_between(rolling_mean.index, 
                    lower_band.values * 100, 
                    upper_band.values * 100, 
                    alpha=0.2, color='#3B82F6')
    
    # 标记超出布林带的点
    above_upper = log_returns > upper_band
    below_lower = log_returns < lower_band
    
    if above_upper.any():
        ax.scatter(log_returns[above_upper].index, 
                  log_returns[above_upper].values * 100,
                  color='#EF4444', s=30, label='Above Upper Band', zorder=5)
    
    if below_lower.any():
        ax.scatter(log_returns[below_lower].index, 
                  log_returns[below_lower].values * 100,
                  color='#10B981', s=30, label='Below Lower Band', zorder=5)
    
    ax.set_title(f'{index_name} - Log Returns with Bollinger Bands ({window}d, {num_std}σ)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Log Returns (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=9)
    
    # 添加统计信息
    mean_return = log_returns.mean() * 100 * 252  # 年化
    std_return = log_returns.std() * 100 * np.sqrt(252)  # 年化
    
    # 计算超出布林带的百分比
    pct_above_upper = (above_upper.sum() / len(log_returns)) * 100 if len(log_returns) > 0 else 0
    pct_below_lower = (below_lower.sum() / len(log_returns)) * 100 if len(log_returns) > 0 else 0
    
    stats_text = f'Ann. Return: {mean_return:.2f}%\nAnn. Vol: {std_return:.2f}%\n'
    stats_text += f'Above Upper: {pct_above_upper:.1f}%\nBelow Lower: {pct_below_lower:.1f}%'
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 主应用逻辑
if analyze_button and selected_indices:
    
    # 获取数据
    with st.spinner("📊 Fetching market data..."):
        data = fetch_index_data(selected_indices, start_date, end_date)
    
    if data.empty or len(data) < 50:
        st.error("❌ Insufficient data retrieved. Please check your selections and try again.")
        st.stop()
    
    # 前向填充处理
    data_ffilled = data.ffill().bfill().dropna(axis=1, how='all')
    
    if data_ffilled.empty:
        st.error("❌ No valid data after processing.")
        st.stop()
    
    # 计算指标
    with st.spinner("📈 Calculating quantitative metrics..."):
        metrics_df = calculate_performance_metrics(data_ffilled, risk_free_rate)
    
    if metrics_df.empty:
        st.error("❌ Could not calculate metrics. Please try different parameters.")
        st.stop()
    
    # 存储到会话状态
    st.session_state.data = data_ffilled
    st.session_state.metrics = metrics_df
    st.session_state.bb_params = {'window': bb_window, 'std': bb_std}
    
elif 'data' in st.session_state:
    data_ffilled = st.session_state.data
    metrics_df = st.session_state.metrics
    bb_params = st.session_state.bb_params
else:
    st.info("👈 Configure your analysis in the sidebar and click 'Run Quantitative Analysis'")
    st.stop()

# 显示概览信息
st.markdown('<div class="sub-header">📊 Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Indices Analyzed", len(data_ffilled.columns))
with col2:
    st.metric("Date Range", f"{data_ffilled.index[0].date()} to {data_ffilled.index[-1].date()}")
with col3:
    st.metric("Trading Days", len(data_ffilled))
with col4:
    st.metric("Risk-Free Rate", f"{risk_free_rate*100:.1f}%")

# 显示成功信息
st.markdown(f"""
<div class="success-box">
✅ Successfully analyzed {len(data_ffilled.columns)} indices with {len(data_ffilled)} trading days of data.
</div>
""", unsafe_allow_html=True)

# 显示绩效指标
st.markdown('<div class="sub-header">📈 Performance Metrics</div>', unsafe_allow_html=True)

# 排序选项
sort_options = ['CAGR (%)', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (%)', 'Volatility (%)', 'Omega Ratio']
sort_by = st.selectbox("Sort metrics by:", sort_options, index=0)

if not metrics_df.empty:
    # 按选择排序（降序，除了最大回撤）
    if sort_by == 'Max Drawdown (%)':
        sorted_metrics = metrics_df.sort_values(sort_by)
    else:
        sorted_metrics = metrics_df.sort_values(sort_by, ascending=False)
    
    # 选择要显示的主要指标
    display_metrics = sorted_metrics[['CAGR (%)', 'Volatility (%)', 'Sharpe Ratio', 
                                     'Sortino Ratio', 'Max Drawdown (%)', 'Calmar Ratio',
                                     'Win Rate (%)']].copy()
    
    # 格式化显示
    for col in display_metrics.columns:
        if '%' in col:
            display_metrics[col] = display_metrics[col].apply(lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A")
        else:
            display_metrics[col] = display_metrics[col].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
    
    st.dataframe(display_metrics, use_container_width=True)

# 单独显示每个指数的图表
st.markdown('<div class="sub-header">📊 Individual Index Analysis</div>', unsafe_allow_html=True)

# 创建标签页用于显示每个指数
tab_names = list(data_ffilled.columns)
tabs = st.tabs(tab_names)

for i, (tab, index_name) in enumerate(zip(tabs, tab_names)):
    with tab:
        if index_name not in data_ffilled.columns:
            st.warning(f"No data available for {index_name}")
            continue
        
        st.markdown(f'<div class="section-header">{index_name} Analysis</div>', unsafe_allow_html=True)
        
        # 创建两列布局：图表和指标
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("##### 📈 Normalized Price Chart")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            plot_normalized_price_single(index_name, data_ffilled, ax1)
            st.pyplot(fig1)
            plt.close(fig1)
        
        with col_chart2:
            st.markdown(f"##### 📊 Log Returns with Bollinger Bands ({bb_window}d, {bb_std}σ)")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            plot_bollinger_bands_log_returns_single(
                index_name, data_ffilled, ax2, 
                window=bb_window, num_std=bb_std
            )
            st.pyplot(fig2)
            plt.close(fig2)
        
        # 显示该指数的详细指标
        st.markdown("##### 📋 Detailed Performance Metrics")
        if index_name in metrics_df.index:
            index_metrics = metrics_df.loc[index_name]
            
            # 创建三列布局显示指标
            col1, col2, col3 = st.columns(3)
            
            # 第一列：收益指标
            with col1:
                metrics_group1 = {
                    'Total Return (%)': 'Total Return',
                    'CAGR (%)': 'Annual Return',
                    'Volatility (%)': 'Annual Volatility',
                    'Sharpe Ratio': 'Sharpe Ratio'
                }
                
                for metric_key, metric_label in metrics_group1.items():
                    if metric_key in index_metrics:
                        value = index_metrics[metric_key]
                        if '%' in metric_key:
                            display_value = f"{value:.2f}%"
                        else:
                            display_value = f"{value:.3f}"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong>{metric_label}</strong><br>
                            <span style="font-size: 1.2rem; color: #1E40AF;">{display_value}</span>
                        </div>
                        """, unsafe_allow_html=True)
            
            # 第二列：风险指标
            with col2:
                metrics_group2 = {
                    'Sortino Ratio': 'Sortino Ratio',
                    'Max Drawdown (%)': 'Max Drawdown',
                    'Calmar Ratio': 'Calmar Ratio',
                    'VaR 95% (%)': 'VaR 95%'
                }
                
                for metric_key, metric_label in metrics_group2.items():
                    if metric_key in index_metrics:
                        value = index_metrics[metric_key]
                        if '%' in metric_key:
                            display_value = f"{value:.2f}%"
                        else:
                            display_value = f"{value:.3f}"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong>{metric_label}</strong><br>
                            <span style="font-size: 1.2rem; color: #1E40AF;">{display_value}</span>
                        </div>
                        """, unsafe_allow_html=True)
            
            # 第三列：统计指标
            with col3:
                metrics_group3 = {
                    'Omega Ratio': 'Omega Ratio',
                    'Win Rate (%)': 'Win Rate',
                    'Profit Factor': 'Profit Factor',
                    'Skewness': 'Skewness'
                }
                
                for metric_key, metric_label in metrics_group3.items():
                    if metric_key in index_metrics:
                        value = index_metrics[metric_key]
                        if '%' in metric_key:
                            display_value = f"{value:.2f}%"
                        elif metric_key == 'Profit Factor':
                            display_value = f"{value:.2f}x" if not pd.isna(value) else "N/A"
                        else:
                            display_value = f"{value:.3f}" if not pd.isna(value) else "N/A"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong>{metric_label}</strong><br>
                            <span style="font-size: 1.2rem; color: #1E40AF;">{display_value}</span>
                        </div>
                        """, unsafe_allow_html=True)

# 风险回报分析
st.markdown('<div class="sub-header">🎯 Risk-Return Analysis</div>', unsafe_allow_html=True)

if not metrics_df.empty:
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    
    # 创建散点图
    scatter = ax3.scatter(
        metrics_df['Volatility (%)'], 
        metrics_df['CAGR (%)'],
        c=metrics_df['Sharpe Ratio'], 
        s=200, 
        cmap='RdYlGn', 
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    
    # 添加指数标签
    for idx in metrics_df.index:
        if pd.notna(metrics_df.loc[idx, 'Volatility (%)']) and pd.notna(metrics_df.loc[idx, 'CAGR (%)']):
            ax3.annotate(idx, 
                        (metrics_df.loc[idx, 'Volatility (%)'], 
                         metrics_df.loc[idx, 'CAGR (%)']),
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=9,
                        alpha=0.8)
    
    ax3.set_xlabel('Annual Volatility (%)', fontsize=12)
    ax3.set_ylabel('CAGR (%)', fontsize=12)
    ax3.set_title('Risk-Return Profile (Color = Sharpe Ratio)', 
                  fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Sharpe Ratio', fontsize=12)
    
    st.pyplot(fig3)
    plt.close(fig3)

# 回撤分析
st.markdown('<div class="sub-header">📉 Maximum Drawdown Analysis</div>', unsafe_allow_html=True)

if not metrics_df.empty:
    fig4, (ax4_1, ax4_2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 最大回撤条形图
    max_dd_data = metrics_df['Max Drawdown (%)'].dropna()
    if len(max_dd_data) > 0:
        max_dd_sorted = max_dd_data.sort_values()
        colors_dd = plt.cm.RdYlGn_r(
            (max_dd_sorted - max_dd_sorted.min()) / 
            (max_dd_sorted.max() - max_dd_sorted.min() + 1e-10)
        )
        
        ax4_1.barh(range(len(max_dd_sorted)), max_dd_sorted.values, color=colors_dd)
        ax4_1.set_yticks(range(len(max_dd_sorted)))
        ax4_1.set_yticklabels(max_dd_sorted.index, fontsize=10)
        ax4_1.set_xlabel('Maximum Drawdown (%)', fontsize=12)
        ax4_1.set_title('Maximum Drawdown by Index', fontsize=14, fontweight='bold')
        ax4_1.grid(True, alpha=0.3, axis='x')
    
    # 回撤 vs 收益散点图
    valid_data = metrics_df.dropna(subset=['Max Drawdown (%)', 'CAGR (%)'])
    if len(valid_data) > 0:
        scatter_dd = ax4_2.scatter(
            valid_data['Max Drawdown (%)'].abs(),
            valid_data['CAGR (%)'],
            c=valid_data['Calmar Ratio'],
            s=150,
            cmap='RdYlGn',
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        for idx in valid_data.index:
            ax4_2.annotate(idx,
                          (abs(valid_data.loc[idx, 'Max Drawdown (%)']),
                           valid_data.loc[idx, 'CAGR (%)']),
                          xytext=(5, 5),
                          textcoords='offset points',
                          fontsize=9,
                          alpha=0.8)
        
        ax4_2.set_xlabel('Maximum Drawdown (%)', fontsize=12)
        ax4_2.set_ylabel('CAGR (%)', fontsize=12)
        ax4_2.set_title('Return vs Drawdown (Color = Calmar Ratio)', 
                        fontsize=14, fontweight='bold')
        ax4_2.grid(True, alpha=0.3, linestyle='--')
        
        plt.colorbar(scatter_dd, ax=ax4_2, label='Calmar Ratio')
    
    st.pyplot(fig4)
    plt.close(fig4)

# 数据下载选项
st.markdown('<div class="sub-header">💾 Download Analysis Results</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    metrics_csv = metrics_df.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download All Metrics",
        data=metrics_csv,
        file_name=f"quantedge_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col2:
    prices_csv = data_ffilled.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Price Data",
        data=prices_csv,
        file_name=f"quantedge_prices_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col3:
    # 计算收益率数据
    returns = data_ffilled.pct_change().dropna()
    returns_csv = returns.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Returns Data",
        data=returns_csv,
        file_name=f"quantedge_returns_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; padding: 1rem;'>
    <p style='font-size: 1.1rem; font-weight: bold;'>QUANTEDGE Advanced Quantitative Analysis</p>
    <p>Data from Yahoo Finance | All calculations performed internally</p>
    <p>Developed by LabGen25 | {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
