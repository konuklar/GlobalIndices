import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import quantstats as qs
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
    .index-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
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
    
    # 指数选择
    st.markdown("### 🌐 Select Indices")
    
    # 预定义的指数列表
    all_indices = {
        '^HSI': 'Hang Seng',
        '^IXIC': 'Nasdaq',
        '^GSPC': 'S&P 500',
        '^RUT': 'US Russell 2000',
        '^BVSP': 'Bovespa',
        'MSCIW.PA': "MSCI All-World",
        '^N225': "Nikkei 225",
        '^STOXX': 'STOXX 600',
        '^GDAXI': 'DAX',
        '^FTSE': 'FTSE 100',
        '^FCHI': 'CAC 40',
        '^STOXX50E': 'EURO STOXX 50',
        '^SSMI': 'SWISS SMI',
        'XU100.IS': 'BIST 100',
        '^AXJO': 'S&P/ASX 200',
        '000001.SS': 'Shanghai Index',
        '399001.SZ': 'SZSE Component',
        '^SSE50': 'China A50',
        '^KS11': 'KOSPI',
        '^TWII': 'Taiwan Weighted',
        '^NSEI': 'NIFTY 50'
    }
    
    # 让用户选择要分析的指数
    selected_indices = {}
    for ticker, name in all_indices.items():
        if st.checkbox(name, value=True):
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
    for global financial indices using **QuantStats**.
    
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
    
    progress_text = st.sidebar.empty()
    
    for i, (ticker, name) in enumerate(tickers_dict.items()):
        try:
            progress_text.text(f"📥 Downloading {name}...")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if not data.empty and len(data) > 10:
                if 'Adj Close' in data.columns:
                    price_series = data['Adj Close']
                else:
                    price_series = data['Close']
                
                price_series.name = name
                all_data = pd.concat([all_data, price_series], axis=1)
            else:
                st.sidebar.warning(f"⚠️ Insufficient data for {name}")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Error downloading {name}")
    
    progress_text.empty()
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

# 使用QuantStats计算绩效指标
def calculate_quantstats_metrics(prices, risk_free_rate=0.02):
    """使用QuantStats计算绩效指标"""
    metrics_dict = {}
    
    for idx_name in prices.columns:
        try:
            # 获取价格序列
            idx_prices = prices[idx_name].dropna()
            
            if len(idx_prices) < 50:  # 数据太少
                continue
            
            # 计算收益率
            returns = idx_prices.pct_change().dropna()
            
            # 使用QuantStats计算指标
            # 年化收益率
            cagr = qs.stats.cagr(returns) * 100
            
            # 年化波动率
            vol = qs.stats.volatility(returns) * 100
            
            # 夏普比率
            sharpe = qs.stats.sharpe(returns, risk_free=risk_free_rate)
            
            # 索提诺比率
            sortino = qs.stats.sortino(returns, risk_free=risk_free_rate)
            
            # Calmar比率
            calmar = qs.stats.calmar(returns)
            
            # 最大回撤
            max_dd = qs.stats.max_drawdown(returns) * 100
            
            # Omega比率
            omega = qs.stats.omega(returns, risk_free=risk_free_rate)
            
            # 偏度
            skew = qs.stats.skew(returns)
            
            # 峰度
            kurtosis = qs.stats.kurtosis(returns)
            
            # 索提诺比率
            sortino = qs.stats.sortino(returns, risk_free=risk_free_rate)
            
            # VaR (95%)
            var_95 = qs.stats.value_at_risk(returns) * 100
            
            # CVaR (95%)
            cvar_95 = qs.stats.conditional_value_at_risk(returns) * 100
            
            metrics_dict[idx_name] = {
                'CAGR (%)': cagr,
                'Volatility (%)': vol,
                'Sharpe Ratio': sharpe,
                'Sortino Ratio': sortino,
                'Calmar Ratio': calmar,
                'Max Drawdown (%)': max_dd,
                'Omega Ratio': omega,
                'Skewness': skew,
                'Kurtosis': kurtosis,
                'VaR 95% (%)': var_95,
                'CVaR 95% (%)': cvar_95,
                'Win Rate (%)': (returns > 0).mean() * 100
            }
            
        except Exception as e:
            st.warning(f"Error calculating metrics for {idx_name}: {str(e)[:50]}")
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
    ax.text(0.02, 0.98, f'Total Return: {total_return:.2f}%',
            transform=ax.transAxes, verticalalignment='top',
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
    
    ax.set_title(f'{index_name} - Log Returns with Bollinger Bands', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Log Returns (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend()
    
    # 添加统计信息
    mean_return = log_returns.mean() * 100 * 252  # 年化
    std_return = log_returns.std() * 100 * np.sqrt(252)  # 年化
    ax.text(0.02, 0.98, f'Ann. Return: {mean_return:.2f}%\nAnn. Vol: {std_return:.2f}%',
            transform=ax.transAxes, verticalalignment='top',
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
    
    # 计算QuantStats指标
    with st.spinner("📈 Calculating quantitative metrics..."):
        quant_metrics = calculate_quantstats_metrics(data_ffilled, risk_free_rate)
    
    # 存储到会话状态
    st.session_state.data = data_ffilled
    st.session_state.metrics = quant_metrics
    st.session_state.bb_params = {'window': bb_window, 'std': bb_std}
    
elif 'data' in st.session_state:
    data_ffilled = st.session_state.data
    quant_metrics = st.session_state.metrics
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

# 显示QuantStats绩效指标
st.markdown('<div class="sub-header">📈 Quantitative Performance Metrics</div>', unsafe_allow_html=True)

# 排序选项
sort_options = ['CAGR (%)', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (%)', 'Volatility (%)']
sort_by = st.selectbox("Sort metrics by:", sort_options, index=0)

if not quant_metrics.empty:
    sorted_metrics = quant_metrics.sort_values(sort_by, ascending=False)
    
    # 格式化显示
    display_df = sorted_metrics.copy()
    
    # 格式化百分比列
    percent_cols = [col for col in display_df.columns if '%' in col]
    for col in percent_cols:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A")
    
    # 格式化比率列
    ratio_cols = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Omega Ratio']
    for col in ratio_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
    
    # 格式化统计列
    stat_cols = ['Skewness', 'Kurtosis']
    for col in stat_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")
    
    st.dataframe(display_df, use_container_width=True)

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
            st.markdown("##### Normalized Price Chart")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            plot_normalized_price_single(index_name, data_ffilled, ax1)
            st.pyplot(fig1)
            plt.close(fig1)
        
        with col_chart2:
            st.markdown("##### Log Returns with Bollinger Bands")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            plot_bollinger_bands_log_returns_single(
                index_name, data_ffilled, ax2, 
                window=bb_window, num_std=bb_std
            )
            st.pyplot(fig2)
            plt.close(fig2)
        
        # 显示该指数的详细指标
        st.markdown("##### Detailed Performance Metrics")
        if index_name in quant_metrics.index:
            index_metrics = quant_metrics.loc[index_name]
            
            # 创建指标卡片
            cols = st.columns(4)
            metric_groups = [
                ['CAGR (%)', 'Volatility (%)', 'Max Drawdown (%)'],
                ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'],
                ['Omega Ratio', 'VaR 95% (%)', 'CVaR 95% (%)'],
                ['Skewness', 'Kurtosis', 'Win Rate (%)']
            ]
            
            for col, metrics in zip(cols, metric_groups):
                for metric in metrics:
                    if metric in index_metrics:
                        value = index_metrics[metric]
                        if '%' in metric:
                            display_value = f"{value:.2f}%"
                        else:
                            display_value = f"{value:.3f}"
                        
                        col.markdown(f"""
                        <div class="metric-card">
                            <strong>{metric}</strong><br>
                            <span style="font-size: 1.2rem; color: #1E40AF;">{display_value}</span>
                        </div>
                        """, unsafe_allow_html=True)

# 风险回报分析
st.markdown('<div class="sub-header">🎯 Risk-Return Analysis</div>', unsafe_allow_html=True)

if not quant_metrics.empty:
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    
    # 创建散点图
    scatter = ax3.scatter(
        quant_metrics['Volatility (%)'], 
        quant_metrics['CAGR (%)'],
        c=quant_metrics['Sharpe Ratio'], 
        s=200, 
        cmap='RdYlGn', 
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    
    # 添加指数标签
    for idx in quant_metrics.index:
        ax3.annotate(idx, 
                    (quant_metrics.loc[idx, 'Volatility (%)'], 
                     quant_metrics.loc[idx, 'CAGR (%)']),
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

if not quant_metrics.empty:
    fig4, (ax4_1, ax4_2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 最大回撤条形图
    max_dd_sorted = quant_metrics['Max Drawdown (%)'].sort_values()
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
    scatter_dd = ax4_2.scatter(
        quant_metrics['Max Drawdown (%)'].abs(),
        quant_metrics['CAGR (%)'],
        c=quant_metrics['Calmar Ratio'],
        s=150,
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    
    for idx in quant_metrics.index:
        ax4_2.annotate(idx,
                      (abs(quant_metrics.loc[idx, 'Max Drawdown (%)']),
                       quant_metrics.loc[idx, 'CAGR (%)']),
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
    metrics_csv = quant_metrics.to_csv().encode('utf-8')
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
    <p>Powered by QuantStats | Data from Yahoo Finance</p>
    <p>Developed by LabGen25 | {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
