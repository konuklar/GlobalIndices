import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="QUANTEDGE - Global Indices Performance Analysis",
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
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .stDataFrame {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# 应用标题
st.markdown('<h1 class="main-header">📊 QUANTEDGE By LabGen25</h1>', unsafe_allow_html=True)
st.markdown("### Global Indices Performance Analysis Dashboard")

# 侧边栏配置
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/stock-exchange.png", width=80)
    st.markdown("## Configuration")
    
    # 日期范围选择
    st.markdown("### Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", 
                                  value=datetime(2020, 1, 1),
                                  min_value=datetime(2000, 1, 1))
    with col2:
        end_date = st.date_input("End Date", 
                                value=datetime.now(),
                                min_value=datetime(2000, 1, 1))
    
    # 风险参数
    st.markdown("### Risk Parameters")
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0) / 100
    
    # 分析选项
    st.markdown("### Analysis Options")
    show_correlation = st.checkbox("Show Correlation Matrix", value=True)
    show_drawdown = st.checkbox("Show Drawdown Analysis", value=True)
    
    # 更新按钮
    analyze_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown("""
    **QUANTEDGE** provides comprehensive performance analysis 
    for global equity indices using quantitative methods.
    
    Developed by **LabGen25**
    """)

# 定义指数列表
indices = {
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

# 数据处理函数
@st.cache_data(ttl=3600)
def fetch_index_data(tickers_dict, start_date, end_date):
    """
    获取多个指数的历史数据
    """
    all_data = pd.DataFrame()
    
    with st.spinner("📥 Downloading market data..."):
        progress_bar = st.progress(0)
        total_tickers = len(tickers_dict)
        
        for i, (ticker, name) in enumerate(tickers_dict.items()):
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    # 使用调整后的收盘价
                    if 'Adj Close' in data.columns:
                        price_series = data['Adj Close']
                    else:
                        price_series = data['Close']
                    
                    price_series.name = name
                    all_data = pd.concat([all_data, price_series], axis=1)
            except Exception as e:
                st.warning(f"Could not download {ticker}: {str(e)[:50]}...")
            
            progress_bar.progress((i + 1) / total_tickers)
    
    return all_data

# 计算绩效指标的修复版本
def calculate_performance_metrics(prices, returns, risk_free_rate=0.02):
    """
    计算各项绩效指标（修复版本）
    """
    # 年化因子
    trading_days = 252
    years = len(prices) / trading_days
    
    results = {}
    
    for idx in prices.columns:
        try:
            # 获取该指数的价格和收益率
            idx_prices = prices[idx].dropna()
            idx_returns = returns[idx].dropna()
            
            if len(idx_prices) < 10:  # 数据太少
                continue
            
            # 总收益率
            total_return = (idx_prices.iloc[-1] / idx_prices.iloc[0] - 1) * 100
            
            # 年化收益率
            annual_return = ((1 + total_return/100) ** (1/years) - 1) * 100
            
            # 年化波动率
            annual_volatility = idx_returns.std() * np.sqrt(trading_days) * 100
            
            # 夏普比率
            sharpe_ratio = (annual_return - risk_free_rate * 100) / annual_volatility
            
            # 最大回撤
            cumulative = (1 + idx_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative / running_max - 1) * 100
            max_drawdown = drawdown.min()
            
            # 索提诺比率 - 修复的版本
            negative_returns = idx_returns[idx_returns < 0]
            if len(negative_returns) > 0:
                downside_std = negative_returns.std() * np.sqrt(trading_days) * 100
                if downside_std > 0:
                    sortino_ratio = (annual_return - risk_free_rate * 100) / downside_std
                else:
                    sortino_ratio = np.nan
            else:
                sortino_ratio = np.nan
            
            # Calmar比率
            if max_drawdown != 0:
                calmar_ratio = (annual_return - risk_free_rate * 100) / abs(max_drawdown)
            else:
                calmar_ratio = np.nan
            
            # 胜率
            win_rate = (idx_returns > 0).sum() / len(idx_returns) * 100
            
            # VaR (95%)
            var_95 = np.percentile(idx_returns, 5) * 100
            
            # CVaR (95%)
            cvar_95 = idx_returns[idx_returns <= np.percentile(idx_returns, 5)].mean() * 100
            
            results[idx] = {
                'Total Return (%)': total_return,
                'Annual Return (%)': annual_return,
                'Annual Volatility (%)': annual_volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Sortino Ratio': sortino_ratio,
                'Calmar Ratio': calmar_ratio,
                'Max Drawdown (%)': max_drawdown,
                'Win Rate (%)': win_rate,
                'VaR 95% (%)': var_95,
                'CVaR 95% (%)': cvar_95
            }
            
        except Exception as e:
            st.warning(f"Error calculating metrics for {idx}: {str(e)[:50]}")
            continue
    
    return pd.DataFrame(results).T

# 创建可视化图表
def create_performance_charts(data_ffilled, returns, performance_metrics):
    """
    创建绩效图表
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 前5名表现者 - 标准化价格
    top_5 = performance_metrics['Annual Return (%)'].nlargest(5).index
    ax1 = axes[0, 0]
    normalized_prices = data_ffilled[top_5] / data_ffilled[top_5].iloc[0] * 100
    for idx in top_5:
        ax1.plot(normalized_prices.index, normalized_prices[idx], label=idx, linewidth=2)
    ax1.set_title('Top 5 Performers - Normalized Price', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Normalized Price (Start=100)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 风险回报散点图
    ax2 = axes[0, 1]
    scatter = ax2.scatter(performance_metrics['Annual Volatility (%)'], 
                         performance_metrics['Annual Return (%)'],
                         c=performance_metrics['Sharpe Ratio'], 
                         s=100, cmap='RdYlGn', alpha=0.7)
    
    # 标注重要指数
    important_indices = ['Nasdaq', 'S&P 500', 'Hang Seng', 'Nikkei 225', 'Shanghai Index']
    for idx in important_indices:
        if idx in performance_metrics.index:
            vol = performance_metrics.loc[idx, 'Annual Volatility (%)']
            ret = performance_metrics.loc[idx, 'Annual Return (%)']
            ax2.annotate(idx, (vol, ret), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Annual Volatility (%)')
    ax2.set_ylabel('Annual Return (%)')
    ax2.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Sharpe Ratio')
    
    # 3. 最大回撤条形图
    ax3 = axes[1, 0]
    max_dd = performance_metrics['Max Drawdown (%)'].sort_values()
    colors = plt.cm.RdYlGn_r((max_dd - max_dd.min()) / (max_dd.max() - max_dd.min() + 1e-10))
    bars = ax3.barh(range(len(max_dd)), max_dd.values, color=colors)
    ax3.set_yticks(range(len(max_dd)))
    ax3.set_yticklabels(max_dd.index)
    ax3.set_xlabel('Max Drawdown (%)')
    ax3.set_title('Maximum Drawdown by Index', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. 夏普比率排名
    ax4 = axes[1, 1]
    sharpe_sorted = performance_metrics['Sharpe Ratio'].sort_values(ascending=False)
    colors_sharpe = plt.cm.RdYlGn((sharpe_sorted - sharpe_sorted.min()) / 
                                 (sharpe_sorted.max() - sharpe_sorted.min() + 1e-10))
    ax4.barh(range(len(sharpe_sorted)), sharpe_sorted.values, color=colors_sharpe)
    ax4.set_yticks(range(len(sharpe_sorted)))
    ax4.set_yticklabels(sharpe_sorted.index)
    ax4.set_xlabel('Sharpe Ratio')
    ax4.set_title('Sharpe Ratio Ranking', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('QUANTEDGE - Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

# 主应用逻辑
if analyze_button or 'data_loaded' not in st.session_state:
    
    # 获取数据
    data = fetch_index_data(indices, start_date, end_date)
    
    if data.empty:
        st.error("❌ No data retrieved. Please check your internet connection and try again.")
        st.stop()
    
    # 前向填充处理
    data_ffilled = data.ffill().bfill()
    
    if data_ffilled.isnull().all().any():
        st.error("❌ Some indices have no available data for the selected period.")
        st.stop()
    
    # 计算收益率
    returns = data_ffilled.pct_change().dropna()
    
    if returns.empty:
        st.error("❌ Insufficient data to calculate returns.")
        st.stop()
    
    # 计算绩效指标
    with st.spinner("📊 Calculating performance metrics..."):
        performance_metrics = calculate_performance_metrics(data_ffilled, returns, risk_free_rate)
    
    st.session_state.data_loaded = True
    st.session_state.data = data_ffilled
    st.session_state.returns = returns
    st.session_state.metrics = performance_metrics
    
else:
    if 'data_loaded' in st.session_state:
        data_ffilled = st.session_state.data
        returns = st.session_state.returns
        performance_metrics = st.session_state.metrics
    else:
        st.info("👈 Configure your analysis in the sidebar and click 'Run Analysis'")
        st.stop()

# 显示数据概览
st.markdown('<div class="sub-header">📈 Data Overview</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Number of Indices", len(data_ffilled.columns))
with col2:
    st.metric("Date Range", f"{data_ffilled.index[0].date()} to {data_ffilled.index[-1].date()}")
with col3:
    st.metric("Trading Days", len(data_ffilled))

# 显示绩效指标表格
st.markdown('<div class="sub-header">🏆 Performance Metrics</div>', unsafe_allow_html=True)

# 排序选项
sort_by = st.selectbox("Sort by:", 
                       ['Annual Return (%)', 'Sharpe Ratio', 'Annual Volatility (%)', 
                        'Max Drawdown (%)', 'Win Rate (%)'],
                       index=0)

# 显示排序后的表格
sorted_metrics = performance_metrics.sort_values(sort_by, ascending=False)

# 格式化显示
display_metrics = sorted_metrics.copy()
for col in display_metrics.columns:
    if '%' in col:
        display_metrics[col] = display_metrics[col].apply(lambda x: f"{x:.2f}%")
    else:
        display_metrics[col] = display_metrics[col].apply(lambda x: f"{x:.3f}")

st.dataframe(display_metrics, use_container_width=True)

# 创建并显示图表
st.markdown('<div class="sub-header">📊 Performance Visualization</div>', unsafe_allow_html=True)
try:
    fig = create_performance_charts(data_ffilled, returns, performance_metrics)
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error creating charts: {str(e)}")

# 额外分析
st.markdown('<div class="sub-header">🔍 Additional Analysis</div>', unsafe_allow_html=True)

if show_correlation:
    st.markdown("##### Correlation Matrix")
    
    # 选择主要指数进行相关性分析
    major_indices = ['Nasdaq', 'S&P 500', 'Hang Seng', 'Nikkei 225', 
                    'STOXX 600', 'FTSE 100', 'Shanghai Index']
    available_indices = [idx for idx in major_indices if idx in returns.columns]
    
    if available_indices:
        corr_matrix = returns[available_indices].corr()
        
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        im = ax_corr.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax_corr.set_xticks(range(len(available_indices)))
        ax_corr.set_yticks(range(len(available_indices)))
        ax_corr.set_xticklabels(available_indices, rotation=45, ha='right')
        ax_corr.set_yticklabels(available_indices)
        ax_corr.set_title('Correlation Matrix - Major Indices', fontsize=14, fontweight='bold')
        
        # 添加数值标签
        for i in range(len(available_indices)):
            for j in range(len(available_indices)):
                ax_corr.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                           ha='center', va='center', 
                           color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black',
                           fontsize=10)
        
        plt.colorbar(im, ax=ax_corr, shrink=0.8)
        st.pyplot(fig_corr)

if show_drawdown:
    st.markdown("##### Worst Drawdown Periods")
    
    # 计算并显示最大回撤
    worst_dd = performance_metrics[['Max Drawdown (%)', 'Annual Return (%)']].sort_values('Max Drawdown (%)')
    worst_dd = worst_dd.head(10)  # 最差的10个
    
    fig_dd, ax_dd = plt.subplots(figsize=(12, 6))
    colors_dd = plt.cm.RdYlGn_r((worst_dd['Max Drawdown (%)'] - worst_dd['Max Drawdown (%)'].min()) / 
                               (worst_dd['Max Drawdown (%)'].max() - worst_dd['Max Drawdown (%)'].min() + 1e-10))
    
    bars = ax_dd.barh(range(len(worst_dd)), worst_dd['Max Drawdown (%)'], color=colors_dd)
    ax_dd.set_yticks(range(len(worst_dd)))
    ax_dd.set_yticklabels(worst_dd.index)
    ax_dd.set_xlabel('Maximum Drawdown (%)')
    ax_dd.set_title('Worst Drawdown Periods (Top 10)', fontsize=14, fontweight='bold')
    ax_dd.grid(True, alpha=0.3, axis='x')
    
    # 在条形图右侧添加年化收益率
    for i, (idx, row) in enumerate(worst_dd.iterrows()):
        ax_dd.text(row['Max Drawdown (%)'] + 0.5, i, 
                  f"Return: {row['Annual Return (%)']:.1f}%", 
                  va='center', fontsize=9)
    
    st.pyplot(fig_dd)

# 数据下载选项
st.markdown('<div class="sub-header">💾 Download Data</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

# 准备CSV数据
metrics_csv = performance_metrics.to_csv().encode('utf-8')
returns_csv = returns.to_csv().encode('utf-8')
prices_csv = data_ffilled.to_csv().encode('utf-8')

with col1:
    st.download_button(
        label="📥 Download Performance Metrics",
        data=metrics_csv,
        file_name=f"quantedge_performance_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col2:
    st.download_button(
        label="📥 Download Returns Data",
        data=returns_csv,
        file_name=f"quantedge_returns_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col3:
    st.download_button(
        label="📥 Download Price Data",
        data=prices_csv,
        file_name=f"quantedge_prices_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280;'>
    <p>QUANTEDGE Performance Analysis Dashboard | Developed by LabGen25</p>
    <p>Data provided by Yahoo Finance | Last updated: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
