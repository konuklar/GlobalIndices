import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 定义投资标的 ====================
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

# ==================== 2. 获取数据 ====================
def fetch_index_data(tickers_dict, start_date='2018-01-01', end_date=None):
    """
    获取多个指数的历史数据
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    all_data = pd.DataFrame()
    
    for ticker, name in tickers_dict.items():
        try:
            print(f"Downloading {name} ({ticker})...")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                # 使用调整后的收盘价
                if 'Adj Close' in data.columns:
                    price_series = data['Adj Close']
                else:
                    price_series = data['Close']
                
                price_series.name = name
                all_data = pd.concat([all_data, price_series], axis=1)
            else:
                print(f"No data for {ticker}")
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    
    return all_data

# 获取数据（这里使用示例时间段）
print("="*50)
print("开始下载数据...")
print("="*50)
data = fetch_index_data(indices, start_date='2020-01-01')
print(f"数据下载完成，形状: {data.shape}")

# ==================== 3. 前向填充处理缺失值 ====================
print("\n处理缺失值...")
print(f"处理前缺失值数量: {data.isnull().sum().sum()}")

# 前向填充（forward fill）
data_ffilled = data.ffill()

# 对于开头仍然缺失的值，使用后向填充
data_ffilled = data_ffilled.bfill()

print(f"处理后缺失值数量: {data_ffilled.isnull().sum().sum()}")
print(f"数据期间: {data_ffilled.index[0].date()} 至 {data_ffilled.index[-1].date()}")

# ==================== 4. 计算收益率 ====================
returns = data_ffilled.pct_change().dropna()
log_returns = np.log(data_ffilled / data_ffilled.shift(1)).dropna()

# ==================== 5. 计算绩效指标 ====================
def calculate_performance_metrics(prices, returns, risk_free_rate=0.02):
    """
    计算各项绩效指标
    """
    # 年化因子
    trading_days = 252
    years = len(prices) / trading_days
    
    # 总收益率
    total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    
    # 年化收益率
    annual_return = ((1 + total_return/100) ** (1/years) - 1) * 100
    
    # 年化波动率
    annual_volatility = returns.std() * np.sqrt(trading_days) * 100
    
    # 夏普比率
    excess_returns = returns - risk_free_rate/trading_days
    sharpe_ratio = (excess_returns.mean() * trading_days) / (returns.std() * np.sqrt(trading_days))
    
    # 最大回撤
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative / running_max - 1) * 100
    max_drawdown = drawdown.min()
    
    # 索提诺比率
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0:
        downside_std = negative_returns.std() * np.sqrt(trading_days)
        sortino_ratio = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else np.nan
    else:
        sortino_ratio = np.nan
    
    # Calmar比率
    calmar_ratio = (annual_return - risk_free_rate) / abs(max_drawdown) if max_drawdown != 0 else np.nan
    
    # 胜率（正收益天数比例）
    win_rate = (returns > 0).sum() / len(returns) * 100
    
    # VaR (95%)
    var_95 = np.percentile(returns, 5) * 100
    
    # CVaR (95%)
    cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
    
    # 创建结果DataFrame
    metrics_df = pd.DataFrame({
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
    })
    
    return metrics_df.T  # 转置以便每个指数一列

# 计算所有指数的绩效指标
print("\n计算绩效指标...")
performance_metrics = calculate_performance_metrics(data_ffilled, returns)

# ==================== 6. 生成类似示例的表格 ====================
def create_summary_table(metrics_df):
    """
    创建类似示例的汇总表格
    """
    # 选择最重要的几个指标进行展示
    summary = pd.DataFrame({
        'Index': metrics_df.columns,
        'Ann. Return (%)': metrics_df.loc['Annual Return (%)'].values,
        'Ann. Vol (%)': metrics_df.loc['Annual Volatility (%)'].values,
        'Sharpe Ratio': metrics_df.loc['Sharpe Ratio'].values.round(2),
        'Max DD (%)': metrics_df.loc['Max Drawdown (%)'].values.round(1),
        'Win Rate (%)': metrics_df.loc['Win Rate (%)'].values.round(1)
    })
    
    # 按年化收益率排序
    summary = summary.sort_values('Ann. Return (%)', ascending=False)
    
    return summary

# 创建汇总表格
summary_table = create_summary_table(performance_metrics)

print("\n" + "="*50)
print("绩效指标汇总表")
print("="*50)
print(summary_table.to_string(index=False))

# ==================== 7. 生成图表 ====================
def create_performance_charts(data_ffilled, returns, performance_metrics):
    """
    创建绩效图表
    """
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 价格走势图（标准化）
    ax1 = plt.subplot(3, 2, 1)
    normalized_prices = data_ffilled / data_ffilled.iloc[0] * 100
    top_5 = performance_metrics.loc['Annual Return (%)'].nlargest(5).index
    for idx in top_5:
        ax1.plot(normalized_prices.index, normalized_prices[idx], label=idx, linewidth=2)
    ax1.set_title('Top 5 Performers - Normalized Price', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Normalized Price (起始=100)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. 累计收益率
    ax2 = plt.subplot(3, 2, 2)
    cumulative_returns = (1 + returns).cumprod() - 1
    for idx in top_5:
        ax2.plot(cumulative_returns.index, cumulative_returns[idx]*100, label=idx, linewidth=2)
    ax2.set_title('Top 5 Performers - Cumulative Return', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cumulative Return (%)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. 年化收益率 vs 年化波动率（散点图）
    ax3 = plt.subplot(3, 2, 3)
    scatter = ax3.scatter(performance_metrics.loc['Annual Volatility (%)'], 
                          performance_metrics.loc['Annual Return (%)'],
                          c=performance_metrics.loc['Sharpe Ratio'], 
                          s=100, cmap='RdYlGn', alpha=0.7)
    
    # 标注部分重要指数
    important_indices = ['Nasdaq', 'S&P 500', 'Hang Seng', 'Nikkei 225', 'Shanghai Index']
    for idx in important_indices:
        if idx in performance_metrics.columns:
            vol = performance_metrics.loc['Annual Volatility (%)', idx]
            ret = performance_metrics.loc['Annual Return (%)', idx]
            ax3.annotate(idx, (vol, ret), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax3.set_xlabel('Annual Volatility (%)')
    ax3.set_ylabel('Annual Return (%)')
    ax3.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Sharpe Ratio')
    
    # 4. 最大回撤热图
    ax4 = plt.subplot(3, 2, 4)
    max_dd = performance_metrics.loc['Max Drawdown (%)'].sort_values()
    colors = plt.cm.RdYlGn_r((max_dd - max_dd.min()) / (max_dd.max() - max_dd.min()))
    bars = ax4.barh(range(len(max_dd)), max_dd.values, color=colors)
    ax4.set_yticks(range(len(max_dd)))
    ax4.set_yticklabels(max_dd.index)
    ax4.set_xlabel('Max Drawdown (%)')
    ax4.set_title('Maximum Drawdown by Index', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. 夏普比率排名
    ax5 = plt.subplot(3, 2, 5)
    sharpe_sorted = performance_metrics.loc['Sharpe Ratio'].sort_values(ascending=False)
    colors_sharpe = plt.cm.RdYlGn((sharpe_sorted - sharpe_sorted.min()) / 
                                 (sharpe_sorted.max() - sharpe_sorted.min()))
    bars_sharpe = ax5.barh(range(len(sharpe_sorted)), sharpe_sorted.values, color=colors_sharpe)
    ax5.set_yticks(range(len(sharpe_sorted)))
    ax5.set_yticklabels(sharpe_sorted.index)
    ax5.set_xlabel('Sharpe Ratio')
    ax5.set_title('Sharpe Ratio Ranking', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # 6. 相关系数热图
    ax6 = plt.subplot(3, 2, 6)
    corr_matrix = returns.corr()
    # 选择部分重要指数展示相关性
    important_idx = ['Nasdaq', 'S&P 500', 'Hang Seng', 'Nikkei 225', 
                    'STOXX 600', 'FTSE 100', 'Shanghai Index']
    available_idx = [idx for idx in important_idx if idx in corr_matrix.columns]
    if available_idx:
        corr_subset = corr_matrix.loc[available_idx, available_idx]
        im = ax6.imshow(corr_subset, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax6.set_xticks(range(len(available_idx)))
        ax6.set_yticks(range(len(available_idx)))
        ax6.set_xticklabels(available_idx, rotation=45, ha='right')
        ax6.set_yticklabels(available_idx)
        ax6.set_title('Correlation Matrix (Selected Indices)', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax6, shrink=0.8)
        
        # 添加数值标签
        for i in range(len(available_idx)):
            for j in range(len(available_idx)):
                ax6.text(j, i, f'{corr_subset.iloc[i, j]:.2f}', 
                        ha='center', va='center', color='white', fontsize=8)
    
    plt.suptitle('Global Indices Performance Analysis', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig

# 生成图表
print("\n生成图表...")
fig = create_performance_charts(data_ffilled, returns, performance_metrics)

# 保存图表
fig.savefig('global_indices_performance_analysis.png', dpi=150, bbox_inches='tight')
print("图表已保存为 'global_indices_performance_analysis.png'")

# ==================== 8. 生成详细绩效表格（类似示例格式） ====================
def create_detailed_performance_table(metrics_df):
    """
    创建详细的绩效表格，类似示例格式
    """
    detailed_table = pd.DataFrame({
        'Index': metrics_df.columns,
        'Annual Return (%)': metrics_df.loc['Annual Return (%)'].values.round(1),
        'Annual Volatility (%)': metrics_df.loc['Annual Volatility (%)'].values.round(1),
        'Sharpe Ratio': metrics_df.loc['Sharpe Ratio'].values.round(2),
        'Sortino Ratio': metrics_df.loc['Sortino Ratio'].values.round(2),
        'Calmar Ratio': metrics_df.loc['Calmar Ratio'].values.round(2),
        'Max Drawdown (%)': metrics_df.loc['Max Drawdown (%)'].values.round(1),
        'Win Rate (%)': metrics_df.loc['Win Rate (%)'].values.round(1),
        'VaR 95% (%)': metrics_df.loc['VaR 95% (%)'].values.round(2),
        'CVaR 95% (%)': metrics_df.loc['CVaR 95% (%)'].values.round(2)
    })
    
    # 按年化收益率排序
    detailed_table = detailed_table.sort_values('Annual Return (%)', ascending=False)
    
    return detailed_table

# 创建详细表格
detailed_table = create_detailed_performance_table(performance_metrics)

print("\n" + "="*50)
print("详细绩效指标表")
print("="*50)
print(detailed_table.to_string(index=False))

# 保存数据到CSV
detailed_table.to_csv('global_indices_performance_metrics.csv', index=False)
print("\n详细绩效指标已保存为 'global_indices_performance_metrics.csv'")

# 显示图表
plt.show()

print("\n" + "="*50)
print("分析完成!")
print("="*50)
