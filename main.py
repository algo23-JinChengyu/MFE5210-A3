import math
from typing import Dict
import datetime 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def modify_statistics(df) -> Dict[str, float]:
    '''
    输入：
    df: DataFrame类型，包含价格数据和仓位、开平仓标志
        position列：仓位标志位，0表示空仓，1表示持有标的
        flag列：买入卖出标志位，1表示在该时刻买入，-1表示在该时刻卖出
        close列：日收盘价
        
    输出：dict类型，包含夏普比率、最大回撤等策略结果的统计数据
    '''
    #净值序列
    df['net_asset_pct_chg'] = df['net_asset_value'].pct_change(1).fillna(0)
    
    #总收益率与年化收益率
    total_return = (df['net_asset_value'].iloc[-1] - 1) * 100
    annual_return = ((df['net_asset_value'].iloc[-1] - 1) ** (1 / (df.shape[0] / 252)) - 1) * 100
    
    #夏普比率
    df['ex_pct_chg'] = df['net_asset_pct_chg']
    sharp_ratio = (df['ex_pct_chg'].mean() * math.sqrt(252) / df['ex_pct_chg'].std())
    
    #回撤
    df['high_level'] = df['net_asset_value'].rolling(window=len(df), min_periods=1, center=False).max()
    df['draw_down'] = df['net_asset_value'] - df['high_level']
    df['draw_down_percent'] = df['draw_down'] / df['high_level'] * 100
    max_draw_down = df['draw_down'].min()
    max_draw_percent = df['draw_down_percent'].min()
    
    #持仓总天数
    hold_days = df['position'].sum()
    
    #交易次数
    trade_count = df[df['flag'] != 0].shape[0] / 2
    
    #平均持仓天数
    avg_hold_days = int(hold_days / trade_count)
    
    #获利天数
    profit_days = df[df['net_asset_pct_chg'] > 0].shape[0]
    #亏损天数
    loss_days = df[df['net_asset_pct_chg'] < 0].shape[0]
    
    #胜率(按天)
    winrate_by_day = profit_days / (profit_days + loss_days) * 100
    #平均盈利率(按天)
    avg_profit_rate_day = df[df['net_asset_pct_chg'] > 0]['net_asset_pct_chg'].mean() * 100
    #平均亏损率(按天)
    avg_loss_rate_day = df[df['net_asset_pct_chg'] < 0]['net_asset_pct_chg'].mean() * 100
    #平均盈亏比(按天)
    avg_profit_loss_ratio_day = avg_profit_rate_day / abs(avg_loss_rate_day)
    
    #每一次交易情况
    buy_trades = df[df['flag'] == 1].reset_index()
    sell_trades = df[df['flag'] == -1].reset_index()
    result_by_trade = {
        'buy': buy_trades['close'],
        'sell': sell_trades['close'],
        'pct_chg': (sell_trades['close'] - buy_trades['close']) / buy_trades['close']
    }
    result_by_trade = pd.DataFrame(result_by_trade)

    #盈利次数 
    profit_trades = result_by_trade[result_by_trade['pct_chg'] > 0].shape[0]
    #亏损次数
    loss_trades = result_by_trade[result_by_trade['pct_chg'] < 0].shape[0]
    #单次最大盈利
    max_profit_trade = result_by_trade['pct_chg'].max() * 100
    #单次最大亏损
    max_loss_trade = result_by_trade['pct_chg'].min() * 100
    #胜率(按次)
    winrate_by_trade = profit_trades / (profit_trades + loss_trades) * 100
    #平均盈利率(按次)
    avg_profit_rate_trade = result_by_trade[result_by_trade['pct_chg'] > 0]['pct_chg'].mean() * 100
    #平均亏损率(按次)
    avg_loss_rate_trade = result_by_trade[result_by_trade['pct_chg'] < 0]['pct_chg'].mean() * 100
    #平均盈亏比(按次)
    avg_profit_loss_ratio_trade = avg_profit_rate_trade / abs(avg_loss_rate_trade)

    statistics_result = {
        'net_asset_value': df['net_asset_value'].iloc[-1],  # 最终净值
        'total_return': total_return,  # 收益率
        'annual_return': annual_return,  # 年化收益率
        'sharp_ratio': sharp_ratio,  # 夏普比率
        'max_draw_percent': max_draw_percent,  # 最大回撤
        'hold_days': hold_days,  # 持仓天数
        'trade_count': trade_count,  # 交易次数
        'avg_hold_days': avg_hold_days,  # 平均持仓天数
        'profit_days': profit_days,  # 盈利天数
        'loss_days': loss_days,  # 亏损天数
        'winrate_by_day': winrate_by_day,  # 胜率（按天）
        'avg_profit_rate_day': avg_profit_rate_day,  # 平均盈利率（按天）
        'avg_loss_rate_day': avg_loss_rate_day,  # 平均亏损率（按天）
        'avg_profit_loss_ratio_day': avg_profit_loss_ratio_day,  # 平均盈亏比（按天）
        'profit_trades': profit_trades,  # 盈利次数
        'loss_trades': loss_trades,  # 亏损次数
        'max_profit_trade': max_profit_trade,  # 单次最大盈利
        'max_loss_trade': max_loss_trade,  # 单次最大亏损
        'winrate_by_trade': winrate_by_trade,  #胜率（按次）
        'avg_profit_rate_trade': avg_profit_rate_trade, # 平均盈利率（按次）
        'avg_loss_rate_trade': avg_loss_rate_trade, # 平均亏损率（按次）
        'avg_profit_loss_ratio_trade': avg_profit_loss_ratio_trade # 平均盈亏比（按次）
    }

    return statistics_result


#当日斜率指标计算方式，线性回归

def modify_nbeta(df, n):
    nbeta = []
    trade_days = len(df.index)
    
    df['position'] = 0
    df['flag'] = 0
    position = 0
    
    # 计算斜率值
    for i in range(n-1, trade_days):
        x = df['low'].iloc[i-n+1:i+1]
        x = sm.add_constant(x)
        y = df['high'].iloc[i-n+1:i+1]
        regr = sm.OLS(y, x)
        res = regr.fit()
        beta = round(res.params[1], 2)  # 斜率指标          
        nbeta.append(beta)    
    df1 = df.iloc[n-1:]
    df1['beta'] = nbeta
    
    # 执行交易策略
    for i in range(len(df1.index)-1):
        if df1['beta'].iloc[i] > 1 and position == 0:
            df1.loc[df1.index[i], 'flag'] = 1  # 开仓标志
            df1.loc[df1.index[i+1], 'position'] = 1  # 仓位不为空
            position = 1
        elif df1['beta'].iloc[i] < 0.8 and position == 1:
            df1.loc[df1.index[i], 'flag'] = -1  # 平仓标志
            df1.loc[df1.index[i+1], 'position'] = 0  # 仓位为空
            position = 0
        else:
            df1.loc[df1.index[i+1], 'position'] = df1.loc[df1.index[i], 'position']
    
    # 计算净值序列     
    df1['net_asset_value'] = (1 + df1['close'].pct_change(1).fillna(0) * df1['position']).cumprod()
    
    return df1

#标准分策略
def modify_stdbeta(df, n):
    df['position'] = 0
    df['flag'] = 0
    position = 0
    
    df1 = modify_nbeta(df, n)
    pre_stdbeta = df1['beta']
    pre_stdbeta = np.array(pre_stdbeta)
    # 转化为数组，可以对整个数组进行操作
    sigma = np.std(pre_stdbeta)
    mu = np.mean(pre_stdbeta)
    # 标准化
    stdbeta = (pre_stdbeta - mu) / sigma
    
    df1['stdbeta'] = stdbeta
    
    for i in range(len(df1.index)-1):
        if df1['stdbeta'].iloc[i] > 0.7 and position == 0:
            df1.loc[df1.index[i], 'flag'] = 1
            df1.loc[df1.index[i+1], 'position'] = 1
            position = 1
        elif df1['stdbeta'].iloc[i] < -0.7 and position == 1:
            df1.loc[df1.index[i], 'flag'] = -1
            df1.loc[df1.index[i+1], 'position'] = 0
            position = 0
        else:
            df1.loc[df1.index[i+1], 'position'] = df1.loc[df1.index[i], 'position']
            
    df1['net_asset_value'] = (1 + df1['close'].pct_change(1).fillna(0) * df1['position']).cumprod()
    return df1


#RSRS 标准分指标优化,修正标准分
def modify_better_stdbeta(df, n):
    nbeta = []
    R2 = []
    trade_days = len(df.index)
    
    for i in range(n-1, trade_days):
        x = df['low'].iloc[i-n+1:i+1]
        x = sm.add_constant(x)
        y = df['high'].iloc[i-n+1:i+1]
        regr = sm.OLS(y, x)
        res = regr.fit()
        beta = round(res.params[1], 2)
        
        R2.append(res.rsquared)
        nbeta.append(beta)
        
    prebeta = np.array(nbeta)
    sigma = np.std(prebeta)
    mu = np.mean(prebeta)
    stdbeta = (prebeta - mu) / sigma
    
    r2 = np.array(R2)
    better_stdbeta = r2 * stdbeta  # 修正标准分
    
    df1 = df.iloc[n-1:]
    df1['beta'] = nbeta
    df1['flag'] = 0
    df1['position'] = 0
    position = 0
    df1['better_stdbeta'] = better_stdbeta
    
    for i in range(len(df1.index)-1):
        if df1['better_stdbeta'].iloc[i] > 0.7 and position == 0:
            df1.loc[df1.index[i], 'flag'] = 1
            df1.loc[df1.index[i+1], 'position'] = 1
            position = 1
        elif df1['better_stdbeta'].iloc[i] < -0.7 and position == 1:
            df1.loc[df1.index[i], 'flag'] = -1
            df1.loc[df1.index[i+1], 'position'] = 0
            position = 0
        else:
            df1.loc[df1.index[i+1], 'position'] = df1.loc[df1.index[i], 'position']
    
    df1['net_asset_value'] = (1 + df1['close'].pct_change(1).fillna(0) * df1['position']).cumprod()
    
    return df1

#右偏标准分 此时N取16
def modify_right_stdbeta(df, n):
    df1 = modify_better_stdbeta(df, n)
    df1['position'] = 0
    df1['flag'] = 0
    df1['net_value'] = 0  
    position = 0
    
    df1['right_stdbeta'] = df1['better_stdbeta'] * df1['beta']        
    # 修正标准分与斜率值相乘能够达到使原有分布右偏的效果
    
    for i in range(len(df1.index)-1):
        if df1['right_stdbeta'].iloc[i] > 0.7 and position == 0:
            df1.loc[df1.index[i], 'flag'] = 1
            df1.loc[df1.index[i+1], 'position'] = 1
            position = 1
        elif df1['right_stdbeta'].iloc[i] < -0.7 and position == 1:
            df1.loc[df1.index[i], 'flag'] = -1
            df1.loc[df1.index[i+1], 'position'] = 0
            position = 0
        else:
            df1.loc[df1.index[i+1], 'position'] = df1.loc[df1.index[i], 'position']
    df1['net_asset_value'] = (1 + df1['close'].pct_change(1).fillna(0) * df1['position']).cumprod()    
    return df1

#RSRS 指标配合价格数据优化策略
def modify_ma_beta(df, n):
    df1 = modify_stdbeta(df, n)
    df1['position'] = 0
    df1['flag'] = 0
    df1['net_asset_value'] = 0
    position = 0

    ma = df['close'].rolling(window=n).mean()
    df1['ma'] = ma

    for i in range(len(df1.index)-1):
        if df1['stdbeta'].iloc[i] > 0.7 and position == 0 and df1['close'].iloc[i] > df1['ma'].iloc[i]:
            df1.loc[df1.index[i], 'flag'] = 1
            df1.loc[df1.index[i+1], 'position'] = 1
            position = 1
        elif df1['stdbeta'].iloc[i] < -0.7 and position == 1:
            df1.loc[df1.index[i], 'flag'] = -1
            df1.loc[df1.index[i+1], 'position'] = 0
            position = 0
        else:
            df1.loc[df1.index[i+1], 'position'] = df1.loc[df1.index[i], 'position']

    # 考虑交易成本
    transaction_cost = 0 #可以调节
    df1['transaction_cost'] = transaction_cost * (df1['flag'].diff() != 0)
    df1['net_asset_value'] = (1 + df1['close'].pct_change(1).fillna(0) * df1['position'] - df1['transaction_cost']).cumprod()

    return df1



#基于 RSRS 指标与交易量相关性的优化
def modify_vol_beta(df, n):
    df1 = modify_stdbeta(df, n)
    
    df1['position'] = 0
    df1['flag'] = 0
    df1['net_asset_value'] = 0  
    position = 0
    
    for i in range(10, len(df1.index)-1):
        pre_volume = df1['volume'].iloc[i-10:i]
        series_beta = df1['stdbeta'].iloc[i-10:i]
        # 计算相关系数需要数据为series格式
        corr = series_beta.corr(pre_volume, method='pearson')
        if df1['stdbeta'].iloc[i] > 0.7 and corr > 0 and position == 0:
            df1.loc[df1.index[i], 'flag'] = 1
            df1.loc[df1.index[i+1], 'position'] = 1
            position = 1
        elif df1['stdbeta'].iloc[i] < -0.7 and position == 1:
            df1.loc[df1.index[i], 'flag'] = -1
            df1.loc[df1.index[i+1], 'position'] = 0
            position = 0
        else:
            df1.loc[df1.index[i+1], 'position'] = df1.loc[df1.index[i], 'position']

    # 考虑交易成本
    transaction_cost = 0 #可以调节
    df1['transaction_cost'] = transaction_cost * (df1['flag'].diff() != 0)
    df1['net_asset_value'] = (1 + df1['close'].pct_change(1).fillna(0) * df1['position'] - df1['transaction_cost']).cumprod()


    return df1
