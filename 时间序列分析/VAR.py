# %%
# load data
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.filterwarnings("ignore")
ChinaBank = pd.read_csv('ChinaBank.csv', index_col='Date', parse_dates=[
                        'Date']).drop(labels="Unnamed: 0", axis=1)
#ChinaBank.index = pd.to_datetime(ChinaBank.index)
sub = ChinaBank['2014-01':'2014-04']
train = sub.loc['2014-01':'2014-03']
testDf = sub.loc['2014-04-01':'2014-04-30']
plt.figure(figsize=(10, 10))
print(sub.loc['2014-01':'2014-03'])
plt.plot(train["Close"], color='red', label="Close")
plt.plot(train["Open"], color='green', label="Open")
plt.plot(train["High"], color='blue', label="High")
plt.plot(train["Low"], color='skyblue', label="Low")
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
plt.yticks(np.arange(2.4, 2.7, 0.01))
plt.show()
plt.figure(figsize=(10, 10))
plt.plot(train["Volume"], color='black', label="Volum")
plt.legend()
plt.yticks(np.arange(0, train["Volume"].max()+1e7, 1e7))
plt.show()


# %%
from sklearn.metrics import r2_score
'''训练数据维度'''
dimension = 3
nptrain = train.to_numpy()
# print(nptrain)
u, s, vt = np.linalg.svd(nptrain)
'''降维'''
svdtrain = np.dot(vt[:dimension, :], np.mat(nptrain).T).T
predict_sunspots02 = np.dot(svdtrain, vt[:dimension, :])
print(r2_score(nptrain, predict_sunspots02))
# print(predict_sunspots02)
plt.figure(figsize=(10, 10))
plt.plot(predict_sunspots02[..., 3], color='red', label="Close")
plt.plot(predict_sunspots02[..., 0], color='green', label="Open")
plt.plot(predict_sunspots02[..., 1], color='blue', label="High")
plt.plot(predict_sunspots02[..., 2], color='skyblue', label="Low")
# plt.plot(train["Volume"],color='blacl')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
plt.figure(figsize=(10, 10))
plt.plot(predict_sunspots02[..., 4], color='black', label="Volum")
plt.legend()
plt.show()
# print(svdtrain)
dataFrame = pd.DataFrame(svdtrain).set_index(train.index)
print(dataFrame.shape)
plt.figure(figsize=(10, 10))
plt.plot(dataFrame)
plt.show()


# %%
#格兰杰检验
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.stattools import durbin_watson

maxlag = 12
test = 'ssr_chi2test'
variables = dataFrame.columns


def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))),
                      columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1], 4)for i in range(maxlag)]
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [str(var) + '_x' for var in variables]
    df.index = [str(var) + '_y' for var in variables]
    return df


grangers_causation_matrix(train, variables=train.columns)


# %%
# ADF检验
def adfuller_test(series, signif=0.05, name='', verbose=False):
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic': round(r[0], 4), 'pvalue': round(
        r[1], 4), 'n_lags': round(r[2], 4), 'n_obs': r[3]}
    p_value = output['pvalue']
    def adjust(val, length=6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key, val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(
            f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")


for name, column in dataFrame.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')


# %%
# 协整检验
def cointegration_test(df, alpha=0.05):
    out = coint_johansen(df, -1, 5)
    d = {'0.90': 0, '0.95': 1, '0.99': 2}
    """Trace statistic"""
    traces = out.lr1
    """Critical values (90%, 95%, 99%) of trace statistic"""
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length=6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace, 2), 9),
              ">", adjust(cvt, 8), ' =>  ', trace > cvt)


cointegration_test(ChinaBank)


# %%
# 差分
df_differenced = dataFrame.diff().dropna()
for name, column in df_differenced.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')
#df_differenced = dataFrame

# %%
# 选择模型阶数
model = VAR(df_differenced)
ss=pd.DataFrame(columns={'AIC','BIC','FPE','HQIC'})
for i in range(40):
    result = model.fit(i+1)
    ss.loc[str(i+1)+'阶']=[[result.aic],[result.bic],[result.fpe],[result.hqic]]
    # print('Lag Order =', i)
    # print('AIC : ', result.aic, '\n')
    # print('BIC : ', result.bic, '\n')
    # print('FPE : ', result.fpe, '\n')
    # print('HQIC : ', result.hqic, '\n')
ss

# %%
# 拟合模型
model_fitted = model.fit(8)
model_fitted.summary()


# %%
#durbin watson test
out = durbin_watson(model_fitted.resid)
a=0
b=0
for col, val in zip(sub.columns, out):
    a+=round(val, 4)
    b=b+1
    print(b,round(val, 4))  # 检验值越接近2，说明模型越好
a=a/b
print("平均数：",a)

# %%

lag_order = model_fitted.k_ar
forecast_input = df_differenced.values[-lag_order:]
fc = model_fitted.forecast(y=forecast_input, steps=22)
df_forecast = pd.DataFrame(fc, index=sub.index[-22:])
df_forecast


# %%
#差分还原
def invert_transformation(df_train, df_forecast):
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        #df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        df_fc[col] = df_train[col].iloc[-1] + df_fc[col].cumsum()
    return df_fc

#df_results=df_forecast
df_results = invert_transformation(dataFrame, df_forecast)
df_results
# df_results.loc[:, ['rgnp_forecast', 'pgnp_forecast', 'ulc_forecast', 'gdfco_forecast',
#                    'gdf_forecast', 'gdfim_forecast', 'gdfcf_forecast', 'gdfce_forecast']]


# %%
df_results=np.dot(df_results,vt[:dimension,:])
df_results=pd.DataFrame(df_results,index=testDf.index,columns=testDf.columns+"_Forecast")
df_results

# %%
def forecase_accuracy(forecast,actual):
    mape=np.mean(np.abs(forecast-actual)/np.abs(actual))
    me=np.mean(forecast-actual)
    mae=np.mean(np.abs(forecast-actual))
    mpe=np.mean((forecast-actual)/actual)
    rmse=np.mean((forecast-actual)**2)**.5
    corr=np.corrcoef(forecast,actual)[0,1]
    # mins=np.amin(np.hstack([forecast[:,None],actual[:,None]]),axis=1)
    # maxs=np.amax(np.hstack([forecast[:,None],actual[:,None]]),axis=1)
    # minmax=1-np.mean(mins/maxs)
    return ({'mape':mape,'me':me,'mae':mae,'mpe':mpe,'rmse':rmse,'corr':corr})

#print(forecase_accuracy(df_results,testDf))
fig, axes = plt.subplots(nrows=5, ncols=1, dpi=150, figsize=(5, 10))
for i, (col, ax) in enumerate(zip(sub.columns, axes.flatten())):
    df_results[str(col)+"_Forecast"].plot(legend=True,
                                          ax=ax).autoscale(axis='x', tight=True)
    testDf[str(col)].plot(legend=True, ax=ax)
    ax.set_title(col + ": Forecast vs Actuals:"+str(r2_score(df_results[str(col)+"_Forecast"].T,testDf[str(col)].T)))
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()



