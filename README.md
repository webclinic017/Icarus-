![icarus-visual](https://github.com/bilkosem/Icarus/assets/40933377/6d9ea5e7-f39d-44c4-a602-263ea87c3379)
# Icarus
Icarus is a trading platform that enable users to do technical analysis on financial markets and develop strategies with an engineering approach. It enable users to do backtest, live-trade, report, monitor, and develop custom trading strategies.



# Table of content

- [Installation](#installation)
- [Philosophy](#philosophy)
- [Applications](#applications)
- [Live Trade](#live-trade)

# Installation
```
git clone https://github.com/bilkosem/Icarus.git
cd Icarus
sudo ./install.sh
```
Install mongodb from official website: https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/, then
```
sudo chown -R mongodb:mongodb /var/lib/mongodb
sudo chown mongodb:mongodb /tmp/mongodb-27017.sock
sudo service mongod restart
```
# Philosophy
Icarus tries to overcome downsides of traditional trading strategy creation process:
<p align="center">
  <img src="/docs/readme/strategy_dev_old.png?raw=true" alt="strategy_dev_old"/>
</p>

## Problems
### Strategy Creation

A hypothesis is not equal to a strategy. Strategies are compose of multiple hypothesis considering entry/exit rules, pozition sizing and risk management. When a hypothesis is converted to a strategy and tested, the results also contains the affect of other components. Thus the evaluation of the results might be misleading and does not reflect the real validity of hypothesis.

### Evaluate Results

As a result of the backtest, statistics are created. These statistics contains general metrics like, total profit, win rate, average return, average duration etc. These metrics may only measure the validity of the hypothesis indirectly.

### Optimize

Since the hypothesis itself is not the direct subject of the statistic evaluation, how to decide what to optimize.

## Scientific Approach for Strategy Development
Proposed solution:

<p align="center">
  <img src="/docs/readme/strategy_dev_new.png?raw=true" alt="strategy_dev_new"/>
</p>

# Applications
## Backtest
### 1. Configure 🛠
Choose an existing configuration file (such as [configs/quick-start/config.json](configs/quick-start/config.json)) or just create a new one to experiment.

By doing backtest with 'quick-start' config file, you run 3 strategies that works on pairs together and independent from their decisions. These strategies uses Market, Limit, and OCO orders to enter and exit trades.

### 2. Run 🚀
Run the backtest script by providing the config file as argument:

`python icarus/backtest.py configs/quick-start/config.json`
### 3. Create and analyze statistics of backtest 📊
Use reporting tool to create preconfigured but also custom statistics regarding backtest session or any kind of analysis on the market.

`python icarus/generate_report.py configs/quick-start/config.json`

<details close>
  <summary>Backtest Statistics</summary>

# Backtest

## Balance

|    |   start |     end |   absolute_profit |   percentage_profit |   max_drawdown |
|---:|--------:|--------:|------------------:|--------------------:|---------------:|
|  0 |   10000 | 9875.73 |           -124.27 |               -1.24 |           1.95 |


## Strategies

### count

| strategy                          |   live |   closed |   None |   enter_expire |   manual_change |   market |   limit |   stop_limit |   not_updated |   updated |   win |   lose |
|:----------------------------------|-------:|---------:|-------:|---------------:|----------------:|---------:|--------:|-------------:|--------------:|----------:|------:|-------:|
| HODL-BTCUSDT                      |      0 |        1 |      0 |              0 |               0 |        1 |       0 |            0 |             1 |         0 |     0 |      1 |
| HODL-XRPUSDT                      |      0 |        1 |      0 |              0 |               0 |        1 |       0 |            0 |             1 |         0 |     0 |      1 |
| SREventsPredictiveVanilla-BTCUSDT |      0 |       52 |      0 |             34 |               0 |        0 |       6 |           12 |            14 |         4 |     6 |     12 |
| SREventsPredictiveVanilla-XRPUSDT |      0 |       47 |      0 |             29 |               0 |        0 |       5 |           13 |            16 |         2 |     5 |     13 |


### absolute_profit

| strategy                          |    best |   worst |   total |   total_updated |   total_not_updated |   average |   average_updated |   average_not_updated |
|:----------------------------------|--------:|--------:|--------:|----------------:|--------------------:|----------:|------------------:|----------------------:|
| HODL-BTCUSDT                      | -56.616 | -56.616 | -56.616 |           0     |             -56.616 |   -56.616 |           nan     |               -56.616 |
| HODL-XRPUSDT                      | -47.985 | -47.985 | -47.985 |           0     |             -47.985 |   -47.985 |           nan     |               -47.985 |
| SREventsPredictiveVanilla-BTCUSDT |  36.288 | -12.138 |  -0.832 |         111.618 |            -112.45  |    -0.046 |            27.904 |                -8.032 |
| SREventsPredictiveVanilla-XRPUSDT |  58.897 | -12.569 | -18.84  |          46.466 |             -65.306 |    -1.047 |            23.233 |                -4.082 |


### percentage_profit

| strategy                          |   best |   worst |   total |   total_updated |   total_not_updated |   average |   average_updated |   average_not_updated |
|:----------------------------------|-------:|--------:|--------:|----------------:|--------------------:|----------:|------------------:|----------------------:|
| HODL-BTCUSDT                      | -5.662 |  -5.662 |  -5.662 |           0     |              -5.662 |    -5.662 |           nan     |                -5.662 |
| HODL-XRPUSDT                      | -4.8   |  -4.8   |  -4.8   |           0     |              -4.8   |    -4.8   |           nan     |                -4.8   |
| SREventsPredictiveVanilla-BTCUSDT |  3.629 |  -1.214 |  -0.083 |          11.163 |             -11.246 |    -0.005 |             2.791 |                -0.803 |
| SREventsPredictiveVanilla-XRPUSDT |  5.892 |  -1.258 |  -1.886 |           4.648 |              -6.534 |    -0.105 |             2.324 |                -0.408 |


### price_change

| strategy                          |   best |   worst |   total |   total_updated |   total_not_updated |   average |   average_updated |   average_not_updated |
|:----------------------------------|-------:|--------:|--------:|----------------:|--------------------:|----------:|------------------:|----------------------:|
| HODL-BTCUSDT                      | -5.459 |  -5.459 |  -5.459 |           0     |              -5.459 |    -5.459 |           nan     |                -5.459 |
| HODL-XRPUSDT                      | -4.602 |  -4.602 |  -4.602 |           0     |              -4.602 |    -4.602 |           nan     |                -4.602 |
| SREventsPredictiveVanilla-BTCUSDT |  3.842 |  -1     |   3.713 |          12.016 |              -8.303 |     0.206 |             3.004 |                -0.593 |
| SREventsPredictiveVanilla-XRPUSDT |  6.155 |  -1.023 |   2.295 |           5.135 |              -2.84  |     0.128 |             2.568 |                -0.178 |


### duration

| strategy                          |   max |   min |   total |   average |   average_not_updated |   average_updated |
|:----------------------------------|------:|------:|--------:|----------:|----------------------:|------------------:|
| HODL-BTCUSDT                      | 1.461 | 1.461 |   1.461 |     1.461 |                 1.461 |           nan     |
| HODL-XRPUSDT                      | 1.461 | 1.461 |   1.461 |     1.461 |                 1.461 |           nan     |
| SREventsPredictiveVanilla-BTCUSDT | 0.198 | 0.001 |   0.458 |     0.025 |                 0.008 |             0.085 |
| SREventsPredictiveVanilla-XRPUSDT | 0.297 | 0.001 |   0.562 |     0.031 |                 0.014 |             0.17  |


### rates

| strategy                          |   win |   lose |   enter |
|:----------------------------------|------:|-------:|--------:|
| HODL-BTCUSDT                      | 0     |  1     |   1     |
| HODL-XRPUSDT                      | 0     |  1     |   1     |
| SREventsPredictiveVanilla-BTCUSDT | 0.333 |  0.667 |   0.346 |
| SREventsPredictiveVanilla-XRPUSDT | 0.278 |  0.722 |   0.383 |


### risk

| strategy                          |   expectancy |     SQN |
|:----------------------------------|-------------:|--------:|
| HODL-BTCUSDT                      |      nan     | nan     |
| HODL-XRPUSDT                      |      nan     | nan     |
| SREventsPredictiveVanilla-BTCUSDT |       -0.628 |  -1.983 |
| SREventsPredictiveVanilla-XRPUSDT |       -0.415 |  -1.031 |


### others

| strategy                          |   total_fee |
|:----------------------------------|------------:|
| HODL-BTCUSDT                      |       1.944 |
| HODL-XRPUSDT                      |       1.952 |
| SREventsPredictiveVanilla-BTCUSDT |      36.013 |
| SREventsPredictiveVanilla-XRPUSDT |      35.987 |

</details>
    

### 4. Visualize Trades 📈
Use Icarus Developer Dashboard to
* analyze backtested strategies
* analyze live trades
* observe internal variables
* visualize analyzers

`streamlit run icarus/developer_dashboard.py configs/quick-start/config.json`

<p align="center">
  <img src="/docs/readme/developer-dashboard.png?raw=true" alt="Icarus Developer Dashboard"/>
</p>

### 5. Visualize Indicators 📉
If you are developing a custom indicator you can still use the Icarus Developer Dashboard without any backtest.

`streamlit run icarus/developer_dashboard.py configs/quick-start/config.json`

<p align="center">
  <img src="/docs/readme/custom-indicators.png?raw=true" alt="Icarus Developer Dashboard Indicators"/>
</p>

## Live-Trade

### 1. Run 🚀
`python icarus/live-trade.py configs/quick-start/config.json`

### 2. Visualize 📈
Visualize live-trades and events on the Developer Dashboard

### 3. Monitor 🔍
Get notifications on certain events.
<p align="center"><img src="/docs/readme/telegram-bot-interface.png" width="225" height="400"></p>
