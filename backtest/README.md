# bee ICT bot backtest

Python port of `bee-ict-bot.pine` that emulates TradingView's broker (signals at bar close,
fills on later bars, intrabar path, commission and slippage).

Data: OANDA NAS100 1-minute bars 2010–2020 from the public repo
`FutureSharks/financial-data` (`pyfinancialdata/data/currencies/oanda/NAS100_USD`).
Prices are rescaled per day to ~20,000 so point-based costs match today's NQ/MNQ.

Result (1m, defaults): 288 trades, win 49.3%, profit factor 1.00, +0.01R per trade,
max drawdown 8.8%. Without the macro bias filter: PF 0.91, −74R, max drawdown 40%.
