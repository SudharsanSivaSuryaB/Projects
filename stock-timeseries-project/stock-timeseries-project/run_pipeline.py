# from __future__ import annotations
# import argparse, os
# import pandas as pd
# from pathlib import Path
# from src.data_loader import load_from_yfinance, load_from_csv
# from src.preprocess import basic_clean, train_test_split_series
# from src.eda import plot_close, plot_train_test
# from src.evaluate import evaluate_forecast
# from src.compare import comparison_table
# from src.models.arima_sarima import fit_arima, fit_sarima, forecast as sarimax_forecast
# from src.models.prophet_model import fit_prophet, forecast as prophet_forecast
# from src.models.lstm_model import fit_lstm, forecast_lstm

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ticker', type=str, default='AAPL')
#     parser.add_argument('--start', type=str, default='2015-01-01')
#     parser.add_argument('--end', type=str, default='2025-08-31')
#     parser.add_argument('--horizon', type=int, default=30)
#     parser.add_argument('--csv', type=str, default=None, help='Optional path to local CSV with Date,OHLCV')
#     parser.add_argument('--output', type=str, default='artifacts')
#     parser.add_argument('--epochs', type=int, default=10, help='LSTM epochs')
#     args = parser.parse_args()

#     outdir = Path(args.output)
#     (outdir / 'figures').mkdir(parents=True, exist_ok=True)

#     # Load
#     if args.csv:
#         df = load_from_csv(args.csv)
#     else:
#         df = load_from_yfinance(args.ticker, args.start, args.end)

#     # Clean
#     df = basic_clean(df)

#     # EDA plots
#     plot_close(df, str(outdir / 'figures' / 'close.png'))

#     # Split
#     train, test = train_test_split_series(df, test_size=max(30, args.horizon))
#     plot_train_test(train, test, str(outdir / 'figures' / 'train_test.png'))

#     # Prepare series
#     y_train = train['Close'].reset_index(drop=True)
#     y_test = test['Close'].reset_index(drop=True)

#     results = {}
#     preds_out = {}

#     # ARIMA
#     try:
#         arima_fit = fit_arima(y_train, order=(5,1,0))
#         arima_preds = sarimax_forecast(arima_fit, steps=len(y_test))
#         metrics = evaluate_forecast(y_test.values, arima_preds)
#         results['ARIMA(5,1,0)'] = metrics
#         preds_out['ARIMA'] = arima_preds
#     except Exception as e:
#         print('ARIMA failed:', e)

#     # SARIMA
#     try:
#         sarima_fit = fit_sarima(y_train, order=(1,1,1), seasonal_order=(1,1,1,12))
#         sarima_preds = sarimax_forecast(sarima_fit, steps=len(y_test))
#         metrics = evaluate_forecast(y_test.values, sarima_preds)
#         results['SARIMA(1,1,1)x(1,1,1,12)'] = metrics
#         preds_out['SARIMA'] = sarima_preds
#     except Exception as e:
#         print('SARIMA failed:', e)

#     # Prophet
#     try:
#         m = fit_prophet(train[['Date','Close']])
#         fc = prophet_forecast(m, horizon=len(y_test), last_date=train['Date'].iloc[-1])
#         prophet_preds = fc['yhat'].values[:len(y_test)]
#         metrics = evaluate_forecast(y_test.values, prophet_preds)
#         results['Prophet'] = metrics
#         preds_out['Prophet'] = prophet_preds
#     except Exception as e:
#         print('Prophet failed:', e)

#     # LSTM
#     try:
#         model, lookback, mean, std = fit_lstm(y_train.values, lookback=30, epochs=args.epochs)
#         lstm_preds = forecast_lstm(model, history=y_train.values, steps=len(y_test), lookback=lookback, mean=mean, std=std)
#         metrics = evaluate_forecast(y_test.values, lstm_preds)
#         results['LSTM'] = metrics
#         preds_out['LSTM'] = lstm_preds
#     except Exception as e:
#         print('LSTM failed:', e)

#     # Save metrics
#     comp = comparison_table(results)
#     comp.to_csv(outdir / 'comparison.csv', index=False)
#     print('\nModel Comparison:')
#     print(comp)

#     # Save test predictions for each model
#     idx = test['Date'].reset_index(drop=True)
#     for name, preds in preds_out.items():
#         pd.DataFrame({'Date': idx, 'y_true': y_test, 'y_pred': preds}).to_csv(outdir / f'predictions_{name}.csv', index=False)

# if __name__ == '__main__':
#     main()
from __future__ import annotations
import argparse, os
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from src.data_loader import load_from_yfinance, load_from_csv
from src.preprocess import basic_clean, train_test_split_series
from src.eda import plot_close, plot_train_test
from src.evaluate import evaluate_forecast
from src.compare import comparison_table
from src.models.arima_sarima import fit_arima, fit_sarima, forecast as sarimax_forecast
from src.models.prophet_model import fit_prophet, forecast as prophet_forecast
from src.models.lstm_model import fit_lstm, forecast_lstm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='AAPL')
    parser.add_argument('--start', type=str, default='2015-01-01')
    parser.add_argument('--end', type=str, default='2025-08-31')
    parser.add_argument('--horizon', type=int, default=30)
    parser.add_argument('--csv', type=str, default=None, help='Optional path to local CSV with Date,OHLCV')
    parser.add_argument('--output', type=str, default='artifacts')
    parser.add_argument('--epochs', type=int, default=10, help='LSTM epochs')
    parser.add_argument('--auto-week', action='store_true',
                        help='Auto mode: last 5 years of data, forecast 5 days ahead')
    parser.add_argument('--current-week', action='store_true',
                        help='Use only recent ~3 months of data, forecast 5 days ahead')

    args = parser.parse_args()

    # ✅ Auto-week mode (longer history)
    if args.auto_week:
        args.start = (date.today() - timedelta(days=365*5)).strftime("%Y-%m-%d")
        args.end = date.today().strftime("%Y-%m-%d")
        args.horizon = 5
        print(f"[AUTO-WEEK] Using data from {args.start} to {args.end}, forecasting {args.horizon} days.")

    # ✅ Current-week mode (short recent history)
    if args.current_week:
        args.start = (date.today() - timedelta(days=90)).strftime("%Y-%m-%d")  # last 3 months only
        args.end = date.today().strftime("%Y-%m-%d")
        args.horizon = 5
        print(f"[CURRENT-WEEK] Using recent data ({args.start} to {args.end}), forecasting this week ({args.horizon} days).")

    outdir = Path(args.output)
    (outdir / 'figures').mkdir(parents=True, exist_ok=True)

    # Load
    if args.csv:
        df = load_from_csv(args.csv)
    else:
        df = load_from_yfinance(args.ticker, args.start, args.end)

    # Clean
    df = basic_clean(df)

    # EDA plots
    plot_close(df, str(outdir / 'figures' / 'close.png'))

    # Split
    train, test = train_test_split_series(df, test_size=max(30, args.horizon))
    plot_train_test(train, test, str(outdir / 'figures' / 'train_test.png'))

    # Prepare series
    y_train = train['Close'].reset_index(drop=True)
    y_test = test['Close'].reset_index(drop=True)

    results = {}
    preds_out = {}

    # ARIMA
    try:
        arima_fit = fit_arima(y_train, order=(5,1,0))
        arima_preds = sarimax_forecast(arima_fit, steps=len(y_test))
        metrics = evaluate_forecast(y_test.values, arima_preds)
        results['ARIMA(5,1,0)'] = metrics
        preds_out['ARIMA'] = arima_preds
    except Exception as e:
        print('ARIMA failed:', e)

    # SARIMA
    try:
        sarima_fit = fit_sarima(y_train, order=(1,1,1), seasonal_order=(1,1,1,12))
        sarima_preds = sarimax_forecast(sarima_fit, steps=len(y_test))
        metrics = evaluate_forecast(y_test.values, sarima_preds)
        results['SARIMA(1,1,1)x(1,1,1,12)'] = metrics
        preds_out['SARIMA'] = sarima_preds
    except Exception as e:
        print('SARIMA failed:', e)

    # Prophet
    try:
        m = fit_prophet(train[['Date','Close']])
        fc = prophet_forecast(m, horizon=len(y_test), last_date=train['Date'].iloc[-1])
        prophet_preds = fc['yhat'].values[:len(y_test)]
        metrics = evaluate_forecast(y_test.values, prophet_preds)
        results['Prophet'] = metrics
        preds_out['Prophet'] = prophet_preds
    except Exception as e:
        print('Prophet failed:', e)

    # LSTM
    try:
        model, lookback, mean, std = fit_lstm(y_train.values, lookback=30, epochs=args.epochs)
        lstm_preds = forecast_lstm(model, history=y_train.values, steps=len(y_test), lookback=lookback, mean=mean, std=std)
        metrics = evaluate_forecast(y_test.values, lstm_preds)
        results['LSTM'] = metrics
        preds_out['LSTM'] = lstm_preds
    except Exception as e:
        print('LSTM failed:', e)

    # Save metrics
    comp = comparison_table(results)
    comp.to_csv(outdir / 'comparison.csv', index=False)
    print('\nModel Comparison:')
    print(comp)

    # Save test predictions for each model
    idx = test['Date'].reset_index(drop=True)
    for name, preds in preds_out.items():
        pd.DataFrame({'Date': idx, 'y_true': y_test, 'y_pred': preds}).to_csv(outdir / f'predictions_{name}.csv', index=False)


if __name__ == '__main__':
    main()
