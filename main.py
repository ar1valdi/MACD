import pandas as pd
import data_functions as df
import numpy as np
import matplotlib.pyplot as plt


def calc_EMA(dataset, n):
    alpha = 2/(n+1)
    prefixes = np.power((1 - alpha), np.arange(0, n+1))
    up = dataset[::-1] * prefixes
    return sum(up)/sum(prefixes)


def calc_MACD_and_SIGNAL(dataset):
    macd = [0] * 27
    signal = [0] * 27
    for i in range(27, dataset.shape[0]):
        ema12 = calc_EMA(dataset[(i - 12):(i+1)], 12)
        ema26 = calc_EMA(dataset[(i - 26):(i+1)], 26)
        macd.append(ema12 - ema26)
        signal.append(calc_EMA(macd[(i - 9):(i+1)], 9))

    return macd, signal


def generate_buy_sell_signals(macd, signal):
    macd, signal = np.array(macd), np.array(signal)
    delta = np.array(macd - signal)
    delta = np.sign(delta)
    changes = np.where(np.diff(delta) != 0)[0] + 1
    sells, buys = np.array([], dtype=int), np.array([], dtype=int)

    skip = False
    for change in changes:
        if skip:
            skip = False
            continue
        if delta[change] == 0:
            skip = True

        if delta[change] > 0:
            buys = np.append(buys, change)
        else:
            sells = np.append(sells, change)

    min_sells = min(sells)
    max_buys = max(buys)
    buys = buys[np.where(buys > min_sells)]
    sells = sells[np.where(sells < max_buys)]
    return buys, sells


def show_plots(dataset, macd, signal, buy_signals, sell_signals, hist_m, hist_s, hist_a, buy_id, sell_id, name):
    x = pd.to_datetime(dataset['Date'])

    plt.subplot(2, 2, 1)
    plt.plot(x, macd, label="MACD")
    plt.plot(x, signal, label="SIGNAL")
    plt.title("MACD and SIGNAL")
    plt.xlabel("Date")
    plt.ylabel("MACD / SIGNAL")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(x, macd, label="MACD")
    plt.plot(x, signal, label="SIGNAL")
    plt.title("Sell and buy moments")
    plt.xlabel("Date")
    plt.ylabel("MACD / SIGNAL")
    plt.vlines(buy_signals, ymin=min(macd), ymax=max(macd), colors="green")
    plt.vlines(sell_signals, ymin=min(macd), ymax=max(macd), colors="red")
    plt.scatter(pd.to_datetime(dataset['Date']), macd)
    plt.scatter(pd.to_datetime(dataset['Date']), signal)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(x, dataset["Open"].to_numpy(), label="STOCK")
    plt.scatter(buy_signals, dataset["Open"].to_numpy()[buy_id], c="green", label="BUY")
    plt.scatter(sell_signals, dataset["Open"].to_numpy()[sell_id], c="red", label="SELL")
    plt.title("STOCK")
    plt.xlabel("Date")
    plt.ylabel("Stock")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(x, np.full_like(hist_a, hist_a[0]), c="black")
    plt.plot(x, hist_a, c="green")
    plt.title("Profit made")
    plt.xlabel("Date")
    plt.ylabel("Profit")

    plt.subplots_adjust(hspace=0.7)
    plt.suptitle(name, fontsize=16)
    plt.show()


def to_datetime(buy, sell, dates):
    d1 = pd.to_datetime(dates[buy])
    d2 = pd.to_datetime(dates[sell])
    return d1, d2


def calc_profit_graph(buy_signals, sell_signals, data, start_stocks):
    curr_money = 0
    curr_stocks = start_stocks
    buy_iter, sell_iter = 0, 0
    result_money = np.zeros_like(data)
    result_stocks = np.zeros_like(data)
    result_stocks[0] = start_stocks

    while sell_iter < len(sell_signals):
        sell_ID = sell_signals[sell_iter]
        buy_ID = buy_signals[buy_iter]

        money_profit = curr_stocks * data[sell_ID]
        curr_money += money_profit
        result_money[sell_ID] = money_profit
        result_stocks[sell_ID] = -curr_stocks

        curr_stocks = round(curr_money / data[buy_ID], 0)
        stocks_price = curr_stocks * data[buy_ID]
        curr_money -= stocks_price
        result_money[buy_ID] = -stocks_price
        result_stocks[buy_ID] = curr_stocks

        buy_iter += 1
        sell_iter += 1

    result_stocks = np.cumsum(result_stocks)
    result_money = np.cumsum(result_money)
    result_final = result_stocks * data + result_money

    return result_money - result_money[0], result_stocks - result_stocks[0], result_final - result_final[0]


def run_MACD(vector_name, data_amount, start_stocks, name):
    dataset = df.get_data(vector_name)[0:data_amount]
    opendata = dataset['Open'].to_numpy()
    macd, signal = calc_MACD_and_SIGNAL(opendata)
    buy_signals, sell_signals = generate_buy_sell_signals(macd, signal)
    profit_money, profit_stocks, profit_all = calc_profit_graph(buy_signals, sell_signals, opendata, start_stocks)
    buy_signals_d, sell_signals_d = to_datetime(buy_signals, sell_signals, dataset["Date"])
    show_plots(dataset, macd, signal, buy_signals_d, sell_signals_d, profit_money, profit_stocks, profit_all, buy_signals, sell_signals, name)
    return dataset, profit_all


def get_macd_profit_and_stocks(vector_name, data_amount, start_stocks):
    dataset = df.get_data(vector_name)[0:data_amount]
    opendata = dataset['Open'].to_numpy()
    macd, signal = calc_MACD_and_SIGNAL(opendata)
    buy_signals, sell_signals = generate_buy_sell_signals(macd, signal)
    profit_money, profit_stocks, profit_all = calc_profit_graph(buy_signals, sell_signals, opendata, start_stocks)
    return profit_all, opendata, dataset['Date'][0], dataset['Date'][999]


def run_profit_comp(data_to_run):
    data_len = len(data_to_run)
    datasets, profits, dates_start, dates_end = [], [], [], []

    for i in range(data_len):
        p, d, dt1, dt2 = get_macd_profit_and_stocks(data_to_run[i][0], data_to_run[i][1], data_to_run[i][2])
        datasets.append(d)
        profits.append(p)
        dates_start.append(dt1)
        dates_end.append(dt2)

    datasets = np.array(datasets)
    profits = np.array(profits)
    dates_start = np.array(dates_start)
    dates_end = np.array(dates_end)

    x = np.linspace(0, 999, 1000)
    for i in range(data_len):
        data_to_run[i][3] = f'{data_to_run[i][3]}: {dates_start[i]} - {dates_end[i]}'
        plt.subplot(data_len, 2, i * 2 + 1)
        plt.plot(x, datasets[i])
        plt.title(data_to_run[i][3])
        plt.subplot(data_len, 2, i * 2 + 2)
        plt.plot(x, np.zeros_like(profits[i]), c="black")
        plt.plot(x, profits[i] - profits[i][0], c="green", label="MACD")
        plt.plot(x, (datasets[i] - datasets[i][0]) * 1000, c="red", label="HOLD")
        plt.legend()
        plt.title("PROFIT")

    plt.subplots_adjust(hspace=1.0)
    plt.show()


def run_mult_MACD(data_to_run):
    data_len = len(data_to_run)
    for i in range(data_len):
        run_MACD(data_to_run[i][0], data_to_run[i][1], data_to_run[i][2], data_to_run[i][3])


def main():
    data_to_run_full_macd = [
        ['cdr.csv', 1000, 1000, 'CD Projekt Red'],
        ['aapl_us_d.csv', 1000, 1000, "Apple"],
        ['intc_us_d.csv', 1000, 1000, "Intel"]
    ]

    data_to_run_comp = [
        ['cdr.csv', 1000, 1000, 'CD Projekt Red'],
        ['aapl_us_d.csv', 1000, 1000, "Apple"],
        ['intc_us_d.csv', 1000, 1000, "Intel"],
        ['5_check/gold_d.csv', 1000, 1000, "Gold"],
        ['5_check/ibm_us_d.csv', 1000, 1000, "IBM"],
        ['5_check/amzn_us_d.csv', 1000, 1000, "Amazon"],
        ['5_check/msft_us_d.csv', 1000, 1000, "Microsoft"],
        ['5_check/googl_us_d.csv', 1000, 1000, "Google"],
    ]
    run_mult_MACD(data_to_run_full_macd)
    run_profit_comp(data_to_run_comp)


main()
