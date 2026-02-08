import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

# --- 페이지 설정 ---
st.set_page_config(page_title="ETF Backtester", layout="wide")
st.title("📈 배당 재투자 백테스터")

# --- 사이드바 설정 ---
st.sidebar.header("⚙️ 투자 설정")
my_tickers_input = st.sidebar.text_input("나의 티커 (쉼표 구분)", "SCHD, DGRO")
my_weights_input = st.sidebar.text_input("나의 비중 (쉼표 구분)", "0.5, 0.5")
bench_tickers_input = st.sidebar.text_input("벤치마크 티커", "SPY")
bench_weights_input = st.sidebar.text_input("벤치마크 비중", "1.0")

initial_capital = st.sidebar.number_input("초기 자본 ($)", value=10000)
monthly_investment = st.sidebar.number_input("월 투자액 ($)", value=1000)
salary_growth = st.sidebar.slider("연봉 상승률 (%)", 0, 15, 5) / 100
start_date = st.sidebar.date_input("시작일", datetime.date(2015, 1, 1))
rebalance_freq = st.sidebar.selectbox("리밸런싱 주기", ["M", "Q", "H", "Y"], index=1)

# ---------------- 데이터 수집 ----------------
@st.cache_data
def get_data_safe(tickers, start):
    price_dict, div_dict, valid_starts = {}, {}, []

    for t in tickers:
        t = t.strip().upper()
        tk = yf.Ticker(t)
        h = tk.history(start=start, interval="1d")

        if h is None or h.empty or 'Close' not in h.columns:
            continue

        price_dict[t] = h['Close']
        div_dict[t] = h['Dividends'] if 'Dividends' in h.columns else pd.Series(0, index=h.index)
        valid_starts.append(h.index[0])

    if not price_dict:
        return None, None, None

    price_df = pd.DataFrame(price_dict).ffill()
    max_start = max(valid_starts)
    price_df = price_df.loc[max_start:].dropna()

    div_df = pd.DataFrame(div_dict).reindex(price_df.index, fill_value=0)
    div_df.index = div_df.index.tz_localize(None)
    price_df.index = price_df.index.tz_localize(None)

    return price_df, div_df, max_start

# ---------------- 백테스트 ----------------
def run_backtest(price_data, div_data, tickers, weights, is_bench=False):
    portfolio_value = []
    dates = price_data.index
    annual_divs = {y: 0.0 for y in dates.year.unique()}

    shares = {t: 0.0 for t in tickers}
    cash = float(initial_capital)
    current_pay = float(monthly_investment)

    for i, date in enumerate(dates):
        if i > 0 and date.month != dates[i-1].month:
            cash += current_pay
            if date.month == 1:
                current_pay *= (1 + salary_growth)

        for t in tickers:
            d_amt = div_data.loc[date, t]
            if d_amt > 0:
                received = shares[t] * d_amt
                annual_divs[date.year] += received
                cash += received

        is_rebal = False
        if is_bench:
            if i == 0 or date.month != dates[i-1].month:
                is_rebal = True
        else:
            if i == 0:
                is_rebal = True
            elif rebalance_freq == 'M' and date.month != dates[i-1].month:
                is_rebal = True
            elif rebalance_freq == 'Q' and (date.month-1)//3 != (dates[i-1].month-1)//3:
                is_rebal = True
            elif rebalance_freq == 'H' and (date.month-1)//6 != (dates[i-1].month-1)//6:
                is_rebal = True
            elif rebalance_freq == 'Y' and date.year != dates[i-1].year:
                is_rebal = True

        if is_rebal:
            total = cash + sum(shares[t] * price_data.loc[date, t] for t in tickers)
            for t, w in zip(tickers, weights):
                shares[t] = (total * w) / price_data.loc[date, t]
            cash = 0.0

        daily_total = sum(shares[t] * price_data.loc[date, t] for t in tickers) + cash
        portfolio_value.append(float(daily_total))

    return pd.Series(portfolio_value, index=dates), annual_divs

# ---------------- 지표 함수 ----------------
def calc_mdd(series):
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min()

def get_summary_metrics(h):
    years = (h.index[-1] - h.index[0]).days / 365
    cagr = ((h.iloc[-1] / h.iloc[0]) ** (1/years) - 1) * 100
    vol = h.pct_change().std() * np.sqrt(252) * 100
    sharpe = (h.pct_change().mean() / h.pct_change().std()) * np.sqrt(252)
    mdd = calc_mdd(h) * 100
    return round(cagr,2), round(vol,2), round(sharpe,2), round(mdd,2)

def get_sum_df(h, d):
    rows = []
    prev_asset, prev_div = None, None

    for y in sorted(d.keys()):
        y_h = h.loc[str(y)]
        start_asset = y_h.iloc[0]
        end_asset = y_h.iloc[-1]

        yoy = ((end_asset / prev_asset) - 1) * 100 if prev_asset else 0
        div = d[y]
        div_yoy = ((div / prev_div) - 1) * 100 if prev_div and prev_div != 0 else 0
        mdd = calc_mdd(y_h) * 100

        rows.append({
            "Year": y,
            "Start Asset": round(start_asset, 2),
            "Final Asset": round(end_asset, 2),
            "YoY %": round(yoy, 2),
            "Annual Div": round(div, 2),
            "Div YoY %": round(div_yoy, 2),
            "MDD %": round(mdd, 2)
        })

        prev_asset, prev_div = end_asset, div

    return pd.DataFrame(rows).set_index("Year")

# ---------------- 실행 ----------------
my_t = [x.strip().upper() for x in my_tickers_input.split(",")]
my_w = [float(x.strip()) for x in my_weights_input.split(",")]
bn_t = [x.strip().upper() for x in bench_tickers_input.split(",")]
bn_w = [float(x.strip()) for x in bench_weights_input.split(",")]

if st.sidebar.button("백테스팅 시작"):
    p_df, d_df, adj_start = get_data_safe(list(set(my_t + bn_t)), start_date)

    if p_df is not None:
        my_h, my_divs = run_backtest(p_df, d_df, my_t, my_w)
        bn_h, bn_divs = run_backtest(p_df, d_df, bn_t, bn_w, is_bench=True)

        # 그래프
        st.subheader("📊 자산 성장 곡선")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(my_h.index, my_h.values, label="My")
        ax.plot(bn_h.index, bn_h.values, label="Bench", ls="--")
        ax.legend()
        st.pyplot(fig)

        # 전체 요약 카드
        st.subheader("📌 전체 성과")
        c1,c2,c3,c4 = st.columns(4)
        cagr, vol, sharpe, mdd = get_summary_metrics(my_h)
        c1.metric("CAGR %", cagr)
        c2.metric("Volatility %", vol)
        c3.metric("Sharpe", sharpe)
        c4.metric("MDD %", mdd)

        # 연도별 표
        st.subheader("📋 연도별 성과")
        col1, col2 = st.columns(2)
        with col1:
            st.write("### My Strategy")
            st.dataframe(get_sum_df(my_h, my_divs))
        with col2:
            st.write("### Benchmark")
            st.dataframe(get_sum_df(bn_h, bn_divs))
