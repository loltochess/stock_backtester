import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

# ---------------- 페이지 ----------------
st.set_page_config(page_title="ETF Backtester", layout="wide")
st.title("📈 배당 재투자 백테스터")

# ---------------- 사이드바 ----------------
st.sidebar.header("⚙️ 투자 설정")

my_tickers_input = st.sidebar.text_input("나의 티커", "SCHD, DGRO")
my_weights_input = st.sidebar.text_input("나의 비중", "0.5, 0.5")
bench_tickers_input = st.sidebar.text_input("벤치마크 티커", "SPY")
bench_weights_input = st.sidebar.text_input("벤치마크 비중", "1.0")

initial_capital = st.sidebar.number_input("초기 자본 ($)", value=10000)
monthly_investment = st.sidebar.number_input("월 투자액 ($)", value=1000)
salary_growth = st.sidebar.slider("연봉 상승률 (%)", 0, 15, 5) / 100
start_date = st.sidebar.date_input("시작일", datetime.date(2015, 1, 1))

log_scale = st.sidebar.checkbox("Log Scale")
reinvest_div = st.sidebar.checkbox("배당 재투자", value=True)

# ---------------- 데이터 ----------------
@st.cache_data
def get_data_safe(tickers, start):
    price_dict, div_dict, valid_starts = {}, {}, []

    for t in tickers:
        tk = yf.Ticker(t)
        h = tk.history(start=start)
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

    return price_df, div_df, max_start

# ---------------- 백테스트 ----------------
def run_backtest(price_data, div_data, tickers, weights):
    portfolio_value = []
    annual_divs = {y: 0.0 for y in price_data.index.year.unique()}
    shares = {t: 0.0 for t in tickers}
    cash = float(initial_capital)
    current_pay = float(monthly_investment)

    for i, date in enumerate(price_data.index):
        if i > 0 and date.month != price_data.index[i-1].month:
            cash += current_pay
            if date.month == 1:
                current_pay *= (1 + salary_growth)

        for t in tickers:
            d_amt = div_data.loc[date, t]
            if d_amt > 0:
                received = shares[t] * d_amt
                annual_divs[date.year] += received
                if reinvest_div:
                    cash += received

        if i == 0 or date.month != price_data.index[i-1].month:
            total = cash + sum(shares[t] * price_data.loc[date, t] for t in tickers)
            for t, w in zip(tickers, weights):
                shares[t] = (total * w) / price_data.loc[date, t]
            cash = 0.0

        total_val = sum(shares[t] * price_data.loc[date, t] for t in tickers) + cash
        portfolio_value.append(total_val)

    return pd.Series(portfolio_value, index=price_data.index), annual_divs

# ---------------- 지표 ----------------

def build_yearly_table(price_df, history_series, annual_divs, tickers):
    rows = []

    div_series = pd.Series(annual_divs, dtype="float64").sort_index()

    prev_price = None
    prev_div = None

    for year in sorted(div_series.index):
        y_hist = history_series.loc[str(year)]
        if y_hist.empty:
            continue

        # 연말 자산을 총자산으로 본다.
        end_asset = y_hist.iloc[-1]

        # 연말 종가 (대표 ETF 첫번째 기준)
        y_price = price_df[tickers[0]].loc[str(year)]
        year_end_price = y_price.iloc[-1]

        # Price YoY
        price_yoy = ((year_end_price / prev_price) - 1) * 100 if prev_price else 0

        # Dividend
        div = div_series.loc[year]
        div_yoy = ((div / prev_div) - 1) * 100 if prev_div and prev_div != 0 else 0

        # MDD
        roll_max = y_hist.cummax()
        dd = (y_hist - roll_max) / roll_max
        mdd = dd.min() * 100

        rows.append({
            "Year": int(year),
            "Asset": round(float(end_asset), 2),
            "Price": round(float(year_end_price), 2),
            "Price YoY %": round(float(price_yoy), 2),
            "Dividend": round(float(div), 2),
            "Dividend YoY %": round(float(div_yoy), 2),
            "MDD %": round(float(mdd), 2)
        })

        prev_price = year_end_price
        prev_div = div

    df = pd.DataFrame(rows)
    return df

def calc_drawdown(series):
    roll_max = series.cummax()
    return (series - roll_max) / roll_max

def get_total_metrics(history_series, annual_divs):
    years = (history_series.index[-1] - history_series.index[0]).days / 365.25
    cagr = ((history_series.iloc[-1] / history_series.iloc[0]) ** (1/years) - 1) * 100

    div_series = pd.Series(annual_divs, dtype="float64").sort_index()
    div_yoy = div_series.pct_change().dropna() * 100
    avg_div_growth = div_yoy.mean() if len(div_yoy) > 0 else 0.0

    roll_max = history_series.cummax()
    dd = (history_series - roll_max) / roll_max
    mdd = dd.min() * 100

    return round(cagr,2), round(avg_div_growth,2), round(mdd,2)

# ---------------- 실행 ----------------
my_t = [x.strip().upper() for x in my_tickers_input.split(",")]
my_w = np.array([float(x) for x in my_weights_input.split(",")])
my_w = my_w / my_w.sum()

bn_t = [x.strip().upper() for x in bench_tickers_input.split(",")]
bn_w = np.array([float(x) for x in bench_weights_input.split(",")])
bn_w = bn_w / bn_w.sum()

if st.sidebar.button("백테스팅 시작"):
    p_df, d_df, _ = get_data_safe(list(set(my_t + bn_t)), start_date)

    if p_df is None:
        st.error("데이터 오류")
    else:
        my_h, my_divs = run_backtest(p_df, d_df, my_t, my_w)
        bn_h, bn_divs = run_backtest(p_df, d_df, bn_t, bn_w)

        # -------- 자산 --------
        st.subheader("자산 성장")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(my_h, label="My")
        ax.plot(bn_h, label="Bench")
        if log_scale:
            ax.set_yscale("log")
        ax.legend()
        st.pyplot(fig)

        # -------- Drawdown --------
        st.subheader("Drawdown")
        fig2, ax2 = plt.subplots(figsize=(8,3))
        ax2.plot(calc_drawdown(my_h)*100, label="My DD")
        ax2.plot(calc_drawdown(bn_h)*100, label="Bench DD")
        ax2.legend()
        st.pyplot(fig2)

        # -------- 전체 요약 --------
        st.subheader("📌 전체 기간 요약")
        c1,c2 = st.columns(2)

        with c1:
            st.write("### My Strategy")
            cagr, avg_div, mdd = get_total_metrics(my_h, my_divs)
            a,b,c = st.columns(3)
            a.metric("CAGR %", cagr)
            b.metric("Avg Div Growth %", avg_div)
            c.metric("MDD %", mdd)

        with c2:
            st.write("### Benchmark")
            cagr, avg_div, mdd = get_total_metrics(bn_h, bn_divs)
            a,b,c = st.columns(3)
            a.metric("CAGR %", cagr)
            b.metric("Avg Div Growth %", avg_div)
            c.metric("MDD %", mdd)

        # -------- 연도별 종합 Spreadsheet --------
        st.subheader("📋 Yearly Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### My Strategy")
            my_table = build_yearly_table(p_df, my_h, my_divs, my_t)
            st.dataframe(my_table, use_container_width=True)

        with col2:
            st.write("### Benchmark")
            bn_table = build_yearly_table(p_df, bn_h, bn_divs, bn_t)
            st.dataframe(bn_table, use_container_width=True)

