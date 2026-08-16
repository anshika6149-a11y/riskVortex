import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(page_title="RISKVORTEX", page_icon="⚡", layout="wide")

# Custom Styling & Mobile Responsiveness
st.markdown(
    """
    <style>
        .main-title {
            text-align: center; 
            padding: 15px 0; 
            background: linear-gradient(135deg, #0F172A, #1E293B); 
            border-radius: 10px; 
            margin-bottom: 15px; 
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            color: #F8FAFC; 
            font-weight: 800; 
            font-size: 24px; 
            letter-spacing: 1.5px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 18px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='main-title'>⚡ RISKVORTEX</div>", unsafe_allow_html=True)


# =====================================================================
# OPTIMIZED BACKEND ENGINE WITH CACHING
# =====================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(ticker: str, benchmark: str):
  try:
    data = yf.download(
        [ticker, benchmark], period="max", progress=False, group_by="column"
    )
    if "Close" in data.columns:
      df = data["Close"]
    else:
      df = data

    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    return df
  except Exception as e:
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def search_tickers(query: str):
  if not query or len(query.strip()) < 1:
    return [("Apple Inc. (AAPL) [NASDAQ]", "AAPL")]
  try:
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query.strip()}&quotesCount=8&newsCount=0"
    resp = requests.get(
        url, headers={"User-Agent": "Mozilla/5.0"}, timeout=4
    ).json()
    choices = []
    for q in resp.get("quotes", []):
      symbol = q.get("symbol")
      longname = q.get("longname") or q.get("shortname") or symbol
      exch = q.get("exchDisp") or q.get("exchange") or ""
      if symbol and q.get("typeDisp") in ["Equity", "ETF"]:
        choices.append((f"{longname} ({symbol}) [{exch}]", symbol))
    return (
        choices
        if choices
        else [("Apple Inc. (AAPL) [NASDAQ]", "AAPL")]
    )
  except Exception:
    return [("Apple Inc. (AAPL) [NASDAQ]", "AAPL")]


class RiskVortexEngine:
  TIMEFRAME_DAYS = {
      "1 Week (5 Days)": 5,
      "1 Month (21 Days)": 21,
      "3 Months (63 Days)": 63,
      "6 Months (126 Days)": 126,
      "1 Year (252 Days)": 252,
      "3 Years (756 Days)": 756,
      "5 Years (1260 Days)": 1260,
  }

  def __init__(
      self,
      ticker: str,
      initial_investment: float = 100000.0,
      timeframe_label: str = "1 Year (252 Days)",
      benchmark_ticker: str = "^GSPC",
  ):
    self.ticker = ticker.strip().upper()
    self.benchmark_ticker = benchmark_ticker
    self.initial_investment = initial_investment
    self.horizon_days = self.TIMEFRAME_DAYS.get(timeframe_label, 252)

    self.df = fetch_market_data(self.ticker, self.benchmark_ticker)

    if self.df.empty or self.ticker not in self.df.columns:
      raise ValueError(
          f"Ticker '{self.ticker}' invalid or no market data found."
      )

    if self.benchmark_ticker not in self.df.columns:
      # Fallback to asset only if benchmark fails
      self.bench_prices = self.df[self.ticker].dropna()
    else:
      self.bench_prices = self.df[self.benchmark_ticker].dropna()

    self.close_prices = self.df[self.ticker].dropna()

    self.log_returns = np.log(
        self.close_prices / self.close_prices.shift(1)
    ).dropna()
    self.bench_log_returns = np.log(
        self.bench_prices / self.bench_prices.shift(1)
    ).dropna()

    aligned_data = pd.concat(
        [self.log_returns, self.bench_log_returns],
        axis=1,
        keys=["asset", "bench"],
    ).dropna()
    self.asset_aligned_returns = aligned_data["asset"]
    self.bench_aligned_returns = aligned_data["bench"]

    self.current_price = float(self.close_prices.iloc[-1])

  def get_tickertape_summary(self):
    try:
      ticker_obj = yf.Ticker(self.ticker)
      info = ticker_obj.info or {}

      mkt_cap = info.get("marketCap", 0) or 0
      if mkt_cap >= 1e12:
        mkt_cap_fmt = f"{mkt_cap / 1e12:.2f} T"
      elif mkt_cap >= 1e9:
        mkt_cap_fmt = f"{mkt_cap / 1e9:.2f} B"
      elif mkt_cap >= 1e6:
        mkt_cap_fmt = f"{mkt_cap / 1e6:.2f} M"
      else:
        mkt_cap_fmt = f"{mkt_cap:,}" if mkt_cap > 0 else "N/A"

      mean_target = info.get("targetMeanPrice", 0.0) or 0.0
      upside_pct = 0.0
      if mean_target > 0 and self.current_price > 0:
        upside_pct = (
            (mean_target - self.current_price) / self.current_price
        ) * 100

      div_raw = info.get("dividendYield", 0.0) or 0.0
      if div_raw > 1.0:
        div_raw = div_raw / 100.0
      div_yield_str = (
          f"{div_raw * 100:.2f}%" if 0 < div_raw <= 0.25 else "N/A"
      )

      profit_margin = info.get("profitMargins", 0.0)
      profit_margin_str = (
          f"{profit_margin * 100:.2f}%"
          if profit_margin is not None
          else "N/A"
      )

      return {
          "Company_Name": info.get("longName")
          or info.get("shortName")
          or self.ticker,
          "Sector": info.get("sector", "N/A") or "N/A",
          "Industry": info.get("industry", "N/A") or "N/A",
          "Country": info.get("country", "N/A") or "N/A",
          "Market_Cap": mkt_cap_fmt,
          "PE_Ratio": (
              f"{info.get('trailingPE', 0.0):.2f}"
              if info.get("trailingPE")
              else "N/A"
          ),
          "PB_Ratio": (
              f"{info.get('priceToBook', 0.0):.2f}"
              if info.get("priceToBook")
              else "N/A"
          ),
          "Dividend_Yield": div_yield_str,
          "Beta": f"{info.get('beta', 0.0):.2f}" if info.get("beta") else "N/A",
          "Profit_Margins": profit_margin_str,
          "52W_High": (
              f"{info.get('fiftyTwoWeekHigh', 0.0):,.2f}"
              if info.get("fiftyTwoWeekHigh")
              else "N/A"
          ),
          "52W_Low": (
              f"{info.get('fiftyTwoWeekLow', 0.0):,.2f}"
              if info.get("fiftyTwoWeekLow")
              else "N/A"
          ),
          "Mean_Target": f"{mean_target:,.2f}" if mean_target > 0 else "N/A",
          "Target_Upside": f"{upside_pct:+.2f}%" if mean_target > 0 else "N/A",
          "Summary": info.get(
              "longBusinessSummary", "No business description available."
          ),
          "Info_Dict": info,
      }
    except Exception:
      return {
          "Company_Name": self.ticker,
          "Sector": "N/A",
          "Industry": "N/A",
          "Country": "N/A",
          "Market_Cap": "N/A",
          "PE_Ratio": "N/A",
          "PB_Ratio": "N/A",
          "Dividend_Yield": "N/A",
          "Beta": "N/A",
          "Profit_Margins": "N/A",
          "52W_High": "N/A",
          "52W_Low": "N/A",
          "Mean_Target": "N/A",
          "Target_Upside": "N/A",
          "Summary": "Fundamental details could not be loaded.",
          "Info_Dict": {},
      }

  def calculate_altman_z_score(self, info: dict):
    try:
      total_assets = info.get("totalAssets", 0) or 0
      total_liab = info.get("totalDebt", 0) or 0
      mkt_cap = info.get("marketCap", 0) or 0
      retained_earnings = info.get("retainedEarnings", 0) or 0
      ebit = info.get("ebit", 0) or 0
      revenue = info.get("totalRevenue", 0) or 0

      if not total_assets or total_assets == 0:
        pb = info.get("priceToBook", 2.0) or 2.0
        pe = info.get("trailingPE", 20.0) or 20.0
        z_score = (
            1.2
            + (1.4 / max(pb, 0.5))
            + (3.3 * (1.0 / max(pe, 5.0)))
            + 0.6 * 1.5
            + 0.999 * 1.2
        )
        return z_score, "Estimated (Partial Data)"

      x1 = 0.25
      x2 = retained_earnings / total_assets if total_assets else 0.1
      x3 = ebit / total_assets if total_assets else 0.1
      x4 = mkt_cap / total_liab if total_liab and total_liab > 0 else 1.5
      x5 = revenue / total_assets if total_assets else 1.0

      z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5
      return float(z_score), "Calculated"
    except Exception:
      return 2.7, "Default Estimation"

  def get_chart_data(self, period_str: str, chart_type: str):
    period_map = {
        "1 Week": "5d",
        "1 Month": "1mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "5 Years": "5y",
        "10 Years": "10y",
        "Max": "max",
    }
    yf_period = period_map.get(period_str, "1y")

    hist_df = yf.download(self.ticker, period=yf_period, progress=False)
    if isinstance(hist_df.columns, pd.MultiIndex):
      hist_df.columns = hist_df.columns.get_level_values(0)

    hist_df = hist_df.dropna()
    if hist_df.empty:
      fig = go.Figure()
      fig.update_layout(title="No chart data available for this period.")
      return fig

    fig = go.Figure()
    if chart_type == "Candlestick":
      fig.add_trace(
          go.Candlestick(
              x=hist_df.index,
              open=hist_df["Open"],
              high=hist_df["High"],
              low=hist_df["Low"],
              close=hist_df["Close"],
              name="Candlestick",
          )
      )
      fig.update_layout(xaxis_rangeslider_visible=False)
    else:
      fig.add_trace(
          go.Scatter(
              x=hist_df.index,
              y=hist_df["Close"],
              mode="lines",
              line=dict(color="#0284C7", width=2),
              name="Close Price",
          )
      )

    fig.update_layout(
        title=f"<b>{self.ticker} Price Chart ({period_str}) - {chart_type}</b>",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig

  def calculate_risk_metrics(self, risk_free_rate: float = 0.045):
    period_return = float(np.mean(self.log_returns)) * self.horizon_days
    period_volatility = float(np.std(self.log_returns)) * np.sqrt(
        self.horizon_days
    )
    excess_returns = self.log_returns - (risk_free_rate / 252.0)
    ann_vol = np.std(self.log_returns) * np.sqrt(252)
    sharpe_ratio = (
        (np.mean(excess_returns) * 252) / ann_vol if ann_vol != 0 else 0
    )
    downside_returns = self.log_returns[self.log_returns < 0]
    downside_vol = (
        np.std(downside_returns) * np.sqrt(252)
        if len(downside_returns) > 0
        else 1e-6
    )
    sortino_ratio = (np.mean(excess_returns) * 252) / downside_vol
    cum_returns = (1 + self.close_prices.pct_change()).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = float(drawdown.min()) * 100

    return {
        "Period_Return": period_return * 100,
        "Period_Volatility": period_volatility * 100,
        "Sharpe_Ratio": sharpe_ratio,
        "Sortino_Ratio": sortino_ratio,
        "Max_Drawdown": max_drawdown,
    }

  def run_monte_carlo(self, simulations: int = 5000, confidence_level: float = 0.95):
    days = self.horizon_days
    dt = 1 / 252
    mu = float(np.mean(self.log_returns))
    sigma = float(np.std(self.log_returns))

    drift = (mu - 0.5 * sigma**2) * dt
    Z = np.random.normal(0, 1, (days, simulations))
    daily_log_returns = drift + (sigma * np.sqrt(dt) * Z)

    price_paths = np.zeros((days + 1, simulations))
    price_paths[0] = self.current_price

    for t in range(1, days + 1):
      price_paths[t] = price_paths[t - 1] * np.exp(daily_log_returns[t - 1])

    final_prices = price_paths[-1]
    simulated_port_values = (
        final_prices / self.current_price
    ) * self.initial_investment

    percentile_idx = int((1 - confidence_level) * 100)
    var_portfolio_val = np.percentile(simulated_port_values, percentile_idx)
    var_dollar = self.initial_investment - var_portfolio_val
    var_pct = (var_dollar / self.initial_investment) * 100

    tail_losses = simulated_port_values[
        simulated_port_values <= var_portfolio_val
    ]
    cvar_dollar = (
        self.initial_investment - np.mean(tail_losses)
        if len(tail_losses) > 0
        else var_dollar
    )
    cvar_pct = (cvar_dollar / self.initial_investment) * 100

    percentiles = {
        "p5_worst": np.percentile(simulated_port_values, 5),
        "p50_median": np.percentile(simulated_port_values, 50),
        "p95_best": np.percentile(simulated_port_values, 95),
    }

    return (
        price_paths,
        simulated_port_values,
        var_dollar,
        var_pct,
        cvar_dollar,
        cvar_pct,
        percentiles,
    )

  def run_stress_testing(self):
    timeframe_multiplier = np.sqrt(self.horizon_days / 252.0)
    scenarios = {
        "2008 Financial Crisis": -0.45 * timeframe_multiplier,
        "2020 COVID Liquidity Crash": -0.30 * timeframe_multiplier,
        "Rate Hike Shock": -0.15 * timeframe_multiplier,
        "Tech Sector Selloff": -0.22 * timeframe_multiplier,
    }
    stress_results = {}
    for name, drop in scenarios.items():
      drop = max(drop, -0.95)
      projected_val = self.initial_investment * (1 + drop)
      loss_val = self.initial_investment - projected_val
      stress_results[name] = {
          "Projected": projected_val,
          "Loss": loss_val,
          "Drop_Pct": drop * 100,
      }
    return stress_results

  def calculate_factor_attribution(self, risk_free_rate: float = 0.045):
    cov_matrix = np.cov(self.asset_aligned_returns, self.bench_aligned_returns)
    covariance = cov_matrix[0][1]
    bench_variance = np.var(self.bench_aligned_returns)
    beta = covariance / bench_variance if bench_variance != 0 else 1.0
    ann_asset_return = np.mean(self.asset_aligned_returns) * 252
    ann_bench_return = np.mean(self.bench_aligned_returns) * 252
    capm_expected = risk_free_rate + beta * (
        ann_bench_return - risk_free_rate
    )
    jensens_alpha = ann_asset_return - capm_expected
    return {
        "Beta": beta,
        "CAPM_Expected": capm_expected * 100,
        "Jensens_Alpha": jensens_alpha * 100,
        "Benchmark_Return": ann_bench_return * 100,
    }


# =====================================================================
# UI LAYOUT & CONTROLS
# =====================================================================
col_c1, col_c2, col_c3, col_c4 = st.columns([2, 2, 1.5, 1])

with col_c1:
  search_query = st.text_input("🔍 Search Company / Asset", value="AAPL")

ticker_options = search_tickers(search_query)
ticker_labels = [opt[0] for opt in ticker_options]
ticker_values = [opt[1] for opt in ticker_options]

with col_c2:
  selected_ticker_label = st.selectbox("📌 Select Security", ticker_labels)
  ticker = ticker_values[ticker_labels.index(selected_ticker_label)]

with col_c3:
  timeframe = st.selectbox(
      "⏱️ Horizon",
      [
          "1 Week (5 Days)",
          "1 Month (21 Days)",
          "3 Months (63 Days)",
          "6 Months (126 Days)",
          "1 Year (252 Days)",
          "3 Years (756 Days)",
          "5 Years (1260 Days)",
      ],
      index=4,
  )

with col_c4:
  capital = st.number_input(
      "Capital ($/₹)", value=100000.0, step=10000.0, format="%.0f"
  )

sub_c1, sub_c2, sub_c3 = st.columns([1, 1, 1])
with sub_c1:
  chart_type = st.selectbox("📈 Chart Style", ["Line Chart", "Candlestick"])
with sub_c2:
  chart_period = st.selectbox(
      "📅 Chart Range",
      ["1 Week", "1 Month", "6 Months", "1 Year", "5 Years", "Max"],
      index=3,
  )
with sub_c3:
  st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
  run_btn = st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True)

st.divider()

# =====================================================================
# EXECUTION & TABS
# =====================================================================
if run_btn or "analyzed" not in st.session_state:
  st.session_state["analyzed"] = True
  try:
    benchmark = (
        "^NSEI"
        if ticker.endswith(".NS") or ticker.endswith(".BO")
        else "^GSPC"
    )
    engine = RiskVortexEngine(
        ticker,
        initial_investment=capital,
        timeframe_label=timeframe,
        benchmark_ticker=benchmark,
    )
    st.session_state["engine"] = engine
    st.session_state["benchmark"] = benchmark
  except Exception as e:
    st.error(f"Execution Error: {str(e)}")
    st.stop()

if "engine" in st.session_state and st.session_state["engine"] is not None:
  engine = st.session_state["engine"]
  benchmark = st.session_state["benchmark"]

  tt_info = engine.get_tickertape_summary()
  metrics = engine.calculate_risk_metrics()
  paths, final_vals, var_d, var_p, cvar_d, cvar_p, percentiles = (
      engine.run_monte_carlo()
  )
  stress_res = engine.run_stress_testing()
  factor_res = engine.calculate_factor_attribution()

  z_score, z_status = engine.calculate_altman_z_score(
      tt_info.get("Info_Dict", {})
  )

  if z_score > 2.99:
    z_zone = "Safe Zone"
    z_color = "#16A34A"
  elif 1.81 <= z_score <= 2.99:
    z_zone = "Grey Zone"
    z_color = "#D97706"
  else:
    z_zone = "Distress Zone"
    z_color = "#DC2626"

  curr_sym = "₹" if ticker.endswith(".NS") or ticker.endswith(".BO") else "$"

  tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
      "🏢 Overview",
      "📈 Chart",
      "📊 Risk",
      "🎲 Simulation",
      "📉 Drawdown",
      "⚠️ Crisis",
      "🏛️ Attribution",
  ])

  with tab1:
    st.markdown(
        f"### {tt_info['Company_Name']} ({ticker}) — "
        f"**{curr_sym}{engine.current_price:,.2f}**"
    )
    st.caption(
        f"Sector: {tt_info['Sector']} | Industry: {tt_info['Industry']} |"
        f" Country: {tt_info['Country']}"
    )

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market Cap", f"{curr_sym}{tt_info['Market_Cap']}")
    c2.metric("P/E Ratio", tt_info["PE_Ratio"])
    c3.metric("P/B Ratio", tt_info["PB_Ratio"])
    c4.metric("Div Yield", tt_info["Dividend_Yield"])

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Beta", tt_info["Beta"])
    c6.metric("Profit Margin", tt_info["Profit_Margins"])
    c7.metric("52W High", f"{curr_sym}{tt_info['52W_High']}")
    c8.metric("Analyst Upside", tt_info["Target_Upside"])

    st.markdown("---")
    st.write("**Business Summary:**")
    st.write(tt_info["Summary"])

  with tab2:
    dynamic_chart = engine.get_chart_data(chart_period, chart_type)
    st.plotly_chart(dynamic_chart, use_container_width=True)

  with tab3:
    col1, col2 = st.columns(2)
    with col1:
      st.markdown(
          f"""
            <div style='background: #FFFFFF; padding: 15px; border-radius: 10px; border: 1px solid #E5E7EB;'>
                <h4 style='color: #1E3A8A; margin-top:0;'>📊 Risk-Return Profile</h4>
                <p><b>Expected Return ({timeframe}):</b> <span style='color: #16A34A;'>{metrics['Period_Return']:.2f}%</span></p>
                <p><b>Expected Volatility:</b> {metrics['Period_Volatility']:.2f}%</p>
                <p><b>Sharpe Ratio:</b> <b style='color:#2563EB;'>{metrics['Sharpe_Ratio']:.2f}</b></p>
                <p><b>Sortino Ratio:</b> <b>{metrics['Sortino_Ratio']:.2f}</b></p>
                <p><b>Max Drawdown:</b> <span style='color:#DC2626;'>{metrics['Max_Drawdown']:.2f}%</span></p>
            </div>
            """,
          unsafe_allow_html=True,
      )
    with col2:
      st.markdown(
          f"""
            <div style='background: #FFFFFF; padding: 15px; border-radius: 10px; border: 1px solid #E5E7EB;'>
                <h4 style='color: #EA580C; margin-top:0;'>🎲 Tail-Risk Metrics</h4>
                <p><b>95% VaR:</b> <span style='color:#DC2626;'>-{var_p:.2f}% ({curr_sym}{var_d:,.2f})</span></p>
                <p><b>95% CVaR:</b> <span style='color:#991B1B;'>-{cvar_p:.2f}% ({curr_sym}{cvar_d:,.2f})</span></p>
                <p><b>Median Outcome:</b> {curr_sym}{percentiles['p50_median']:,.2f}</p>
                <p><b>95th Percentile Bull:</b> <span style='color:#16A34A;'>{curr_sym}{percentiles['p95_best']:,.2f}</span></p>
            </div>
            """,
          unsafe_allow_html=True,
      )

  with tab4:
    path_fig = go.Figure()
    step = max(1, paths.shape[1] // 100)
    for i in range(0, paths.shape[1], step):
      path_fig.add_trace(
          go.Scatter(
              y=paths[:, i],
              mode="lines",
              line=dict(width=0.6),
              opacity=0.25,
              showlegend=False,
          )
      )
    path_fig.update_layout(
        title=f"<b>Monte Carlo Paths ({timeframe})</b>",
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(path_fig, use_container_width=True)

    dist_fig = go.Figure()
    dist_fig.add_trace(
        go.Histogram(
            x=final_vals, nbinsx=50, marker_color="#6366F1", opacity=0.75
        )
    )
    dist_fig.add_vline(
        x=capital - var_d, line_dash="dash", line_color="orange"
    )
    dist_fig.update_layout(
        title="<b>Ending Value Distribution</b>",
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(dist_fig, use_container_width=True)

  with tab5:
    cum_returns = (1 + engine.close_prices.pct_change()).cumprod()
    drawdown = (cum_returns - cum_returns.cummax()) / cum_returns.cummax()
    dd_fig = go.Figure()
    dd_fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            fill="tozeroy",
            line=dict(color="#DC2626", width=1),
        )
    )
    dd_fig.update_layout(
        title="<b>Historical Underwater Profile</b>",
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(dd_fig, use_container_width=True)

  with tab6:
    stress_fig = go.Figure()
    scenarios_list = list(stress_res.keys())
    projected_vals = [v["Projected"] for v in stress_res.values()]
    stress_fig.add_trace(
        go.Bar(x=scenarios_list, y=projected_vals, marker_color="#B91C1C")
    )
    stress_fig.add_hline(y=capital, line_dash="dash", line_color="green")
    stress_fig.update_layout(
        title="<b>Macro Crisis Replay Shocks</b>",
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(stress_fig, use_container_width=True)

  with tab7:
    col1, col2 = st.columns(2)
    with col1:
      st.markdown(
          f"""
            <div style='background: #FFFFFF; padding: 15px; border-radius: 10px; border: 1px solid #E5E7EB;'>
                <h4 style='color: #0D9488; margin-top:0;'>🏛️ Factor Attribution</h4>
                <p><b>Market Beta (β):</b> <b>{factor_res['Beta']:.2f}</b></p>
                <p><b>CAPM Expected:</b> {factor_res['CAPM_Expected']:.2f}%</p>
                <p><b>Jensen's Alpha (α):</b> <span style='color: {"#16A34A" if factor_res["Jensens_Alpha"] >= 0 else "#DC2626"};'><b>{factor_res['Jensens_Alpha']:.2f}%</b></span></p>
                <p><b>Benchmark Return:</b> {factor_res['Benchmark_Return']:.2f}%</p>
            </div>
            """,
          unsafe_allow_html=True,
      )
    with col2:
      st.markdown(
          f"""
            <div style='background: #FFFFFF; padding: 15px; border-radius: 10px; border: 1px solid #E5E7EB;'>
                <h4 style='color: #7C3AED; margin-top:0;'>⚖️ Altman Z-Score</h4>
                <p><b>Z-Score:</b> <b style='color: {z_color}; font-size: 16px;'>{z_score:.2f}</b></p>
                <p><b>Status:</b> <span style='color: {z_color}; font-weight: bold;'>{z_zone}</span></p>
                <p><b>Model Info:</b> {z_status}</p>
            </div>
            """,
          unsafe_allow_html=True,
      )
