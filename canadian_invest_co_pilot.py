import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
from scipy.optimize import minimize
import requests

# ====================== Sidebar ======================
st.sidebar.title("Co-Pilot Settings")
api_provider = st.sidebar.selectbox("AI Provider", ["Grok 3 (xAI)", "Claude 3.5 (Anthropic)", "Gemini 1.5 (Google)"])
api_key = st.sidebar.text_input("Your Free API Key", type="password", help="Get from grok.com / claude.ai / ai.google.dev")
risk_tolerance = st.sidebar.slider("Risk Level", 1, 10, 5, help="1=Conservative, 10=Aggressive")
investment_amount = st.sidebar.number_input("Investment Amount (CAD)", 1000, 1000000, 50000)

# ====================== Data & Optimizer ======================
@st.cache_data(ttl=300)
def fetch_data(tickers, period="1mo"):
    data = yf.download(tickers, period=period)["Adj Close"]
    returns = data.pct_change().dropna()
    return data, returns

def optimize_portfolio(returns, risk_free_rate=0.02):
    n_assets = returns.shape[1]
    def neg_sharpe(weights):
        port_return = np.sum(returns.mean() * weights) * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        return -(port_return - risk_free_rate) / port_vol
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    result = minimize(neg_sharpe, np.array([1/n_assets]*n_assets), method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# ====================== AI Query ======================
def query_ai(prompt, provider, key):
    if not key:
        return "Add your API key in sidebar for AI responses!"
    headers = {"Content-Type": "application/json"}
    if provider == "Grok 3 (xAI)":
        url = "https://api.x.ai/v1/chat/completions"
        payload = {"model": "grok-3", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1500}
        headers["Authorization"] = f"Bearer {key}"
    elif provider == "Claude 3.5 (Anthropic)":
        url = "https://api.anthropic.com/v1/messages"
        payload = {"model": "claude-3-5-sonnet-20241022", "max_tokens": 1500, "messages": [{"role": "user", "content": prompt}]}
        headers["x-api-key"] = key
        headers["anthropic-version"] = "2023-06-01"
    elif provider == "Gemini 1.5 (Google)":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        if provider == "Gemini 1.5 (Google)":
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"API error: {str(e)}"

# ====================== Main App ======================
st.title("Canadian Investing Co-Pilot")
st.write(f"Today: {datetime.now().strftime('%Y-%m-%d')} | BoC Rate ~2.25% → expected 2.0% in 2026")

col1, col2, col3 = st.columns(3)
if col1.button("Lazy $50k TFSA Portfolio"):
    tickers = ["XEQT.TO", "ZAG.TO", "HXS.TO"]
    data, _ = fetch_data(tickers)
    st.success("70% XEQT.TO • 20% ZAG.TO • 10% HXS.TO → Simple & TFSA-friendly")
    st.line_chart(data)
if col2.button("Recession-Proof"):
    tickers = ["ZAG.TO", "ZGD.TO", "GC=F"]
    data, _ = fetch_data(tickers)
    st.info("Defensive mix — hedges CAD weakness")
    st.line_chart(data)
if col3.button("Dividend Focus"):
    tickers = ["CDZ.TO", "XDIV.TO", "ZWC.TO"]
    data, _ = fetch_data(tickers)
    st.balloons()
    st.bar_chart(data.iloc[-1])

# ====================== Chat ======================
st.subheader("Ask FinCo Anything")
user_question = st.text_input("Your question (e.g., retirement, house down-payment, etc.):", key="q")

if user_question:
      # ——— Safe ticker selection (never empty) ———
    common_tickers = ["XEQT.TO","VEQT.TO","VCN.TO","ZAG.TO","GC=F","CADUSD=X","ZGD.TO","HXS.TO","VGRO.TO","VBAL.TO"]
    mentioned = [t for t in common_tickers if t.split(".")[0] in user_question.upper().replace(" ","")]
    relevant_tickers = mentioned if mentioned else common_tickers[:4]   # ← always at least 4 tickers
    data, returns = fetch_data(relevant_tickers)

    if len(data.columns) > 1:
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Price History", "Daily Returns %"))
        for col in data.columns:
            fig.add_trace(go.Scatter(y=data[col], name=col), row=1, col=1)
            fig.add_trace(go.Bar(y=returns[col]*100, name=col, opacity=0.6), row=2, col=1)
        fig.update_layout(height=600, title=f"Live Data: {', '.join(data.columns)}")
        st.plotly_chart(fig, use_container_width=True)

    prompt = f"""
You are FinCo, the ultimate Canadian fiduciary co-pilot — always 100% in my best interest.

User Profile:
• Budget: ${investment_amount:,} CAD
• Goal/Question: {user_question}
• Risk Tolerance: {risk_tolerance}/10
• Prioritize: TFSA → RRSP → FHSA → non-reg

Live Data ({datetime.now().strftime('%B %d, %Y')}):
• Latest prices: {dict(data.iloc[-1].round(2))}
• 1-month avg return: {returns.mean().mean()*30:.1%}

Current Context (Dec 2025):
• BoC rate 2.25% → cutting to 2.0%
• CAD/USD ≈ 0.72 (weak loonie)
• Key themes: EV/hydrogen push, slowing immigration, tariff risk

Give a clear, low-cost, tax-smart plan in bullets + Markdown table.
5–7 specific CAD ETFs/GICs only. Realistic returns & risks. Rebalancing triggers. Next touch-base.
No jargon. No high-fee funds.
"""

    with st.spinner("FinCo is building your plan…"):
        ai_response = query_ai(prompt, api_provider, api_key)

    st.markdown("### FinCo's Personalized Advice")
    st.markdown(ai_response)

    if any(w in user_question.lower() for w in ["portfolio", "optimize", "weights", "allocation"]) or st.button("Show Optimization & Frontier", type="primary"):
        if len(returns.columns) >= 2:
            weights = optimize_portfolio(returns)
            port_return = np.sum(returns.mean() * weights) * 252
            port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

            st.markdown("### Optimized Portfolio (Max Sharpe)")
            weight_df = pd.DataFrame({"Ticker": returns.columns, "Allocation %": (weights*100).round(1)})
            weight_df = weight_df.sort_values("Allocation %", ascending=False)
            st.dataframe(weight_df, use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            with c1: st.metric("Expected Annual Return", f"{port_return:.1%}")
            with c2: st.metric("Annual Risk", f"{port_vol:.1%}")

            target_returns = np.linspace(returns.mean().sum()*252*0.5, returns.mean().sum()*252*1.5, 30)
            vols = []
            for tr in target_returns:
                cons = (
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: np.sum(returns.mean() * x) * 252 - tr}
                )
                res = minimize(lambda w: np.dot(w.T, np.dot(returns.cov()*252, w)),
                               np.array([1/len(returns.columns)]*len(returns.columns)),
                               method='SLSQP', bounds=tuple((0,1) for _ in returns.columns), constraints=cons)
                if res.success:
                    vols.append(np.sqrt(res.fun))
                else:
                    vols.append(np.nan)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=vols, y=target_returns, mode='lines', name='Efficient Frontier', line=dict(color='#00CED1')))
            fig.add_trace(go.Scatter(x=[port_vol], y=[port_return], mode='markers', marker=dict(size=16, color='red'), name='Your Portfolio'))
            fig.update_layout(title="Efficient Frontier", height=500, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("""
### Pro Tips
- TFSA first → tax-free forever
- VGRO/VBAL = perfect all-in-one ETFs
- Hedge weak CAD with HXS.TO or unhedged global
- Rebalance yearly or at 10% drift
""")
