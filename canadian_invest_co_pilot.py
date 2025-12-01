{\rtf1\ansi\ansicpg1252\cocoartf2513
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 AppleColorEmoji;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import yfinance as yf\
import pandas as pd\
import plotly.graph_objects as go\
from plotly.subplots import make_subplots\
import plotly.express as px\
import numpy as np\
from datetime import datetime\
from scipy.optimize import minimize\
import requests  # For AI API calls\
import json\
\
# Sidebar for config\
st.sidebar.title("
\f1 \uc0\u55356 \u56808 \u55356 \u56806 
\f0  Co-Pilot Settings")\
api_provider = st.sidebar.selectbox("AI Provider", ["Grok 3 (xAI)", "Claude 3.5 (Anthropic)", "Gemini 1.5 (Google)"])\
api_key = st.sidebar.text_input("Your Free API Key", type="password", help="Get from grok.com / claude.ai / ai.google.dev")\
risk_tolerance = st.sidebar.slider("Risk Level", 1, 10, 5, help="1=Conservative (bonds heavy), 10=Aggressive (stocks heavy)")\
investment_amount = st.sidebar.number_input("Investment Amount (CAD)", 1000, 1000000, 50000)\
\
# Function to get real-time data\
@st.cache_data(ttl=300)  # Refresh every 5 mins\
def fetch_data(tickers, period="1mo"):\
    data = yf.download(tickers, period=period)["Adj Close"]\
    returns = data.pct_change().dropna()\
    return data, returns\
\
# Simple portfolio optimizer (max Sharpe ratio)\
def optimize_portfolio(returns, risk_free_rate=0.02):  # 2% risk-free (BoC rate approx)\
    n_assets = returns.shape[1]\
    def neg_sharpe(weights):\
        port_return = np.sum(returns.mean() * weights) * 252\
        port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))\
        return -(port_return - risk_free_rate) / port_vol\
    constraints = (\{'type': 'eq', 'fun': lambda x: np.sum(x) - 1\})\
    bounds = tuple((0, 1) for _ in range(n_assets))\
    result = minimize(neg_sharpe, np.array([1/n_assets]*n_assets), method='SLSQP', bounds=bounds, constraints=constraints)\
    return result.x\
\
# AI Query Function (pick your provider)\
def query_ai(prompt, provider, key):\
    if not key:\
        return "Add your API key in sidebar for AI responses!"\
    headers = \{"Content-Type": "application/json"\}\
    if provider == "Grok 3 (xAI)":\
        url = "https://api.x.ai/v1/chat/completions"\
        payload = \{"model": "grok-3", "messages": [\{"role": "user", "content": prompt\}], "max_tokens": 1000\}\
        headers["Authorization"] = f"Bearer \{key\}"\
    elif provider == "Claude 3.5 (Anthropic)":\
        url = "https://api.anthropic.com/v1/messages"\
        payload = \{"model": "claude-3-5-sonnet-20241022", "max_tokens": 1000, "messages": [\{"role": "user", "content": prompt\}]\}\
        headers["x-api-key"] = key\
        headers["anthropic-version"] = "2023-06-01"\
    elif provider == "Gemini 1.5 (Google)":\
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"\
        payload = \{"contents": [\{"parts": [\{"text": prompt\}]\}]\}\
        url += f"?key=\{key\}"\
        headers = \{\}\
    try:\
        response = requests.post(url, json=payload, headers=headers)\
        if provider == "Gemini 1.5 (Google)":\
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]\
        else:\
            return response.json()["choices"][0]["message"]["content"]\
    except:\
        return "API call failed \'97 check key or try another provider."\
\
# Main App\
st.title("
\f1 \uc0\u55356 \u56808 \u55356 \u56806 
\f0  Canadian Investing Co-Pilot")\
st.write(f"Today's Date: \{datetime.now().strftime('%Y-%m-%d')\} | BoC Rate: ~2.0% (check latest)")\
\
# Quick buttons for strategies\
col1, col2, col3 = st.columns(3)\
if col1.button("
\f1 \uc0\u55357 \u56508 
\f0  Lazy $50k TFSA Portfolio"):\
    tickers = ["XEQT.TO", "ZAG.TO", "HXS.TO"]  # Global equity, bonds, US stocks\
    weights = [0.7, 0.2, 0.1]\
    st.success("**Recommendation**: 70% XEQT.TO (global ETFs, low fee), 20% ZAG.TO (CAD bonds for stability), 10% HXS.TO (hedged US exposure). TFSA-friendly (no taxes on growth). Expected return: 6-8% annual, moderate risk.")\
    data, returns = fetch_data(tickers)\
    fig = px.line(data, title="1-Month Performance")\
    st.plotly_chart(fig, use_container_width=True)\
    port_value = investment_amount * (1 + returns.mean().sum() * 30)  # Rough 1-mo projection\
    st.metric("Projected Value (1 mo)", f"$\{port_value:,.0f\}")\
\
if col2.button("
\f1 \uc0\u55357 \u57057 \u65039 
\f0  Recession-Proof (Bonds + Gold)"):\
    tickers = ["ZAG.TO", "ZGD.TO", "GC=F"]  # Bonds, gold miners, gold futures\
    st.info("**Strategy**: Heavy on defensives. Hedge CAD weakness with gold. RRSP eligible for US exposure. Watch yield curve (currently normal).")\
    data, returns = fetch_data(tickers)\
    fig = go.Figure()\
    for col in data.columns:\
        fig.add_trace(go.Scatter(y=data[col], name=col))\
    fig.update_layout(title="Defensive Assets Trend")\
    st.plotly_chart(fig)\
\
if col3.button("
\f1 \uc0\u55357 \u56520 
\f0  Dividend Focus (High Yield CAD)"):\
    tickers = ["CDZ.TO", "XDIV.TO", "ZWC.TO"]  # Dividend ETFs\
    st.balloons()  # Fun!\
    st.write("**Picks**: CDZ.TO (top dividends, ~4.5% yield), XDIV (active picks), ZWC (covered calls for extra income). Great for non-reg accounts.")\
    data, returns = fetch_data(tickers)\
    fig = px.bar(data.iloc[-1], title="Current Yields Approx (check prospectus)")\
    st.plotly_chart(fig)\
\
# Chat Interface\
st.subheader("Ask Anything (e.g., 'Compare VEQT vs XEQT for FHSA' or 'CAD/USD hedge strategy')")\
user_question = st.text_input("Your Question:", key="input")\
\
if user_question:\
    # Auto-detect tickers in question (simple regex-like)\
    common_tickers = ["XEQT.TO", "VEQT.TO", "VCN.TO", "ZAG.TO", "GC=F", "CL=F", "CADUSD=X", "ZGD.TO", "HXS.TO"]\
    relevant_tickers = [t for t in common_tickers if any(word in user_question.upper() for word in t.split())] or common_tickers[:3]\
    \
    data, returns = fetch_data(relevant_tickers)\
    \
    # Show quick chart\
    if len(data.columns) > 1:\
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Price History", "Daily Returns"))\
        for col in data.columns:\
            fig.add_trace(go.Scatter(y=data[col], name=col, line=dict(width=2)), row=1, col=1)\
            fig.add_trace(go.Bar(y=returns[col]*100, name=col, opacity=0.6), row=2, col=1)\
        fig.update_layout(height=600, title=f"Data for: \{', '.join(relevant_tickers)\}")\
        st.plotly_chart(fig)\
    \
    # Build prompt\
    prompt = f"""\
    You are a CFA-certified Canadian investment strategist. Focus on bonds, ETFs, currencies, commodities, stocks.\
    User: \{user_question\}\
    Amount: $\{investment_amount:,\} CAD | Risk: \{risk_tolerance\}/10 | Accounts: Assume TFSA/RRSP unless specified.\
    Latest Data (\{datetime.now().strftime('%Y-%m-%d')\}):\
    Prices: \{data.iloc[-1].to_dict()\}\
    1-Mo Returns: \{returns.mean()*30:.2%\} avg.\
    \
    Respond: Practical advice, 3-5 bullet steps, risks/taxes (e.g., FHSA for first home). No jargon overload.\
    """\
    \
    # Get AI response\
    with st.spinner("Strategizing with AI..."):\
        ai_response = query_ai(prompt, api_provider, api_key)\
    st.markdown(f"**AI Advice:** \{ai_response\}")\
    \
    # Bonus: Optimize if relevant\
    if "portfolio" in user_question.lower() or st.button("Optimize This Portfolio"):\
        if len(returns.columns) >= 2:\
            weights = optimize_portfolio(returns)\
            port_return = np.sum(returns.mean() * weights) * 252\
            port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))\
            st.write("**Optimized Weights:**", dict(zip(returns.columns, weights.round(3))))\
            st.metric("Expected Annual Return", f"\{port_return:.1%\}", delta=f"\{port_return - 0.05:.1%\}")\
            st.metric("Annual Volatility", f"\{port_vol:.1%\}")\
            \
            # Efficient frontier plot\
            target_returns = np.linspace(returns.mean().sum()*252*0.5, returns.mean().sum()*252*1.5, 20)\
            vols = []\
            for tr in target_returns:\
                cons = (\{'type': 'eq', 'fun': lambda x: np.sum(x) - 1\},\
                        \{'type': 'eq', 'fun': lambda x: np.sum(returns.mean() * x) * 252 - tr\})\
                res = minimize(lambda w: np.dot(w, np.dot(returns.cov() * 252, w)), np.array([1/len(returns.columns)]*len(returns.columns)),\
                               method='SLSQP', bounds=tuple((0,1) for _ in returns.columns), constraints=cons)\
                vols.append(np.sqrt(res.fun))\
            fig_ef = go.Figure()\
            fig_ef.add_trace(go.Scatter(x=vols, y=target_returns, mode='lines', name='Efficient Frontier'))\
            fig_ef.add_trace(go.Scatter(x=[port_vol], y=[port_return], mode='markers', name='Your Optimal', marker=dict(size=10, color='red')))\
            fig_ef.update_layout(title="Portfolio Efficient Frontier", xaxis_title="Risk (Vol %)", yaxis_title="Return %")\
            st.plotly_chart(fig_ef)\
\
# Footer tips\
st.sidebar.markdown("""\
### Pro Tips\
- **Bonds**: ZAG.TO for short-term, XBB.TO for broad.\
- **ETFs**: VGRO for growth, VBAL for balanced (Vanguard low fees).\
- **Currency**: Watch CAD/USD=X; hedge with HXS.TO.\
- **Commodities**: GC=F (gold), CL=F (oil) via futures ETFs.\
- **Taxes**: TFSA for flexibility, RRSP for deduction. Use Morningstar.ca for yields.\
- **Alerts**: Add email via code (e.g., if yield curve inverts: 2yr - 10yr < 0).\
- Update data: Hit R in browser to refresh.\
""")\
\
if st.sidebar.button("Export Portfolio to CSV"):\
    data.to_csv("portfolio_data.csv")\
    st.sidebar.success("Downloaded! Check your folder.")}
