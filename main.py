import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import networkx as nx
import joblib

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(layout="wide")
st.title("⚡ AI Self-Healing Microgrid (REAL SIMULATION)")

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_all():
    df = pd.read_csv("processed_microgrid.csv")
    graph_data = pickle.load(open("microgrid_graph.pkl", "rb"))
    X = np.load("X.npy")
    y = np.load("y.npy")
    scaler = joblib.load("scaler.pkl")

    try:
        q_table = pickle.load(open("q_table.pkl", "rb"))
    except:
        q_table = {}

    return df, graph_data, X, y, scaler, q_table

df, graph_data, X, y, scaler, q_table = load_all()
G = graph_data["graph"]
positions = graph_data["positions"]

# ----------------------------
# SESSION STATE
# ----------------------------
if "faults" not in st.session_state:
    st.session_state.faults = set(np.where(y == 1)[0])

if "run_rl" not in st.session_state:
    st.session_state.run_rl = False

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.header("⚙️ Controls")

inject_fault = st.sidebar.button("💥 Inject Fault")
auto_heal = st.sidebar.button("🤖 Auto Heal (RL)")
reset = st.sidebar.button("🔄 Reset System")

# ----------------------------
# RESET
# ----------------------------
if reset:
    st.session_state.faults = set(np.where(y == 1)[0])

# ----------------------------
# FAULT INJECTION
# ----------------------------
if inject_fault:
    non_fault_nodes = list(set(range(len(df))) - st.session_state.faults)
    if non_fault_nodes:
        idx = df.loc[non_fault_nodes]["stress_index"].idxmax()
        st.session_state.faults.add(int(idx))

# ----------------------------
# GRAPH (SAMPLED)
# ----------------------------
SAMPLE_SIZE = 800
sample_nodes = list(G.nodes())[:SAMPLE_SIZE]
subG = G.subgraph(sample_nodes)
sub_positions = {i: positions[i] for i in sample_nodes}
sample_faults = set(sample_nodes) & st.session_state.faults

def plot_graph(G, positions, faults):
    edge_x, edge_y = [], []

    for e in G.edges():
        x0, y0, _ = positions[e[0]]
        x1, y1, _ = positions[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines"
    )

    node_x, node_y, colors = [], [], []

    for i in G.nodes():
        x, y_, _ = positions[i]
        node_x.append(x)
        node_y.append(y_)
        colors.append("red" if i in faults else "green")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(size=6, color=colors),
        text=[f"Node {i}" for i in G.nodes()],
        hoverinfo="text"
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False)
    return fig

# ----------------------------
# DETECTION
# ----------------------------
threshold_value = df["stress_index"].quantile(0.75)
detected_faults = set(df[df["stress_index"] > threshold_value].index)

# ----------------------------
# SMART HEALING (FINAL)
# ----------------------------
def rl_step():
    healed = []
    logs = []

    TOP_K = 50

    fault_list = sorted(
        list(st.session_state.faults),
        key=lambda x: df.loc[x, "stress_index"],
        reverse=True
    )

    stress_thr = df["stress_index"].quantile(0.7)
    load_thr = df["load_ratio"].quantile(0.7)
    imb_thr = df["imbalance"].quantile(0.7)

    for node in fault_list[:TOP_K]:
        stress = df.loc[node, "stress_index"]
        load = df.loc[node, "load_ratio"]
        imbalance = df.loc[node, "imbalance"]

        score = 0
        if stress > stress_thr:
            score += 2
        if load > load_thr:
            score += 1
        if imbalance > imb_thr:
            score += 1

        # 🔥 rank-based healing (FINAL FIX)
        HEAL_K = 15   # only heal top 15 out of 50

        if len(healed) < HEAL_K:
            healed.append(node)
            logs.append(f"✅ Healed node {node}")
        elif len(healed) < 30:
            logs.append(f"⚠️ Node {node} monitored")
        else:
            logs.append(f"❌ Node {node} skipped")

    for h in healed:
        st.session_state.faults.remove(h)

    return healed, logs

# ----------------------------
# BUTTON HANDLING
# ----------------------------
if auto_heal:
    st.session_state.run_rl = True

healed_nodes = []
rl_logs = []

if st.session_state.run_rl:
    with st.spinner("🤖 AI is healing the grid..."):
        healed_nodes, rl_logs = rl_step()
    st.session_state.run_rl = False

# ----------------------------
# LAYOUT
# ----------------------------
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🌐 Microgrid Network")
    st.plotly_chart(plot_graph(subG, sub_positions, sample_faults), use_container_width=True)

with col2:
    st.subheader("📊 Metrics")

    st.metric("Total Nodes", len(df))
    st.metric("Actual Faults", len(st.session_state.faults))
    st.metric("Detected Faults", len(detected_faults))

    st.metric("Avg Stress (Abs)", round(df["stress_index"].abs().mean(), 3))
    st.metric("Avg Load Ratio (Abs)", round(df["load_ratio"].abs().mean(), 3))

# ----------------------------
# DETECTION METRICS
# ----------------------------
st.subheader("🧠 Fault Detection")

precision = len(detected_faults & st.session_state.faults) / max(1, len(detected_faults))
recall = len(detected_faults & st.session_state.faults) / max(1, len(st.session_state.faults))

colA, colB = st.columns(2)
colA.metric("Precision", round(precision, 3))
colB.metric("Recall", round(recall, 3))

# ----------------------------
# CLEAN OUTPUT SECTION
# ----------------------------
st.subheader("🧾 AI Decision Summary")

total_processed = len(rl_logs)
total_healed = len(healed_nodes)

col1, col2, col3 = st.columns(3)
col1.metric("Nodes Processed", total_processed)
col2.metric("Nodes Healed", total_healed)

efficiency = total_healed / max(1, total_processed)
col3.metric("Efficiency", round(efficiency, 3))

# Top healed nodes
st.subheader("🔥 Top Critical Nodes Healed")
st.write([int(i) for i in healed_nodes[:10]])

# Limited logs
st.subheader("📜 Key Decisions (Top 10)")
for log in rl_logs[:10]:
    st.write(log)

st.info(f"Showing 10 of {total_processed} decisions")

# Progress bar
progress = min(1.0, total_healed / 200)
st.progress(progress)