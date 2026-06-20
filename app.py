import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import joblib

st.set_page_config(layout="wide")
st.title("⚡ AI Self-Healing Microgrid (Advanced UI)")

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
    return df, graph_data, X, y, scaler

df, graph_data, X, y, scaler = load_all()
G = graph_data["graph"]
positions = graph_data["positions"]

# ----------------------------
# STATE
# ----------------------------
if "faults" not in st.session_state:
    st.session_state.faults = set(np.where(y == 1)[0])

if "healed" not in st.session_state:
    st.session_state.healed = set()

# ----------------------------
# CONTROLS
# ----------------------------
st.sidebar.header("⚙️ Controls")

if st.sidebar.button("💥 Inject Fault"):
    non_faults = list(set(range(len(df))) - st.session_state.faults)
    if non_faults:
        idx = df.loc[non_faults]["stress_index"].idxmax()
        st.session_state.faults.add(int(idx))

if st.sidebar.button("🤖 Auto Heal"):
    fault_list = sorted(
        list(st.session_state.faults),
        key=lambda x: df.loc[x, "stress_index"],
        reverse=True
    )

    HEAL_K = 15
    healed_now = set(fault_list[:HEAL_K])

    st.session_state.healed.update(healed_now)
    st.session_state.faults -= healed_now

if st.sidebar.button("🔄 Reset"):
    st.session_state.faults = set(np.where(y == 1)[0])
    st.session_state.healed = set()

# ----------------------------
# GRAPH
# ----------------------------
SAMPLE_SIZE = 800
nodes = list(G.nodes())[:SAMPLE_SIZE]
subG = G.subgraph(nodes)
pos = {i: positions[i] for i in nodes}

def plot_graph():
    edge_x, edge_y = [], []

    for e in subG.edges():
        x0, y0, _ = pos[e[0]]
        x1, y1, _ = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="#aaa"),
        mode="lines"
    )

    node_x, node_y, colors = [], [], []

    for i in subG.nodes():
        x, y_, _ = pos[i]
        node_x.append(x)
        node_y.append(y_)

        if i in st.session_state.healed:
            colors.append("blue")
        elif i in st.session_state.faults:
            colors.append("red")
        else:
            colors.append("green")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        marker=dict(size=6, color=colors),
        text=[f"Node {i}" for i in subG.nodes()]
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False)
    return fig

# ----------------------------
# LAYOUT
# ----------------------------
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🌐 Microgrid Visualization")
    st.plotly_chart(plot_graph(), use_container_width=True)

with col2:
    st.subheader("📊 System Metrics")

    total = len(df)
    faults = len(st.session_state.faults)
    healed = len(st.session_state.healed)

    st.metric("Total Nodes", total)
    st.metric("Active Faults", faults)
    st.metric("Healed Nodes", healed)

    # Health score
    health = 1 - (faults / total)
    st.metric("System Health", round(health, 3))

    st.progress(health)

# ----------------------------
# NODE INSIGHT PANEL
# ----------------------------
st.subheader("🧠 Node Insights")

node_id = st.selectbox("Select Node", list(range(len(df))))

stress = df.loc[node_id, "stress_index"]
load = df.loc[node_id, "load_ratio"]
imbalance = df.loc[node_id, "imbalance"]

if node_id in st.session_state.healed:
    status = "Healed 🔵"
elif node_id in st.session_state.faults:
    status = "Fault 🔴"
else:
    status = "Normal 🟢"

colA, colB, colC, colD = st.columns(4)
colA.metric("Stress", round(stress, 3))
colB.metric("Load", round(load, 3))
colC.metric("Imbalance", round(imbalance, 3))
colD.metric("Status", status)

# ----------------------------
# DETECTION SUMMARY
# ----------------------------
st.subheader("🧠 Detection Summary")

threshold = df["stress_index"].quantile(0.75)
detected = set(df[df["stress_index"] > threshold].index)

precision = len(detected & st.session_state.faults) / max(1, len(detected))
recall = len(detected & st.session_state.faults) / max(1, len(st.session_state.faults))

c1, c2 = st.columns(2)
c1.metric("Precision", round(precision, 3))
c2.metric("Recall", round(recall, 3))