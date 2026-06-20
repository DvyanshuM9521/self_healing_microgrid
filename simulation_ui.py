import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import joblib
from datetime import datetime
import time

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================
st.set_page_config(
    page_title="Self-Healing Microgrid Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .status-healthy { color: #00ff00; font-weight: bold; }
    .status-warning { color: #ffaa00; font-weight: bold; }
    .status-critical { color: #ff0000; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_all():
    """Load all required data files"""
    try:
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
    except FileNotFoundError as e:
        st.error(f"Missing file: {str(e)}")
        return None, None, None, None, None, None

df, graph_data, X, y, scaler, q_table = load_all()

if df is None:
    st.stop()

G = graph_data["graph"]
positions = graph_data["positions"]

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    """Initialize or reset session state"""
    if "faults" not in st.session_state:
        st.session_state.faults = set(np.where(y == 1)[0])
    
    if "healed" not in st.session_state:
        st.session_state.healed = set()
    
    if "history" not in st.session_state:
        st.session_state.history = {
            "timestamp": [],
            "action": [],
            "node_id": [],
            "total_faults": [],
            "total_healed": [],
            "system_health": []
        }
    
    if "auto_heal_active" not in st.session_state:
        st.session_state.auto_heal_active = False
    
    if "simulation_speed" not in st.session_state:
        st.session_state.simulation_speed = 1.0

init_session_state()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def calculate_metrics():
    """Calculate comprehensive system metrics"""
    total_nodes = len(df)
    total_faults = len(st.session_state.faults)
    total_healed = len(st.session_state.healed)
    healthy_nodes = total_nodes - total_faults - total_healed
    
    system_health = (healthy_nodes + total_healed) / total_nodes
    avg_stress = df.loc[list(st.session_state.faults), "stress_index"].mean() if st.session_state.faults else 0
    
    # Detection metrics
    threshold = df["stress_index"].quantile(0.75)
    detected = set(df[df["stress_index"] > threshold].index)
    
    true_positives = len(detected & st.session_state.faults)
    false_positives = len(detected - st.session_state.faults)
    false_negatives = len(st.session_state.faults - detected)
    
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / max(1e-6, precision + recall)
    
    return {
        "total_nodes": total_nodes,
        "total_faults": total_faults,
        "total_healed": total_healed,
        "healthy_nodes": healthy_nodes,
        "system_health": system_health,
        "avg_stress": avg_stress,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "detected_faults": len(detected)
    }

def log_action(action, node_id=None):
    """Log actions to history"""
    metrics = calculate_metrics()
    st.session_state.history["timestamp"].append(datetime.now())
    st.session_state.history["action"].append(action)
    st.session_state.history["node_id"].append(node_id)
    st.session_state.history["total_faults"].append(metrics["total_faults"])
    st.session_state.history["total_healed"].append(metrics["total_healed"])
    st.session_state.history["system_health"].append(metrics["system_health"])

def inject_fault_smart():
    """Inject fault on high-stress node"""
    non_faults = list(set(range(len(df))) - st.session_state.faults)
    if non_faults:
        idx = df.loc[non_faults, "stress_index"].idxmax()
        st.session_state.faults.add(int(idx))
        log_action("FAULT_INJECTED", int(idx))
        return int(idx)
    return None

def auto_heal_algorithm():
    """Advanced healing algorithm"""
    healed_this_round = []
    
    if not st.session_state.faults:
        return healed_this_round
    
    fault_list = sorted(
        list(st.session_state.faults),
        key=lambda x: df.loc[x, "stress_index"],
        reverse=True
    )
    
    # Multi-criteria healing strategy
    stress_thr = df["stress_index"].quantile(0.7)
    load_thr = df["load_ratio"].quantile(0.7)
    imb_thr = df["imbalance"].quantile(0.7)
    
    scored_nodes = []
    for node in fault_list[:50]:  # Consider top 50
        stress = df.loc[node, "stress_index"]
        load = df.loc[node, "load_ratio"]
        imbalance = df.loc[node, "imbalance"]
        
        score = 0
        if stress > stress_thr: score += 3
        if load > load_thr: score += 2
        if imbalance > imb_thr: score += 1
        
        scored_nodes.append((node, score))
    
    # Heal top candidates
    scored_nodes.sort(key=lambda x: x[1], reverse=True)
    HEAL_COUNT = min(15, len(scored_nodes))
    
    for node, _ in scored_nodes[:HEAL_COUNT]:
        st.session_state.faults.remove(node)
        st.session_state.healed.add(node)
        healed_this_round.append(node)
        log_action("HEALED", node)
    
    return healed_this_round

def plot_network(sample_size=800):
    """Plot microgrid network with status colors"""
    nodes = list(G.nodes())[:sample_size]
    subG = G.subgraph(nodes)
    pos = {i: positions[i] for i in nodes}
    
    # Edges
    edge_x, edge_y = [], []
    for edge in subG.edges():
        x0, y0, _ = pos[edge[0]]
        x1, y1, _ = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="rgba(125,125,125,0.2)"),
        mode="lines",
        hoverinfo="none",
        showlegend=False
    )
    
    # Nodes with color coding
    node_x, node_y, colors, node_text, node_hover = [], [], [], [], []
    for node in subG.nodes():
        x, y_, _ = pos[node]
        node_x.append(x)
        node_y.append(y_)
        
        if node in st.session_state.healed:
            colors.append("#4169E1")  # Blue
            status = "HEALED"
        elif node in st.session_state.faults:
            colors.append("#FF0000")  # Red
            status = "FAULT"
        else:
            colors.append("#00AA00")  # Green
            status = "HEALTHY"
        
        stress = df.loc[node, "stress_index"]
        hover_text = f"<b>Node {node}</b><br>Status: {status}<br>Stress: {stress:.3f}"
        node_hover.append(hover_text)
        node_text.append(f"Node {node}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        marker=dict(
            size=7,
            color=colors,
            line=dict(width=2, color="white")
        ),
        text=node_hover,
        hoverinfo="text",
        showlegend=False
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="rgba(240,240,240,0.5)",
        height=600
    )
    
    return fig

def plot_metrics_over_time():
    """Plot simulation history"""
    if not st.session_state.history["timestamp"]:
        return None
    
    history_df = pd.DataFrame(st.session_state.history)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=history_df["timestamp"],
        y=history_df["total_faults"],
        mode="lines+markers",
        name="Active Faults",
        line=dict(color="#FF0000", width=2),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=history_df["timestamp"],
        y=history_df["total_healed"],
        mode="lines+markers",
        name="Healed Nodes",
        line=dict(color="#4169E1", width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="System Status Over Time",
        xaxis_title="Time",
        yaxis_title="Count",
        hovermode="x unified",
        height=400
    )
    
    return fig

def plot_health_distribution():
    """Plot stress distribution of faults"""
    if not st.session_state.faults:
        return None
    
    fault_stresses = df.loc[list(st.session_state.faults), "stress_index"].values
    
    fig = go.Figure(data=[
        go.Histogram(
            x=fault_stresses,
            nbinsx=20,
            name="Fault Stress Distribution",
            marker_color="#FF6B6B"
        )
    ])
    
    fig.update_layout(
        title="Stress Distribution of Current Faults",
        xaxis_title="Stress Index",
        yaxis_title="Count",
        height=400,
        showlegend=False
    )
    
    return fig

# =============================================================================
# SIDEBAR: SIMULATION CONTROLS
# =============================================================================
with st.sidebar:
    st.markdown("## ⚙️ SIMULATION CONTROLS")
    st.divider()
    
    # Simulation speed
    speed = st.slider("Simulation Speed", 0.5, 3.0, 1.0, step=0.1)
    st.session_state.simulation_speed = speed
    
    # Basic controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💥 Inject Fault", use_container_width=True):
            injected = inject_fault_smart()
            if injected is not None:
                st.success(f"Fault injected on Node {injected}")
            else:
                st.warning("All nodes are already faulty!")
    
    with col2:
        if st.button("🤖 Auto Heal", use_container_width=True):
            healed = auto_heal_algorithm()
            if healed:
                st.success(f"Healed {len(healed)} nodes")
            else:
                st.info("No faults to heal")
    
    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.faults = set(np.where(y == 1)[0])
            st.session_state.healed = set()
            st.session_state.history = {
                "timestamp": [],
                "action": [],
                "node_id": [],
                "total_faults": [],
                "total_healed": [],
                "system_health": []
            }
            st.success("System reset!")
    
    st.divider()
    
    # Advanced controls
    st.markdown("### 🎯 Advanced Options")
    
    num_faults = st.slider("Inject Multiple Faults", 1, 20, 1)
    if st.button("Batch Inject Faults", use_container_width=True):
        count = 0
        for _ in range(num_faults):
            if inject_fault_smart():
                count += 1
        st.success(f"Injected {count} faults")
    
    # Manual node selection for healing
    st.markdown("### 🔧 Manual Healing")
    fault_nodes = sorted(list(st.session_state.faults))
    if fault_nodes:
        selected_nodes = st.multiselect(
            "Select nodes to heal manually",
            fault_nodes,
            default=[]
        )
        if st.button("Heal Selected", use_container_width=True):
            for node in selected_nodes:
                st.session_state.faults.discard(node)
                st.session_state.healed.add(node)
                log_action("MANUAL_HEAL", node)
            st.success(f"Manually healed {len(selected_nodes)} nodes")
    
    st.divider()
    
    # System info
    st.markdown("### ℹ️ System Info")
    metrics = calculate_metrics()
    st.metric("Total Nodes", metrics["total_nodes"])
    st.metric("Active Faults", metrics["total_faults"])
    st.metric("Healed Nodes", metrics["total_healed"])

# =============================================================================
# MAIN CONTENT: DASHBOARD
# =============================================================================
st.markdown("# ⚡ Self-Healing Microgrid Simulator")
st.markdown("Advanced simulation and monitoring of microgrid fault detection and healing")

# Key metrics row
metrics = calculate_metrics()
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    health_color = "🟢" if metrics["system_health"] > 0.8 else "🟡" if metrics["system_health"] > 0.5 else "🔴"
    st.metric(
        "System Health",
        f"{metrics['system_health']*100:.1f}%",
        help="Percentage of nodes operating normally or healed"
    )

with col2:
    st.metric(
        "Active Faults",
        metrics["total_faults"],
        help="Number of nodes currently faulty"
    )

with col3:
    st.metric(
        "Healed Nodes",
        metrics["total_healed"],
        help="Number of nodes that have been healed"
    )

with col4:
    st.metric(
        "F1-Score",
        f"{metrics['f1_score']:.3f}",
        help="Harmonic mean of precision and recall"
    )

with col5:
    st.metric(
        "Detection Accuracy",
        f"{metrics['precision']*100:.1f}%",
        help="Precision of fault detection"
    )

st.divider()

# Network visualization and analysis
tab1, tab2, tab3, tab4 = st.tabs(["🌐 Network", "📊 Analytics", "📝 History", "🔍 Node Details"])

# Tab 1: Network Visualization
with tab1:
    col_net1, col_net2 = st.columns([3, 1])
    
    with col_net1:
        st.subheader("Microgrid Topology")
        fig_network = plot_network(sample_size=800)
        st.plotly_chart(fig_network, use_container_width=True)
    
    with col_net2:
        st.subheader("Legend")
        st.info("""
        🟢 **Green**: Healthy nodes
        
        🔴 **Red**: Faulty nodes
        
        🔵 **Blue**: Healed nodes
        """)
        
        # Fault statistics
        st.subheader("Fault Statistics")
        fault_list = sorted(
            list(st.session_state.faults),
            key=lambda x: df.loc[x, "stress_index"],
            reverse=True
        )[:10]
        
        if fault_list:
            st.write("**Top 10 Faults by Stress:**")
            for i, node in enumerate(fault_list, 1):
                stress = df.loc[node, "stress_index"]
                st.write(f"{i}. Node {node}: {stress:.3f}")
        else:
            st.success("No faults detected!")

# Tab 2: Analytics
with tab2:
    # Time series
    fig_history = plot_metrics_over_time()
    if fig_history:
        st.plotly_chart(fig_history, use_container_width=True)
    else:
        st.info("No history yet. Start the simulation to see trends.")
    
    # Distribution
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        fig_stress = plot_health_distribution()
        if fig_stress:
            st.plotly_chart(fig_stress, use_container_width=True)
    
    with col_a2:
        # Detection metrics
        st.subheader("Detection Performance")
        c_p1, c_p2 = st.columns(2)
        c_p1.metric("Precision", f"{metrics['precision']:.3f}")
        c_p2.metric("Recall", f"{metrics['recall']:.3f}")
        
        c_p3, c_p4 = st.columns(2)
        c_p3.metric("True Positives", metrics["detected_faults"])
        c_p4.metric("F1-Score", f"{metrics['f1_score']:.3f}")

# Tab 3: History Log
with tab3:
    st.subheader("Action History")
    
    if st.session_state.history["timestamp"]:
        history_df = pd.DataFrame(st.session_state.history)
        history_df["timestamp"] = history_df["timestamp"].dt.strftime("%H:%M:%S")
        
        st.dataframe(
            history_df[["timestamp", "action", "node_id", "total_faults", "system_health"]],
            use_container_width=True,
            hide_index=True
        )
        
        # Export history
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name=f"microgrid_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No actions recorded yet.")

# Tab 4: Node Details
with tab4:
    st.subheader("Node Analysis")
    
    selected_node = st.selectbox(
        "Select a node to analyze",
        range(len(df)),
        format_func=lambda x: f"Node {x}"
    )
    
    if selected_node is not None:
        node_data = df.loc[selected_node]
        
        col_nd1, col_nd2, col_nd3, col_nd4 = st.columns(4)
        
        with col_nd1:
            status = "🔴 FAULT"
            if selected_node in st.session_state.healed:
                status = "🔵 HEALED"
            elif selected_node not in st.session_state.faults:
                status = "🟢 HEALTHY"
            
            st.metric("Status", status)
        
        with col_nd2:
            st.metric("Stress Index", f"{node_data['stress_index']:.4f}")
        
        with col_nd3:
            st.metric("Load Ratio", f"{node_data['load_ratio']:.4f}")
        
        with col_nd4:
            st.metric("Imbalance", f"{node_data['imbalance']:.4f}")
        
        # Actions
        col_act1, col_act2 = st.columns(2)
        
        with col_act1:
            if selected_node not in st.session_state.faults and selected_node not in st.session_state.healed:
                if st.button("Inject Fault on This Node"):
                    st.session_state.faults.add(selected_node)
                    log_action("MANUAL_FAULT_INJECT", selected_node)
                    st.success("Fault injected!")
        
        with col_act2:
            if selected_node in st.session_state.faults:
                if st.button("Heal This Node"):
                    st.session_state.faults.remove(selected_node)
                    st.session_state.healed.add(selected_node)
                    log_action("MANUAL_HEAL", selected_node)
                    st.success("Node healed!")

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
📊 Self-Healing Microgrid Simulator v1.0 | Last Updated: 2026 | 
Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
