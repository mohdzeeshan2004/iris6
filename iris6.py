import streamlit as st
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

# ✅ MUST be first Streamlit command
st.set_page_config(
    page_title="IRIS Dataset EDA",
    page_icon="🌸",
    layout="wide"
)

# -------------------- DATA LOADING --------------------
@st.cache_data(show_spinner=False)
def load_data():
    return sn.load_dataset("iris")

data = load_data()

# -------------------- UI --------------------
st.title("🌸 IRIS Dataset – Exploratory Data Analysis")
st.caption("Optimized for Streamlit Cloud deployment")

st.sidebar.header("EDA Options")
eda_option = st.sidebar.selectbox(
    "Select Analysis Type",
    (
        "Dataset Overview",
        "Statistical Summary",
        "Distribution Plot",
        "Joint Plot",
        "Pair Plot",
        "Boxen Plot",
        "Strip Plot",
        "Swarm Plot"
    )
)

# -------------------- DATASET OVERVIEW --------------------
if eda_option == "Dataset Overview":
    st.subheader("📄 Dataset Preview")
    st.dataframe(data, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", data.shape[0])
    c2.metric("Columns", data.shape[1])
    c3.metric("Missing Values", data.isnull().sum().sum())

    st.subheader("Column Data Types")
    st.write(data.dtypes)

# -------------------- STATISTICS --------------------
elif eda_option == "Statistical Summary":
    st.subheader("📊 Descriptive Statistics")
    st.dataframe(data.describe(), use_container_width=True)

# -------------------- DISTRIBUTION --------------------
elif eda_option == "Distribution Plot":
    st.subheader("📈 Distribution Plot")

    column = st.selectbox(
        "Select Feature",
        data.select_dtypes("number").columns
    )

    fig, ax = plt.subplots()
    sn.histplot(data[column], kde=True, ax=ax)
    ax.set_title(f"Distribution of {column}")
    st.pyplot(fig, clear_figure=True)

# -------------------- JOINT PLOT --------------------
elif eda_option == "Joint Plot":
    st.subheader("🔗 Joint Plot")

    num_cols = data.select_dtypes("number").columns.tolist()
    x_col = st.selectbox("X Axis", num_cols)
    y_col = st.selectbox("Y Axis", num_cols, index=1)

    if x_col == y_col:
        st.warning("Please select different features")
    else:
        kind = st.selectbox("Plot Type", ["scatter", "reg", "hex", "kde"])

        if kind in ["scatter", "kde"]:
            g = sn.jointplot(data=data, x=x_col, y=y_col, hue="species", kind=kind)
        else:
            g = sn.jointplot(data=data, x=x_col, y=y_col, kind=kind)

        st.pyplot(g.fig, clear_figure=True)

# -------------------- PAIR PLOT --------------------
elif eda_option == "Pair Plot":
    st.subheader("🔀 Pair Plot")
    st.info("Color coded by species")

    g = sn.pairplot(data, hue="species")
    st.pyplot(g.fig, clear_figure=True)

# -------------------- BOXEN PLOT --------------------
elif eda_option == "Boxen Plot":
    st.subheader("📦 Boxen Plot")

    y_col = st.selectbox("Feature", data.select_dtypes("number").columns)

    fig, ax = plt.subplots(figsize=(8, 5))
    sn.boxenplot(data=data, x="species", y=y_col, ax=ax)
    ax.set_title(f"{y_col} by Species")
    st.pyplot(fig, clear_figure=True)

# -------------------- STRIP PLOT --------------------
elif eda_option == "Strip Plot":
    st.subheader("📌 Strip Plot")

    y_col = st.selectbox("Feature", data.select_dtypes("number").columns)

    fig, ax = plt.subplots(figsize=(8, 5))
    sn.stripplot(data=data, x="species", y=y_col, ax=ax)
    ax.set_title(f"{y_col} by Species")
    st.pyplot(fig, clear_figure=True)

# -------------------- SWARM PLOT --------------------
elif eda_option == "Swarm Plot":
    st.subheader("🐝 Swarm Plot")

    y_col = st.selectbox("Feature", data.select_dtypes("number").columns)

    fig, ax = plt.subplots(figsize=(8, 5))
    sn.swarmplot(data=data, x="species", y=y_col, ax=ax)
    ax.set_title(f"{y_col} by Species")
    st.pyplot(fig, clear_figure=True)

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("🚀 **Deployed on Streamlit Cloud**")
