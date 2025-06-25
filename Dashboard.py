import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import VehicleStateAnalyzer
from visuals import extract_trips_from_gps_data

# App config
st.set_page_config(
    page_title="GPS Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)

st.title("Upload CSV File")

# Initialize placeholders
df = st.session_state.get("data", None)
trips = st.session_state.get("trips", None)

# Upload if not already loaded
if df is None:
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.write(df.head())

            # --- Preprocess Function ---
            def preprocess_data(df):
                unneeded_columns = ['Odometer', 'Other Ports', 'Data Received Time', 'IGN']
                week_filtered_df = df.drop(columns=unneeded_columns, errors='ignore')
                week_filtered_df = week_filtered_df[week_filtered_df["Data String"].str.contains('Satellites')]
                week_filtered_df['Satellites'] = week_filtered_df['Data String'].str.extract(r'\[Satellites=(\d+)\]', expand=False).astype(int)
                week_filtered_df = week_filtered_df[week_filtered_df["Satellites"] >= 10]
                df = week_filtered_df.copy()
                df = df[df["Data String"].str.contains('Angle')]
                df['Angle'] = df['Data String'].str.extract(r'\[Angle=(\d+)\]', expand=False).astype(int)
                df = df[df["Data String"].str.contains('Altitude')]
                df['Altitude'] = df['Data String'].str.extract(r'\[Altitude=(\d+)\]', expand=False).astype(int)
                df['Latitude'] = df['Latitude'] / 1e7
                df['Longitude'] = df['Longitude'] / 1e7
                df['Data Actual Time'] = pd.to_datetime(df['Data Actual Time'])
                return df

            # Preprocess and analyze
            df = preprocess_data(df)
            analyzer = VehicleStateAnalyzer(idle_threshold=2, accel_threshold=2, time_window=2)
            df = analyzer.analyze_vehicle_states(df)
            analyzer.generate_summary_stats(df)

            # Trip extraction
            trips = extract_trips_from_gps_data(
                df,
                proximity_threshold=0.1,
                min_trip_duration_minutes=30,
                min_trip_points=100
            )

            # Store in session state
            st.session_state["data"] = df
            st.session_state["trips"] = trips

        except Exception as e:
            st.error(f"Error loading CSV: {e}")
else:
    st.info("Using uploaded data from session. No need to upload again.")

# Show dashboard if data exists
if df is not None:

    # --- Metric Plot Function ---
    def plot_metric(label, value, suffix=""):
        fig = go.Figure(go.Indicator(
            mode="number",
            value=value,
            number={"suffix": suffix, "font": {"size": 28}},
            title={"text": label, "font": {"size": 20}},
        ))
        fig.update_layout(height=120, margin=dict(t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # --- Bottom Left Line Chart ---
    def plot_bottom_left_line():
        fig = px.line(
            df.sort_values("Data Actual Time"),
            x="Data Actual Time",
            y="Speed",
            title="Speed Over Time",
            labels={"Speed": "Speed (km/h)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Bottom Right Box Chart ---
    def plot_bottom_right_box():
        fig = px.box(
            df,
            x="vehicle_state",
            y="confidence",
            title="Confidence Distribution by Vehicle State",
            labels={"confidence": "Confidence"}
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Top Metrics Layout ---
    top_left_column, top_right_column = st.columns(2)

    with top_left_column:
        col1, col2 = st.columns(2)
        with col1:
            plot_metric("Avg. Speed", df['Speed'].mean(), " km/h")
            plot_metric("Avg. Altitude", df['Altitude'].mean(), " m")
        with col2:
            plot_metric("Avg. Angle", df['Angle'].mean(), "°")
            plot_metric("Avg. Satellites", df['Satellites'].mean(), "")

    # --- Bar Chart: Full Width ---
    st.markdown("### Total Distance Moved by Vehicle State")
    distance_by_state = df.groupby('vehicle_state')['distance_moved'].sum().reset_index()
    fig_bar = px.bar(
        distance_by_state,
        x='vehicle_state',
        y='distance_moved',
        title=None,
        labels={'distance_moved': 'Distance (m)'},
        text_auto='.2s',
        color='vehicle_state'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Pie Chart: Full Width ---
    st.markdown("### Vehicle State Distribution")
    vehicle_state_counts = df['vehicle_state'].value_counts().reset_index()
    vehicle_state_counts.columns = ['vehicle_state', 'count']
    fig_pie = px.pie(
        vehicle_state_counts,
        names='vehicle_state',
        values='count',
        title=None,
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # --- Bottom Charts Layout ---
    bottom_left_column, bottom_right_column = st.columns(2)

    with bottom_left_column:
        plot_bottom_left_line()

    with bottom_right_column:
        plot_bottom_right_box()

# Reset button if needed
st.markdown("---")
if st.button("Reset Dashboard"):
    for key in ['data', 'trips']:
        if key in st.session_state:
            del st.session_state[key]
    st.experimental_rerun()
