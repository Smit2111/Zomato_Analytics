import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Zomato Delivery Insights Dashboard",
    page_icon="🍽️",
    layout="wide",
)


COLOR_PRIMARY = "#D94F30"
COLOR_SECONDARY = "#1F3C88"
COLOR_ACCENT = "#F4B942"
COLOR_BG = "#FFF8F4"

st.markdown(
    f"""
    <style>
        .stApp {{
            background: linear-gradient(180deg, {COLOR_BG} 0%, #FFFFFF 45%, #FFF1E8 100%);
        }}
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }}
        [data-testid="stMetricValue"] {{
            color: {COLOR_SECONDARY};
        }}
        h1, h2, h3 {{
            color: #202124;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("Zomato Dataset.csv", encoding="latin1")
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )

    numeric_columns = [
        "delivery_person_age",
        "delivery_person_ratings",
        "vehicle_condition",
        "multiple_deliveries",
        "time_taken_min",
        "restaurant_latitude",
        "restaurant_longitude",
        "delivery_location_latitude",
        "delivery_location_longitude",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    text_columns = [
        "weather_conditions",
        "road_traffic_density",
        "type_of_order",
        "type_of_vehicle",
        "festival",
        "city",
    ]
    for column in text_columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .replace({"NaN": np.nan, "nan": np.nan, "": np.nan})
        )

    df["order_date"] = pd.to_datetime(df["order_date"], format="%d-%m-%Y", errors="coerce")

    for column in ["time_orderd", "time_order_picked"]:
        cleaned = (
            df[column]
            .astype(str)
            .str.strip()
            .replace({"NaN": np.nan, "nan": np.nan, "": np.nan})
        )
        df[column] = pd.to_datetime(cleaned, format="%H:%M", errors="coerce")

    df["prep_time_min"] = (
        (df["time_order_picked"] - df["time_orderd"]).dt.total_seconds() / 60
    )
    df.loc[df["prep_time_min"] < 0, "prep_time_min"] += 24 * 60

    df["distance_km"] = (
        ((df["restaurant_latitude"] - df["delivery_location_latitude"]) ** 2)
        + ((df["restaurant_longitude"] - df["delivery_location_longitude"]) ** 2)
    ) ** 0.5 * 111

    df["rating_band"] = pd.cut(
        df["delivery_person_ratings"],
        bins=[0, 3.5, 4.0, 4.5, 5.0],
        labels=["Below 3.5", "3.5 - 4.0", "4.0 - 4.5", "4.5 - 5.0"],
        include_lowest=True,
    )

    df = df.dropna(subset=["order_date", "time_taken_min", "city"])
    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Dashboard Filters")

    city_options = sorted(df["city"].dropna().unique().tolist())
    weather_options = sorted(df["weather_conditions"].dropna().unique().tolist())
    traffic_options = sorted(df["road_traffic_density"].dropna().unique().tolist())
    order_options = sorted(df["type_of_order"].dropna().unique().tolist())

    city_filter = st.sidebar.multiselect("City", city_options, default=city_options)
    weather_filter = st.sidebar.multiselect("Weather", weather_options, default=weather_options)
    traffic_filter = st.sidebar.multiselect("Traffic", traffic_options, default=traffic_options)
    order_filter = st.sidebar.multiselect("Order Type", order_options, default=order_options)

    min_rating, max_rating = st.sidebar.slider(
        "Delivery Rating Range",
        min_value=float(df["delivery_person_ratings"].min()),
        max_value=float(df["delivery_person_ratings"].max()),
        value=(
            float(df["delivery_person_ratings"].min()),
            float(df["delivery_person_ratings"].max()),
        ),
    )

    filtered_df = df[
        df["city"].isin(city_filter)
        & df["weather_conditions"].isin(weather_filter)
        & df["road_traffic_density"].isin(traffic_filter)
        & df["type_of_order"].isin(order_filter)
        & df["delivery_person_ratings"].between(min_rating, max_rating)
    ].copy()

    return filtered_df


def metric_row(df: pd.DataFrame) -> None:
    total_orders = int(df.shape[0])
    avg_time = df["time_taken_min"].mean()
    avg_rating = df["delivery_person_ratings"].mean()
    avg_distance = df["distance_km"].mean()
    festival_share = (df["festival"].eq("Yes").mean() * 100) if "festival" in df else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Orders", f"{total_orders:,}")
    m2.metric("Avg Delivery Time", f"{avg_time:.1f} min")
    m3.metric("Avg Rating", f"{avg_rating:.2f}")
    m4.metric("Avg Distance", f"{avg_distance:.1f} km")
    m5.metric("Festival Orders", f"{festival_share:.1f}%")


def build_summary_cards(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        [
            ("Date Range", f"{df['order_date'].min().date()} to {df['order_date'].max().date()}"),
            ("Cities Covered", df["city"].nunique()),
            ("Weather Types", df["weather_conditions"].nunique()),
            ("Vehicle Types", df["type_of_vehicle"].nunique()),
            ("Average Prep Time", f"{df['prep_time_min'].mean():.1f} min"),
            ("Median Delivery Time", f"{df['time_taken_min'].median():.1f} min"),
            ("Best Rated City", df.groupby("city")["delivery_person_ratings"].mean().idxmax()),
            ("Fastest City", df.groupby("city")["time_taken_min"].mean().idxmin()),
        ],
        columns=["Metric", "Value"],
    )
    return summary


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Core performance snapshot")
    metric_row(df)

    st.markdown("### Dataset Summary")
    summary_left, summary_right = st.columns((1.1, 1.4))

    with summary_left:
        st.dataframe(build_summary_cards(df), hide_index=True, use_container_width=True)

    with summary_right:
        stat_df = (
            df[
                [
                    "delivery_person_age",
                    "delivery_person_ratings",
                    "prep_time_min",
                    "distance_km",
                    "time_taken_min",
                ]
            ]
            .describe()
            .transpose()
            .round(2)
        )
        st.dataframe(stat_df, use_container_width=True)

    insight_left, insight_right, insight_third = st.columns(3)
    busiest_city = df["city"].value_counts().idxmax()
    most_common_weather = df["weather_conditions"].mode().iloc[0]
    top_order_type = df["type_of_order"].value_counts().idxmax()

    insight_left.info(
        f"**Busiest location:** {busiest_city} has the highest order volume in the current filtered dataset."
    )
    insight_right.info(
        f"**Most common operating condition:** {most_common_weather} appears most often across deliveries."
    )
    insight_third.info(
        f"**Most popular category:** {top_order_type} is the leading order type in the selected view."
    )

    left, right = st.columns((1.2, 1))

    top_cities = (
        df.groupby("city", as_index=False)
        .agg(orders=("id", "count"), avg_time=("time_taken_min", "mean"))
        .sort_values("orders", ascending=False)
    )

    with left:
        fig_city = px.bar(
            top_cities,
            x="city",
            y="orders",
            color="avg_time",
            color_continuous_scale="YlOrRd",
            text_auto=True,
            title="Top Locations by Order Volume",
        )
        fig_city.update_layout(xaxis_title="", yaxis_title="Orders", coloraxis_colorbar_title="Avg Time")
        st.plotly_chart(fig_city, use_container_width=True)

    with right:
        order_mix = df["type_of_order"].value_counts().reset_index()
        order_mix.columns = ["type_of_order", "orders"]
        fig_mix = px.pie(
            order_mix,
            values="orders",
            names="type_of_order",
            hole=0.55,
            color_discrete_sequence=[COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT, "#7A9E9F"],
            title="Popular Order Categories",
        )
        st.plotly_chart(fig_mix, use_container_width=True)

    trend_left, trend_right = st.columns(2)

    daily_trend = (
        df.groupby("order_date", as_index=False)
        .agg(orders=("id", "count"), avg_rating=("delivery_person_ratings", "mean"))
        .sort_values("order_date")
    )

    with trend_left:
        fig_orders = px.line(
            daily_trend,
            x="order_date",
            y="orders",
            markers=True,
            title="Daily Order Trend",
        )
        fig_orders.update_traces(line_color=COLOR_PRIMARY)
        fig_orders.update_layout(xaxis_title="", yaxis_title="Orders")
        st.plotly_chart(fig_orders, use_container_width=True)

    with trend_right:
        fig_ratings = px.line(
            daily_trend,
            x="order_date",
            y="avg_rating",
            markers=True,
            title="Rating Trend Over Time",
        )
        fig_ratings.update_traces(line_color=COLOR_SECONDARY)
        fig_ratings.update_layout(xaxis_title="", yaxis_title="Average Rating")
        st.plotly_chart(fig_ratings, use_container_width=True)


def answer_data_query(query: str, df: pd.DataFrame) -> str:
    text = query.lower().strip()
    if not text:
        return "Ask about totals, top cities, fastest deliveries, ratings, weather impact, vehicles, or order categories."

    if any(keyword in text for keyword in ["total records", "records", "rows", "entries", "total orders"]):
        return (
            f"The filtered dataset currently contains {len(df):,} records with an average delivery time of "
            f"{df['time_taken_min'].mean():.1f} minutes."
        )

    if any(keyword in text for keyword in ["top city", "top location", "busiest city", "most orders city"]):
        city_stats = df["city"].value_counts()
        city = city_stats.idxmax()
        return f"{city} is the busiest city with {int(city_stats.iloc[0]):,} orders in the current selection."

    if any(keyword in text for keyword in ["popular order", "popular cuisine", "top category", "top order type"]):
        order_counts = df["type_of_order"].value_counts()
        order_type = order_counts.idxmax()
        return (
            f"{order_type} is the most popular order category with {int(order_counts.iloc[0]):,} orders. "
            "This dataset does not include a cuisine column, so order type is used as the nearest preference signal."
        )

    if any(keyword in text for keyword in ["best rating", "highest rating", "top rated city"]):
        rating_by_city = df.groupby("city")["delivery_person_ratings"].mean().sort_values(ascending=False)
        city = rating_by_city.index[0]
        rating = rating_by_city.iloc[0]
        return f"{city} has the highest average delivery rating at {rating:.2f}."

    if any(keyword in text for keyword in ["fastest city", "lowest delivery time", "quickest city"]):
        time_by_city = df.groupby("city")["time_taken_min"].mean().sort_values()
        city = time_by_city.index[0]
        avg_time = time_by_city.iloc[0]
        return f"{city} is the fastest city on average with deliveries completed in {avg_time:.1f} minutes."

    if any(keyword in text for keyword in ["slowest city", "highest delivery time"]):
        time_by_city = df.groupby("city")["time_taken_min"].mean().sort_values(ascending=False)
        city = time_by_city.index[0]
        avg_time = time_by_city.iloc[0]
        return f"{city} is the slowest city on average with delivery time around {avg_time:.1f} minutes."

    if any(keyword in text for keyword in ["weather impact", "weather", "traffic impact", "traffic"]):
        pivot = df.pivot_table(
            index="weather_conditions",
            columns="road_traffic_density",
            values="time_taken_min",
            aggfunc="mean",
        )
        worst_weather = (
            df.groupby("weather_conditions")["time_taken_min"].mean().sort_values(ascending=False)
        )
        weather = worst_weather.index[0]
        avg_time = worst_weather.iloc[0]
        jam_time = pivot.max().max()
        return (
            f"{weather} causes the longest average delivery time at {avg_time:.1f} minutes. "
            f"Across weather-traffic combinations, the worst observed average is {jam_time:.1f} minutes."
        )

    if any(keyword in text for keyword in ["vehicle", "best vehicle", "fastest vehicle"]):
        vehicle_stats = df.groupby("type_of_vehicle")["time_taken_min"].mean().sort_values()
        vehicle = vehicle_stats.index[0]
        avg_time = vehicle_stats.iloc[0]
        return f"{vehicle.title()} is the fastest vehicle type on average at {avg_time:.1f} minutes per delivery."

    if any(keyword in text for keyword in ["festival", "festival impact"]):
        festival_stats = df.groupby("festival")["time_taken_min"].mean()
        if {"Yes", "No"}.issubset(set(festival_stats.index)):
            diff = festival_stats["Yes"] - festival_stats["No"]
            direction = "higher" if diff > 0 else "lower"
            return (
                f"During festivals, average delivery time is {abs(diff):.1f} minutes {direction} "
                f"than non-festival periods."
            )
        return "Festival comparison is limited because one of the festival categories is missing in the filtered data."

    if any(keyword in text for keyword in ["summary", "overview", "insights"]):
        city = df["city"].value_counts().idxmax()
        order_type = df["type_of_order"].value_counts().idxmax()
        avg_rating = df["delivery_person_ratings"].mean()
        avg_time = df["time_taken_min"].mean()
        return (
            f"Summary: {len(df):,} records are selected. {city} leads order volume, {order_type} is the top order type, "
            f"average rating is {avg_rating:.2f}, and average delivery time is {avg_time:.1f} minutes."
        )

    return (
        "I can answer questions about record counts, top cities, popular order categories, rating leaders, "
        "delivery-time trends, traffic or weather impact, festival effect, and vehicle performance."
    )


def render_operations(df: pd.DataFrame) -> None:
    st.subheader("Delivery performance and operating conditions")

    left, right = st.columns(2)

    traffic_weather = (
        df.pivot_table(
            index="weather_conditions",
            columns="road_traffic_density",
            values="time_taken_min",
            aggfunc="mean",
        )
        .sort_index()
    )

    with left:
        heatmap = go.Figure(
            data=go.Heatmap(
                z=traffic_weather.values,
                x=traffic_weather.columns.tolist(),
                y=traffic_weather.index.tolist(),
                colorscale="YlOrRd",
                text=np.round(traffic_weather.values, 1),
                texttemplate="%{text}",
            )
        )
        heatmap.update_layout(
            title="Average Delivery Time by Weather and Traffic",
            xaxis_title="Traffic Density",
            yaxis_title="Weather",
        )
        st.plotly_chart(heatmap, use_container_width=True)

    with right:
        fig_box = px.box(
            df,
            x="road_traffic_density",
            y="time_taken_min",
            color="weather_conditions",
            title="Delivery Time Spread Across Traffic Conditions",
        )
        fig_box.update_layout(xaxis_title="Traffic Density", yaxis_title="Time Taken (min)")
        st.plotly_chart(fig_box, use_container_width=True)

    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        fig_scatter = px.scatter(
            df,
            x="distance_km",
            y="time_taken_min",
            color="type_of_vehicle",
            size="delivery_person_ratings",
            hover_data=["city", "type_of_order"],
            opacity=0.65,
            title="Distance vs Delivery Time",
        )
        fig_scatter.update_layout(xaxis_title="Estimated Distance (km)", yaxis_title="Time Taken (min)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with bottom_right:
        vehicle_summary = (
            df.groupby("type_of_vehicle", as_index=False)
            .agg(
                avg_time=("time_taken_min", "mean"),
                avg_rating=("delivery_person_ratings", "mean"),
                avg_distance=("distance_km", "mean"),
            )
            .sort_values("avg_time")
        )
        fig_vehicle = px.bar(
            vehicle_summary,
            x="type_of_vehicle",
            y="avg_time",
            color="avg_rating",
            color_continuous_scale="Blues",
            text_auto=".1f",
            title="Vehicle Type Performance",
        )
        fig_vehicle.update_layout(xaxis_title="", yaxis_title="Avg Delivery Time (min)")
        st.plotly_chart(fig_vehicle, use_container_width=True)


def render_customer_patterns(df: pd.DataFrame) -> None:
    st.subheader("Patterns in order mix, ratings, and fulfillment")

    left, right = st.columns(2)

    with left:
        cuisine_note = (
            "This dataset does not include a cuisine column, so the chart below uses "
            "order categories as the closest proxy for customer preference."
        )
        st.caption(cuisine_note)
        cuisine_proxy = (
            df.groupby(["type_of_order", "city"], as_index=False)
            .size()
            .rename(columns={"size": "orders"})
        )
        fig_treemap = px.treemap(
            cuisine_proxy,
            path=["city", "type_of_order"],
            values="orders",
            color="orders",
            color_continuous_scale="Oranges",
            title="Demand Mix by City and Order Category",
        )
        st.plotly_chart(fig_treemap, use_container_width=True)

    with right:
        rating_distribution = (
            df["rating_band"].value_counts(dropna=False).reset_index()
        )
        rating_distribution.columns = ["rating_band", "orders"]
        fig_hist = px.bar(
            rating_distribution,
            x="rating_band",
            y="orders",
            color="rating_band",
            title="Rating Band Distribution",
        )
        fig_hist.update_layout(xaxis_title="Rating Band", yaxis_title="Orders", showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        prep_analysis = (
            df.groupby("type_of_order", as_index=False)
            .agg(
                avg_prep_time=("prep_time_min", "mean"),
                avg_delivery_time=("time_taken_min", "mean"),
            )
            .sort_values("avg_delivery_time", ascending=False)
        )
        fig_prep = px.bar(
            prep_analysis,
            x="type_of_order",
            y=["avg_prep_time", "avg_delivery_time"],
            barmode="group",
            title="Preparation Time vs Delivery Time",
        )
        fig_prep.update_layout(xaxis_title="", yaxis_title="Minutes")
        st.plotly_chart(fig_prep, use_container_width=True)

    with bottom_right:
        festival_summary = (
            df.groupby(["festival", "city"], as_index=False)
            .agg(avg_time=("time_taken_min", "mean"))
        )
        fig_festival = px.bar(
            festival_summary,
            x="city",
            y="avg_time",
            color="festival",
            barmode="group",
            title="Festival Effect on Delivery Time",
        )
        fig_festival.update_layout(xaxis_title="", yaxis_title="Avg Delivery Time (min)")
        st.plotly_chart(fig_festival, use_container_width=True)


def render_data_view(df: pd.DataFrame) -> None:
    st.subheader("Dataset explorer")
    query = st.text_input("Search any value in the filtered dataset")

    display_df = df.copy()
    if query:
        mask = display_df.astype(str).apply(
            lambda row: row.str.contains(query, case=False, na=False)
        ).any(axis=1)
        display_df = display_df[mask]

    st.dataframe(display_df.head(500), use_container_width=True)
    st.caption(f"Showing {min(len(display_df), 500):,} rows out of {len(display_df):,} matching rows.")


def render_data_assistant(df: pd.DataFrame) -> None:
    st.subheader("Data Assistant")
    st.write(
        "Ask command-style questions about the filtered dataset. Examples: "
        "`total records`, `busiest city`, `top order type`, `weather impact`, `fastest vehicle`."
    )

    col1, col2 = st.columns((2.2, 1))
    with col1:
        user_query = st.text_input("Enter a dataset question")
    with col2:
        preset = st.selectbox(
            "Quick command",
            [
                "Choose a command",
                "summary",
                "total records",
                "busiest city",
                "top order type",
                "top rated city",
                "fastest city",
                "weather impact",
                "fastest vehicle",
                "festival impact",
            ],
        )

    final_query = user_query.strip() if user_query.strip() else ("" if preset == "Choose a command" else preset)

    if final_query:
        st.success(answer_data_query(final_query, df))

    st.markdown("### Supported questions")
    helper_df = pd.DataFrame(
        {
            "Command": [
                "total records",
                "busiest city",
                "top order type",
                "top rated city",
                "fastest city",
                "weather impact",
                "fastest vehicle",
                "festival impact",
                "summary",
            ],
            "What it returns": [
                "Filtered row count and average delivery time",
                "City with the highest order volume",
                "Most popular order category",
                "City with the best average rating",
                "City with the shortest average delivery time",
                "Weather and traffic effect on delivery duration",
                "Fastest vehicle category by average time",
                "Festival vs non-festival delivery comparison",
                "Compact overview of the current selection",
            ],
        }
    )
    st.dataframe(helper_df, hide_index=True, use_container_width=True)


def main() -> None:
    st.title("Zomato Delivery Insights Dashboard")
    st.write(
        "Explore delivery behavior across cities, operating conditions, and customer demand patterns "
        "with interactive filters and advanced charts."
    )

    df = load_data()
    filtered_df = apply_filters(df)

    if filtered_df.empty:
        st.warning("No data matches the current filters. Adjust the sidebar selections to continue.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Operations", "Customer Patterns", "Dataset", "Data Assistant"]
    )

    with tab1:
        render_overview(filtered_df)

    with tab2:
        render_operations(filtered_df)

    with tab3:
        render_customer_patterns(filtered_df)

    with tab4:
        render_data_view(filtered_df)

    with tab5:
        render_data_assistant(filtered_df)


if __name__ == "__main__":
    main()
