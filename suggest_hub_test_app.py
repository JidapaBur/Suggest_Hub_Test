import streamlit as st
import pandas as pd
import folium
from folium import FeatureGroup, LayerControl
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from geopy.distance import great_circle

st.set_page_config(layout="wide")
st.title("📦 Customer & Hub Visualization Tool")
# Footer note
st.markdown("<div style='text-align:right; font-size:12px; color:gray;'>Developed by Jidapa Buranachan</div>", unsafe_allow_html=True)

# Downloadable template section
st.markdown("### 📥 Download Template Files")
cust_template = pd.DataFrame(columns=["Customer_Code", "Lat", "Long", "Type", "Province"])
dc_template = pd.DataFrame(columns=["Hub_Name", "Lat", "Long", "Type", "Province"])

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="⬇️ Download Customer Template",
        data=cust_template.to_csv(index=False).encode('utf-8-sig'),
        file_name='Customer_Template.csv',
        mime='text/csv'
    )
with col2:
    st.download_button(
        label="⬇️ Download Hub Template",
        data=dc_template.to_csv(index=False).encode('utf-8-sig'),
        file_name='Hub_Template.csv',
        mime='text/csv'
    )

# Upload files
cust_file = st.file_uploader("Upload Customer File (.csv with Lat, Long, Customer_Code, Type, Province)", type="csv")
dc_file = st.file_uploader("Upload Hub File (.csv with Lat, Long, Hub_Name, Type, Province)", type="csv")

if cust_file:
    try:
        cust_data = pd.read_csv(cust_file)
        if cust_data.empty:
            st.error("🚫 Customer CSV file is empty. Please upload a file with data.")
            st.stop()
        cust_data = cust_data.dropna(subset=['Lat', 'Long'])
    except pd.errors.EmptyDataError:
        st.error("🚫 Customer CSV file is empty or corrupted.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load customer file: {e}")
        st.stop()

    locations = cust_data[['Lat', 'Long']]

    # Filter by Type
    customer_types = cust_data['Type'].dropna().unique().tolist()
    selected_types = st.multiselect("Filter Customer Types:", options=customer_types, default=customer_types)
    cust_data = cust_data[cust_data['Type'].isin(selected_types)]

    st.subheader("📍 Nearest Hub for Each Customer")

    if dc_file:
        dc_data = pd.read_csv(dc_file).dropna(subset=['Lat', 'Long'])

        # Filter Hub types
        dc_types = dc_data['Type'].dropna().unique().tolist()
        selected_dc_types = st.multiselect("Filter Hub Types:", options=dc_types, default=dc_types)
        dc_data = dc_data[dc_data['Type'].isin(selected_dc_types)]

        # Find nearest hub for each customer
        def find_nearest_dc(cust_lat, cust_lon, dc_df):
            distances = dc_df.apply(
                lambda row: great_circle((cust_lat, cust_lon), (row['Lat'], row['Long'])).km,
                axis=1
            )
            idx = distances.idxmin()
            return dc_df.loc[idx, 'Hub_Name'], distances.min()

        results = []
        for _, row in cust_data.iterrows():
            nearest_dc, min_dist = find_nearest_dc(row['Lat'], row['Long'], dc_data)
            results.append({
                'Customer_Code': row['Customer_Code'],
                'Type': row.get('Type', 'Unknown'),
                'Province': row.get('Province', 'N/A'),
                'Nearest_Hub': nearest_dc,
                'Distance_km': round(min_dist, 2)
            })

        nearest_df = pd.DataFrame(results)
        st.dataframe(nearest_df)

    # Suggested hub clustering (KMeans)
    st.subheader("🌐 Suggested Hub Locations (KMeans)")
    n_dc = st.slider("Select number of suggested hubs (KMeans):", 1, 10, 5)
    kmeans = KMeans(n_clusters=n_dc, random_state=42)
    kmeans.fit(cust_data[['Lat', 'Long']])
    kmeans_dc_locations = kmeans.cluster_centers_

    # HDBSCAN-based Hub Suggestion
    st.subheader("🧭 Radius-based Hub Grouping (HDBSCAN)")
    min_cluster_size = st.slider("Minimum cluster size:", 2, 50, 10)

    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    hdb_labels = clusterer.fit_predict(cust_data[['Lat', 'Long']])

    # Calculate cluster centers
    hdbscan_centers = []
    for label in set(hdb_labels):
        if label == -1:
            continue  # skip noise
        cluster_points = cust_data[hdb_labels == label]
        center = cluster_points[['Lat', 'Long']].mean().values
        hdbscan_centers.append(center)

# ---------------------------
# 🗺️ KMeans Visualization
# ---------------------------
st.subheader("🗺️ KMeans Hub Visualization")
m_kmeans = folium.Map(location=[13.75, 100.5], zoom_start=6)
for i, (lat, lon) in enumerate(kmeans_dc_locations):
    folium.Marker(
        location=[lat, lon],
        popup=f"KMeans Hub #{i+1}",
        icon=folium.Icon(color='blue', icon='star', prefix='fa')
    ).add_to(m_kmeans)
st_folium(m_kmeans, width=1100, height=400, key="map_kmeans")

# ---------------------------
# 🧭 HDBSCAN Visualization
# ---------------------------
st.subheader("🧭 HDBSCAN Hub Visualization")
m_hdbscan = folium.Map(location=[13.75, 100.5], zoom_start=6)
for i, (lat, lon) in enumerate(hdbscan_centers):
    folium.Marker(
        location=[lat, lon],
        popup=f"HDBSCAN Hub #{i+1}",
        icon=folium.Icon(color='green', icon='star', prefix='fa')
    ).add_to(m_hdbscan)
st_folium(m_hdbscan, width=1100, height=400, key="map_hdbscan")


    # Create folium map for KMeans
    st.subheader("🗺️ KMeans Visualization")
    m_kmeans = folium.Map(location=[13.75, 100.5], zoom_start=6)
    for i, (lat, lon) in enumerate(kmeans_dc_locations):
        folium.Marker(
            location=[lat, lon],
            popup=f"KMeans Hub #{i+1}",
            icon=folium.Icon(color='blue', icon='star', prefix='fa')
        ).add_to(m_kmeans)
    st_folium(m_kmeans, width=1100, height=400, key="map_kmeans")

    # Create folium map for HDBSCAN
    st.subheader("🗺️ HDBSCAN Visualization")
    m_hdbscan = folium.Map(location=[13.75, 100.5], zoom_start=6)
    for i, (lat, lon) in enumerate(hdbscan_centers):
        folium.Marker(
            location=[lat, lon],
            popup=f"HDBSCAN Hub #{i+1}",
            icon=folium.Icon(color='green', icon='star', prefix='fa')
        ).add_to(m_hdbscan)
    st_folium(m_hdbscan, width=1100, height=400, key="map_hdbscan")
    LayerControl().add_to(m)

    # Show final map
    st.subheader("🗺️ Visualization")
    st_folium(m, width=1100, height=600, returned_objects=[], key="main_map")
