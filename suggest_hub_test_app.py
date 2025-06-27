import streamlit as st
import pandas as pd
import folium
from folium import FeatureGroup, LayerControl
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from geopy.distance import great_circle, geodesic
import numpy as np

from sklearn.neighbors import BallTree
from scipy.spatial.distance import cdist
from math import radians

import geopandas as gpd
from shapely.geometry import Point

#------------------------------------------------------------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("Customer & Hub Visualization Tool")
# Footer note
st.markdown("<div style='text-align:right; font-size:12px; color:gray;'>Version 1.0.4 Developed by Jidapa Buranachan</div>", unsafe_allow_html=True)

#------------------------------------------------------------------------------------------------------------------------


# Downloadable template section
st.markdown("### Download Template Files")
cust_template = pd.DataFrame(columns=["Customer_Code", "Lat", "Long", "Type", "Province"])
store_template = pd.DataFrame(columns=["Store_Code", "Lat", "Long", "Type", "Province"])
dc_template = pd.DataFrame(columns=["Hub_Name", "Lat", "Long", "Type", "Province"])

col1, col2, col3 = st.columns(3)

with col1:
    st.download_button(
        label="⬇️ Download Customer Template",
        data=cust_template.to_csv(index=False).encode('utf-8-sig'),
        file_name='Customer_Template.csv',
        mime='text/csv'
    )

with col2:
    st.download_button(
        label="⬇️ Download Store Template",
        data=store_template.to_csv(index=False).encode('utf-8-sig'),
        file_name='Store_Template.csv',
        mime='text/csv'
    )

with col3:
    st.download_button(
        label="⬇️ Download Hub Template",
        data=dc_template.to_csv(index=False).encode('utf-8-sig'),
        file_name='Hub_Template.csv',
        mime='text/csv'
    )

#------------------------------------------------------------------------------------------------------------------------

# ------------------------------ Upload files ------------------------------
cust_file = st.file_uploader("📤 Upload Customer File (.csv with Lat, Long, Customer_Code, Type, Province)", type="csv")
store_file = st.file_uploader("📤 Upload Store File (.csv with Lat, Long, Store_Code, Type, Province)", type="csv")
dc_file = st.file_uploader("📤 Upload Hub File (.csv with Lat, Long, Hub_Name, Type, Province)", type="csv")

# ------------------------------ Load Customer File ------------------------------
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

# ------------------------------ Load Store File ------------------------------
if store_file:
    try:
        store_data = pd.read_csv(store_file)
        if store_data.empty:
            st.error("🚫 Store CSV file is empty. Please upload a file with data.")
            st.stop()
        store_data = store_data.dropna(subset=['Lat', 'Long'])
    except pd.errors.EmptyDataError:
        st.error("🚫 Store CSV file is empty or corrupted.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load store file: {e}")
        st.stop()

#------------------------------------------------------------------------------------------------------------------------
    
    # โหลดแผนที่ประเทศไทย
    thailand = gpd.read_file("thailand.geojson")
    thailand_union = thailand.unary_union
    
    provinces_gdf = gpd.read_file("provinces.geojson")
    
    # แปลงลูกค้าเป็น GeoDataFrame
    cust_data['geometry'] = cust_data.apply(lambda row: Point(row['Long'], row['Lat']), axis=1)
    cust_gdf = gpd.GeoDataFrame(cust_data, geometry='geometry', crs="EPSG:4326")
    
    # กรองเฉพาะลูกค้าที่อยู่ในขอบเขตประเทศไทยจริง
    cust_gdf = cust_gdf[cust_gdf.geometry.within(thailand.unary_union)]
    
    # คืนค่า DataFrame กลับไปใช้งานต่อ
    cust_data = pd.DataFrame(cust_gdf.drop(columns='geometry'))

#------------------------------------------------------------------------------------------------------------------------
    
    # Filter by Type
    customer_types = cust_data['Type'].dropna().unique().tolist()
    selected_types = st.multiselect("Filter Customer Types:", options=customer_types, default=customer_types)
    cust_data = cust_data[cust_data['Type'].isin(selected_types)]

#------------------------------------------------------------------------------------------------------------------------

    st.subheader("📍 Nearest Hub for Each Customer")
    
    if dc_file:
        # โหลด Hub และทำความสะอาดข้อมูล
        dc_data = pd.read_csv(dc_file)
        dc_data[['Lat', 'Long']] = dc_data[['Lat', 'Long']].apply(pd.to_numeric, errors='coerce')
        dc_data = dc_data.dropna(subset=['Lat', 'Long'])
    
        # Filter Hub types
        dc_types = dc_data['Type'].dropna().unique().tolist()
        selected_dc_types = st.multiselect("Filter Hub Types:", options=dc_types, default=dc_types)
        dc_data = dc_data[dc_data['Type'].isin(selected_dc_types)]
        
        # เตรียมข้อมูลลูกค้า
        cust_data[['Lat', 'Long']] = cust_data[['Lat', 'Long']].apply(pd.to_numeric, errors='coerce')
        cust_data = cust_data.dropna(subset=['Lat', 'Long'])
        cust_data['Province'] = cust_data['Province'].fillna("").astype(str)
        
        # เลือกลูกค้าที่ Province เป็น NaN หรือ "Unknown"
        cust_unknown = cust_data[cust_data['Province'].str.lower().isin(["", "unknown"])].copy()
        
        # แปลงเป็นจุด geometry
        cust_unknown['geometry'] = cust_unknown.apply(lambda row: Point(row['Long'], row['Lat']), axis=1)
        cust_unknown_gdf = gpd.GeoDataFrame(cust_unknown, geometry='geometry', crs="EPSG:4326")
        
        # เชื่อมกับ polygon จังหวัดด้วย spatial join
        cust_with_province = gpd.sjoin(cust_unknown_gdf, provinces_gdf, how="left", predicate="within")
        
        # ใส่ชื่อจังหวัดจาก polygon (เปลี่ยนชื่อคอลัมน์ให้ตรงกับ geojson ที่คุณใช้)
        cust_with_province['Province'] = cust_with_province['pro_en']
        
        # เตรียมข้อมูลลูกค้าที่รู้จังหวัดอยู่แล้ว
        cust_known = cust_data[~cust_data.index.isin(cust_unknown.index)].copy()
        
        # รวมทั้งหมดกลับมาเป็น cust_data
        cust_data = pd.concat([cust_known, cust_with_province[cust_known.columns]], ignore_index=True)
    
        # ตรวจว่าคอลัมน์ Hub_Name มีจริง
        if 'Hub_Name' not in dc_data.columns:
            st.error("❌ 'Hub_Name' column is missing in hub data.")
            st.stop()
    
        # แปลงพิกัดเป็น radians
        cust_coords = np.radians(cust_data[['Lat', 'Long']].values)
        dc_coords = np.radians(dc_data[['Lat', 'Long']].values)
    
        # สร้าง BallTree จากจุด Hub
        hub_tree = BallTree(dc_coords, metric='haversine')
    
        # คำนวณระยะทางไป hub ที่ใกล้ที่สุด
        distances, indices = hub_tree.query(cust_coords, k=1)
        distances_km = distances.flatten() * 6371  # Earth radius in km
    
        # เตรียมตารางแสดงผล
        nearest_df = cust_data[['Customer_Code', 'Type', 'Province']].copy()
        nearest_df['Nearest_Hub'] = dc_data.iloc[indices.flatten()]['Hub_Name'].values
        nearest_df['Distance_km'] = np.round(distances_km, 2)
    
        st.success(f"✅ Calculated nearest hubs for {len(nearest_df)} customers.")
        st.dataframe(nearest_df)
    
        # ✅ (Optional) ปุ่มดาวน์โหลด
        csv = nearest_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="⬇️ Download Nearest Hub Results",
            data=csv,
            file_name='nearest_hub_results.csv',
            mime='text/csv'
        )
        
    #------------------------------------------------------------------------------------------------------------------------
        
        # Suggest New Hubs for Out-of-Radius Customers
        st.subheader("Suggest New Hubs Based on Radius & Existing Hubs")
        radius_threshold_km = st.slider("Set Radius Threshold from Existing Hubs (km):", 10, 500, 100)
        
    
    #------------------------------------------------------------------------------------------------------------------------
        
    def kmeans_within_thailand(data, n_clusters, thailand_polygon, max_retry=10):
        # ✅ แนะนำ: simplify polygon เพื่อให้ within() เร็วขึ้น (ค่า 0.01 = ประมาณ 1 กม.)
        simplified_polygon = thailand_polygon.simplify(0.01)
    
        for i in range(max_retry):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42 + i)
            kmeans.fit(data[['Lat', 'Long']])
            centers = kmeans.cluster_centers_
    
            # ✅ Vectorized check: แปลงศูนย์กลางเป็น GeoSeries
            centers_geometry = gpd.GeoSeries(
                [Point(lon, lat) for lat, lon in centers],
                crs="EPSG:4326"
            )
    
            # ✅ ตรวจว่า center ทุกจุดอยู่ใน polygon ไทยที่ simplify แล้ว
            if centers_geometry.within(simplified_polygon).all():
                return [(lat, lon) for lat, lon in centers]
    
        # ❗ ถ้าทำไม่สำเร็จใน max_retry → ยอมใช้ค่าที่ได้ (อาจหลุดทะเล)
        return [(lat, lon) for lat, lon in centers]
            
        #------------------------ Main Block ------------------------
        
        # -------------------- สร้าง BallTree และคำนวณระยะใกล้ที่สุด --------------------
        hub_tree = BallTree(dc_coords, metric='haversine')
        distances, _ = hub_tree.query(cust_coords, k=1)  # ระยะใกล้ที่สุดจากแต่ละลูกค้า
        distances_km = distances.flatten() * 6371  # คูณรัศมีโลกให้เป็น km
        
        # -------------------- ตัดสินว่าอยู่นอกระยะ hub เดิมหรือไม่ --------------------
        cust_data['Outside_Hub'] = distances_km > radius_threshold_km
        
        # -------------------- แปลงเป็น GeoDataFrame เพื่อตรวจสอบว่าอยู่ในประเทศไทย --------------------
        cust_data['geometry'] = cust_data.apply(lambda row: Point(row['Long'], row['Lat']), axis=1)
        cust_gdf = gpd.GeoDataFrame(cust_data, geometry='geometry', crs="EPSG:4326")
        cust_gdf = cust_gdf[cust_gdf.geometry.within(thailand_union)]
        
        # ✅ ลูกค้าที่อยู่ในประเทศไทยทั้งหมด (จะรวมทั้งในระยะและนอกระยะ)
        cluster_data = cust_gdf.copy()
                
        st.markdown(
            f"<b>{len(cluster_data)} customers</b> will be used for new hub suggestions (in and out of coverage, inside Thailand).",
            unsafe_allow_html=True
        )
        
        if not cluster_data.empty:
            n_new_hubs = st.slider("How many new hubs to suggest from all customers?", 1, 10, 3)
            new_hub_locations = kmeans_within_thailand(cluster_data, n_new_hubs, thailand_union)
            
            st.subheader("New Hub Suggestions Map")
            m_new = folium.Map(location=[13.75, 100.5], zoom_start=6, control_scale=True)
            
     #------------------------------------------------------------------------------------------------------------------------
                
        # Layer visibility controls
            with st.expander("🧭 Layer Visibility Controls"):
                show_heatmap = st.checkbox("Show Heatmap", value=True)
                show_customer_markers = st.checkbox("Show Customer Markers", value=True)
                show_existing_hubs = st.checkbox("Show Existing Hubs", value=True)
                show_suggested_hubs = st.checkbox("Show Suggested Hubs", value=True)
                show_hub_radius_layer = st.checkbox("Show Existing Hub Radius Zones", value=True)
                           
      
    #------------------------------------------------------------------------------------------------------------------------
    
            # Existing hub layer
            existing_layer = FeatureGroup(name="Existing Hubs")
            for _, row in dc_data.iterrows():
                folium.Marker(
                    location=[row['Lat'], row['Long']],
                    popup=row['Hub_Name'],
                    icon=folium.Icon(color = 'red' if row.get('Type', '').lower() == 'makro' else 'blue', icon='store', prefix='fa')
                ).add_to(existing_layer)
            if show_existing_hubs:
                existing_layer.add_to(m_new)

            # Existing hub radius circles
            if show_hub_radius_layer:
                radius_layer = FeatureGroup(name="Existing Hub Radius")
                for _, row in dc_data.iterrows():
                    folium.Circle(
                        location=[row['Lat'], row['Long']],
                        radius=radius_threshold_km * 1000,
                        color='gray',
                        fill=False,
                        dash_array="5"
                    ).add_to(radius_layer)
                radius_layer.add_to(m_new)

            # Outside customer layer with brand-based color
            outside_customers = cust_gdf[cust_gdf['Outside_Hub'] == True]
            outside_layer = FeatureGroup(name="Outside Customers")
            for _, row in outside_customers.iterrows():
                color = 'red' if row.get('Type', '').lower() == 'makro' else 'blue'
                folium.CircleMarker(
                    location=[row['Lat'], row['Long']],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_opacity=0.5,
                    popup=row['Customer_Code']
                ).add_to(outside_layer)
            if show_customer_markers:
                outside_layer.add_to(m_new)

            # Suggested hub layer
            suggest_layer = FeatureGroup(name="Suggested New Hubs")
            for i, (lat, lon) in enumerate(new_hub_locations):
                point = Point(lon, lat)
            
                # หา province name
                province_name = "Unknown"
                for _, prov in provinces_gdf.iterrows():
                    if point.within(prov['geometry']):
                        province_name = prov.get("pro_en", "Unknown")  # หรือเปลี่ยนชื่อคอลัมน์ตาม geojson ของคุณ
                        break
            
                # แสดง marker + popup จังหวัด
                folium.Marker(
                    location=[lat, lon],
                    popup=f"Suggest New Hub #{i+1}<br>Province: {province_name}",
                    icon=folium.Icon(color='darkgreen', icon='star', prefix='fa')
                ).add_to(suggest_layer)
            
                # วงรัศมี
                folium.Circle(
                    location=[lat, lon],
                    radius=radius_threshold_km * 1000,
                    color='darkgreen',
                    fill=True,
                    fill_opacity=0.1,
                    popup=f"Radius {radius_threshold_km} km"
                ).add_to(suggest_layer)
            
            if show_suggested_hubs:
                suggest_layer.add_to(m_new)

            # Combined heatmap
            if show_heatmap:
                heatmap_layer = FeatureGroup(name="Customer Heatmap")
                HeatMap(
                    cust_data[['Lat', 'Long']].values.tolist(),
                    radius=10,
                    gradient={0.2: '#FFE5B4', 0.6: '#FFA500', 1: '#FF8C00'}
                ).add_to(heatmap_layer)
                heatmap_layer.add_to(m_new)

            LayerControl().add_to(m_new)

        #------------------------------------------------------------------------------------------------------------------------
            
            st_folium(m_new, width=1100, height=600, key="new_hub_map", returned_objects=[], feature_group_to_add=None, center=[13.75, 100.5], zoom=6)
