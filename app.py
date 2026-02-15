import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter
import re

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="Athens Airbnb Analyst Portfolio | Le Quy Phat",
    page_icon="🇬🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. STYLE & THEME (IBCS STYLE & BRANDING) ---
IBCS_ACTUAL = "#404040"
IBCS_GOOD = "#92D050"
IBCS_BAD = "#C00000"
IBCS_HIGHLIGHT = "#FF4D00"
BG_COLOR = "#F5F7F9"
CARD_COLOR = "#FFFFFF"

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG_COLOR}; font-family: 'Segoe UI', sans-serif; }}
    section[data-testid="stSidebar"] {{ background-color: #FFFFFF; border-right: 1px solid #E0E0E0; }}
    
    /* Card Style */
    .css-card {{ background-color: {CARD_COLOR}; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-bottom: 20px; border: 1px solid #EBEBEB; }}
    
    /* Metrics Style */
    div[data-testid="stMetric"] {{ background-color: #FFFFFF; padding: 15px; border-radius: 10px; border: 1px solid #F0F0F0; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }}
    div[data-testid="stMetricLabel"] {{ font-size: 14px; color: #666; font-weight: 600; text-transform: uppercase; }}
    div[data-testid="stMetricValue"] {{ font-size: 26px; color: #333; font-weight: 700; }}
    
    /* Chart Container */
    .chart-container {{ background-color: #FFFFFF; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); border: 1px solid #E5E7EB; margin-bottom: 20px; height: 100%; }}
    .ibcs-title {{ font-size: 16px; font-weight: 700; color: #111827; margin-bottom: 5px; }}
    .ibcs-subtitle {{ font-size: 13px; color: #6B7280; margin-bottom: 15px; }}
    
    /* Section Headers */
    .section-header {{ font-size: 18px; font-weight: 800; color: #2C3E50; margin-top: 30px; margin-bottom: 15px; border-left: 5px solid {IBCS_HIGHLIGHT}; padding-left: 10px; }}
    
    /* --- COPYRIGHT FOOTER --- */
    .footer {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #FFFFFF;
        color: #555;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #E0E0E0;
        z-index: 9999;
    }}
    .footer b {{ color: {IBCS_HIGHLIGHT}; }}
</style>
""", unsafe_allow_html=True)

# --- 3. HÀM XỬ LÝ DỮ LIỆU ---
def haversine_distance(lat1, lon1, lat2, lon2):
    """Tính khoảng cách (km) giữa 2 điểm tọa độ"""
    R = 6371 
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv('Athens_Airbnb_Data.csv')
    except FileNotFoundError:
        st.warning("⚠️ Không tìm thấy file dữ liệu thực tế. Đang sử dụng Dữ liệu Giả lập (Dummy Data) để demo.")
        # Tạo dữ liệu giả lập để app không bị crash
        np.random.seed(42)
        n_rows = 500
        data = {
            'price': np.random.randint(20, 500, n_rows),
            'minimum_nights': np.random.randint(1, 10, n_rows),
            'availability_365': np.random.randint(0, 365, n_rows),
            'number_of_reviews': np.random.randint(0, 300, n_rows),
            'reviews_per_month': np.random.uniform(0, 5, n_rows),
            'latitude': np.random.uniform(37.95, 38.00, n_rows),
            'longitude': np.random.uniform(23.70, 23.75, n_rows),
            'room_type': np.random.choice(['Entire home/apt', 'Private room', 'Shared room'], n_rows),
            'neighbourhood': np.random.choice(['Plaka', 'Koukaki', 'Monastiraki', 'Syntagma', 'Exarcheia'], n_rows),
            'name': [f"Apartment {i} by Host" for i in range(n_rows)],
            'host_name': [f"Host {i}" for i in range(n_rows)],
            'calculated_host_listings_count': np.random.randint(1, 10, n_rows),
            'last_review': pd.date_range(start='1/1/2023', periods=n_rows).astype(str)
        }
        df = pd.DataFrame(data)

    # 1. Drop cột rỗng (nếu có)
    if 'neighbourhood_group' in df.columns:
        df = df.drop(columns=['neighbourhood_group'])
        
    # 2. Xử lý Missing Values
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    df['name'] = df['name'].fillna("Unknown")
    df['host_name'] = df['host_name'].fillna("Unknown")
    
    # 3. Lọc Outliers & Data sai lệch
    df = df[(df['price'] >= 10) & (df['price'] <= 800)] 
    df = df[df['minimum_nights'] <= 30]
    df = df[df['availability_365'] > 0]
    
    # 4. Feature Engineering
    ACROPOLIS_LAT = 37.9715
    ACROPOLIS_LON = 23.7257
    df['dist_to_center'] = haversine_distance(df['latitude'], df['longitude'], ACROPOLIS_LAT, ACROPOLIS_LON)
    df['name_length'] = df['name'].astype(str).apply(len)
    
    return df

# --- 4. MACHINE LEARNING ENGINE ---
@st.cache_resource
def train_model_and_evaluate(df):
    """Huấn luyện và đánh giá mô hình"""
    # Chọn features
    features = ['dist_to_center', 'minimum_nights', 'number_of_reviews', 
                'availability_365', 'calculated_host_listings_count', 'reviews_per_month']
    
    # Encode Categorical Data
    le_room = LabelEncoder()
    df['room_type_encoded'] = le_room.fit_transform(df['room_type'])
    features.append('room_type_encoded')
    
    le_neigh = LabelEncoder()
    df['neighbourhood_encoded'] = le_neigh.fit_transform(df['neighbourhood'])
    features.append('neighbourhood_encoded')
    
    X = df[features]
    y = df['price']
    
    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }
    
    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    
    return model, le_room, le_neigh, features, metrics, comparison_df

def run_kmeans(df, n_clusters=4):
    """Phân cụm vị trí và giá"""
    X = df[['latitude', 'longitude', 'price', 'dist_to_center']].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(X_scaled)

def get_keywords(text_series):
    text = " ".join(text_series.astype(str).tolist()).lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    stop_words = {'in', 'the', 'at', 'of', 'and', 'with', 'to', 'a', 'is', 'for', 'near', 'from', 'apt', 'apartment', 'athens', 'room', 'flat', 'unknown'}
    filtered = [w for w in words if w not in stop_words and len(w) > 2]
    return Counter(filtered).most_common(15)

# --- 5. GIAO DIỆN CHÍNH ---

df = load_and_clean_data()
if df.empty: st.stop()

# SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2111/2111320.png", width=50)
    st.title("ATHENS ANALYTICS")
    st.caption("Data Analyst Portfolio")
    st.markdown("---")
    
    st.subheader("🛠️ Bộ Lọc Dữ Liệu")
    neigh_filter = st.multiselect("Khu vực (Neighbourhood)", sorted(df['neighbourhood'].unique()))
    room_filter = st.multiselect("Loại phòng", df['room_type'].unique())
    
    min_price, max_price = int(df['price'].min()), int(df['price'].max())
    price_filter = st.slider("Khoảng giá (€)", min_price, max_price, (min_price, max_price))

    # --- BRANDING SECTION TRONG SIDEBAR ---
    st.markdown("---")
    st.markdown("### 👨‍💻 Author Profile")
    st.info("**Lê Quý Phát**\n\nData Scientist & Analyst")
    st.markdown("© 2026 **lequyphat**. All rights reserved.")


# Áp dụng bộ lọc
filtered_df = df.copy()
if neigh_filter: filtered_df = filtered_df[filtered_df['neighbourhood'].isin(neigh_filter)]
if room_filter: filtered_df = filtered_df[filtered_df['room_type'].isin(room_filter)]
filtered_df = filtered_df[(filtered_df['price'] >= price_filter[0]) & (filtered_df['price'] <= price_filter[1])]

# MAIN HEADER
st.title("📊 Athens Airbnb Market Analysis Dashboard")
st.markdown("""
**Mục tiêu dự án:** Phân tích toàn diện thị trường Airbnb tại Athens dưới góc độ kinh doanh và kỹ thuật.
Dự án được phát triển và tối ưu hóa bởi **Lê Quý Phát**.
""")
st.markdown("---")

# TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 Tổng Quan Thị Trường", 
    "💰 Phân Tích Giá Chuyên Sâu", 
    "📍 Phân Tích Địa Lý & NLP",
    "🤖 Machine Learning Lab"
])

# --- TAB 1: TỔNG QUAN ---
with tab1:
    st.markdown('<div class="section-header">1. Market Health Check</div>', unsafe_allow_html=True)
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Tổng Listing Active", f"{len(filtered_df):,.0f}")
    kpi2.metric("Giá trung bình (ADR)", f"€{filtered_df['price'].mean():.1f}")
    kpi3.metric("Số Review trung bình", f"{filtered_df['number_of_reviews'].mean():.0f}")
    kpi4.metric("Tỷ lệ lấp đầy (Ước tính)", f"{((365-filtered_df['availability_365'])/365*100).mean():.1f}%")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Bản đồ phân bổ Listing & Giá</div>', unsafe_allow_html=True)
        fig_map = px.scatter_mapbox(
            filtered_df, lat="latitude", lon="longitude", color="price", size="number_of_reviews",
            color_continuous_scale="Jet", zoom=10, height=450, mapbox_style="carto-positron"
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Cơ cấu loại phòng</div>', unsafe_allow_html=True)
        fig_pie = px.pie(filtered_df, names='room_type', hole=0.5, color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown(f'<div class="ibcs-title" style="margin-top:20px">Top Hosts</div>', unsafe_allow_html=True)
        top_hosts = filtered_df['host_name'].value_counts().head(5).reset_index()
        top_hosts.columns = ['Host', 'Listings']
        fig_host = px.bar(top_hosts, x='Listings', y='Host', orientation='h', color='Listings', color_continuous_scale='Blues')
        st.plotly_chart(fig_host, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: GIÁ ---
with tab2:
    st.markdown('<div class="section-header">2. Price Sensitivity Analysis</div>', unsafe_allow_html=True)
    
    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Giá theo Khu Vực (Boxplot)</div>', unsafe_allow_html=True)
        top_15_neigh = filtered_df['neighbourhood'].value_counts().head(15).index
        df_top15 = filtered_df[filtered_df['neighbourhood'].isin(top_15_neigh)]
        fig_box = px.box(df_top15, x="price", y="neighbourhood", color="neighbourhood", points="outliers", orientation='h')
        fig_box.update_layout(showlegend=False, yaxis={'categoryorder':'median ascending'})
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with p2:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Giá vs Khoảng Cách (Trend)</div>', unsafe_allow_html=True)
        fig_trend = px.scatter(filtered_df, x="dist_to_center", y="price", opacity=0.3, trendline="lowess", trendline_color_override=IBCS_HIGHLIGHT)
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: ĐỊA LÝ & NLP ---
with tab3:
    st.markdown('<div class="section-header">3. Location Clustering & NLP</div>', unsafe_allow_html=True)
    
    # K-MEANS INTEGRATION
    col_k1, col_k2 = st.columns([3, 1])
    
    # Check data volume before running KMeans
    if len(filtered_df) > 10:
        df_cluster = filtered_df.copy()
        df_cluster['Cluster'] = run_kmeans(df_cluster, n_clusters=4)
        df_cluster['Cluster'] = df_cluster['Cluster'].astype(str)
        
        with col_k1:
            st.markdown(f'<div class="chart-container"><div class="ibcs-title">Phân cụm thị trường (K-Means Clustering)</div><div class="ibcs-subtitle">Gom nhóm dựa trên Vị trí, Giá và Khoảng cách</div>', unsafe_allow_html=True)
            fig_cluster = px.scatter_mapbox(
                df_cluster, lat="latitude", lon="longitude", color="Cluster",
                hover_data=['price', 'neighbourhood'],
                zoom=10, height=500, mapbox_style="carto-positron",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_k2:
             st.markdown(f'<div class="chart-container"><div class="ibcs-title">Thống kê cụm</div>', unsafe_allow_html=True)
             cluster_stats = df_cluster.groupby('Cluster')['price'].mean().reset_index()
             st.dataframe(cluster_stats.style.format({"price": "€{:.2f}"}), hide_index=True)
             st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Cần ít nhất 10 điểm dữ liệu để chạy K-Means.")

    # NLP Keywords
    st.markdown("### 📝 Phân tích từ khóa mô tả")
    k1, k2 = st.columns(2)
    with k1:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Top từ khóa: Cao Cấp (High-end)</div>', unsafe_allow_html=True)
        high_end = filtered_df[filtered_df['price'] > filtered_df['price'].quantile(0.75)]['name']
        if not high_end.empty:
            kw_high = pd.DataFrame(get_keywords(high_end), columns=['Word', 'Count'])
            fig_k1 = px.bar(kw_high, x='Count', y='Word', orientation='h', color='Count', color_continuous_scale='Greens')
            fig_k1.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_k1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Top từ khóa: Bình Dân (Budget)</div>', unsafe_allow_html=True)
        budget = filtered_df[filtered_df['price'] < filtered_df['price'].quantile(0.25)]['name']
        if not budget.empty:
            kw_budget = pd.DataFrame(get_keywords(budget), columns=['Word', 'Count'])
            fig_k2 = px.bar(kw_budget, x='Count', y='Word', orientation='h', color='Count', color_continuous_scale='Oranges')
            fig_k2.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_k2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 4: ML LAB ---
with tab4:
    st.markdown('<div class="section-header">4. Predictive Modeling Lab</div>', unsafe_allow_html=True)
    
    # Train model (using full dataset for better training, filters apply only to analysis tabs)
    model, le_room, le_neigh, features, metrics, comparison_df = train_model_and_evaluate(df)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"€{metrics['MAE']:.2f}")
    m2.metric("RMSE", f"€{metrics['RMSE']:.2f}")
    m3.metric("R² Score", f"{metrics['R2']:.2%}")
    
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Actual vs Predicted</div>', unsafe_allow_html=True)
        fig_diag = px.scatter(comparison_df, x="Actual", y="Predicted", opacity=0.5)
        max_val = max(comparison_df.max())
        fig_diag.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="Red", width=2, dash="dash"))
        st.plotly_chart(fig_diag, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with d2:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Feature Importance</div>', unsafe_allow_html=True)
        imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
        fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', color_discrete_sequence=[IBCS_ACTUAL])
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 🔮 Dự đoán giá phòng (Live Demo)")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            inp_neigh = st.selectbox("Khu vực", le_neigh.classes_)
            inp_room = st.selectbox("Loại phòng", le_room.classes_)
        with c2:
            inp_dist = st.number_input("Khoảng cách tới TT (km)", 0.0, 20.0, 2.0)
            inp_min_nights = st.number_input("Số đêm tối thiểu", 1, 30, 2)
        with c3:
            inp_reviews = st.number_input("Số review giả định", 0, 500, 50)
            inp_avail = st.slider("Availability (ngày/năm)", 0, 365, 200)
            
        submitted = st.form_submit_button("Dự đoán ngay")
        
        if submitted:
            # Tạo DataFrame input với đúng tên cột như khi train
            input_data = pd.DataFrame({
                'dist_to_center': [inp_dist],
                'minimum_nights': [inp_min_nights],
                'number_of_reviews': [inp_reviews],
                'availability_365': [inp_avail],
                'calculated_host_listings_count': [1], # Giá trị mặc định
                'reviews_per_month': [0.5],             # Giá trị mặc định
                'room_type_encoded': [le_room.transform([inp_room])[0]],
                'neighbourhood_encoded': [le_neigh.transform([inp_neigh])[0]]
            })
            
            # Đảm bảo thứ tự cột đúng với feature list
            input_data = input_data[features]
            
            pred = model.predict(input_data)[0]
            st.success(f"💰 Mức giá khuyến nghị cho căn hộ này là: **€{pred:.2f}** / đêm")
