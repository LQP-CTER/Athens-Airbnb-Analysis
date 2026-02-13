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
    page_title="Athens Airbnb Analyst Portfolio",
    page_icon="🇬🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. STYLE & THEME (IBCS STYLE) ---
# Màu sắc chuẩn mực cho báo cáo kinh doanh
IBCS_ACTUAL = "#404040"  # Màu xám đậm cho thực tế
IBCS_GOOD = "#92D050"    # Màu xanh cho tích cực
IBCS_BAD = "#C00000"     # Màu đỏ cho tiêu cực
IBCS_HIGHLIGHT = "#FF4D00" # Màu cam tạo điểm nhấn
BG_COLOR = "#F5F7F9"
CARD_COLOR = "#FFFFFF"

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG_COLOR}; font-family: 'Segoe UI', sans-serif; }}
    section[data-testid="stSidebar"] {{ background-color: #FFFFFF; border-right: 1px solid #E0E0E0; }}
    .css-card {{ background-color: {CARD_COLOR}; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-bottom: 20px; border: 1px solid #EBEBEB; }}
    div[data-testid="stMetric"] {{ background-color: #FFFFFF; padding: 15px; border-radius: 10px; border: 1px solid #F0F0F0; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }}
    div[data-testid="stMetricLabel"] {{ font-size: 14px; color: #666; font-weight: 600; text-transform: uppercase; }}
    div[data-testid="stMetricValue"] {{ font-size: 26px; color: #333; font-weight: 700; }}
    .chart-container {{ background-color: #FFFFFF; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); border: 1px solid #E5E7EB; margin-bottom: 20px; height: 100%; }}
    .ibcs-title {{ font-size: 16px; font-weight: 700; color: #111827; margin-bottom: 5px; }}
    .ibcs-subtitle {{ font-size: 13px; color: #6B7280; margin-bottom: 15px; }}
    .section-header {{ font-size: 18px; font-weight: 800; color: #2C3E50; margin-top: 30px; margin-bottom: 15px; border-left: 5px solid {IBCS_HIGHLIGHT}; padding-left: 10px; }}
    h1, h2, h3 {{ font-family: 'Segoe UI', sans-serif; }}
</style>
""", unsafe_allow_html=True)

# --- 3. HÀM XỬ LÝ DỮ LIỆU ---
def haversine_distance(lat1, lon1, lat2, lon2):
    """Tính khoảng cách (km) giữa 2 điểm tọa độ theo công thức Haversine"""
    R = 6371  # Bán kính trái đất (km)
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
        
        # 1. Drop cột rỗng
        if 'neighbourhood_group' in df.columns:
            df = df.drop(columns=['neighbourhood_group'])
            
        # 2. Xử lý Missing Values
        df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
        df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
        df['name'] = df['name'].fillna("Unknown")
        df['host_name'] = df['host_name'].fillna("Unknown")
        
        # 3. Lọc Outliers & Data sai lệch (Data Cleaning)
        # Giá < 10 (lỗi) hoặc > 800 (biệt thự quá khủng làm lệch mô hình chung)
        df = df[(df['price'] >= 10) & (df['price'] <= 800)] 
        # Số đêm tối thiểu <= 30 (chỉ lấy short-term rental)
        df = df[df['minimum_nights'] <= 30]
        # Availability > 0 (chỉ lấy căn còn hoạt động)
        df = df[df['availability_365'] > 0]
        
        # 4. Feature Engineering (Tạo đặc trưng mới)
        # Khoảng cách đến Acropolis (Trung tâm du lịch)
        ACROPOLIS_LAT = 37.9715
        ACROPOLIS_LON = 23.7257
        df['dist_to_center'] = haversine_distance(df['latitude'], df['longitude'], ACROPOLIS_LAT, ACROPOLIS_LON)
        
        # Độ dài tiêu đề (Title Length)
        df['name_length'] = df['name'].astype(str).apply(len)
        
        return df
    except FileNotFoundError:
        st.error("LỖI: Không tìm thấy file 'Athens_Airbnb_Data.csv'. Hãy upload file vào thư mục gốc.")
        return pd.DataFrame()

# --- 4. MACHINE LEARNING ENGINE ---
@st.cache_resource
def train_model_and_evaluate(df):
    """Huấn luyện và đánh giá mô hình với đầy đủ chỉ số cho Data Analyst"""
    # Chọn features
    features = ['dist_to_center', 'minimum_nights', 'number_of_reviews', 'availability_365', 'calculated_host_listings_count', 'reviews_per_month']
    
    # Encode Categorical Data
    le_room = LabelEncoder()
    df['room_type_encoded'] = le_room.fit_transform(df['room_type'])
    features.append('room_type_encoded')
    
    le_neigh = LabelEncoder()
    df['neighbourhood_encoded'] = le_neigh.fit_transform(df['neighbourhood'])
    features.append('neighbourhood_encoded')
    
    X = df[features]
    y = df['price']
    
    # Split Train/Test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    model = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }
    
    # Dataframe so sánh thực tế vs dự báo (để vẽ chart)
    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    
    return model, le_room, le_neigh, features, metrics, comparison_df

def run_kmeans(df, n_clusters=4):
    X = df[['latitude', 'longitude', 'price', 'dist_to_center']].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(X_scaled)

def get_keywords(text_series):
    text = " ".join(text_series.astype(str).tolist()).lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    stop_words = {'in', 'the', 'at', 'of', 'and', 'with', 'to', 'a', 'is', 'for', 'near', 'from', 'apt', 'apartment', 'athens', 'room', 'flat'}
    filtered = [w for w in words if w not in stop_words and len(w) > 2]
    return Counter(filtered).most_common(15)

# --- 5. GIAO DIỆN CHÍNH ---

df = load_and_clean_data()
if df.empty: st.stop()

# SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2111/2111320.png", width=50)
    st.title("ATHENS ANALYTICS")
    st.caption("Data Analyst Portfolio Project")
    st.markdown("---")
    
    st.subheader("🛠️ Bộ Lọc Dữ Liệu")
    neigh_filter = st.multiselect("Khu vực (Neighbourhood)", sorted(df['neighbourhood'].unique()))
    room_filter = st.multiselect("Loại phòng", df['room_type'].unique())
    price_filter = st.slider("Khoảng giá (€)", int(df['price'].min()), int(df['price'].max()), (10, 200))

# Áp dụng bộ lọc
filtered_df = df.copy()
if neigh_filter: filtered_df = filtered_df[filtered_df['neighbourhood'].isin(neigh_filter)]
if room_filter: filtered_df = filtered_df[filtered_df['room_type'].isin(room_filter)]
filtered_df = filtered_df[(filtered_df['price'] >= price_filter[0]) & (filtered_df['price'] <= price_filter[1])]

# MAIN HEADER
st.title("📊 Athens Airbnb Market Analysis Dashboard")
st.markdown("""
**Mục tiêu dự án:** Phân tích các yếu tố ảnh hưởng đến giá thuê và hiệu suất kinh doanh của thị trường Airbnb tại Athens.
Sử dụng các kỹ thuật: *Exploratory Data Analysis (EDA)*, *Geospatial Analysis*, *NLP*, và *Machine Learning (Random Forest)*.
""")
st.markdown("---")

# TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 Tổng Quan Thị Trường", 
    "💰 Phân Tích Giá Chuyên Sâu", 
    "📍 Phân Tích Địa Lý & NLP",
    "🤖 Machine Learning Lab"
])

# --- TAB 1: TỔNG QUAN THỊ TRƯỜNG ---
with tab1:
    st.markdown('<div class="section-header">1. Market Health Check (Sức Khỏe Thị Trường)</div>', unsafe_allow_html=True)
    
    # KPIs Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Tổng Listing Active", f"{len(filtered_df):,.0f}", help="Số lượng căn hộ sau khi lọc")
    kpi2.metric("Giá trung bình (ADR)", f"€{filtered_df['price'].mean():.1f}", help="Average Daily Rate")
    kpi3.metric("Số Review trung bình", f"{filtered_df['number_of_reviews'].mean():.0f}", help="Mức độ phổ biến")
    kpi4.metric("Tỷ lệ lấp đầy (Proxy)", f"{((365-filtered_df['availability_365'])/365*100).mean():.1f}%", help="Ước tính dựa trên Availability")

    # Chart Row 1
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Bản đồ phân bổ Listing & Giá</div><div class="ibcs-subtitle">Kích thước = Lượng Review, Màu sắc = Giá</div>', unsafe_allow_html=True)
        fig_map = px.scatter_mapbox(
            filtered_df, lat="latitude", lon="longitude", color="price", size="number_of_reviews",
            color_continuous_scale="Jet", zoom=11, height=450, mapbox_style="carto-positron"
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Cơ cấu loại phòng (Market Share)</div>', unsafe_allow_html=True)
        fig_pie = px.pie(filtered_df, names='room_type', hole=0.5, color_discrete_sequence=px.colors.sequential.RdBu)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown(f'<div class="ibcs-title" style="margin-top:20px">Top 5 Super Hosts</div>', unsafe_allow_html=True)
        top_hosts = filtered_df['host_name'].value_counts().head(5).reset_index()
        top_hosts.columns = ['Host', 'Listings']
        fig_host = px.bar(top_hosts, x='Listings', y='Host', orientation='h', color='Listings', color_continuous_scale='Blues')
        fig_host.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_host, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Chart Row 2
    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Phân phối tình trạng phòng trống (Availability)</div><div class="ibcs-subtitle">Listing trống nhiều (365) hay full khách (0)?</div>', unsafe_allow_html=True)
        fig_avail = px.histogram(filtered_df, x="availability_365", nbins=30, color_discrete_sequence=[IBCS_ACTUAL])
        fig_avail.update_layout(xaxis_title="Số ngày trống trong năm", yaxis_title="Số lượng Listing")
        st.plotly_chart(fig_avail, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c4:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Ma trận Tương quan (Correlation Matrix)</div><div class="ibcs-subtitle">Các chỉ số ảnh hưởng lẫn nhau như thế nào?</div>', unsafe_allow_html=True)
        corr_cols = ['price', 'number_of_reviews', 'minimum_nights', 'availability_365', 'dist_to_center', 'calculated_host_listings_count']
        corr_matrix = filtered_df[corr_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: PHÂN TÍCH GIÁ ---
with tab2:
    st.markdown('<div class="section-header">2. Price Sensitivity Analysis (Phân Tích Nhạy Cảm Giá)</div>', unsafe_allow_html=True)
    
    # Row 1
    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Biểu đồ Hộp: Giá theo Khu Vực (Boxplot)</div><div class="ibcs-subtitle">Phát hiện khoảng giá và các giá trị ngoại lai (outliers)</div>', unsafe_allow_html=True)
        # Lấy top 15 khu vực đông đúc nhất để vẽ cho đỡ rối
        top_15_neigh = filtered_df['neighbourhood'].value_counts().head(15).index
        df_top15 = filtered_df[filtered_df['neighbourhood'].isin(top_15_neigh)]
        
        fig_box = px.box(df_top15, x="price", y="neighbourhood", color="neighbourhood", 
                         points="outliers", orientation='h')
        fig_box.update_layout(showlegend=False, xaxis_title="Giá (€)", yaxis={'categoryorder':'median ascending'})
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with p2:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Xu hướng Giá theo Khoảng Cách đến Acropolis</div><div class="ibcs-subtitle">Càng xa trung tâm giá càng giảm? (Trendline)</div>', unsafe_allow_html=True)
        # Binning khoảng cách để vẽ line chart cho mượt
        df_trend = filtered_df.copy()
        df_trend['dist_bin'] = pd.cut(df_trend['dist_to_center'], bins=20)
        trend_data = df_trend.groupby('dist_bin')['price'].mean().reset_index()
        trend_data['dist_center'] = trend_data['dist_bin'].apply(lambda x: x.mid)
        
        fig_trend = px.scatter(filtered_df, x="dist_to_center", y="price", opacity=0.3, color_discrete_sequence=['#cccccc'])
        fig_trend.add_traces(px.line(trend_data, x="dist_center", y="price").data[0])
        fig_trend.update_traces(line_color=IBCS_HIGHLIGHT, line_width=4, selector=dict(type='scatter', mode='lines'))
        fig_trend.update_layout(xaxis_title="Khoảng cách (km)", yaxis_title="Giá (€)")
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2
    p3, p4 = st.columns(2)
    with p3:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Phân phối Giá (Histogram)</div><div class="ibcs-subtitle">Đa số các căn hộ nằm ở mức giá nào?</div>', unsafe_allow_html=True)
        fig_hist_p = px.histogram(filtered_df, x="price", nbins=50, marginal="violin", color_discrete_sequence=[IBCS_ACTUAL])
        st.plotly_chart(fig_hist_p, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with p4:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Giá vs Số lượng Review</div><div class="ibcs-subtitle">Giá rẻ có thực sự hút nhiều review hơn?</div>', unsafe_allow_html=True)
        fig_scatter_rev = px.scatter(filtered_df, x="price", y="number_of_reviews", color="room_type", size="minimum_nights", hover_name="name")
        st.plotly_chart(fig_scatter_rev, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: ĐỊA LÝ & NLP ---
with tab3:
    st.markdown('<div class="section-header">3. Location Intelligence & NLP</div>', unsafe_allow_html=True)
    
    l1, l2 = st.columns([2, 1])
    with l1:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Treemap: Cấu trúc thị trường theo Quận</div><div class="ibcs-subtitle">Diện tích = Số lượng Listing, Màu sắc = Giá trung bình</div>', unsafe_allow_html=True)
        neigh_stats = filtered_df.groupby('neighbourhood').agg({'price':'mean', 'name':'count'}).reset_index()
        fig_tree = px.treemap(neigh_stats, path=['neighbourhood'], values='name', color='price', 
                              color_continuous_scale='RdBu_r', hover_data=['price'])
        st.plotly_chart(fig_tree, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with l2:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Độ dài tiêu đề vs Giá</div><div class="ibcs-subtitle">Viết tiêu đề dài hơn có bán giá cao hơn?</div>', unsafe_allow_html=True)
        len_price = filtered_df.groupby('name_length')['price'].mean().reset_index()
        fig_len = px.scatter(len_price, x='name_length', y='price', trendline="lowess", trendline_color_override=IBCS_HIGHLIGHT)
        st.plotly_chart(fig_len, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # NLP Keywords
    st.markdown('<div class="section-header">Phân tích từ khóa (Keyword Analysis)</div>', unsafe_allow_html=True)
    k1, k2 = st.columns(2)
    
    high_end = filtered_df[filtered_df['price'] > filtered_df['price'].quantile(0.75)]['name']
    budget = filtered_df[filtered_df['price'] < filtered_df['price'].quantile(0.25)]['name']
    
    kw_high = pd.DataFrame(get_keywords(high_end), columns=['Word', 'Count'])
    kw_budget = pd.DataFrame(get_keywords(budget), columns=['Word', 'Count'])
    
    with k1:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Top từ khóa: Phân khúc Cao Cấp (High-end)</div>', unsafe_allow_html=True)
        fig_k1 = px.bar(kw_high, x='Count', y='Word', orientation='h', color='Count', color_continuous_scale='Greens')
        fig_k1.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_k1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with k2:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Top từ khóa: Phân khúc Bình Dân (Budget)</div>', unsafe_allow_html=True)
        fig_k2 = px.bar(kw_budget, x='Count', y='Word', orientation='h', color='Count', color_continuous_scale='Oranges')
        fig_k2.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_k2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 4: MACHINE LEARNING LAB ---
with tab4:
    st.markdown('<div class="section-header">4. Predictive Modeling Lab (Phòng Thí Nghiệm ML)</div>', unsafe_allow_html=True)
    
    # Train model
    model, le_room, le_neigh, features, metrics, comparison_df = train_model_and_evaluate(df)
    
    # Model Performance Metrics
    st.markdown("### 🔍 Đánh giá hiệu suất Mô hình (Model Evaluation)")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE (Sai số tuyệt đối TB)", f"€{metrics['MAE']:.2f}", help="Trung bình mô hình lệch bao nhiêu Euro?")
    m2.metric("RMSE (Căn bậc 2 sai số TB)", f"€{metrics['RMSE']:.2f}", help="Mức phạt nặng hơn cho các sai số lớn")
    m3.metric("R² Score (Độ phù hợp)", f"{metrics['R2']:.2%}", help="Mô hình giải thích được bao nhiêu % sự biến thiên của giá")
    
    # Diagnostic Plots
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Actual vs Predicted Prices</div><div class="ibcs-subtitle">Đường chéo đỏ là dự đoán hoàn hảo. Các điểm càng gần đường đỏ càng tốt.</div>', unsafe_allow_html=True)
        fig_diag = px.scatter(comparison_df, x="Actual", y="Predicted", opacity=0.5)
        fig_diag.add_shape(type="line", x0=0, y0=0, x1=800, y1=800, line=dict(color="Red", width=2, dash="dash"))
        fig_diag.update_layout(xaxis_title="Giá thực tế", yaxis_title="Giá dự đoán")
        st.plotly_chart(fig_diag, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with d2:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Feature Importance (Tầm quan trọng biến số)</div><div class="ibcs-subtitle">Yếu tố nào quyết định giá nhà?</div>', unsafe_allow_html=True)
        imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
        fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', color_discrete_sequence=[IBCS_ACTUAL])
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Live Prediction Tool
    st.markdown("### 🔮 Công cụ Dự đoán Giá (Live Demo)")
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
            # Prepare input
            vec = pd.DataFrame([[
                inp_dist, inp_min_nights, inp_reviews, inp_avail, 1, 0.5, # host_listings, reviews_per_month (dummy)
                le_room.transform([inp_room])[0],
                le_neigh.transform([inp_neigh])[0]
            ]], columns=features)
            
            pred = model.predict(vec)[0]
            st.success(f"💰 Mức giá khuyến nghị: **€{pred:.2f}** / đêm")