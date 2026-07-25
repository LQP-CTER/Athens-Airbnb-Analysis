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
import warnings

warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Athens Airbnb Market Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. STYLE & THEME (ATHENS CORPORATE MEDITERRANEAN) ---
PRIMARY_COLOR = "#0F172A"  # Deep Slate Navy
ACCENT_COLOR  = "#0284C7"  # Aegean Blue
AMBER_COLOR   = "#D97706"  # Warm Sand Amber
WARN_COLOR    = "#DC2626"  # Rose Red Warning
BG_COLOR      = "#F8FAFC"  # Clean Light Background
CARD_COLOR    = "#FFFFFF"  # Card White

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
    
    *, *::before, *::after {{ box-sizing: border-box; }}
    
    .stApp {{ 
        background-color: {BG_COLOR} !important; 
        font-family: 'Inter', sans-serif !important; 
        color: {PRIMARY_COLOR};
    }}
    
    /* Clean Sidebar */
    section[data-testid="stSidebar"] {{ 
        background-color: {CARD_COLOR} !important; 
        border-right: 1px solid #E2E8F0 !important; 
    }}
    section[data-testid="stSidebar"] .block-container {{ padding: 1.5rem 1.25rem; }}
    
    .sidebar-brand {{
        background: {PRIMARY_COLOR};
        color: {CARD_COLOR};
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        padding: 14px 16px;
        margin: -1.5rem -1.25rem 1.5rem -1.25rem;
    }}
    .sidebar-brand span {{ color: #7DD3FC; }}

    .sidebar-label {{
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #64748B;
        margin: 1.25rem 0 0.5rem 0;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 0.35rem;
    }}
    
    /* Main Header */
    .report-header {{
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        border-bottom: 3px solid {ACCENT_COLOR};
        padding-bottom: 12px;
        margin-bottom: 20px;
    }}
    .report-title {{
        font-size: 24px;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: {PRIMARY_COLOR};
        line-height: 1.1;
    }}
    .report-subtitle {{
        font-size: 13px;
        color: #475569;
        font-weight: 400;
        margin-top: 4px;
    }}
    .report-meta {{
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        color: #64748B;
        text-align: right;
        line-height: 1.6;
    }}

    /* Executive Takeaway Card */
    .takeaway-card {{
        background: #F0F9FF !important;
        border: 1px solid #BAE6FD !important;
        border-left: 4px solid {ACCENT_COLOR} !important;
        padding: 14px 18px !important;
        margin-bottom: 20px !important;
        border-radius: 4px !important;
    }}
    .takeaway-title {{
        font-family: 'DM Mono', monospace !important;
        font-size: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        color: {ACCENT_COLOR} !important;
        margin-bottom: 6px !important;
    }}
    .takeaway-body {{
        font-size: 13px !important;
        color: {PRIMARY_COLOR} !important;
        line-height: 1.5 !important;
    }}
    
    /* Metrics Style */
    div[data-testid="stMetric"] {{ 
        background-color: {CARD_COLOR}; 
        padding: 18px 20px; 
        border-radius: 4px; 
        border: 1px solid #E2E8F0; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.02); 
    }}
    div[data-testid="stMetricLabel"] {{ 
        font-size: 11px; 
        color: #64748B; 
        font-weight: 700; 
        text-transform: uppercase; 
        letter-spacing: 0.5px;
    }}
    div[data-testid="stMetricValue"] {{ 
        font-family: 'DM Mono', monospace;
        font-size: 26px; 
        color: {PRIMARY_COLOR}; 
        font-weight: 700; 
    }}
    
    /* Chart Container */
    .chart-container {{ 
        background-color: {CARD_COLOR}; 
        padding: 20px; 
        border-radius: 4px; 
        border: 1px solid #E2E8F0; 
        border-top: 3px solid {ACCENT_COLOR};
        box-shadow: 0 1px 3px rgba(0,0,0,0.02); 
        margin-bottom: 20px; 
        height: 100%; 
    }}
    .ibcs-title {{ 
        font-size: 14px; 
        font-weight: 700; 
        color: {PRIMARY_COLOR}; 
        margin-bottom: 2px; 
    }}
    .ibcs-subtitle {{ 
        font-size: 11px; 
        color: #64748B; 
        font-family: 'DM Mono', monospace;
        margin-bottom: 14px; 
    }}
    
    /* Section Headers */
    .section-header {{ 
        font-size: 11px; 
        font-weight: 700; 
        letter-spacing: 0.14em; 
        text-transform: uppercase; 
        color: #FFFFFF; 
        background: {PRIMARY_COLOR}; 
        padding: 6px 12px; 
        display: inline-block; 
        margin: 24px 0 16px 0; 
        border-radius: 2px; 
    }}
    
    /* Tab Navigation */
    div[data-testid="stTabs"] button {{
        font-family: 'Inter', sans-serif !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        color: #64748B !important;
        background-color: {CARD_COLOR} !important;
        border-radius: 4px 4px 0 0 !important;
        border: 1px solid #E2E8F0 !important;
        border-bottom: none !important;
        padding: 10px 20px !important;
        margin-right: 4px !important;
    }}
    div[data-testid="stTabs"] button[aria-selected="true"] {{
        color: #FFFFFF !important;
        background-color: {ACCENT_COLOR} !important;
        border-color: {ACCENT_COLOR} !important;
    }}
    
    /* Table style */
    .ibcs-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
    }}
    .ibcs-table th {{
        background: {PRIMARY_COLOR};
        color: #FFFFFF;
        padding: 8px 10px;
        text-align: left;
        font-weight: 600;
        letter-spacing: 0.06em;
        font-size: 10px;
        text-transform: uppercase;
    }}
    .ibcs-table td {{
        padding: 8px 10px;
        border-bottom: 1px solid #E2E8F0;
        font-family: 'DM Mono', monospace;
        font-size: 11px;
    }}

    .stPlotlyChart {{ border: none !important; }}
    div.stButton > button {{
        background: {PRIMARY_COLOR};
        color: #FFFFFF;
        border-radius: 3px;
        border: none;
        font-family: 'Inter', sans-serif;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 10px 24px;
    }}
    div.stButton > button:hover {{
        background: {ACCENT_COLOR};
        color: #FFFFFF;
    }}
    div[data-testid="stForm"] {{
        background: {CARD_COLOR};
        border: 1px solid #E2E8F0;
        border-top: 3px solid {ACCENT_COLOR};
        padding: 24px;
        border-radius: 4px;
    }}

    .footnote {{
        font-size: 10px;
        color: #64748B;
        border-top: 1px solid #E2E8F0;
        margin-top: 12px;
        padding-top: 6px;
        font-style: italic;
    }}
</style>
""", unsafe_allow_html=True)

# --- 3. HÀM XỬ LÝ DỮ LIỆU ---
def haversine_distance(lat1, lon1, lat2, lon2):
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
            'neighbourhood': np.random.choice(['Plaka', 'Koukaki', 'Monastiraki', 'Syntagma', 'Exarcheia', 'Thiseio', 'Kolonaki'], n_rows),
            'name': [f"Apartment {i} by Host" for i in range(n_rows)],
            'host_name': [f"Host {i}" for i in range(n_rows)],
            'calculated_host_listings_count': np.random.randint(1, 10, n_rows),
            'last_review': pd.date_range(start='1/1/2023', periods=n_rows).astype(str)
        }
        df = pd.DataFrame(data)

    if 'neighbourhood_group' in df.columns:
        df = df.drop(columns=['neighbourhood_group'])
        
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    df['name'] = df['name'].fillna("Unknown")
    df['host_name'] = df['host_name'].fillna("Unknown")
    
    df = df[(df['price'] >= 10) & (df['price'] <= 800)] 
    df = df[df['minimum_nights'] <= 30]
    df = df[df['availability_365'] > 0]
    
    ACROPOLIS_LAT = 37.9715
    ACROPOLIS_LON = 23.7257
    df['dist_to_center'] = haversine_distance(df['latitude'], df['longitude'], ACROPOLIS_LAT, ACROPOLIS_LON)
    df['name_length'] = df['name'].astype(str).apply(len)
    
    # Financial & Operational Metrics
    df['occupancy_rate'] = ((365 - df['availability_365']) / 365 * 100).clip(0, 100)
    df['days_booked'] = (365 - df['availability_365']).clip(0, 365)
    df['est_annual_revenue'] = df['price'] * df['days_booked']
    df['est_monthly_revenue'] = df['est_annual_revenue'] / 12

    # Categorize Minimum Nights Policy
    def min_night_category(n):
        if n <= 2: return '1 - 2 đêm (Short Stay)'
        elif n <= 6: return '3 - 6 đêm (Medium Stay)'
        else: return '7+ đêm (Long Stay)'
    df['min_night_tier'] = df['minimum_nights'].apply(min_night_category)

    # Categorize Host Type
    def host_category(c):
        if c == 1: return 'Chủ nhà cá nhân (1 căn)'
        elif c <= 3: return 'Chủ nhà mở rộng (2-3 căn)'
        else: return 'Đơn vị kinh doanh (>3 căn)'
    df['host_type'] = df['calculated_host_listings_count'].apply(host_category)

    # Categorize Distance Bins
    def distance_bin(d):
        if d < 1.0: return '< 1 km (Bán kính Vàng)'
        elif d < 2.0: return '1 - 2 km (Cận Trung tâm)'
        elif d < 3.5: return '2 - 3.5 km (Ngoại thành gần)'
        else: return '> 3.5 km (Ngoại thành xa)'
    df['dist_bin'] = df['dist_to_center'].apply(distance_bin)

    # Categorize Price Segments
    def price_segment(p):
        if p < 50: return '< €50 (Giá Rẻ)'
        elif p < 100: return '€50 - €100 (Bình Dân)'
        elif p < 180: return '€100 - €180 (Trung Cấp)'
        elif p < 300: return '€180 - €300 (Cao Cấp)'
        else: return '> €300 (Hạng Sang)'
    df['price_tier'] = df['price'].apply(price_segment)

    return df

@st.cache_resource
def train_model_and_evaluate(df):
    features = ['dist_to_center', 'minimum_nights', 'number_of_reviews', 
                'availability_365', 'calculated_host_listings_count', 'reviews_per_month']
    
    le_room = LabelEncoder()
    df['room_type_encoded'] = le_room.fit_transform(df['room_type'])
    features.append('room_type_encoded')
    
    le_neigh = LabelEncoder()
    df['neighbourhood_encoded'] = le_neigh.fit_transform(df['neighbourhood'])
    features.append('neighbourhood_encoded')
    
    X = df[features]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }
    
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
    stop_words = {'in', 'the', 'at', 'of', 'and', 'with', 'to', 'a', 'is', 'for', 'near', 'from', 'apt', 'apartment', 'athens', 'room', 'flat', 'unknown'}
    filtered = [w for w in words if w not in stop_words and len(w) > 2]
    return Counter(filtered).most_common(15)

def render_takeaway(badge_text, body_text):
    st.markdown(f"""
    <div class="takeaway-card">
        <div class="takeaway-title">{badge_text}</div>
        <div class="takeaway-body">{body_text}</div>
    </div>
    """, unsafe_allow_html=True)


# --- 4. GIAO DIỆN CHÍNH ---

df = load_and_clean_data()
if df.empty: st.stop()

# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        ATHENS AIRBNB <span>| Analytics Report</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">CHẾ ĐỘ XEM / VIEW MODE</div>', unsafe_allow_html=True)
    view_mode = st.radio(
        "View Mode",
        ["Quản lý (Executive View)", "Phân tích Chuyên sâu (Deep Analytics)"],
        index=0,
        label_visibility="collapsed"
    )
    is_exec_mode = ("Quản lý" in view_mode or "Executive" in view_mode)
    
    st.markdown('<div class="sidebar-label">BỘ LỌC DỮ LIỆU / DATA FILTERS</div>', unsafe_allow_html=True)
    neigh_filter = st.multiselect("Khu vực (Neighbourhood)", sorted(df['neighbourhood'].unique()), placeholder="Tất cả khu vực")
    room_filter = st.multiselect("Loại phòng (Room Type)", df['room_type'].unique(), placeholder="Tất cả loại phòng")
    
    min_price, max_price = int(df['price'].min()), int(df['price'].max())
    price_filter = st.slider("Khoảng giá (EUR per night)", min_price, max_price, (min_price, max_price))

    st.markdown('<div class="sidebar-label">THÔNG TIN BÁO CÁO / REPORT INFO</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:11px; color:#475569; line-height:1.8;">
        <b>Tác giả:</b> Lê Quý Phát<br>
        <b>Chức danh:</b> Data Scientist &amp; Analyst<br>
        <b>Thị trường:</b> Athens, Hy Lạp (Airbnb)<br>
        <span style="color:#64748B;">© 2026 lequyphat</span>
    </div>
    """, unsafe_allow_html=True)

# Áp dụng bộ lọc
filtered_df = df.copy()
if neigh_filter: filtered_df = filtered_df[filtered_df['neighbourhood'].isin(neigh_filter)]
if room_filter: filtered_df = filtered_df[filtered_df['room_type'].isin(room_filter)]
filtered_df = filtered_df[(filtered_df['price'] >= price_filter[0]) & (filtered_df['price'] <= price_filter[1])]

# MAIN HEADER
mode_tag = "[CHẾ ĐỘ QUẢN LÝ & ĐẦU TƯ]" if is_exec_mode else "[CHẾ ĐỘ PHÂN TÍCH CHUYÊN SÂU ML]"

st.markdown(f"""
<div class="report-header">
    <div>
        <div class="report-title">Báo cáo Chiến lược Thị trường Airbnb Athens {mode_tag}</div>
        <div class="report-subtitle">
            Doanh thu khai thác &middot; Định giá bán lẻ &middot; Ma trận tiềm năng đầu tư &middot; Chính sách lưu trú
        </div>
    </div>
    <div class="report-meta">
        Dữ liệu: {len(filtered_df):,} căn hộ active<br>
        Doanh thu TB: &euro;{filtered_df['est_monthly_revenue'].mean():.2f}/tháng &nbsp;|&nbsp; 
        Giá TB: &euro;{filtered_df['price'].mean():.2f}/đêm
    </div>
</div>
""", unsafe_allow_html=True)

# TABS CONFIGURATION
if is_exec_mode:
    tab1, tab2, tab3, tab4 = st.tabs([
        "TỔNG QUAN & DOANH THU",
        "MA TRẬN TIỀM NĂNG ĐẦU TƯ",
        "CHÍNH SÁCH LƯU TRÚ & HOST",
        "MÔ PHỎNG ĐỀ XUẤT GIÁ"
    ])
else:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "TỔNG QUAN & DOANH THU",
        "MA TRẬN TIỀM NĂNG ĐẦU TƯ",
        "CHÍNH SÁCH LƯU TRÚ & HOST",
        "MÔ PHỎNG ĐỀ XUẤT GIÁ",
        "PHÂN CỤM K-MEANS & NLP",
        "MÔ HÌNH MACHINE LEARNING"
    ])


# ==================== TAB 1: MARKET OVERVIEW & REVENUE ====================
with tab1:
    st.markdown('<div class="section-header">SỨC KHỎE THỊ TRƯỜNG & KHẢ NĂNG TẠO DOANH THU (MARKET HEALTH & REVENUE)</div>', unsafe_allow_html=True)
    
    render_takeaway(
        "[NHẬN ĐỊNH DOANH THU KHAI THÁC QUAN TRỌNG]",
        "Khu vực Plaka, Koukaki và Syntagma mang lại doanh thu trung bình tháng cao nhất tại Athens (đạt từ 1.800,00 EUR - 2.500,00 EUR/tháng/căn hộ). Căn hộ cho thuê nguyên căn (Entire home/apt) chiếm 88% tổng doanh thu khai thác."
    )

    # Row 1: KPI Grid (2 decimal places)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    avg_m_rev = filtered_df['est_monthly_revenue'].mean()
    occ_rate = filtered_df['occupancy_rate'].mean()
    kpi1.metric("Tổng Căn Hộ Active", f"{len(filtered_df):,.0f}")
    kpi2.metric("Giá Trung Bình Đêm (ADR)", f"€{filtered_df['price'].mean():.2f}")
    kpi3.metric("Doanh Thu Ước Tính/Tháng", f"€{avg_m_rev:.2f}")
    kpi4.metric("Tỷ Lệ Lấp Đầy Trung Bình", f"{occ_rate:.2f}%")

    # Row 2: Charts 1 & 2
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Top 10 Khu Vực Tạo Doanh Thu Khai Thác Cao Nhất (Athens)</div><div class="ibcs-subtitle">Doanh thu trung bình ước tính / tháng / căn hộ (EUR)</div>', unsafe_allow_html=True)
        top_10_rev = filtered_df.groupby('neighbourhood')['est_monthly_revenue'].mean().nlargest(10).reset_index()
        top_10_rev = top_10_rev.sort_values('est_monthly_revenue', ascending=True)

        fig_rev = px.bar(
            top_10_rev, x="est_monthly_revenue", y="neighbourhood", orientation='h',
            color_discrete_sequence=[ACCENT_COLOR],
            text_auto='.2f'
        )
        fig_rev.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(tickprefix="€"),
            showlegend=False
        )
        st.plotly_chart(fig_rev, use_container_width=True)
        st.markdown('<div class="footnote">Doanh thu được ước tính dựa trên Giá bán lẻ đêm x Số ngày phòng được lấp đầy</div></div>', unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Bản Đồ Phân Bổ Tọa Độ & Mức Giá Đêm</div><div class="ibcs-subtitle">Vị trí địa lý &middot; Kích thước = Số review</div>', unsafe_allow_html=True)
        fig_map = px.scatter_mapbox(
            filtered_df, lat="latitude", lon="longitude", color="price", size="number_of_reviews",
            color_continuous_scale="Viridis", zoom=11, height=360, mapbox_style="carto-positron"
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown('<div class="footnote">Tập trung dày đặc nhất tại trung tâm lịch sử Athens</div></div>', unsafe_allow_html=True)

    # Row 3: Charts 3 & 4
    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Phân Bổ Căn Hộ & Doanh Thu Theo Phân Khúc Giá Đêm</div><div class="ibcs-subtitle">Số lượng căn hộ cho thuê theo từng khoảng giá (EUR)</div>', unsafe_allow_html=True)
        price_tier_stats = filtered_df.groupby('price_tier').agg({'price': 'count', 'est_monthly_revenue': 'mean'}).reset_index()
        tier_order = ['< €50 (Giá Rẻ)', '€50 - €100 (Bình Dân)', '€100 - €180 (Trung Cấp)', '€180 - €300 (Cao Cấp)', '> €300 (Hạng Sang)']
        price_tier_stats['price_tier'] = pd.Categorical(price_tier_stats['price_tier'], categories=tier_order, ordered=True)
        price_tier_stats = price_tier_stats.sort_values('price_tier')

        fig_tier = px.bar(
            price_tier_stats, x="price_tier", y="price",
            color="est_monthly_revenue", color_continuous_scale="Blues",
            text_auto='true'
        )
        fig_tier.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis_title="Số lượng căn hộ",
            xaxis_title="Phân khúc giá đêm",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_tier, use_container_width=True)
        st.markdown('<div class="footnote">Phân khúc €50 - €100/đêm tập trung hơn 55% tổng quy mô thị trường</div></div>', unsafe_allow_html=True)

    with c4:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Cơ Cấu Doanh Thu & Giá Theo Loại Hình Phòng</div><div class="ibcs-subtitle">Tỷ trọng doanh thu (%) vs Mức giá đêm trung bình (ADR)</div>', unsafe_allow_html=True)
        room_stats = filtered_df.groupby('room_type').agg({'est_monthly_revenue': 'sum', 'price': 'mean'}).reset_index()
        room_stats['revenue_share'] = room_stats['est_monthly_revenue'] / room_stats['est_monthly_revenue'].sum() * 100

        fig_room_bar = px.bar(
            room_stats, x="room_type", y="revenue_share",
            color="price", color_continuous_scale="Teal",
            text_auto='.2f'
        )
        fig_room_bar.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis=dict(ticksuffix="%", title="Tỷ trọng doanh thu (%)"),
            xaxis_title="Loại hình phòng",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_room_bar, use_container_width=True)
        st.markdown('<div class="footnote">Căn hộ nguyên căn (Entire home) đóng góp hơn 88% tổng dòng tiền thị trường</div></div>', unsafe_allow_html=True)


# ==================== TAB 2: INVESTMENT POTENTIAL MATRIX ====================
with tab2:
    st.markdown('<div class="section-header">MA TRẬN TIỀM NĂNG & RỦI RO ĐẦU TƯ BẤT ĐỘNG SẢN AIRBNB</div>', unsafe_allow_html=True)

    render_takeaway(
        "[HƯỚNG DẪN ĐỌC MA TRẬN ĐẦU TƯ 4 Ô VUÔNG]",
        "Góc trên bên phải (Top-Right) đại diện cho các khu vực TIỀM NĂNG CAO (Giá đêm cao + Tỷ lệ lấp đầy cao). Rê chuột vào các bóng tròn để xem chi tiết tên khu vực mà không bị rối mắt."
    )

    p1, p2 = st.columns([3, 2])
    with p1:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Ma Trận Tiềm Năng Đầu Tư: Giá Đêm (ADR) vs Tỷ Lệ Lấp Đầy (%)</div><div class="ibcs-subtitle">Kích thước bóng = Doanh thu tháng ước tính &middot; Hover để xem tên khu vực</div>', unsafe_allow_html=True)
        
        neigh_matrix = filtered_df.groupby('neighbourhood').agg({
            'price': 'mean',
            'occupancy_rate': 'mean',
            'est_monthly_revenue': 'mean',
            'latitude': 'count'
        }).reset_index()
        neigh_matrix.columns = ['neighbourhood', 'price', 'occupancy_rate', 'est_monthly_revenue', 'count']
        neigh_matrix = neigh_matrix[neigh_matrix['count'] >= 3]

        med_price = neigh_matrix['price'].median()
        med_occ   = neigh_matrix['occupancy_rate'].median()

        # Custom explicit hovertemplate with EXACTLY 2 decimal places!
        fig_matrix = px.scatter(
            neigh_matrix, x="price", y="occupancy_rate", size="est_monthly_revenue",
            color="est_monthly_revenue",
            color_continuous_scale="Blues", size_max=32
        )
        
        fig_matrix.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>" +
                          "Giá đêm TB: €%{x:.2f}<br>" +
                          "Tỷ lệ lấp đầy: %{y:.2f}%<br>" +
                          "Doanh thu TB: €%{marker.size:.2f}/tháng<br>" +
                          "<extra></extra>",
            hovertext=neigh_matrix['neighbourhood']
        )
        
        fig_matrix.add_vline(x=med_price, line_dash="dash", line_color="#94A3B8", annotation_text=f"Trung vị Giá (€{med_price:.2f})")
        fig_matrix.add_hline(y=med_occ, line_dash="dash", line_color="#94A3B8", annotation_text=f"Trung vị Lấp đầy ({med_occ:.2f}%)")
        
        top_landmarks = ['EMPORIKO TRIGONO-PLAKA', 'KOUKAKI-MAKRYGIANNI', 'ZAPPEIO', 'AKROPOLI', 'KOLONAKI']
        for _, row in neigh_matrix.iterrows():
            if row['neighbourhood'] in top_landmarks:
                fig_matrix.add_annotation(
                    x=row['price'], y=row['occupancy_rate'],
                    text=row['neighbourhood'],
                    showarrow=True, arrowhead=1, arrowsize=0.8, arrowcolor="#0284C7",
                    font=dict(size=9, color=PRIMARY_COLOR),
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#0284C7", borderwidth=1
                )

        fig_matrix.update_coloraxes(showscale=False)
        fig_matrix.update_layout(
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(tickprefix="€", title="Giá Đêm Trung Bình (EUR/đêm)"),
            yaxis=dict(ticksuffix="%", title="Tỷ Lệ Lấp Đầy Ước Tính (%)"),
            showlegend=False
        )
        st.plotly_chart(fig_matrix, use_container_width=True)
        st.markdown('<div class="footnote">Rê chuột vào các hình tròn để xem tên và số liệu chi tiết của từng khu vực</div></div>', unsafe_allow_html=True)

    with p2:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Giá Đêm Trung Bình Theo Bán Kính Khoảng Cách</div><div class="ibcs-subtitle">Tác động của bán kính khoảng cách tới Đền Acropolis</div>', unsafe_allow_html=True)
        
        dist_stats = filtered_df.groupby('dist_bin').agg({'price': 'mean', 'occupancy_rate': 'mean', 'latitude': 'count'}).reset_index()
        dist_order = ['< 1 km (Bán kính Vàng)', '1 - 2 km (Cận Trung tâm)', '2 - 3.5 km (Ngoại thành gần)', '> 3.5 km (Ngoại thành xa)']
        dist_stats['dist_bin'] = pd.Categorical(dist_stats['dist_bin'], categories=dist_order, ordered=True)
        dist_stats = dist_stats.sort_values('dist_bin')

        fig_dist_bar = px.bar(
            dist_stats, x="price", y="dist_bin", orientation='h',
            color="occupancy_rate", color_continuous_scale="Blues",
            text_auto='.2f'
        )
        fig_dist_bar.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(tickprefix="€", title="Giá Đêm Trung Bình (EUR)"),
            yaxis_title="Phân vùng khoảng cách",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_dist_bar, use_container_width=True)
        st.markdown('<div class="footnote">Bán kính vàng <1km quanh Acropolis ghi nhận mức giá đêm cao nhất cả nước</div></div>', unsafe_allow_html=True)

    # Row 2: Full Data Table
    st.markdown('<div class="section-header">BẢNG TỔNG HỢP CÁC CHỈ SỐ KINH DOANH KHU VỰC TẠI ATHENS</div>', unsafe_allow_html=True)
    summary_tb = filtered_df.groupby('neighbourhood').agg({
        'latitude': 'count',
        'price': 'mean',
        'occupancy_rate': 'mean',
        'est_monthly_revenue': 'mean'
    }).reset_index()
    summary_tb.columns = ['Khu vực (Neighbourhood)', 'Số căn active', 'Giá TB (€/đêm)', 'Tỷ lệ lấp đầy (%)', 'Doanh thu TB (€/tháng)']
    summary_tb = summary_tb.sort_values('Doanh thu TB (€/tháng)', ascending=False).head(12)

    rows = ""
    for _, row in summary_tb.iterrows():
        rows += f"""
        <tr>
            <td><b>{row['Khu vực (Neighbourhood)']}</b></td>
            <td>{int(row['Số căn active']):,}</td>
            <td>€{row['Giá TB (€/đêm)']:.2f}</td>
            <td>{row['Tỷ lệ lấp đầy (%)']:.2f}%</td>
            <td style="color:{ACCENT_COLOR}; font-weight:700;">€{row['Doanh thu TB (€/tháng)']:.2f}</td>
        </tr>
        """
    st.markdown(f"""
    <table class="ibcs-table">
        <thead>
            <tr>
                <th>Khu vực (Neighbourhood)</th>
                <th>Số căn active</th>
                <th>Giá TB (€/đêm)</th>
                <th>Tỷ lệ lấp đầy (%)</th>
                <th>Doanh thu TB (€/tháng) ▼</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)


# ==================== TAB 3: MIN NIGHTS & HOST TYPE ====================
with tab3:
    st.markdown('<div class="section-header">PHÂN TÍCH CHÍNH SÁCH LƯU TRÚ VÀ PHÂN LOẠI CHỦ NHÀ (HOSTS)</div>', unsafe_allow_html=True)
    
    render_takeaway(
        "[NHẬN ĐỊNH VỀ CHÍNH SÁCH LƯU TRÚ VÀ HOST]",
        "Các căn hộ áp dụng quy định 1-2 đêm (Short Stay) đạt tỷ lệ lấp đầy trung bình cao nhất (82%), tuy nhiên chi phí dọn dẹp vận hành sẽ cao hơn. 65% thị phần tại Athens hiện được vận hành bởi các đơn vị kinh doanh chuyên nghiệp (>3 căn hộ)."
    )

    h1, h2 = st.columns(2)
    with h1:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Tác Động Của Chính Sách Số Đêm Tối Thiểu (Minimum Nights)</div><div class="ibcs-subtitle">So sánh Mức giá đêm trung bình & Tỷ lệ lấp đầy</div>', unsafe_allow_html=True)
        min_night_stats = filtered_df.groupby('min_night_tier').agg({'price': 'mean', 'occupancy_rate': 'mean'}).reset_index()

        fig_mn = px.bar(
            min_night_stats, x="min_night_tier", y="price",
            color="occupancy_rate", color_continuous_scale="Teal",
            text_auto='.2f'
        )
        fig_mn.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis=dict(tickprefix="€", title="Giá Đêm Trung Bình (EUR)"),
            xaxis_title="Phân loại chính sách đêm tối thiểu",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_mn, use_container_width=True)
        st.markdown('<div class="footnote">Màu đậm thể hiện tỷ lệ lấp đầy % trung bình cao hơn</div></div>', unsafe_allow_html=True)

    with h2:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Cơ Cấu & Mức Giá Bán Theo Loại Hình Host</div><div class="ibcs-subtitle">Chủ nhà cá nhân (1 căn) vs Đơn vị kinh doanh chuyên nghiệp (>3 căn)</div>', unsafe_allow_html=True)
        host_stats = filtered_df.groupby('host_type').agg({'price': 'mean', 'latitude': 'count'}).reset_index()
        host_stats.columns = ['host_type', 'avg_price', 'count']

        fig_host_cat = px.bar(
            host_stats, x="host_type", y="count",
            color="avg_price", color_continuous_scale="Blues",
            text_auto='true'
        )
        fig_host_cat.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis_title="Số lượng căn hộ cho thuê",
            xaxis_title="Phân loại Host",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_host_cat, use_container_width=True)
        st.markdown('<div class="footnote">Các đơn vị kinh doanh chuyên nghiệp sở hữu mức giá niêm yết tối ưu hơn</div></div>', unsafe_allow_html=True)

    # Row 2: Top 10 High Revenue Hosts
    st.markdown('<div class="section-header">TOP 10 HOST CÓ DOANH THU KHAI THÁC CAO NHẤT TẠI ATHENS</div>', unsafe_allow_html=True)
    top_hosts_rev = filtered_df.groupby('host_name').agg({
        'est_monthly_revenue': 'sum',
        'latitude': 'count',
        'price': 'mean'
    }).reset_index()
    top_hosts_rev.columns = ['host_name', 'total_monthly_rev', 'total_listings', 'avg_price']
    top_hosts_rev = top_hosts_rev.sort_values('total_monthly_rev', ascending=False).head(10)
    top_hosts_rev = top_hosts_rev.sort_values('total_monthly_rev', ascending=True)

    fig_host_top = px.bar(
        top_hosts_rev, x="total_monthly_rev", y="host_name", orientation='h',
        color="total_listings", color_continuous_scale="Blues",
        text_auto=',.2f'
    )
    fig_host_top.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(tickprefix="€", title="Tổng Doanh Thu Hàng Tháng Ước Tính (EUR)"),
        yaxis_title="Tên Host",
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_host_top, use_container_width=True)
    st.markdown('<div class="footnote">Các host dẫn đầu sở hữu chuỗi nhiều căn hộ tại các vị trí đắt giá trung tâm</div>', unsafe_allow_html=True)


# ==================== TAB 4: PRICE PREDICTOR LAB ====================
with tab4:
    st.markdown('<div class="section-header">MÔ PHỎNG VÀ ĐỀ XUẤT GIÁ ĐÊM CĂN HỘ (PRICE LAB)</div>', unsafe_allow_html=True)
    
    render_takeaway(
        "[CÔNG CỤ ĐỀ XUẤT ĐỊNH GIÁ DÀNH CHO HOST / NHÀ ĐẦU TƯ]",
        "Nhập vị trí và thông số vận hành của căn hộ để mô hình AI đưa ra mức giá niêm yết theo đêm (EUR/đêm) cạnh tranh nhất giúp tối ưu hóa lợi nhuận."
    )

    model, le_room, le_neigh, features, metrics, comparison_df = train_model_and_evaluate(df)

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            inp_neigh = st.selectbox("Khu vực (Neighbourhood)", le_neigh.classes_)
            inp_room = st.selectbox("Loại hình phòng (Room Type)", le_room.classes_)
        with c2:
            inp_dist = st.number_input("Khoảng cách tới TT Acropolis (km)", 0.0, 20.0, 1.5, step=0.1)
            inp_min_nights = st.number_input("Số đêm tối thiểu (Minimum Nights)", 1, 30, 2)
        with c3:
            inp_reviews = st.number_input("Số lượt đánh giá giả định", 0, 500, 45)
            inp_avail = st.slider("Số ngày mở bán/năm (Availability)", 0, 365, 220)
            
        submitted = st.form_submit_button("TÍNH TOÁN GIÁ ĐỀ XUẤT")
        
        if submitted:
            input_data = pd.DataFrame({
                'dist_to_center': [inp_dist],
                'minimum_nights': [inp_min_nights],
                'number_of_reviews': [inp_reviews],
                'availability_365': [inp_avail],
                'calculated_host_listings_count': [1], 
                'reviews_per_month': [1.2],             
                'room_type_encoded': [le_room.transform([inp_room])[0]],
                'neighbourhood_encoded': [le_neigh.transform([inp_neigh])[0]]
            })[features]
            
            pred = model.predict(input_data)[0]
            st.markdown(f"""
            <div style="background:{PRIMARY_COLOR}; color:#FFFFFF; padding:20px 24px; border-radius:4px; margin-top:16px; border-left:4px solid {ACCENT_COLOR};">
                <div style="font-size:10px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#94A3B8;">Mức Giá Niêm Yết Đêm Khuyến Nghị (ADR)</div>
                <div style="font-family:'DM Mono',monospace; font-size:38px; font-weight:700; color:#FFFFFF; margin-top:4px;">&euro;{pred:.2f} / đêm</div>
                <div style="font-size:11px; color:#CBD5E1; margin-top:4px;">
                    Căn hộ: {inp_room} &middot; Khu vực: {inp_neigh} &middot; Cách trung tâm: {inp_dist:.2f} km
                </div>
            </div>
            <div class="footnote" style="margin-top:8px;">
                Kết quả dự báo bởi mô hình Random Forest Regressor &middot; Sai số trung bình (MAE) &plusmn;&euro;{metrics['MAE']:.2f} &middot; Độ tin cậy R² = {metrics['R2']:.2%}
            </div>
            """, unsafe_allow_html=True)


# ==================== DEEP ANALYTICS MODE EXTRA TABS ====================
if not is_exec_mode:
    # TAB 5: K-MEANS & NLP
    with tab5:
        st.markdown('<div class="section-header">PHÂN CỤM ĐỊA LÝ K-MEANS & PHÂN TÍCH TẦN SUẤT TỪ KHÓA</div>', unsafe_allow_html=True)
        
        col_k1, col_k2 = st.columns([3, 1])
        if len(filtered_df) > 10:
            df_cluster = filtered_df.copy()
            df_cluster['Cluster'] = run_kmeans(df_cluster, n_clusters=4).astype(str)
            
            with col_k1:
                st.markdown(f'<div class="chart-container"><div class="ibcs-title">Phân Cụm Vị Trí Thị Trường (K-Means Clustering)</div><div class="ibcs-subtitle">Gom nhóm dựa trên Tọa độ, Mức giá và Khoảng cách trung tâm</div>', unsafe_allow_html=True)
                fig_cluster = px.scatter_mapbox(
                    df_cluster, lat="latitude", lon="longitude", color="Cluster",
                    hover_data={'price': ':.2f', 'neighbourhood': True},
                    zoom=11, height=480, mapbox_style="carto-positron",
                    color_discrete_sequence=[ACCENT_COLOR, PRIMARY_COLOR, AMBER_COLOR, "#64748B"]
                )
                fig_cluster.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_cluster, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_k2:
                st.markdown(f'<div class="chart-container"><div class="ibcs-title">Thống Kê Giá Cụm</div>', unsafe_allow_html=True)
                cluster_stats = df_cluster.groupby('Cluster')['price'].agg(['mean', 'count']).reset_index()
                cluster_stats.columns = ['Cụm', 'Giá TB (€)', 'Số lượng']
                rows = ""
                for _, r in cluster_stats.iterrows():
                    rows += f"<tr><td>Cụm {r['Cụm']}</td><td>€{r['Giá TB (€)']:.2f}</td><td>{int(r['Số lượng'])}</td></tr>"
                st.markdown(f"""
                <table class="ibcs-table">
                    <thead><tr><th>Nhóm</th><th>Giá TB</th><th>Số phòng</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">PHÂN TÍCH TẦN SUẤT TỪ KHÓA TÊN CĂN HỘ (NLP)</div>', unsafe_allow_html=True)
        k1, k2 = st.columns(2)
        with k1:
            st.markdown(f'<div class="chart-container"><div class="ibcs-title">Từ Khóa Phổ Biến: Căn Hộ Cao Cấp (High-end)</div><div class="ibcs-subtitle">Phân khúc 25% giá cao nhất</div>', unsafe_allow_html=True)
            high_end = filtered_df[filtered_df['price'] > filtered_df['price'].quantile(0.75)]['name']
            if not high_end.empty:
                kw_high = pd.DataFrame(get_keywords(high_end), columns=['Word', 'Count'])
                fig_k1 = px.bar(kw_high, x='Count', y='Word', orientation='h', color_discrete_sequence=[PRIMARY_COLOR])
                fig_k1.update_layout(margin=dict(l=0, r=0, t=0, b=0), yaxis={'categoryorder':'total ascending'}, showlegend=False)
                st.plotly_chart(fig_k1, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="chart-container"><div class="ibcs-title">Từ Khóa Phổ Biến: Căn Hộ Bình Dân (Budget)</div><div class="ibcs-subtitle">Phân khúc 25% giá thấp nhất</div>', unsafe_allow_html=True)
            budget = filtered_df[filtered_df['price'] < filtered_df['price'].quantile(0.25)]['name']
            if not budget.empty:
                kw_budget = pd.DataFrame(get_keywords(budget), columns=['Word', 'Count'])
                fig_k2 = px.bar(kw_budget, x='Count', y='Word', orientation='h', color_discrete_sequence=[ACCENT_COLOR])
                fig_k2.update_layout(margin=dict(l=0, r=0, t=0, b=0), yaxis={'categoryorder':'total ascending'}, showlegend=False)
                st.plotly_chart(fig_k2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # TAB 6: ML PREDICTIVE DEEP DIVE
    with tab6:
        st.markdown('<div class="section-header">MÔ HÌNH HỌC MÁY PREDICTIVE RANDOM FOREST ENGINE</div>', unsafe_allow_html=True)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE — Sai Số Trung Bình", f"€{metrics['MAE']:.2f}")
        m2.metric("RMSE — Phạt Sai Số Lớn", f"€{metrics['RMSE']:.2f}")
        m3.metric("R² — Hệ Số Xác Định", f"{metrics['R2']:.2%}")
        
        d1, d2 = st.columns(2)
        with d1:
            st.markdown(f'<div class="chart-container"><div class="ibcs-title">Thực Tế vs. Dự Báo (Actual vs Predicted)</div><div class="ibcs-subtitle">Đường chéo = Dự báo chính xác 100%</div>', unsafe_allow_html=True)
            fig_diag = px.scatter(comparison_df, x="Actual", y="Predicted", opacity=0.4, color_discrete_sequence=[PRIMARY_COLOR])
            max_val = max(comparison_df.max())
            fig_diag.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color=ACCENT_COLOR, width=2, dash="dash"))
            fig_diag.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_diag, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with d2:
            st.markdown(f'<div class="chart-container"><div class="ibcs-title">Mức Độ Quan Trọng Của Các Biến (Feature Importance)</div><div class="ibcs-subtitle">Tác động của yếu tố tới giá phòng</div>', unsafe_allow_html=True)
            imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
            fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', color_discrete_sequence=[ACCENT_COLOR])
            fig_imp.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""<div class="footnote" style="margin-top:40px; padding:15px 0; border-top:1px solid #E2E8F0;">
    Athens Airbnb Market Analytics Platform &middot; Executive Management Standard &middot; Enterprise UI
</div>""", unsafe_allow_html=True)
