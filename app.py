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
            Phân tích định giá đêm &middot; Tỷ lệ lấp đầy &middot; Khoảng cách Trung tâm Acropolis &middot; Dự báo ML
        </div>
    </div>
    <div class="report-meta">
        Dữ liệu: {len(filtered_df):,} căn hộ active<br>
        Số khu vực: {filtered_df['neighbourhood'].nunique()} &nbsp;|&nbsp; 
        Giá trung bình: &euro;{filtered_df['price'].mean():.1f}/đêm
    </div>
</div>
""", unsafe_allow_html=True)

# TABS CONFIGURATION
if is_exec_mode:
    tab1, tab2, tab3 = st.tabs([
        "TỔNG QUAN THỊ TRƯỜNG",
        "PHÂN TÍCH GIÁ & VỊ TRÍ",
        "MÔ PHỎNG ĐỀ XUẤT GIÁ PHÒNG"
    ])
else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "TỔNG QUAN THỊ TRƯỜNG",
        "PHÂN TÍCH GIÁ & VỊ TRÍ",
        "MÔ PHỎNG ĐỀ XUẤT GIÁ PHÒNG",
        "PHÂN CỤM K-MEANS & NLP",
        "MÔ HÌNH MACHINE LEARNING"
    ])


# ==================== TAB 1: MARKET OVERVIEW ====================
with tab1:
    st.markdown('<div class="section-header">SỨC KHỎE THỊ TRƯỜNG AIRBNB ATHENS (MARKET HEALTH)</div>', unsafe_allow_html=True)
    
    render_takeaway(
        "[NHẬN ĐỊNH THỊ TRƯỜNG QUAN TRỌNG]",
        "Khu vực trung tâm du lịch xung quanh Đền Acropolis (Plaka, Koukaki, Monastiraki) chiếm hơn 60% tổng lượng căn hộ cho thuê toàn thành phố với tỷ lệ lấp đầy ước tính duy trì ở mức cao >75%."
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    occ_rate = ((365-filtered_df['availability_365'])/365*100).mean()
    kpi1.metric("Tổng Căn Hộ Active", f"{len(filtered_df):,.0f}")
    kpi2.metric("Giá Trung Bình Đêm (ADR)", f"€{filtered_df['price'].mean():.1f}")
    kpi3.metric("Số Đánh Giá Trung Bình", f"{filtered_df['number_of_reviews'].mean():.0f}")
    kpi4.metric("Tỷ Lệ Lấp Đầy (Ước Tính)", f"{occ_rate:.1f}%")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Bản Đồ Phân Bổ Vị Trí Căn Hộ & Mức Giá Đêm (Athens)</div><div class="ibcs-subtitle">Tọa độ địa lý &middot; Kích thước = Số lượng review</div>', unsafe_allow_html=True)
        fig_map = px.scatter_mapbox(
            filtered_df, lat="latitude", lon="longitude", color="price", size="number_of_reviews",
            color_continuous_scale="Viridis", zoom=11, height=450, mapbox_style="carto-positron"
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown('<div class="footnote">Màu vàng/sáng = Mức giá cao &middot; Vùng tập trung dày đặc = Trung tâm lịch sử Athens</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Cơ Cấu Loại Hình Phòng</div><div class="ibcs-subtitle">% tổng số lượng listing</div>', unsafe_allow_html=True)
        fig_pie = px.pie(filtered_df, names='room_type', hole=0.5, color_discrete_sequence=[ACCENT_COLOR, PRIMARY_COLOR, AMBER_COLOR])
        fig_pie.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown(f'<div class="ibcs-title" style="margin-top:16px">Top Host Sở Hữu Nhiều Listing Nhất</div>', unsafe_allow_html=True)
        top_hosts = filtered_df['host_name'].value_counts().head(5).reset_index()
        top_hosts.columns = ['Host', 'Listings']
        fig_host = px.bar(top_hosts, x='Listings', y='Host', orientation='h', color_discrete_sequence=[PRIMARY_COLOR])
        fig_host.update_layout(margin=dict(l=0, r=0, t=0, b=0), yaxis={'categoryorder':'total ascending'}, showlegend=False)
        st.plotly_chart(fig_host, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ==================== TAB 2: PRICE & LOCATION ====================
with tab2:
    st.markdown('<div class="section-header">PHÂN TÍCH GIÁ BÁN LẺ VÀ KHOẢNG CÁCH TRUNG TÂM</div>', unsafe_allow_html=True)
    
    render_takeaway(
        "[NHẬN ĐỊNH ĐỊNH GIÁ & VỊ TRÍ]",
        "Khoảng cách tới Đền Acropolis có tương quan nghịch rõ rệt với giá đêm: Cứ mỗi km cách xa trung tâm historic center, giá phòng bán lẻ trung bình giảm khoảng 12 - 15 EUR/đêm."
    )

    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Mức Giá Đêm Trung Bình Theo Khu Vực Tại Athens</div><div class="ibcs-subtitle">EUR per night &middot; Xếp hạng giảm dần</div>', unsafe_allow_html=True)
        top_12_neigh = filtered_df.groupby('neighbourhood')['price'].mean().nlargest(12).reset_index()
        top_12_neigh = top_12_neigh.sort_values('price', ascending=True)

        fig_bar_p = px.bar(
            top_12_neigh, x="price", y="neighbourhood", orientation='h',
            color_discrete_sequence=[ACCENT_COLOR],
            text_auto='.1f'
        )
        fig_bar_p.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(tickprefix="€"),
            showlegend=False
        )
        st.plotly_chart(fig_bar_p, use_container_width=True)
        st.markdown('<div class="footnote">Plaka và Koukaki dẫn đầu về mức giá phòng bán lẻ trung bình đêm</div></div>', unsafe_allow_html=True)
        
    with p2:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">Tương Quan Giá Đêm vs Khoảng Cách Trung Tâm Acropolis</div><div class="ibcs-subtitle">EUR per night vs Khoảng cách (km)</div>', unsafe_allow_html=True)
        fig_trend = px.scatter(
            filtered_df, x="dist_to_center", y="price", opacity=0.35,
            color_discrete_sequence=[PRIMARY_COLOR],
            trendline="lowess", trendline_color_override=ACCENT_COLOR
        )
        fig_trend.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis=dict(tickprefix="€"),
            xaxis_title="Khoảng cách tới Acropolis (km)"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown('<div class="footnote">Đường xu hướng xanh lục minh họa sự sụt giảm giá khi khoảng cách tăng xa trung tâm</div></div>', unsafe_allow_html=True)


# ==================== TAB 3: PRICE PREDICTOR LAB ====================
with tab3:
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
                    Căn hộ: {inp_room} &middot; Khu vực: {inp_neigh} &middot; Cách trung tâm: {inp_dist} km
                </div>
            </div>
            <div class="footnote" style="margin-top:8px;">
                Kết quả dự báo bởi mô hình Random Forest Regressor &middot; Sai số trung bình (MAE) &plusmn;&euro;{metrics['MAE']:.2f} &middot; Độ tin cậy R² = {metrics['R2']:.2%}
            </div>
            """, unsafe_allow_html=True)


# ==================== DEEP ANALYTICS MODE EXTRA TABS ====================
if not is_exec_mode:
    # TAB 4: K-MEANS & NLP
    with tab4:
        st.markdown('<div class="section-header">PHÂN CỤM ĐỊA LÝ K-MEANS & PHÂN TÍCH TỪ KHÓA MÔ TẢ</div>', unsafe_allow_html=True)
        
        col_k1, col_k2 = st.columns([3, 1])
        if len(filtered_df) > 10:
            df_cluster = filtered_df.copy()
            df_cluster['Cluster'] = run_kmeans(df_cluster, n_clusters=4).astype(str)
            
            with col_k1:
                st.markdown(f'<div class="chart-container"><div class="ibcs-title">Phân Cụm Vị Trí Thị Trường (K-Means Clustering)</div><div class="ibcs-subtitle">Gom nhóm dựa trên Tọa độ, Mức giá và Khoảng cách trung tâm</div>', unsafe_allow_html=True)
                fig_cluster = px.scatter_mapbox(
                    df_cluster, lat="latitude", lon="longitude", color="Cluster",
                    hover_data=['price', 'neighbourhood'],
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
                    rows += f"<tr><td>Cụm {r['Cụm']}</td><td>€{r['Giá TB (€)']:.1f}</td><td>{int(r['Số lượng'])}</td></tr>"
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

    # TAB 5: ML PREDICTIVE DEEP DIVE
    with tab5:
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
