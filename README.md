<div align="center">

ATHENS AIRBNB MARKET ANALYSIS

Dự án Phân tích Dữ liệu & Học máy (Data Analytics & Machine Learning Portfolio)

Phân tích toàn diện thị trường lưu trú ngắn hạn tại Athens, Hy Lạp nhằm tối ưu hóa chiến lược định giá và đầu tư.

GIỚI THIỆU • TÍNH NĂNG • CÔNG NGHỆ • CÀI ĐẶT • INSIGHTS

</div>

1. GIỚI THIỆU

Dự án này là một Dashboard tương tác được xây dựng bằng Python và Streamlit, tập trung vào việc khai phá dữ liệu (EDA) và xây dựng mô hình dự báo giá cho các căn hộ Airbnb tại Athens.

Mục tiêu chính:

Hiểu thị trường: Cung cấp cái nhìn tổng quan về phân bố giá, vị trí và loại phòng.

Phân tích chuyên sâu: Đánh giá tác động của vị trí (khoảng cách tới Acropolis), tiện nghi và cách đặt tên (NLP) đến giá thuê.

Dự báo giá (AI): Xây dựng công cụ định giá tự động giúp chủ nhà (Host) đưa ra mức giá cạnh tranh nhất.

2. TÍNH NĂNG & PHÂN TÍCH

Hệ thống Dashboard được chia thành 4 phân hệ chính:

<table width="100%" border="0">
<tr>
<td width="50%" valign="top" style="border-right: 1px solid #ddd; padding-right: 10px;">
<h3 align="center">I. TỔNG QUAN THỊ TRƯỜNG</h3>
<p align="center"><i>(Market Overview)</i></p>





<ul>
<li><b>KPIs Tracker:</b> Theo dõi tổng số Listing, Giá trung bình (ADR), Tỷ lệ lấp đầy ước tính.</li>
<li><b>Geospatial Analysis:</b> Bản đồ tương tác hiển thị phân bổ giá và mật độ phòng trên nền bản đồ thực địa.</li>
<li><b>Host Analysis:</b> Phân tích thị phần của các Super Host.</li>
</ul>
</td>
<td width="50%" valign="top" style="padding-left: 10px;">
<h3 align="center">II. PHÂN TÍCH GIÁ CHUYÊN SÂU</h3>
<p align="center"><i>(Price Analysis)</i></p>





<ul>
<li><b>Price Sensitivity:</b> Biểu đồ hộp (Boxplot) phát hiện các giá trị ngoại lai (outliers) theo từng khu vực.</li>
<li><b>Correlation Matrix:</b> Ma trận tương quan nhiệt tìm biến số ảnh hưởng mạnh nhất đến giá.</li>
<li><b>Distance Decay:</b> Phân tích xu hướng giảm giá khi vị trí xa trung tâm (Acropolis).</li>
</ul>
</td>
</tr>
<tr><td colspan="2"><hr></td></tr>
<tr>
<td width="50%" valign="top" style="border-right: 1px solid #ddd; padding-right: 10px;">
<h3 align="center">III. PHÂN TÍCH ĐỊA LÝ & NLP</h3>
<p align="center"><i>(Advanced Analytics)</i></p>





<ul>
<li><b>Treemap Visualization:</b> Cấu trúc thị trường theo Quận/Huyện.</li>
<li><b>NLP (Natural Language Processing):</b> Phân tích từ khóa trong tiêu đề. So sánh chiến lược từ khóa giữa phân khúc Cao cấp và Bình dân.</li>
</ul>
</td>
<td width="50%" valign="top" style="padding-left: 10px;">
<h3 align="center">IV. MACHINE LEARNING LAB</h3>
<p align="center"><i>(Predictive Modeling)</i></p>





<ul>
<li><b>Mô hình:</b> Random Forest Regressor.</li>
<li><b>Đánh giá:</b> Hiển thị minh bạch chỉ số MAE, RMSE, R² Score và biểu đồ sai số (Residuals).</li>
<li><b>Live Prediction:</b> Công cụ nhập thông số để AI gợi ý giá thuê ngay lập tức.</li>
</ul>
</td>
</tr>
</table>

3. CÔNG NGHỆ SỬ DỤNG

Hạng mục

Công nghệ / Thư viện

Ngôn ngữ

Python 3.x

Giao diện Web

Streamlit

Xử lý dữ liệu

Pandas, NumPy

Trực quan hóa

Plotly Express, Plotly Graph Objects

Machine Learning

Scikit-learn (Random Forest, Preprocessing)

Khác

Requests, BeautifulSoup (Scraping), Haversine Formula

4. CẤU TRÚC THƯ MỤC

athens-airbnb-analysis/
├── data/
│   └── Athens_Airbnb_Data.csv   # Dữ liệu thô (Raw Data)
├── airbnb_dashboard.py          # Source code chính
├── requirements.txt             # Danh sách thư viện
└── README.md                    # Tài liệu hướng dẫn


5. HƯỚNG DẪN CÀI ĐẶT

Bước 1: Clone dự án

git clone [https://github.com/your-username/athens-airbnb-analysis.git](https://github.com/your-username/athens-airbnb-analysis.git)
cd athens-airbnb-analysis


Bước 2: Tạo môi trường ảo (Khuyến nghị)

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate


Bước 3: Cài đặt thư viện

pip install -r requirements.txt


Bước 4: Chạy ứng dụng

streamlit run airbnb_dashboard.py


6. KEY INSIGHTS

Các phát hiện chính từ quá trình phân tích dữ liệu:

1. Vị trí là yếu tố quyết định




Khoảng cách đến Acropolis có tương quan nghịch biến mạnh với giá thuê. Mỗi 1km xa trung tâm làm giảm trung bình X% giá phòng.

2. Sức mạnh của từ khóa (NLP)




Các căn hộ có tiêu đề chứa từ "View", "Acropolis", "Luxury" có giá cao hơn trung bình 30% so với các căn hộ dùng từ "Cozy", "Metro".

3. Hiệu suất mô hình AI




Mô hình Random Forest đạt độ chính xác R² ~75-80% trên tập kiểm thử, cho thấy khả năng dự báo giá khá tin cậy dựa trên các thuộc tính vật lý và vị trí.

<div align="center">
<p><b>[Tên của bạn]</b>



Data Analyst Portfolio</p>
<p>
<a href="#">LinkedIn</a> &nbsp;|&nbsp;
<a href="#">Email</a> &nbsp;|&nbsp;
<a href="#">Portfolio Website</a>
</p>
</div>
