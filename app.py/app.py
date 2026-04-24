import streamlit as st
import yfinance as yf
import torch
import pandas as pd
import plotly.graph_objects as go
from transformers import PatchTSTConfig, PatchTSTForPrediction
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import PIL.Image as Image
import os

# --- 1. CẤU HÌNH GIAO DIỆN & LOGO ---
st.set_page_config(page_title="VietBTC-NPB Dashboard", layout="wide")

# Hàm load và hiển thị logo
def display_logo():
    logo_path = 'logo.png' # Đảm bảo file logo.png nằm cùng thư mục app.py
    if os.path.exists(logo_path):
        image = Image.open(logo_path)
        # Hiển thị logo ở góc trên bên trái
        st.image(image, width=200)
    else:
        st.sidebar.warning("Không tìm thấy file 'logo.png'. Hãy đặt file ảnh vào cùng thư mục với code.")

# --- 2. GIAO DIỆN CHỌN MÀU NỀN ---
st.sidebar.title("🎨 Cấu hình Giao diện")
# Nút chọn màu nền tổng thể
bg_color = st.sidebar.color_picker("Chọn màu nền trang", "#0e1117")
# Nút chọn màu nền của biểu đồ
chart_bg_color = st.sidebar.color_picker("Chọn màu nền biểu đồ", "#161b22")
# Nút chọn màu của các chữ và số
text_color = st.sidebar.color_picker("Chọn màu chữ", "#ffffff")

# Áp dụng CSS để đổi màu nền trang và màu chữ
st.markdown(f"""
    <style>
    .stApp, .main, [data-testid="stSidebar"] {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .stMetric, div[data-testid="stMarkdownContainer"] p {{
        color: {text_color} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

display_logo()
st.title("💡 Bitcoin AI Analyst by VietBTC-NPB")

# --- 3. XỬ LÝ DỮ LIỆU & AI (GIỮ NGUYÊN TỪ BẢN TRƯỚC) ---
@st.cache_data(ttl=30) # Tối ưu tốc độ tải dữ liệu
def load_data():
    try:
        return yf.download('BTC-USD', period='1d', interval='1m', progress=False)
    except Exception:
        return pd.DataFrame() # Trả về DF rỗng nếu lỗi

# Khởi tạo mô hình AI (Sửa 512 -> 300)
LENGTH = 300
config = PatchTSTConfig(num_input_channels=1, context_length=LENGTH, prediction_length=60)
model = PatchTSTForPrediction(config)
scaler = StandardScaler()

df = load_data()

if not df.empty and len(df) >= LENGTH:
    # Xử lý AI
    data = df['Close'].values[-LENGTH:].reshape(-1, 1)
    scaled = scaler.fit_transform(data)
    input_t = torch.from_numpy(scaled).float().permute(1, 0).unsqueeze(0)
    
    with torch.no_grad():
        pred = model(past_values=input_t).prediction_logits.numpy().reshape(-1, 1)
        pred_actual = scaler.inverse_transform(pred)

    # --- 4. HIỂN THỊ THÔNG SỐ (METRICS) ---
    col1, col2, col3 = st.columns(3)
    curr_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    delta = curr_price - prev_price

    with col1:
        st.metric("Giá Bitcoin (USD)", f"${curr_price:,.2f}", f"{delta:,.2f}")
    with col2:
        st.metric("Trạng thái AI", "On and On", "Live")
    with col3:
        st.write(f"**Cập nhật lúc:** {datetime.now().strftime('%H:%M:%S')}")
        st.write(f"**Dữ liệu:** {len(df)} nến 1m")

    # --- 5. BIỂU ĐỒ TƯƠNG TÁC Plotly (ÁP DỤNG MÀU CHỌN) ---
    fig = go.Figure()

    # Đường giá thực (Màu xanh Neon)
    fig.add_trace(go.Scatter(
        x=df.index[-60:], y=df['Close'].values[-60:],
        mode='lines+markers', name='Giá thực tế (1h)',
        line=dict(color='#00ff88', width=3)
    ))

    # Đường dự báo (Màu đỏ Neon)
    last_time = df.index[-1]
    future_dates = [last_time + timedelta(minutes=i) for i in range(1, 61)]
    fig.add_trace(go.Scatter(
        x=future_dates, y=pred_actual.flatten()[:60],
        mode='lines', name='AI Dự báo (1h tới)',
        line=dict(color='#ff4b4b', width=3, dash='dot')
    ))

    # Áp dụng màu nền biểu đồ đã chọn từ Sidebar
    fig.update_layout(
        paper_bgcolor=chart_bg_color,
        plot_bgcolor=chart_bg_color,
        font=dict(color=text_color),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
        xaxis=dict(showgrid=False, color=text_color),
        yaxis=dict(showgrid=True, gridcolor='#30363d', color=text_color)
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- 6. TỰ ĐỘNG REFRESH ---
    st.empty()
    st.rerun() # Refresh mượt mà

else:
    st.warning(f"Đang tích lũy dữ liệu: {len(df)}/{LENGTH} nến (1m).")
    time.sleep(10)
    st.rerun()
