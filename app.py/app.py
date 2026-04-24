import streamlit as st
import yfinance as yf
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import PatchTSTConfig, PatchTSTForPrediction
from sklearn.preprocessing import StandardScaler
import time

# 1. Cấu hình giao diện Web
st.set_page_config(page_title="Bitcoin AI Predictor", layout="wide")
st.title("📈 Hệ thống Dự báo Bitcoin Real-time (PatchTST)")
st.write("Dữ liệu được cập nhật tự động mỗi 60 giây từ Yahoo Finance.")

# 2. Khởi tạo mô hình AI (Dùng 300 nến để chạy được ngay)
LENGTH = 300 
config = PatchTSTConfig(num_input_channels=1, context_length=LENGTH, prediction_length=60)
model = PatchTSTForPrediction(config)
scaler = StandardScaler()

# Tạo vùng trống để cập nhật biểu đồ mà không bị load lại cả trang
placeholder = st.empty()

while True:
    try:
        # Tải dữ liệu Bitcoin mới nhất
        df = yf.download('BTC-USD', period='1d', interval='1m', progress=False)
        
        if len(df) >= LENGTH:
            # Tiền xử lý dữ liệu cho AI
            data = df['Close'].values[-LENGTH:].reshape(-1, 1)
            scaled = scaler.fit_transform(data)
            # Chuyển đổi cấu trúc dữ liệu cho mô hình (Batch, Channel, Length)
            input_t = torch.from_numpy(scaled).float().permute(1, 0).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(past_values=input_t)
                pred = outputs.prediction_logits.numpy().reshape(-1, 1)
                pred_actual = scaler.inverse_transform(pred)

            # 3. Hiển thị kết quả lên Web
            with placeholder.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    # Vẽ giá thực tế (60 phút gần nhất)
                    ax.plot(df.index[-60:], df['Close'].values[-60:], label='Giá thực tế', color='#1f77b4', linewidth=2)
                    
                    # Vẽ đường dự báo (60 phút tới)
                    last_time = df.index[-1]
                    future_dates = [last_time + pd.Timedelta(minutes=i) for i in range(1, 61)]
                    ax.plot(future_dates, pred_actual[:60], label='Dự báo AI', color='#d62728', linestyle='--')
                    
                    ax.set_ylabel("Giá USD")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                with col2:
                    current_price = df['Close'].iloc[-1]
                    st.metric("Giá hiện tại (BTC)", f"${current_price:,.2f}")
                    st.info(f"Cập nhật lúc: {time.strftime('%H:%M:%S')}")
                    st.write(f"Số lượng nến dữ liệu: {len(df)}")
        else:
            st.warning(f"Đang chờ đủ {LENGTH} nến dữ liệu (Hiện có: {len(df)})")
            
        # Đợi 60 giây trước khi lặp lại
        time.sleep(60)
        
    except Exception as e:
        st.error(f"Đang gặp lỗi: {e}. Hệ thống sẽ thử lại sau 10 giây...")
        time.sleep(10)