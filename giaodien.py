import streamlit as st
import datetime
from ultralytics import YOLO
import cv2
import tempfile
import os
from helper import create_video_writer
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# Thiết lập tiêu đề ứng dụng
st.title('Hệ thống Phát hiện và Theo dõi Đối tượng')

# Cấu hình sidebar
st.sidebar.header('Cài đặt')
CONFIDENCE_THRESHOLD = st.sidebar.slider('Ngưỡng tin cậy', 0.0, 1.0, 0.8)
MAX_AGE = st.sidebar.slider('Thời gian theo dõi tối đa (frames)', 10, 100, 50)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# Tùy chọn tải lên video hoặc sử dụng webcam
source_option = st.sidebar.radio("Nguồn video", ["Tải lên video", "Webcam"])

# Tải model YOLO
@st.cache_resource
def load_model():
    """Tải model YOLO và lưu vào bộ nhớ cache để tăng hiệu suất"""
    return YOLO("yolov8n.pt")

model = load_model()

# Khởi tạo tracker
@st.cache_resource
def create_tracker():
    """Khởi tạo DeepSort tracker và lưu vào bộ nhớ cache"""
    return DeepSort(max_age=MAX_AGE)

tracker = create_tracker()

# Hàm xử lý video
def process_video(video_path, save_output=False):
    # Khởi tạo video capture
    video_cap = cv2.VideoCapture(video_path)
    
    # Kiểm tra video có mở được không
    if not video_cap.isOpened():
        st.error("Không thể mở video. Vui lòng kiểm tra lại tệp.")
        return
    
    # Lấy thông tin về video
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    
    # Khởi tạo video writer nếu cần
    if save_output:
        output_path = "output.mp4"
        writer = create_video_writer(video_cap, output_path)
    
    # Tạo placeholder để hiển thị video
    video_placeholder = st.empty()
    
    # Tạo thanh tiến trình
    progress_bar = st.progress(0)
    
    # Hiển thị thông tin xử lý
    info_container = st.container()
    processing_info = info_container.empty()
    
    # Đếm frame
    frame_count = 0
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Biến để theo dõi số lượng đối tượng
    tracked_objects = []
    
    # Xử lý từng frame
    while True:
        start = datetime.datetime.now()
        
        ret, frame = video_cap.read()
        
        if not ret:
            break
        
        # Phát hiện đối tượng bằng YOLO
        detections = model(frame)[0]
        
        # Danh sách các kết quả phát hiện
        results = []
        
        # Xử lý các phát hiện
        for data in detections.boxes.data.tolist():
            confidence = data[4]
            
            # Lọc các phát hiện có độ tin cậy thấp
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue
                
            # Lấy thông tin bounding box và class id
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])
            
            # Thêm kết quả vào danh sách
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])
        
        # Cập nhật tracker với các phát hiện mới
        tracks = tracker.update_tracks(results, frame=frame)
        
        # Xử lý các track
        current_tracked_ids = []
        for track in tracks:
            # Bỏ qua các track chưa được xác nhận
            if not track.is_confirmed():
                continue
                
            # Lấy track id và bounding box
            track_id = track.track_id
            current_tracked_ids.append(track_id)
            ltrb = track.to_ltrb()
            
            xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            
            # Vẽ bounding box và ID
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
            cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
        
        # Ghi lại đối tượng được theo dõi
        tracked_objects = list(set(tracked_objects + current_tracked_ids))
        
        # Tính toán FPS
        end = datetime.datetime.now()
        fps_text = f"FPS: {1 / (end - start).total_seconds():.2f}"
        processing_time = f"Thời gian xử lý: {(end - start).total_seconds() * 1000:.0f} ms"
        
        # Hiển thị FPS và thời gian xử lý trên frame
        cv2.putText(frame, fps_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Chuyển frame từ BGR sang RGB cho Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Hiển thị frame trong Streamlit
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Cập nhật thông tin xử lý
        processing_info.text(f"""
        {fps_text}
        {processing_time}
        Số đối tượng đang theo dõi: {len(current_tracked_ids)}
        Tổng số đối tượng đã phát hiện: {len(tracked_objects)}
        """)
        
        # Cập nhật thanh tiến trình
        frame_count += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
        
        # Lưu frame nếu cần
        if save_output:
            writer.write(frame)
        
        # Tạm dừng nhẹ để không làm quá tải UI
        cv2.waitKey(1)
    
    # Giải phóng tài nguyên
    video_cap.release()
    if save_output:
        writer.release()
        st.success(f"Video đã được lưu tại: {output_path}")
        
        # Tùy chọn tải về video đã xử lý
        with open(output_path, "rb") as file:
            st.download_button(
                label="Tải về video đã xử lý",
                data=file,
                file_name="output_processed.mp4",
                mime="video/mp4"
            )
    
    # Hiển thị thông tin tổng kết
    st.subheader("Thông tin tổng kết")
    st.write(f"Tổng số frame đã xử lý: {frame_count}")
    st.write(f"Tổng số đối tượng đã phát hiện và theo dõi: {len(tracked_objects)}")

# Phần chính của ứng dụng
if source_option == "Tải lên video":
    # Widget tải lên video
    uploaded_file = st.file_uploader("Chọn video", type=["mp4", "avi", "mov"])
    
    # Lưu tạm thời video đã tải lên
    if uploaded_file is not None:
        # Tạo tệp tạm thời
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
        temp_file.close()
        
        # Tùy chọn lưu video đã xử lý
        save_output = st.checkbox("Lưu video đã xử lý", value=True)
        
        # Nút để bắt đầu xử lý
        if st.button("Bắt đầu xử lý"):
            st.spinner("Đang xử lý video...")
            process_video(temp_file_path, save_output)
            
            # Xóa tập tin tạm thời
            os.unlink(temp_file_path)
else:  # Webcam
    # Sử dụng webcam
    webcam_id = st.sidebar.number_input("ID của webcam", min_value=0, value=0, step=1)
    
    # Hiển thị video từ webcam
    if st.button("Bắt đầu webcam"):
        process_video(int(webcam_id), save_output=False)
        
# Thêm thông tin về ứng dụng
st.sidebar.markdown("---")
st.sidebar.subheader("Thông tin")
st.sidebar.info("""
Ứng dụng này sử dụng YOLOv8 để phát hiện đối tượng và DeepSort để theo dõi.
- YOLOv8 là một mô hình phát hiện đối tượng hiện đại
- DeepSort là thuật toán theo dõi nhiều đối tượng
""")