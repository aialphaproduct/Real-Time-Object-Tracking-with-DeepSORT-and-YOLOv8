import cv2

def create_video_writer(video_cap, output_filename):
    """
    Tạo một VideoWriter để lưu video đầu ra.
    
    Tham số:
    - video_cap: Đối tượng VideoCapture
    - output_filename: Tên file đầu ra
    
    Trả về:
    - Đối tượng VideoWriter
    """
    # Lấy thông tin từ video gốc
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    
    # Định nghĩa codec và tạo VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Sử dụng codec MP4
    
    # Tạo VideoWriter
    return cv2.VideoWriter(output_filename, fourcc, fps, (width, height))