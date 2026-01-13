import cv2
import numpy as np
from ultralytics import YOLO
import yt_dlp

def gerar_heatmap_soccer(url_youtube):
    ydl_opts = {'format': 'best', 'noplaylist': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url_youtube, download=False)
            video_url = info['url']
    except Exception as e:
        print(f"Erro no YouTube: {e}")
        return None

    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_url)
    
    ret, frame = cap.read()
    if not ret: return None
        
    heatmap_overlay = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    # Analisa 300 frames (cerca de 10 seg) para o teste ser rápido
    count = 0
    while cap.isOpened() and count < 300:
        ret, frame = cap.read()
        if not ret: break

        results = model.track(frame, persist=True, classes=[0], verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                x_center = int((box[0] + box[2]) / 2)
                y_bottom = int(box[3])
                cv2.circle(heatmap_overlay, (x_center, y_bottom), 15, 1, -1)
        count += 1

    heatmap_norm = cv2.normalize(heatmap_overlay, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
    
    output_name = "resultado_heatmap.png"
    cv2.imwrite(output_name, heatmap_color)
    cap.release()
    return output_name
