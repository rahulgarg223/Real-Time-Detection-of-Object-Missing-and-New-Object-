import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
from datetime import datetime

class ObjectTracker:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.track_history = defaultdict(lambda: {
            'frames': [],
            'class_name': None,
            'first_seen': None,
            'last_seen': None,
            'total_frames': 0
        })
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.class_names = self.model.names
        
    def process_frame(self, frame):
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            current_time = time.time()
            self.fps = 30 / (current_time - self.last_time)
            self.last_time = current_time
        
        results = self.model.track(frame, persist=True, conf=self.conf_threshold)
        
        current_objects = set()
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                current_objects.add(track_id)
                x1, y1, x2, y2 = map(int, box)
                class_name = self.class_names[class_id]
                
                if track_id not in self.track_history:
                    self.track_history[track_id]['first_seen'] = datetime.now()
                    self.track_history[track_id]['class_name'] = class_name
                
                self.track_history[track_id]['frames'].append(self.frame_count)
                self.track_history[track_id]['last_seen'] = datetime.now()
                self.track_history[track_id]['total_frames'] += 1
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{class_name} ID:{track_id}'
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        previous_objects = set(self.track_history.keys())
        missing_objects = previous_objects - current_objects
        new_objects = current_objects - previous_objects
        
        missing_objects_info = []
        for track_id in missing_objects:
            info = self.track_history[track_id]
            duration = (info['last_seen'] - info['first_seen']).total_seconds()
            missing_objects_info.append({
                'id': track_id,
                'class': info['class_name'],
                'duration': f"{duration:.1f}s",
                'frames': info['total_frames']
            })
        
        new_objects_info = []
        for track_id in new_objects:
            info = self.track_history[track_id]
            new_objects_info.append({
                'id': track_id,
                'class': info['class_name'],
                'first_seen': info['first_seen'].strftime("%H:%M:%S")
            })
        
        cv2.putText(frame, f'FPS: {self.fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Objects: {len(current_objects)}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, missing_objects_info, new_objects_info

def main():
    tracker = ObjectTracker()
    cap = cv2.VideoCapture(0)
    
    print("Starting object tracking...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, missing_objects, new_objects = tracker.process_frame(frame)
        
        if missing_objects:
            print("\nMissing Objects:")
            for obj in missing_objects:
                print(f"ID: {obj['id']}, Class: {obj['class']}, "
                      f"Duration: {obj['duration']}, Frames: {obj['frames']}")
        
        if new_objects:
            print("\nNew Objects:")
            for obj in new_objects:
                print(f"ID: {obj['id']}, Class: {obj['class']}, "
                      f"First Seen: {obj['first_seen']}")
        
        cv2.imshow('Object Tracker', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 