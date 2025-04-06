from ultralytics import YOLO 

model = YOLO('models/football-ball-detection-v2.pt')

results = model.predict('input_videos/08fd33_4.mp4',save=True, conf=0.2)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)