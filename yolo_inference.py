from ultralytics import YOLO 

model = YOLO('models/3k_imgs.pt')

results = model.predict('input_videos/bayern_vs_leverkusen.mp4',save=True, conf=0.2, classes=3)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)
