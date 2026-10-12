from ultralytics import YOLO
import sys
m = sys.argv[1] if len(sys.argv)>1 else 'yolo11s.pt'
name = sys.argv[2] if len(sys.argv)>2 else 'run1'
imgsz = int(sys.argv[3]) if len(sys.argv)>3 else 768
ep = int(sys.argv[4]) if len(sys.argv)>4 else 120
model = YOLO(m)
model.train(data='/Users/abhayindwar/Projects/eris/habitat-contact/yolo/data.yaml',
    epochs=ep, imgsz=imgsz, batch=8, device='mps', workers=4, project='runs', name=name,
    patience=40, cache=False, optimizer='AdamW', lr0=0.001, cos_lr=True, warmup_epochs=3,
    hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, degrees=20.0, translate=0.15, scale=0.6,
    fliplr=0.5, flipud=0.5, mosaic=1.0, close_mosaic=15, mixup=0.0, copy_paste=0.0,
    box=7.5, cls=0.5, dfl=1.5, seed=0, val=True, plots=False, verbose=True)
