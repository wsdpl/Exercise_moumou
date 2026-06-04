from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8m-pose.pt")

    model.train(
        data="fitness_pose.yaml",
        epochs=100,
        patience=0,
        batch=8,
        imgsz=640,
        device=0,
        workers=0,
        amp=True,
        cos_lr=True,
        lr0=0.001,
        warmup_epochs=3,
        mosaic=0.5,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.4,
        project="fitness_pose_train",
        name="coco_finetune_exom",
        exist_ok=True,
        pretrained=True
    )