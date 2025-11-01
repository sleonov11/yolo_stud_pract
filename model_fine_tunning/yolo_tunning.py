from ultralytics import YOLO
import os
import multiprocessing

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_dir, "Hard-Hats-6", "data.yaml")

    print(f"Ищем файл по пути: {data_path}")
    print(f"Файл существует: {os.path.exists(data_path)}")

    model = YOLO("base_model/yolov8s.pt")
    model.train(
        data=data_path,
        epochs=5,
        imgsz=320,
        plots=True,
        workers=0,
        device=0
    )

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
