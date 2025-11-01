from roboflow import Roboflow

rf = Roboflow(api_key="3Ew7VupOGLbZTyijO7J5")
project = rf.workspace("roboflow-universe-projects").project("hard-hats-fhbh5")
version = project.version(6)
dataset = version.download("yolov8")
