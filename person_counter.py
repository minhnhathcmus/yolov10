from ultralytics import YOLOv10
from ultralytics.engine.results import Results


def filter_person_boxes(detection_result:Results):
    person_boxes = list()
    for box in detection_result.boxes:
        if box.cls.item() == 0.0: # person
            person_boxes.extend(box.xyxy.numpy())

    return person_boxes

class PersonCounter:
    def __init__(self, human_detection_model_path='./weights/yolov10n.pt'):        
        self.model = YOLOv10(human_detection_model_path)

    def count(self, frame):
        results = self.model.predict(frame, verbose=False)
        person_boxes = filter_person_boxes(results[0])

        return len(person_boxes), person_boxes