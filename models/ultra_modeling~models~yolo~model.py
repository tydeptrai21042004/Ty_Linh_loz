"""YOLO wrapper limited to OBB and detection tasks used in the experiment."""
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel, OBBModel


class YOLO(Model):
    @property
    def task_map(self):
        return {
            'detect': {
                'model': DetectionModel,
                'trainer': yolo.detect.DetectionTrainer,
                'validator': yolo.detect.DetectionValidator,
                'predictor': yolo.detect.DetectionPredictor,
            },
            'obb': {
                'model': OBBModel,
                'trainer': yolo.obb.OBBTrainer,
                'validator': yolo.obb.OBBValidator,
                'predictor': yolo.obb.OBBPredictor,
            },
        }
