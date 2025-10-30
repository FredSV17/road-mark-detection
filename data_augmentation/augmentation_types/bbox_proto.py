from data_augmentation.augmentation_types.helper import to_tensor


class BoundingBoxesPoly:
    def __init__(self, bboxes, canvas_size):
        self.coords = to_tensor(bboxes)
        self.canvas_size = canvas_size