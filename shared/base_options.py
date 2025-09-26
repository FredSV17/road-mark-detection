import argparse

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Data Analysis Subproject")
        self.initialize()
        
    def initialize(self):
        self.parser.add_argument(
            "--path",
            type=str,
            default="data",
            help="Dataset path (Default: data)"
        )
        self.parser.add_argument(
            "--dataset",
            type=str,
            default="train",
            help="Which dataset to analyze (Default: train)"
        )

        self.parser.add_argument(
            "--save_bboxes",
            action="store_true",
            help="Save images with the visual bounding box representation in a folder (Default: False)"
        )

        self.parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose output"
        )
