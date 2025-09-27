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
            "--verbose",
            action="store_true",
            help="Enable verbose output"
        )
