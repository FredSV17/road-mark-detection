from typing import List

import pandas as pd
from data_loader import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns


plot_labels = {
    'train' : 'training',
    'test' : 'testing',
    'valid' : 'validation'
}

def check_class_representation(dl_list : List[DataLoader]):
    for dl in dl_list:
        print(f"Creating graphs for {plot_labels[dl.dt_type]} dataset")
        # Count the amout of bounding boxes
        class_count = [[key, len(dl.bb_dict[key])] for key in dl.bb_dict.keys()]
        # Convert to a dataframe
        class_representation_df = pd.DataFrame(class_count, columns=["Class", "Count"])
        sns.barplot(class_representation_df, x='Class', y='Count')
        plt.title(f"Class representation in images ({plot_labels[dl.dt_type]} dataset)")
        plt.savefig(f"data_analysis/class_representation_{plot_labels[dl.dt_type]}")
        plt.clf()
        
def check_objects_per_image(dl_list : List[DataLoader]):
    for dl in dl_list:
        print(f"Creating graphs for {plot_labels[dl.dt_type]} dataset")
        # Create histogram for bounding box count in images
        sns.histplot(data=dl.dt_dict['count'])
        plt.title(f"Object count histogram ({plot_labels[dl.dt_type]} dataset)")
        plt.savefig(f"data_analysis/object_count_{plot_labels[dl.dt_type]}")
        plt.clf()