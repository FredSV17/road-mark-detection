from typing import List

import pandas as pd
from data_loader import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

def check_class_representation(dl_list : List[DataLoader]):
    for dl in dl_list:
        print(f"Creating graphs for dataset {dl.dt_type}")
        # Count the amout of bounding boxes
        class_count = [[key, len(dl.bb_dict[key])] for key in dl.bb_dict.keys()]
        # Convert to a dataframe
        class_representation_df = pd.DataFrame(class_count, columns=["Class", "Count"])
        sns.barplot(class_representation_df, x='Class', y='Count')
        plt.title(f"Class representation in images (dataset {dl.dt_type})")
        plt.savefig(f"data_analysis/class_representation_{dl.dt_type}")
        plt.clf()