import argparse
from utils import info, post_processing
from sklearn.metrics import accuracy_score
import numpy as np

# Full body tracking tasks (e.g. gait analysis)
def body_tasks(input_args):
    ts_path = input_args.data_path + 'body/time_series/outputs_finetuned/'

    # Create a gait processor, to get features and indicator functions
    gait_processor = post_processing.gait_processor(ts_path)
    
    # print(gait_feats.feats[:, gait_feats.subjects.index('S04')])

    # use thresholds to get predictions for PD subjects by majority voting
    pd_indices = [i for i, x in enumerate(info.subjects_All) if x in info.subjects_PD]
    pd_gait_feats = gait_processor.feats[:, pd_indices][:-2]
    pd_indicators = gait_processor.indicators(pd_gait_feats)
    pd_predictions = np.mean(pd_indicators, axis=0) > 0.5

    # Results
    print("\nlabels:\n", info.Y_true)
    print("preds:\n", 1 * pd_predictions)
    accuracy = accuracy_score(info.Y_true, np.round(pd_predictions))
    print("acc: ", accuracy)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./auto_UPDRS/data/", help="input data path", type=str)
    parser.add_argument("--output_path", default="./auto_UPDRS/outputs/", help="output data path", type=str)
    return parser.parse_args()

def main():
    input_args = get_args()

    body_tasks(input_args)    
    
if __name__ == '__main__':
    main()
