import argparse
from utils import info, post_processing


# Full body tracking tasks (e.g. gait analysis)
def body_tasks(input_args):
    ts_path = input_args.data_path + 'body/time_series/outputs_finetuned/'

    subjects = info.subjects_All

    # Get the gait features
    gait_feats = post_processing.gait_features(subjects, ts_path)

    step_widths = gait_feats.step_width(subjects)
    step_lengths = gait_feats.step_lengths(subjects)
    cadences_gaitspeeds_gaitspeedvars = gait_feats.cadence_gaitspeed_gaitspeedvar(subjects)
    foot_lifts = gait_feats.foot_lifts(subjects)
    arm_swings = gait_feats.arm_swings(subjects)
    hip_flexions = gait_feats.hip_flexions(subjects)
    knee_flexions = gait_feats.knee_flexions(subjects)
    trunk_rots = gait_feats.trunk_rots(subjects)

    print(trunk_rots)


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
