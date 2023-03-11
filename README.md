# auto_UPDRS
3D body and handpose keypoint extraction from videos of subjects performing UPDRS tasks & post-processing to evaluate their motion features and give UPDRS scores.

## 3D Body Pose Prediction

2D Proposal Network: [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) (Halpe Dataset, 26 keypoints)

### Setup
1. Get the AlphaPose pretrained model weights halpe26_fast_res50_256x192.pth and place into your "auto_UPDRS/AlphaPose/pretrained_models" directory

2. Verify that AlphaPose sample "demo_inference.py" script works on their example data:
```
cd auto_UPDRS/AlphaPose
python scripts/demo_inference.py --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth --indir examples/demo/ --save_img
```

3. Get the pretrained 3D lifter models from [PD_Gait_labeling](https://github.com/mgholamikn/PD_Gait_labeling). Currently we only use "model_lifter.pt", the 3D pose model fine-tuned on our PD walking data using 2 view cameras:
```
cd auto_UPDRS/model_checkpoints/Mohsens
pip install --upgrade --no-cache-dir gdown
gdown --folder --id 1GVjvla21_oXL4KpSbsEAPXRS7N93W1d5?usp=share_link
```

4. Resave the model weight format to be more flexible by uncommenting the lines in "scoring.py":
```
helpers.fix_model_setup(input_args.models_path + 'body_pose/Mohsens/model_lifter.pt', 
                        input_args.models_path + 'body_pose/model_lifter.pt')
```

### Inference
1. Use AlphaPose to extract 2D keypoint proposals from CAMERA data drive using "demo_inference.py" inside "auto_UPDRS/AlphaPose":
```
cd auto_UPDRS/AlphaPose
python3 scripts/demo_inference.py --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth --indir /mnt/CAMERA-data/CAMERA/Other/lbidulka_dataset/<SUBJECT>/<TASK+CHANNEL>/frames --save_img --outdir /mnt/CAMERA-data/CAMERA/Other/lbidulka_dataset/<SUBJECT>/<TASK+CHANNEL>
```

2. Use body_nets.Lifter() to predict 3D keypoints from 2D proposals (scoring.py has example)
