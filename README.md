# auto_UPDRS
3D body and handpose keypoint extraction from videos of subjects performing UPDRS tasks & post-processing to evaluate their motion features and give UPDRS scores.

## Body Pose Prediction

1. Use [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) to extract 2D keypoint proposals from CAMERA data drive using "demo_inference.py" inside:
```
python3 scripts/demo_inference.py --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth --indir /mnt/CAMERA-data/CAMERA/Other/lbidulka_dataset/<SUBJECT>/<TASK+CHANNEL>/frames --save_img --outdir /mnt/CAMERA-data/CAMERA/Other/lbidulka_dataset/<SUBJECT>/<TASK+CHANNEL>
```

2. Use body_nets.Lifter() to predict 3D keypoints from 2D proposals (scoring.py has example)
