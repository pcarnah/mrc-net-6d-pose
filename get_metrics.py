# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:52:08 2024

@author: pcarnah
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

p = Path(r'D:\6DPose\usprobe\test')
gt = dict()
pred_file = Path(r'D:\6DPose\mrcnet_usprobe-test_checkpoints_v2_jan_all.csv')
split = pd.read_csv(r"D:\6DPose\usprobe\test_df.csv", index_col='scene_id')
preds = pd.read_csv(pred_file)

for sc in p.iterdir():
    scene_id = int(sc.name)
    gt[scene_id] = {}
    
    with open(sc.joinpath('scene_gt.json'), 'r') as f:
        scene_gt = json.load(f)
        
    with open(sc.joinpath('scene_gt_info.json'), 'r') as f:
        scene_gt_info = json.load(f)
    
    for k in scene_gt:
        img_id = int(k)
        if not img_id in gt[scene_id]:
            gt[scene_id][img_id] = {}
       
        for det in scene_gt[k]:
            obj_id = int(det['obj_id'])
                        
            gt[scene_id][img_id][obj_id] = {
                    "R_GT": np.reshape(det["cam_R_m2c"], (3,3)),
                    "T_GT": np.array(det["cam_t_m2c"])
                }

            
errors = []

for _, row in preds.iterrows():
    scene_id = row['scene_id']
    img_id = row['im_id']
    obj_id = row['obj_id']
    
    r_pred = np.asarray(row['R'].split(' '), dtype=float).reshape((3,3))
    t_pred = np.asarray(row['t'].split(' '), dtype=float)

    if not scene_id in gt or not img_id in gt[scene_id] or not obj_id in gt[scene_id][img_id]:
        continue

    r_gt = gt[scene_id][img_id][obj_id]['R_GT']
    t_gt = gt[scene_id][img_id][obj_id]['T_GT']
    
    R = np.dot(r_pred, r_gt.T)
    cos_theta = (np.trace(R)-1)/2
    angle = np.arccos(cos_theta) * (180/np.pi)

    point = np.array([0,50,0,1])
    gt_mat = np.eye(4)
    gt_mat[:3,:3] = r_gt
    gt_mat[:3,3] = t_gt

    pred = np.eye(4)
    pred[:3,:3] = r_pred
    pred[:3,3] = t_pred

    point_gt = gt_mat @ point
    point_pred = pred @ point
    
    err = dict(
        scene_id = scene_id,
        img_id = img_id,
        obj_id = obj_id,
        euc = np.linalg.norm(t_pred - t_gt),
        x_err = t_pred[0] - t_gt[0],
        y_err = t_pred[1] - t_gt[1],
        z_err = t_pred[2] - t_gt[2],
        angle = angle,
        euc_point=np.linalg.norm(point_pred - point_gt),
        x_err_point=point_pred[0] - point_gt[0],
        y_err_point=point_pred[1] - point_gt[1],
        z_err_point=point_pred[2] - point_gt[2],
    )
    errors.append(err)

errors = pd.DataFrame(errors)
errors_clip = errors.copy()

errors_clip[errors.abs() > 20] = np.nan
errors_melt = errors_clip.abs().melt('obj_id', ['euc', 'x_err', 'y_err', 'z_err'], 'error')
errors_melt = errors_melt.replace({'euc': 'Euclidean',
                                    'x_err': 'X Error',
                                    'y_err': 'Y Error',
                                    'z_err': 'Z Error'})
errors_melt['obj_id'] = errors_melt['obj_id'].replace({1: 'A',
                                                       2: 'B',
                                                       3: 'C',
                                                       4: 'D'})

sns.set_theme(context='paper', style='whitegrid', font='Verdana',
              palette=sns.color_palette('colorblind'), font_scale=2.5)
fig, ax = plt.subplots(1, 1)
fig.set_size_inches([17,8])

sns.violinplot(data=errors_melt, x='obj_id',y='value', hue='error', ax=ax)
ax.set_ylabel('Error (mm)')
ax.set_xlabel('Ultrasound Probe')
fig.tight_layout()
fig.savefig(pred_file.parent.joinpath(f'{pred_file.stem}_violinplot.pdf'))

fig, ax = plt.subplots(1, 1)
fig.set_size_inches([17,6])

sns.boxplot(data=errors_melt, x='obj_id',y='value', hue='error', ax=ax)
ax.set_ylabel('Error (mm)')
ax.set_xlabel('Ultrasound Probe')
fig.tight_layout()
fig.savefig(pred_file.parent.joinpath(f'{pred_file.stem}_boxplot.pdf'))

fig, ax = plt.subplots(1, 1)
fig.set_size_inches([17,8])

sns.kdeplot(data=errors_clip, x='euc', ax=ax)
ax.set_ylabel('Euclidean Distance (mm)')
plt.show()
# errors.to_csv(pred_file.parent.joinpath(f'{pred_file.stem}_errors.csv'))

# print(errors[['euc', 'x_err', 'y_err', 'z_err', 'angle']].describe())
print("Absolute Errors")
print(errors[['euc', 'x_err', 'y_err', 'z_err', 'angle']].abs().describe())

print("Absolute Errors of Point 50mm away from origin")
print(errors[['euc_point', 'x_err_point', 'y_err_point', 'z_err_point']].abs().describe())


# print(errors[['euc', 'x_err', 'y_err', 'z_err', 'angle']].describe())
err_cam = errors.join(split, 'scene_id')
err_cam.to_csv(pred_file.parent.joinpath(f'{pred_file.stem}_errors.csv'))

err_cam[['euc', 'x_err', 'y_err', 'z_err', 'angle']] = err_cam[['euc', 'x_err', 'y_err', 'z_err', 'angle']].abs()
err_cam = err_cam.groupby('camera')
print("Absolute Errors Grouped")
print(err_cam[['euc']].describe())

print("Absolute Errors of Point 50mm away from origin")
print(err_cam[['euc_point']].describe())

# Calculate Q1, Q3, and IQR
Q1 = errors['euc'].quantile(0.25)
Q3 = errors['euc'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df_cleaned_iqr = errors[(errors['euc'] >= lower_bound) & (errors['euc'] <= upper_bound)]
print("Absolute Errors With Outliers Removed")
print(df_cleaned_iqr[['euc', 'x_err', 'y_err', 'z_err', 'angle']].abs().describe())