import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.camera_utils import get_distortion_params
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils import colormaps
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def extract_img(pipeline, c2ws):
        camera_type = CameraType.PERSPECTIVE
        image_height = 1920
        image_width = 1080
        fov = 50
        fxs = []
        fys = []
        fl_x = 1659.24
        fl_y = 1664.35
        k1 = 0.0464663
        k2 = -0.0542667
        k3 = 0
        k4 = 0
        p1 = -0.00262836
        p2 = 0.00706173
        cx = 580.531
        cy = 986.055
        # fxs.append(focal_length)
        # fys.append(focal_length)
        fxs.append(fl_x)
        fys.append(fl_y)
        camera_to_world = torch.stack([c2ws], dim=0)
        fx = torch.tensor(fxs)
        fy = torch.tensor(fys)
        camera_obj = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_to_worlds=camera_to_world,
            camera_type=camera_type,
            distortion_params=get_distortion_params(
                k1=k1, k2=k2, k3=k3, k4=k4, p1=p1, 
                p2=p2),
            times=None
        )
        camera_obj = camera_obj.to(pipeline.device)

        obb_box = None
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera_obj[0:1], obb_box=obb_box)
        # rendered_output_names = field(default_factory=lambda: ['rgb'])
            rendered_output_names = ['rgb']
        redner_image = []
        for rendered_output_name in rendered_output_names:
            output_image = outputs[rendered_output_name]
            output_image = (
                colormaps.apply_colormap(
                    image=output_image,
                    colormap_options=colormaps.ColormapOptions()
                )
                .cpu()
                .numpy()
            )
            redner_image.append(output_image)
        redner_image_np = np.concatenate(redner_image, axis=1)
        return redner_image_np

def main():
    config_path = '/home/nirmal/project/kitchen/outputs/long_angle_change/instant-ngp/2024-04-03_142353'
    config_file = Path(os.path.join(config_path, 'config.yml'))
    _, pipeline, _, _ = eval_setup(config_path=config_file, 
                                   eval_num_rays_per_chunk=None, 
                                   test_mode='inference')

    file_path = '/home/nirmal/project/kitchen/2024-05-03 06:35:00.547328'
    file_name = 'run_data.json'
    with open(os.path.join(file_path, file_name)) as f:
        data = json.load(f)
    frames = data['frames']
    for iter, frame in enumerate(tqdm(frames)):
        view = torch.tensor(frame['best_particle'])
        render = extract_img(pipeline, view[:3])
        outpath = os.path.join(file_path, 'best_'+str(iter)+'.png')
        plt.imsave(outpath, render)
        
     

if __name__ == "__main__":
    main()