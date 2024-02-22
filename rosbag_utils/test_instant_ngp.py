import torch
from nerfstudio.models.instant_ngp import NGPModel
from nerfstudio.data.scene_box import SceneBox
import yaml
import os
import json
import torch.nn as nn
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.camera_utils import get_distortion_params
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.viewer_legacy.server.utils import three_js_perspective_camera_focal_length
from dataclasses import dataclass, field
from nerfstudio.utils import colormaps
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# def publish_particle_images():
#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#     config_path = '/home/nirmal/project/spot_ros_data/nerf/2023-12-12-21-43-32/instant-ngp/2024-02-14_221838'
#     with open(os.path.join(config_path, 'config.yml'), 'r') as file:
#         nerf_config = yaml.load(file, Loader=yaml.Loader)
#     nerf_model_config = nerf_config.pipeline.model
#     # with open(os.path.join(config_path, 'scene_box.txt'), 'r') as file:
#     #     sb = file.read()
#     # print(sb)
#     sb = torch.tensor([[-1., -1., -1.],
#                        [1., 1., 1.]])
#     sbox = SceneBox(aabb=sb)
#     with open(os.path.join(config_path, 'num_train_data.txt'), 'r') as file:
#         num_data = file.read()
#     # print(num_data)
#     model = NGPModel(nerf_model_config, scene_box=sbox, num_train_data=int(num_data), device=device)
#     # model_parallel = nn.DataParallel(model)
#     checkpoint = torch.load(os.path.join(config_path, 'nerfstudio_models/step-000029999.ckpt'))
#     # print(checkpoint['pipeline'].keys())
#     # print(checkpoint.keys())
#     model.load_state_dict(checkpoint['pipeline'], strict=False)
#     transform = torch.tensor([[
#         [
#           0.8365889327839351,
#           0.046639198749609465,
#           0.5458422322893867,
#           -4.538495107286628
#         ],
#         [
#           0.5468037757006392,
#           -0.1320804788812945,
#           -0.8267771028948423,
#           0.7700811333448259
#         ],
#         [
#           0.03353488181106085,
#           0.9901411678192751,
#           -0.13599955685667334,
#           -0.27963059852122385
#         ],
#         [
#           0.0,
#           0.0,
#           0.0,
#           1.0
#         ]
#       ]])
#     # print(transform)
#     camera_angle_x = 1.1007795572213026
#     camera_angle_y = 8801567899376848
#     fl_x = 521.477
#     fl_y = 509.688
#     k1 = 0.0612242
#     k2 = -0.117107
#     k3 = 0
#     k4 = 0
#     p1 = 0.000493165
#     p2 = 0.000756127
#     is_fisheye = False
#     cx = 314.352
#     cy = 234.32
#     w = 640.0
#     h = 480.0
#     aabb_scale = 32
#     img_path = "./images/40.jpg"
#     sharpness = 436.3734569619072


#     # # test_camera = Cameras(
#     #     camera_to_worlds=transform[:,:3, :],
#     #     fx=fl_x,
#     #     fy= fl_y,
#     #     cx=cx,
#     #     cy=cy,
#     #     width=int(w),
#     #     height=int(h),
#     #     distortion_params=get_distortion_params(
#     #         k1=k1, k2=k2, k3=k3, k4=k4, p1=p1, 
#     #         p2=p2),
#     #     camera_type=CameraType.PERSPECTIVE             
#     # )
#     test_camera=Cameras(camera_to_worlds=torch.tensor([[[ 2.7239e-01, -3.9938e-02, -9.6136e-01, -4.1531e-01],                                                                          
#          [-9.6219e-01, -1.1306e-02, -2.7216e-01, -1.1757e-01],                                      
#          [ 1.6653e-16,  9.9914e-01, -4.1508e-02, -1.7932e-02]]], device='cuda:0'), 
#         fx=torch.tensor([[703.7417]], device='cuda:0', dtype=torch.float64),             
#         fy=torch.tensor([[703.7417]], device='cuda:0', dtype=torch.float64), 
#         cx=torch.tensor([[960.]], device='cuda:0'),
#         cy=torch.tensor([[540.]], device='cuda:0'), 
#         width=torch.tensor([[1920]], device='cuda:0'),                      
#         height=torch.tensor([[1080]], device='cuda:0'), 
#         distortion_params=None, 
#         camera_type=torch.tensor([[1]], device='cuda:0'), 
#         times=None, 
#         metadata=None)  
#     # test_camera = test_camera.to(device=device)
    
#     print(test_camera[0])
    
#     with torch.no_grad():
#       model = model.to(device)
#       test_image = model.get_outputs_for_camera(test_camera[0:1], obb_box=None)

def test_model():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config_path = '/home/nirmal/project/spot_ros_data/nerf/2023-12-12-21-43-32/instant-ngp/2024-02-14_221838'
    config_file = Path(os.path.join(config_path, 'config.yml'))
    _, pipeline, _, _ = eval_setup(config_path=config_file, 
                                   eval_num_rays_per_chunk=None, 
                                   test_mode='inference')
    camera_type = CameraType.PERSPECTIVE
    image_height = 480
    image_width = 640
    c2ws = []
    camera_transform = torch.tensor(
        [[
          0.8365889327839351,
          0.046639198749609465,
          0.5458422322893867,
          -4.538495107286628
        ],
        [
          0.5468037757006392,
          -0.1320804788812945,
          -0.8267771028948423,
          0.7700811333448259
        ],
        [
          0.03353488181106085,
          0.9901411678192751,
          -0.13599955685667334,
          -0.27963059852122385
        ],
        [
          0.0,
          0.0,
          0.0,
          1.0
        ]])
    # camera_transform[:3, 3] /= 10 
    c2w = camera_transform[:3]
    # print(c2w)
    c2ws.append(c2w)
    # print(c2ws)
    fov = 75
    fxs = []
    fys = []
    focal_length = three_js_perspective_camera_focal_length(fov, image_height)
    fl_x = 521.477
    fl_y = 509.688
    k1 = 0.0612242
    k2 = -0.117107
    k3 = 0
    k4 = 0
    p1 = 0.000493165
    p2 = 0.000756127
    cx = 314.352
    cy = 234.32
    aabb_scale = 32
    img_path = "./images/40.jpg"
    sharpness = 436.3734569619072
    # fxs.append(focal_length)
    # fys.append(focal_length)
    fxs.append(fl_x)
    fys.append(fl_y)
    camera_to_world = torch.stack(c2ws, dim=0)
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
    plt.imshow(redner_image_np)
    plt.show()

    
    
def main():
    # publish_particle_images()
    test_model()

if __name__ == "__main__":
    main()