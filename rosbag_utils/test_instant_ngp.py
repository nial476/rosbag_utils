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
from scipy.spatial.transform import Rotation as R
import time

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
    config_path = '/home/nirmal/project/kitchen/outputs/long_angle_change/instant-ngp/2024-04-03_142353'
    config_file = Path(os.path.join(config_path, 'config.yml'))
    _, pipeline, _, _ = eval_setup(config_path=config_file, 
                                   eval_num_rays_per_chunk=None, 
                                   test_mode='inference')
    camera_type = CameraType.PERSPECTIVE
    image_height = 1920
    image_width = 1080
    c2ws = []
    camera_transform = torch.tensor(
        [
        [
          0.7093300205616263,
          0.026335872998407582,
          0.7043842291242147,
          3.034529948194754
        ],
        [
          0.7043808155188771,
          0.010981254028935135,
          -0.7097371921365198,
          -2.4384534989072457
        ],
        [
          -0.026426579188530887,
          0.999592809080648,
          -0.010761057920486976,
          0.2162443440022589
        ],
        [
          0.0,
          0.0,
          0.0,
          1.0
        ]
      ])
    camera_transform[:3, 3] /= 30
    camera_transform[:3,3] = torch.tensor([-1, 1, 1])  
    c2w = camera_transform[:3]
    # print(c2w)
    c2ws.append(c2w)
    particle = torch.tensor(generate_particles(c2w, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1), dtype=torch.float32)
    # print(particle)
    c2ws.append(particle)
    # print(c2ws)
    fov = 50
    fxs = []
    fys = []
    focal_length = three_js_perspective_camera_focal_length(fov, image_height)
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
    aabb_scale = 32
    img_path = "./images/0057.jpg"
    sharpness = 100.44604623081236
    # fxs.append(focal_length)
    # fys.append(focal_length)
    fxs.append(fl_x)
    fys.append(fl_y)
    camera_to_world = torch.stack(c2ws, dim=0)
    # print(camera_to_world)
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
    W = 1161
    H = 1972
    image_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    pixel_offset = 0.5
    image_coords = torch.stack(image_coords, dim=-1)  + pixel_offset
    image_coords_random = image_coords.reshape(-1,2)
    np.random.seed(2024)
    rand_inds = np.random.choice(W*H, size=64, replace=False)
    batch = image_coords_random[rand_inds]
    # print('batch ', batch)
    print('rand_inds ', rand_inds)
    print('image_coords_random.shape ',image_coords_random.shape)
    print('batch.shape ', batch.shape)

    obb_box = None

    with torch.no_grad():
        start = time.time()
        # outputs = pipeline.model.get_outputs_for_camera(camera_obj[0:1], obb_box=obb_box)\
        # outputs = pipeline.model.get_outputs_for_camera_ray_bundle(
        rays_o = camera_obj[0:1].generate_rays(camera_indices=0, 
                                          coords=image_coords)
        # )
        # rays_o = camera_obj[0:1].generate_rays(camera_indices=0,)
        print('rays_o.shape ', rays_o.shape)
        # print(rays_o.reshape(1,1))
        outputs = pipeline.model.get_outputs_for_camera_ray_bundle(rays_o)
        print('outputs.shape ', outputs.keys())
    # rendered_output_names = field(default_factory=lambda: ['rgb'])
        rendered_output_names = ['rgb']
    redner_image = []
    for rendered_output_name in rendered_output_names:
        output_image = outputs[rendered_output_name]
        print('output_image.shape ', output_image.shape)
        output_image = (
            colormaps.apply_colormap(
                image=output_image,
                colormap_options=colormaps.ColormapOptions()
            )
            .cpu()
            .numpy()
        )
        print('output_image.shape after ', output_image.shape)
    # output_image = output_image[np.asarray(batch[:, 0]-0.5, dtype=np.int64), np.asarray(batch[:, 1]-0.5, dtype=np.int64)].reshape(8,8,3)
    # output_image = output_image[np.asarray(batch - 0.5, dtype=np.int64)]
    gt = output_image
    print('time ', time.time()-start)
    plt.imshow(output_image)
    plt.show()

    # with torch.no_grad():
    #     start = time.time()
    #     # outputs = pipeline.model.get_outputs_for_camera(camera_obj[0:1], obb_box=obb_box)\
    #     # outputs = pipeline.model.get_outputs_for_camera_ray_bundle(
    #     rays_o = camera_obj[0:1].generate_rays(camera_indices=0, 
    #                                       coords=torch.tensor(batch.reshape(8,8,2)))
    #     # )
    #     # rays_o = camera_obj[0:1].generate_rays(camera_indices=0,)
    #     print('rays_o.shape ', rays_o.shape)
    #     # print(rays_o.reshape(1,1))
    #     outputs = pipeline.model.get_outputs_for_camera_ray_bundle(rays_o)
    #     print('outputs.shape ', outputs.keys())
    # # rendered_output_names = field(default_factory=lambda: ['rgb'])
    #     rendered_output_names = ['rgb']
    # redner_image = []
    # for rendered_output_name in rendered_output_names:
    #     output_image = outputs[rendered_output_name]
    #     print('output_image.shape ', output_image.shape)
    #     output_image = (
    #         colormaps.apply_colormap(
    #             image=output_image,
    #             colormap_options=colormaps.ColormapOptions()
    #         )
    #         .cpu()
    #         .numpy()
    #     )
    #     print('output_image.shape after ', output_image.shape)
    # # output_image = output_image[np.asarray(batch[:, 0]-0.5, dtype=np.int64), np.asarray(batch[:, 1]-0.5, dtype=np.int64)]
    # # output_image = output_image[np.asarray(batch - 0.5, dtype=np.int64)]
    # mama_mia = output_image
    # print('time ', time.time()-start)
    # plt.imshow(output_image)
    # plt.show()

    # with torch.no_grad():
    #     start = time.time()
    #     # outputs = pipeline.model.get_outputs_for_camera(camera_obj[0:1], obb_box=obb_box)\
    #     # outputs = pipeline.model.get_outputs_for_camera_ray_bundle(
    #     rays_o = camera_obj[0:1].generate_rays(camera_indices=0, 
    #                                       coords=torch.tensor(batch.reshape(8,8,2)))
    #     # )
    #     # rays_o = camera_obj[0:1].generate_rays(camera_indices=0,)
    #     print('rays_o.shape ', rays_o.shape)
    #     # print(rays_o.reshape(1,1))
    #     outputs = pipeline.model.get_outputs_for_camera_ray_bundle(rays_o)
    #     print('outputs.shape ', outputs.keys())
    # # rendered_output_names = field(default_factory=lambda: ['rgb'])
    #     rendered_output_names = ['rgb']
    # redner_image = []
    # for rendered_output_name in rendered_output_names:
    #     output_image = outputs[rendered_output_name]
    #     print('output_image.shape ', output_image.shape)
    #     output_image = (
    #         colormaps.apply_colormap(
    #             image=output_image,
    #             colormap_options=colormaps.ColormapOptions()
    #         )
    #         .cpu()
    #         .numpy()
    #     )
    #     print('output_image.shape after ', output_image.shape)
    # # output_image = output_image[np.asarray(batch[:, 0]-0.5, dtype=np.int64), np.asarray(batch[:, 1]-0.5, dtype=np.int64)]
    # # output_image = output_image[np.asarray(batch - 0.5, dtype=np.int64)]
    # aashan = output_image
    # print('time ', time.time()-start)
    # plt.imshow(output_image)
    # plt.show()

    # print('aashan, gt ',np.mean(aashan, gt))

    # print('aashan, mama_mia ',np.mean(aashan - mama_mia))


    

def generate_particles(pose, p_x, p_y, p_z, r_x, r_y, r_z):
    x = pose[0, 3]
    y = pose[1, 3]
    z = pose[2, 3]
    rot = pose[:3, :3]
    r = R.from_matrix(rot)
    euler = r.as_euler('zyx')
    x += p_x * np.random.normal()
    y += p_y * np.random.normal()
    z += p_z * np.random.normal()
    euler[0] += r_x * np.random.normal()
    euler[1] += r_y * np.random.normal()
    euler[2] += r_z * np.random.normal()
    after_r = R.from_euler('zyx', euler)
    after_rot = after_r.as_matrix()
    final_pose = np.eye(4)
    final_pose[:3, :3] = after_rot
    final_pose[:3, 3] = torch.tensor([x, y, z])
    return final_pose[:3]
    
    
def main():
    # publish_particle_images()
    test_model()

if __name__ == "__main__":
    main()