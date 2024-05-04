import numpy as np
import gtsam
from scipy.spatial.transform import Rotation as R
from multiprocessing import Lock
import torch
from pathlib import Path
import os
import matplotlib.pyplot as plt
import time
import cv2
import json

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.camera_utils import get_distortion_params
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.viewer_legacy.server.utils import three_js_perspective_camera_focal_length
from nerfstudio.utils import colormaps
from datetime import datetime
from tqdm import tqdm
import math

def quaternion_dff(q1, q2):
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)

    del_w = q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]
    del_x = q1[0]*q2[1] - q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    del_y = q1[0]*q2[2] - q1[2]*q2[0] - q1[1]*q2[3] + q1[3]*q2[1]
    del_z = q1[0]*q2[3] - q1[3]*q2[0] + q1[1]*q2[2] - q1[2]*q2[1]

    theta = math.atan(np.sqrt(del_x**2 + del_y**2 + del_z**2) / del_w)
    return theta * 180 / np.pi

class ParticleFilter:
    def __init__(self, initial_particles, ngp_model, gt, path, pixels) -> None:
        self.num_particles = len(initial_particles['position'])
        self.particles = initial_particles
        self.weights = np.ones(self.num_particles)
        self.pipeline = ngp_model
        self.gt = gt
        self.gt_image = self.extract_img(gt[:3])
        self.particle_lock = Lock()
        self.path = path
        self.image_coords = torch.stack(torch.meshgrid(torch.arange(1972), torch.arange(1161), indexing="ij"), dim=-1) + 0.5
        self.pixels = pixels
        self.data_json = {'frames': []}
        # cv2.imwrite(self.path+'/gtcv.png', self.gt_image)
        plt.imsave(self.path+'/gt.png', self.gt_image)

    def reduce_num_particles(self, num_particles):
        self.particle_lock.acquire()
        self.num_particles = num_particles
        self.weights = self.weights[0:num_particles]
        self.particles = self.particles[0:num_particles]
        self.particle_lock.release()

    def predict_no_motion(self, p_x, p_y, p_z, r_x, r_y, r_z):
        self.particle_lock.acquire()
        # print(self.particles['position'][:])
        self.particles['position'][:, 0] += p_x * np.random.normal(size=self.particles['position'].shape[0])
        self.particles['position'][:, 1] += p_y * np.random.normal(size=self.particles['position'].shape[0])
        self.particles['position'][:, 2] += p_z * np.random.normal(size=self.particles['position'].shape[0])

        for i in range(len(self.particles['rotation'])):
            rot = self.particles['rotation'][i]
            r = R.from_matrix(rot)
            euler = r.as_euler('zyx')
            euler[0] += r_z * np.random.normal()
            euler[1] += r_y * np.random.normal()
            euler[2] += r_x * np.random.normal()
            error = R.from_euler('zyx', euler)
            self.particles['rotation'][i] = error.as_matrix()
        self.particle_lock.release()

    def extract_img(self, c2ws):
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
        camera_obj = camera_obj.to(self.pipeline.device)

        obb_box = None
        with torch.no_grad():
            outputs = self.pipeline.model.get_outputs_for_camera(camera_obj[0:1], obb_box=obb_box)
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
    
    def extract_pixels(self, c2ws):
        camera_type = CameraType.PERSPECTIVE
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
        camera_obj = camera_obj.to(self.pipeline.device)

        obb_box = None
        with torch.no_grad():
            rays_o = camera_obj[0:1].generate_rays(camera_indices=0, 
                                          coords=torch.tensor(self.batch.reshape(self.pixels,self.pixels,2)))
            outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(rays_o)
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

    def choose_pixels(self):
        self.rand_inds = np.random.choice(1161*1972, size=self.pixels**2, replace=False)
        image_coords_random = self.image_coords.reshape(-1,2)
        self.batch = image_coords_random[self.rand_inds]

    def likelihood(self, particle):
        weight = 0
        nerf = self.extract_pixels(torch.tensor(particle, dtype=torch.float32))
        img = self.gt_image[
            np.asarray(self.batch[:, 0]-0.5, dtype=np.int64), 
            np.asarray(self.batch[:, 1]-0.5, dtype=np.int64)].reshape(self.pixels,self.pixels,3)
        weight = 1 / np.mean((img - nerf) ** 2)
        return weight
    
    def update(self):
        for i in range(len(self.particles['position'])):
            particle_matrix = np.eye(4)
            particle_matrix[:3, :3] = self.particles['rotation'][i]
            particle_matrix[:3, 3] = self.particles['position'][i]
            self.weights[i] = self.likelihood(particle_matrix[:3])
        
        self.weights = np.square(self.weights)
        self.weights = np.square(self.weights)

        sum_weights=np.sum(self.weights)
        self.weights=self.weights / sum_weights

    def resample(self, convergence_protection, x_noise, y_noise, z_noise, no_particle):
        self.particle_lock.acquire()
        choice = np.random.choice(self.num_particles, self.num_particles, p = self.weights, replace=True)
        # print('choice = ', choice) 
        # print('best particle = ', np.argmax(self.weights))
        temp = {'position':np.copy(self.particles['position'])[choice, :], 'rotation':np.copy(self.particles['rotation'])[choice]}
        self.particles = temp
        if convergence_protection:
            for i in range(no_particle):
                x = np.random.uniform(low=x_noise[0], high=x_noise[1])
                y = np.random.uniform(low=x_noise[0], high=x_noise[1])
                z = np.random.uniform(low=x_noise[0], high=x_noise[1])
                self.particles['position'][i] = np.array([x,y,z])
        self.particle_lock.release()

    def plot_particles(self, iter):
        best_idx = np.argmax(self.weights)
        best_particle = np.eye(4)
        best_particle[:3, :3] = self.particles['rotation'][best_idx]
        best_particle[:3, 3] = self.particles['position'][best_idx]
        plt.clf()
        ax = plt.figure().add_subplot(projection='3d')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.5, 0.5])
        best = np.argmax(self.weights)
        pos = self.gt[:3, 3]
        rot = R.from_matrix(self.gt[:3, :3])
        e = rot.as_euler('zyx')
        for i in range(len(self.particles['position'])):
            x, y, z = self.particles['position'][i]
            r = R.from_matrix(self.particles['rotation'][i])
            z_theta, y_theta, x_theta = r.as_euler('zyx')
            if i == best:
                ax.quiver(x, y, z, x_theta, y_theta, z_theta, length=0.15, normalize=True, color='r')
            else:
                ax.quiver(x, y, z, x_theta, y_theta, z_theta, length=0.1, normalize=True, color='b')
            ax.quiver(pos[0], pos[1], pos[2], e[2], e[1], e[0], length=0.2, normalize=True, color='g')
        ax.plot([], [], [], color='r', label='best particle')
        ax.plot([], [], [], color='b', label='particle')
        ax.plot([], [], [], color='g', label='ground truth')
        ax.legend()
        plt.savefig(self.path+'/iter_'+str(iter)+'.png')
        plt.close()
        return best_particle

    def pos_avg(self):
        avg_pos = np.average(self.particles['position'], axis=0)
        return avg_pos
    
    def weighted_pos_avg(self):
        avg_pos = np.average(self.particles['position'], weights=self.weights, axis=0)
        return avg_pos
    
    def rot_avg(self):
        a = np.zeros((4,4))
        w_sum = 0

        for i in range(self.num_particles):
            rot = R.from_matrix(self.particles['rotation'][i])
            q = rot.as_quat()
            w = self.weights[i]
            a += w * (np.outer(q, q))
            w_sum += w
        
        a /= w_sum
        return np.linalg.eigh(a)[1][:, -1]
    
    def export_run(self, iter):
        self.data_json['frames'].append(
            {
                'iteration': iter

            }
        )


# def initialize_particles():
#     gt = torch.tensor(
#         [
#         [
#           0.7093300205616263,
#           0.026335872998407582,
#           0.7043842291242147,
#           3.034529948194754
#         ],
#         [
#           0.7043808155188771,
#           0.010981254028935135,
#           -0.7097371921365198,
#           -2.4384534989072457
#         ],
#         [
#           -0.026426579188530887,
#           0.999592809080648,
#           -0.010761057920486976,
#           0.2162443440022589
#         ],
#         [
#           0.0,
#           0.0,
#           0.0,
#           1.0
#         ]
#       ])
    
#     gt[:3, 3] /= 30
#     pos = gt[:3, 3]
#     rot_matrix = gt[:3, :3]
#     rot = R.from_matrix(rot_matrix)
#     e = rot.as_euler('zyx')
   
#     x = pos[0] + 1 * np.random.normal(scale=0.1, size=100)
#     y = pos[1] + 1 * np.random.normal(scale=0.1, size=100)
#     z = pos[2] + 1 * np.random.normal(scale=0.1, size=100)
#     x_theta = e[2] + 1 * np.random.normal(size=100)
#     y_theta = e[1] + 1 * np.random.normal(size=100)
#     z_theta = e[0] + 1 * np.random.normal(size=100)
    
#     return gt, x, y, z, x_theta, y_theta, z_theta

def initialize_particles(num_of_particles, x_noise, y_noise, z_noise, x_theta_noise, y_theta_noise, z_theta_noise):
    gt = torch.tensor(
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
    
    gt[:3, 3] /= 30
    
    x = np.random.uniform(low=x_noise[0], high=x_noise[1], size=num_of_particles)
    y = np.random.uniform(low=y_noise[0], high=y_noise[1], size=num_of_particles)
    z = np.random.uniform(low=z_noise[0], high=z_noise[1], size=num_of_particles)
    x_theta = np.random.uniform(low=x_theta_noise[0], high=x_theta_noise[1], size=num_of_particles) * np.pi / 180
    y_theta = np.random.uniform(low=y_theta_noise[0], high=y_theta_noise[1], size=num_of_particles) * np.pi / 180
    z_theta = np.random.uniform(low=z_theta_noise[0], high=z_theta_noise[1], size=num_of_particles) * np.pi / 180

    return gt, x, y, z, x_theta, y_theta, z_theta

def main():

    now = datetime.now()
    print('making directory')
    path = '/home/nirmal/project/kitchen/'+str(now)
    os.makedirs(path)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config_path = '/home/nirmal/project/kitchen/outputs/long_angle_change/instant-ngp/2024-04-03_142353'
    config_file = Path(os.path.join(config_path, 'config.yml'))
    _, pipeline, _, _ = eval_setup(config_path=config_file, 
                                   eval_num_rays_per_chunk=None, 
                                   test_mode='inference')
    
    x_noise=[-0.5, 0.5]
    y_noise=[-0.5, 0.5]
    z_noise=[-0.5, 0.5]
    x_theta_noise=[40, 92]
    y_theta_noise=[17, 81]
    z_theta_noise=[-12, 68]

    motion_x = 0.01
    motion_y = 0.01
    motion_z = 0.01

    convergence_protection = False
    flag = True
    pixel_along_axis = 8

    gt, x, y, z, x_theta, y_theta, z_theta = initialize_particles(100, x_noise, y_noise, z_noise, x_theta_noise, y_theta_noise, z_theta_noise)
    rot_matrix = gt[:3, :3]
    rot = R.from_matrix(rot_matrix)
    particles= {'position': np.zeros((len(x), 3)),
                'rotation': np.zeros((len(x), 3, 3))}
    for i in range(len(x)):
        r = R.from_euler('zyx', torch.tensor([z_theta[i], y_theta[i], x_theta[i]]))
        m = r.as_matrix()
        particles['position'][i] = np.array([x[i], y[i], z[i]])
        particles['rotation'][i] = m
    # print(particles['position'])
    particles['position'] = np.array(particles['position'], dtype='object')
    particles['rotation'] = np.array(particles['rotation'], dtype='object')

    # print(pixel_list)
    filter = ParticleFilter(
        initial_particles=particles,
        ngp_model=pipeline,
        gt=gt,
        path=path,
        pixels = pixel_along_axis
    )
    error_pos = []
    error_rot = []
    data = {'frames': []}
    gt_r = R.from_matrix(gt[:3, :3]) 
    gt_q = gt_r.as_quat() 
    for iter in tqdm(range(500)):
        filter.choose_pixels()
        filter.predict_no_motion(motion_x, motion_y, motion_z, 0.02, 0.02, 0.02)
        filter.update()
        best = filter.plot_particles(iter)
        a_pos = filter.weighted_pos_avg()
        filter.resample(convergence_protection, x_noise=x_noise, y_noise=y_noise, z_noise=z_noise, no_particle=10)
        # print(np.argmin(filter.weights), filter.weights)
        a_q = filter.rot_avg()
        # print((gt[0:3, 3].numpy() - a_pos)*30)
        pos_error = np.linalg.norm((gt[0:3, 3].numpy() - a_pos)*30)
        rot_error = abs(quaternion_dff(gt_q, a_q))
        best_pos_error = np.linalg.norm((gt[0:3, 3].numpy() - best[0:3, 3])*30)
        best_rot = R.from_matrix(best[:3, :3]).as_quat()
        best_rot_error = abs(quaternion_dff(gt_q, best_rot))
        # if rot_error >= 90:
        #     rot_error = 180 - rot_error
        data['frames'].append(
            {
                'iter': iter,
                'avg_pos': pos_error,
                'avg_rot': rot_error,
                'best_particle': best.tolist(),
                'best_pos': best_pos_error,
                'best_rot': best_rot_error
            }
        )
        error_pos.append(pos_error)
        error_rot.append(rot_error)


        if pos_error < 0.13 and flag:
            print('refining')
            motion_x /= 2
            motion_y /= 2
            motion_z /= 2
            flag = False

        # if pos_error < 0.15 and rot_error < 0.2:
        #     break

    plt.clf()
    plt.title('Average position error vs iteration')
    plt.ylabel('avg position error')
    plt.xlabel('iteration')
    plt.plot(error_pos)
    plt.savefig('/home/nirmal/project/kitchen/'+str(now)+'/position_error'+'.png')
    # plt.show()

    plt.clf()
    plt.title('Average rotation error vs iteration')
    plt.ylabel('avg rotation error')
    plt.xlabel('iteration')
    plt.plot(error_rot)
    plt.savefig('/home/nirmal/project/kitchen/'+str(now)+'/rotation_error'+'.png')
    # plt.show()

    save_path = '/home/nirmal/project/kitchen/'+str(now)
    json_obj = json.dumps(data)
    with open(os.path.join(save_path, 'run_data.json'), 'w') as f:
        f.write(json_obj)


if __name__ == "__main__":
    np.random.seed(42)
    main()

