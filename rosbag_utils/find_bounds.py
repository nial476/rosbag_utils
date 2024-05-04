import numpy as np
import json
from scipy.spatial.transform import Rotation as R

file_path = '/home/nirmal/project/kitchen/long_angle_change/transforms.json'
with open(file=file_path) as f:
    data = json.load(f)
frames = data['frames']
x= []
y = []
z = []
x_theta = []
y_theta = []
z_theta = []
for frame in frames:
    transform = frame['transform_matrix']
    x.append(np.array(transform)[0, 3])
    y.append(np.array(transform)[1, 3])
    z.append(np.array(transform)[2, 3])
    rot = R.from_matrix(np.array(transform)[:3, :3])
    theta = rot.as_euler('zyx', degrees=True)
    x_theta.append(theta[2])
    y_theta.append(theta[1])
    z_theta.append(theta[0])

print('min(x) = ', min(x)/30, 'max(x) = ', max(x)/30)
print('min(y) = ', min(y)/30, 'max(y) = ', max(y)/30)
print('min(z) = ', min(z)/30, 'max(z) = ', max(z)/30)
print('min(x_theta) = ', min(x_theta), 'max(x_theta) = ', max(x_theta))
print('min(y_theta) = ', min(y_theta), 'max(y_theta) = ', max(y_theta))
print('min(z_theta) = ', min(z_theta), 'max(z_theta) = ', max(z_theta))
