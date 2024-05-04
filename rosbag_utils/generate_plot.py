import matplotlib.pyplot as plt
import os
import json
import matplotlib.animation as animation
import numpy as np
import cv2
from matplotlib.animation import FFMpegWriter as video
from PIL import Image
from tqdm import tqdm


def main():
    plt.rcParams["figure.autolayout"] = True
    initial = np.array(
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
    file_path = '/home/nirmal/project/kitchen/2024-05-03 06:26:08.867443'
    file_name = 'run_data.json'
    with open(os.path.join(file_path, file_name)) as f:
        data = json.load(f)
    frames = data['frames']
    gt_path = os.path.join(file_path, 'gt.png')
    gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
    gt_border = np.array(Image.new(mode='RGB',
                          size = (gt.shape[0] + 40, gt.shape[1]+40),
                          color=(0, 128, 0)).getdata()
                          ).reshape(gt.shape[0]+40, gt.shape[1]+40, 3)
    gt_border[20:-20, 20:-20] = gt
    fig = plt.figure()
    fig.text(0.48, 0.01, '# of iterations')
    # gs = fig.add_gridspec(8,6
    # plt.gca().set_aspect('equal')
    gs = fig.add_gridspec(16,9)
    fig.set_size_inches(19.2, 10.8, True)
    ax1 = fig.add_subplot(gs[:10, 0:2])
    ax1.axis('off')
    ax1.imshow(gt_border)
    ax1.set_title('ground truth image')
    ax2 = fig.add_subplot(gs[:10, 2:4])
    ax2.axis('off')
    ax2.set_title('view of the best particle')
    ax3 = fig.add_subplot(gs[:, 4:])
    ax3.axis('off')
    ax4 = fig.add_subplot(gs[10:, 0:2])
    ax4.set_title('position error vs # of iteration')
    ax4.set_ylabel('position error')
    ax4.plot([], [], color='b', label='avg particle error')
    ax4.plot([], [], color='r', label='best particle error')
    ax4.legend(loc='upper right')
    ax5 = fig.add_subplot(gs[10:, 2:4])
    ax5.set_title('rotation error vs # of iteration')
    ax5.set_ylabel('arotational error (degrees)')
    ax5.plot([], [], color='b', label='avg particle error')
    ax5.plot([], [], color='r', label='best particle error')
    ax5.legend(loc='upper right')
    # artists = ['pos', 'rot', 'image']
    artist_pos = []
    artist_rot = []
    best_pos = []
    best_rot = []
    iteration = []
    for i in range(len(frames)):
        artist_pos.append(frames[i]['avg_pos'])
        artist_rot.append(frames[i]['avg_rot'])
        best_pos.append(frames[i]['best_pos'])
        best_rot.append(frames[i]['best_rot'])
        iteration.append(frames[i]['iter'])
    
    artist = []
    for i in tqdm(range(len(frames))):
        text_container = fig.text(0.48, 0.98, 'Iteration '+str(i),)
        container_pos = ax4.plot(iteration[:i+1], artist_pos[:i+1], color='b')
        container_best_pos = ax4.plot(iteration[:i+1], best_pos[:i+1], color='r')
        container_rot = ax5.plot(iteration[:i+1], artist_rot[:i+1], color='b')
        container_best_rot = ax5.plot(iteration[:i+1], best_rot[:i+1], color='r')
        frame_path = os.path.join(file_path, 'iter_'+str(i)+'.png')
        frame = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        particle_path = os.path.join(file_path, 'best_'+str(i)+'.png')
        best_img = cv2.cvtColor(cv2.imread(particle_path), cv2.COLOR_BGR2RGB)
        best_img_border = np.array(Image.new(mode='RGB',
                          size = (best_img.shape[0] + 40, best_img.shape[1]+40),
                          color=(255, 0, 0)).getdata()
                          ).reshape(best_img.shape[0]+40, best_img.shape[1]+40, 3)
        best_img_border[20:-20, 20:-20] = best_img
        # print(frame.shape)
        frame_container = ax3.imshow(frame[:, 150:])
        best_img_container = ax2.imshow(best_img_border)
        container = container_pos + container_rot \
            + container_best_pos + container_best_rot + [frame_container] \
            + [best_img_container] + [text_container]
        artist.append(container)
    # print(len(artist))
    ani = animation.ArtistAnimation(fig=fig, artists=artist, interval=400)
    # plt.show()
    outpath = os.path.join(file_path, 'particle_filter.mp4')
    # writer = video(fps=5)
    ani.save(outpath, writer='ffmpeg')






    
    # fig, ax = plt.subplots()
    # rng = np.random.default_rng(19680801)
    # data = np.array([20, 20, 20, 20])
    # x = np.array([1, 2, 3, 4])

    # artists = []
    # colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']
    # for i in range(20):
    #     data += rng.integers(low=0, high=10, size=data.shape)
    #     container = ax.barh(x, data, color=colors)
    #     artists.append(container)
    #     print(i, x, data, len(artists))


    # ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=400)
    # plt.show()


if __name__ == "__main__":
    main()
