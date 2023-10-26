import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from frames_dataset import PairedDataset
from logger import Logger, Visualizer
import imageio
from scipy.spatial import ConvexHull
import numpy as np

from sync_batchnorm import DataParallelWithCallback

# responsible for normalizing keypoints (kp) that are used for aligning and deforming images in the animation generation process
def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    # checks if adapt_movement_scale is True. If it's True, the movement scale is adapted based on the areas enclosed by keypoints.
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    # A copy of the keypoints from the driving video (kp_driving) is made and stored in kp_new
    kp_new = {k: v for k, v in kp_driving.items()}

    # further adjustments are made to the keypoints
    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        # additional adjustment is made to the keypoints using Jacobian matrices
        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new


def animate(config, generator, kp_detector, checkpoint, log_dir, dataset):
    # directory Setup
    log_dir = os.path.join(log_dir, 'animation')
    png_dir = os.path.join(log_dir, 'png')
    # assign configuration parameters
    animate_params = config['animate_params']

    # A dataset is created using PairedDataset. It's initialized with the original dataset, and the number of pairs is determined by animate_params
    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=animate_params['num_pairs'])
    # set up for iterating over the dataset
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # checks if a checkpoint is provided. If so, it loads the model weights for the generator and kp_detector using the Logger.load_cpk method. 
    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='animate'.")

    # checks if both directories exists and if not, it creates them
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    # checks if a GPU is available, if available it wraps the generator and kp_detector models with DataParallelWithCallback
    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    # the generator and keypoint detector are set to evaluation mode
    generator.eval()
    kp_detector.eval()

    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            predictions = []
            visualizations = []

            driving_video = x['driving_video']
            source_frame = x['source_video'][:, :, 0, :, :]

            kp_source = kp_detector(source_frame)
            kp_driving_initial = kp_detector(driving_video[:, :, 0])

            for frame_idx in range(driving_video.shape[2]):
                driving_frame = driving_video[:, :, frame_idx]
                kp_driving = kp_detector(driving_frame)         # key points are generalized
                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                       kp_driving_initial=kp_driving_initial, **animate_params['normalization_params'])
                out = generator(source_frame, kp_source=kp_source, kp_driving=kp_norm)

                out['kp_driving'] = kp_driving
                out['kp_source'] = kp_source
                out['kp_norm'] = kp_norm

                del out['sparse_deformed']

                # prediction frame is added to the predictions list
                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                # Visualization data is created using the Visualizer with the source and driving frames and the out dictionary
                visualization = Visualizer(**config['visualizer_params']).visualize(source=source_frame,
                                                                                    driving=driving_frame, out=out)
                visualization = visualization
                visualizations.append(visualization)

            predictions = np.concatenate(predictions, axis=1)
            result_name = "-".join([x['driving_name'][0], x['source_name'][0]])
            imageio.imsave(os.path.join(png_dir, result_name + '.png'), (255 * predictions).astype(np.uint8))

            image_name = result_name + animate_params['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)
