import torch
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp
from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
import argparse
import os
import src.utils.masking as masking_utils
from utils.mediapipe_utils import run_mediapipe
from datasets.base_dataset import create_mask
import torch.nn.functional as F

from tqdm import tqdm

import pickle

mediapipe_indices = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336,  46,  52,  53,
        55,  63,  65,  66,  70, 105, 107, 249, 263, 362, 373, 374, 380,
       381, 382, 384, 385, 386, 387, 388, 390, 398, 466,   7,  33, 133,
       144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
       168,   6, 197, 195,   5,   4, 129,  98,  97,   2, 326, 327, 358,
         0,  13,  14,  17,  37,  39,  40,  61,  78,  80,  81,  82,  84,
        87,  88,  91,  95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
       308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409,
       415]

def batch_orth_proj(X, camera):
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    # print('-------------')
    # print(camera.shape)
    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn

def batch_draw_keypoints(image, landmarks, color=(255, 255, 255), radius=1):
    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.cpu().numpy()

    if isinstance(image, torch.Tensor):
        image = image.squeeze(0)
        image = image.cpu().numpy().transpose(1,2,0)
        image = (image * 255).astype('uint8')
        image = np.ascontiguousarray(image[..., ::-1])
    
    for point in landmarks:
        image = cv2.circle(image, (int(point[0]), int(point[1])), radius, color, -1)
   
    return image

def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path_source', type=str, default='', help='Path to the input image')
    parser.add_argument('--input_path_target', type=str, default='', help='Path to the input image')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to the checkpoint')
    parser.add_argument('--crop', action='store_true', help='Crop the face using mediapipe')
    parser.add_argument('--out_path', type=str, default='', help='Path to save the output (will be created if not exists)')
    parser.add_argument('--use_smirk_generator', action='store_true', help='Use SMIRK neural image to image translator to reconstruct the image')
    parser.add_argument('--render_orig', action='store_true', help='Present the result w.r.t. the original image/video size')

    args = parser.parse_args()

    image_size = 224
    

    # ----------------------- initialize configuration ----------------------- #
    smirk_encoder = SmirkEncoder().to(args.device)
    checkpoint = torch.load(args.checkpoint)
    checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k} # checkpoint includes both smirk_encoder and smirk_generator

    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()

    if args.use_smirk_generator:
        from src.smirk_generator import SmirkGenerator
        smirk_generator = SmirkGenerator(in_channels=6, out_channels=3, init_features=32, res_blocks=5).to(args.device)

        checkpoint_generator = {k.replace('smirk_generator.', ''): v for k, v in checkpoint.items() if 'smirk_generator' in k} # checkpoint includes both smirk_encoder and smirk_generator
        smirk_generator.load_state_dict(checkpoint_generator)
        smirk_generator.eval()

    # ---- visualize the results ---- #

    flame = FLAME().to(args.device)
    renderer = Renderer().to(args.device)

    with torch.no_grad():
        image = cv2.imread(args.input_path_source)
        orig_image_height, orig_image_width, _ = image.shape

        kpt_mediapipe = run_mediapipe(image)
            
        kpt_mediapipe = kpt_mediapipe[..., :2]

        tform = crop_face(image,kpt_mediapipe,scale=1.4,image_size=image_size)
        
        cropped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)

        cropped_kpt_mediapipe = np.dot(tform.params, np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0],1])]).T).T
        cropped_kpt_mediapipe = cropped_kpt_mediapipe[:,:2]
        
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = cv2.resize(cropped_image, (224,224))
        cropped_image = torch.tensor(cropped_image).permute(2,0,1).unsqueeze(0).float()/255.0
        cropped_image = cropped_image.to(args.device)

        outputs = smirk_encoder(cropped_image)


        flame_output = flame.forward(outputs)
        
    
        renderer_output = renderer.forward(flame_output['vertices'], outputs['cam'],
                                            landmarks_fan=flame_output['landmarks_fan'], landmarks_mp=flame_output['landmarks_mp'])
        
        rendered_img  = renderer_output['rendered_img']
            
            
        #token of target image
        image2 = cv2.imread(args.input_path_target)
        orig_image_height, orig_image_width, _ = image2.shape

        kpt_mediapipe = run_mediapipe(image2)


        kpt_mediapipe = kpt_mediapipe[..., :2]

        tform = crop_face(image2,kpt_mediapipe,scale=1.4,image_size=image_size)
        
        cropped_image_2 = warp(image2, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)

        cropped_kpt_mediapipe_2 = np.dot(tform.params, np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0],1])]).T).T
        cropped_kpt_mediapipe_2 = cropped_kpt_mediapipe_2[:,:2]

        
        cropped_image_2 = cv2.cvtColor(cropped_image_2, cv2.COLOR_BGR2RGB)
        cropped_image_2 = cv2.resize(cropped_image_2, (224,224))
        cropped_image_2 = torch.tensor(cropped_image_2).permute(2,0,1).unsqueeze(0).float()/255.0
        cropped_image_2 = cropped_image_2.to(args.device)

        outputs_2 = smirk_encoder(cropped_image_2)


        # ---- create the neural renderer reconstructed img ---- #
        if args.use_smirk_generator:
            if (kpt_mediapipe is None):
                print('Could not find landmarks for the image using mediapipe and cannot create the hull mask for the smirk generator. Exiting...')
                exit()

            mask_ratio_mul = 5
            mask_ratio = 0.01
            mask_dilation_radius = 10

            hull_mask = create_mask(cropped_kpt_mediapipe, (224, 224))

            rendered_mask = 1 - (rendered_img == 0).all(dim=1, keepdim=True).float()
    
            
            hull_mask = torch.from_numpy(hull_mask).type(dtype = torch.float32).unsqueeze(0).to(args.device)

            masked_img = masking_utils.masking_face(cropped_image, hull_mask, mask_dilation_radius, rendered_mask=rendered_mask)

            smirk_generator_input = torch.cat([rendered_img, masked_img], dim=1)
            
            
            #use target(image2) token
            reconstructed_img = smirk_generator(smirk_generator_input, outputs_2['token'])
            
            grid = torch.cat([cropped_image, reconstructed_img,cropped_image_2], dim=3)

        grid_numpy = grid.squeeze(0).permute(1,2,0).detach().cpu().numpy()*255.0
        grid_numpy = grid_numpy.astype(np.uint8)
        grid_numpy = cv2.cvtColor(grid_numpy, cv2.COLOR_BGR2RGB)

        if not os.path.exists(args.out_path):
            os.makedirs(args.out_path)
        
        image_name = 'swap_token_'+ args.input_path_source.split('/')[-1].split('.')[0] + '_' + args.input_path_target.split('/')[-1]

        cv2.imwrite(f"{args.out_path}/{image_name}", grid_numpy)

        
            
            

