import sys,pdb,os,time
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image
from lavis.models import model_zoo
from lavis.models import load_model_and_preprocess
from lavis.common.gradcam import getAttMap
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
import cv2
import scipy.ndimage
import gzip
import json

def crop_img(image, vis_processors,scales=[1,2,4]):
    """
        Crops the image into multiple patches at different scales and visualizes the patches.

        Parameters:
        - image_path: The path to the image to be processed.
        - scales: A list of scales at which to crop the image.

        Returns:
        - A list of PIL Image objects representing the cropped patches.
        """

    # Initialize a list to hold the cropped images
    cropped_images = []
    cropped_batch=[]
    # Define the crop function using numpy slicing

    def crop_numpy(img, num_x, num_y):
        # Convert to numpy array for slicing
        np_img = np.array(img)
        h, w, _ = np_img.shape
        patches=list(Image.fromarray(np_img[i * h // num_y:(i + 1) * h // num_y,
                                        j * w // num_x:(j + 1) * w // num_x, :])
         for i in range(num_y) for j in range(num_x))
        patches_batch = list(vis_processors(Image.fromarray(np_img[i * h // num_y:(i + 1) * h // num_y,
                                   j * w // num_x:(j + 1) * w // num_x, :])).to(device)
                   for i in range(num_y) for j in range(num_x))

        return patches, patches_batch

    for scale in scales:
        # Calculate number of patches in each axis
        num_x, num_y = scale, scale
        # Crop using numpy and store in dictionary
        patch,batch=crop_numpy(image, num_x, num_y)
        cropped_images.extend(patch)
        cropped_batch.extend(batch)

    return cropped_images,torch.stack(cropped_batch)

def crop_img_padding(image, vis_processors,scales=[0.5, 0.25]):
    w, h = image.size
    patches = [image]
    for scale in scales:
        # Determine the size of the patches
        new_w, new_h = int(w * scale), int(h * scale)

        # Calculate the number of patches to create
        patches_across_width = w // new_w
        patches_across_height = h // new_h

        for i in range(patches_across_width):
            for j in range(patches_across_height):
                # Calculate the left, upper, right, and lower pixel coordinates for cropping
                left = i * new_w
                upper = j * new_h
                right = left + new_w
                lower = upper + new_h

                # Crop the patch and create a masked image of the original size
                patch = image.crop((left, upper, right, lower))
                mask = Image.new("RGB", (w, h), (0, 0, 0))
                mask.paste(patch, (left, upper))
                patches.append(mask)
    return patches #from top to down, from large patch to small patch

def vis_patches(cropped_images,pth):
    # Visualize the cropped images
    fig, axes = plt.subplots(len(cropped_images), 1, figsize=(10, len(cropped_images) * 5))
    if len(cropped_images) == 1:
        axes = [axes]
    for ax, img in zip(axes, cropped_images):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(pth)

def save_images_with_scores(batch, itc_score, attributes,directory):
    for attr in attributes:
        for i, tensor in enumerate(batch):
            # Convert the tensor to a PIL Image
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:,None,None].cuda()
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:,None,None].cuda()
            # pdb.set_trace()
            tensor=tensor*std+mean
            image = to_pil_image(tensor)
            score = itc_score[attr][i].item()
            # Save the image with the score in the filename
            filename = f"image_{i}_attri_{attr}_score_{score:.4f}.png"
            image.save(os.path.join(directory, filename))

def depth_to_colormap(depth_array, colormap=cv2.COLORMAP_JET):
    # Normalize the depth array to the range [0, 255]
    normalized_depth = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
    # Convert to unsigned 8-bit type
    normalized_depth = normalized_depth.astype(np.uint8)
    # Apply colormap
    colored_depth = cv2.applyColorMap(normalized_depth, colormap)
    colored_depth_pil = Image.fromarray(cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB))
    return colored_depth_pil

def get_pyramid_score(image, vis_processors, txt_processors,score_processor,attributes,blip_devices=None,scales=[1, 2, 4]):
    """
    Crops the image into multiple patches at different scales, computes scores for each patch,
    normalizes these scores, and maps them back to the original image pixels.

    Parameters:
    - image: The PIL Image to be processed.
    - vis_processors: A function that processes each patch and returns a tensor suitable for scoring.
    - score_processor: A function that takes a batch of patches and returns scores.
    - scales: A list of scales at which to crop the image.

    Returns:
    - A tuple of (cropped_images, scores_map) where scores_map is a numpy array of the same shape as the original image with scores assigned to each pixel.
    """

    # Initialize lists to hold cropped images and their scores
    cropped_images = []
    cropped_batches = []

    # Define the crop function using numpy slicing
    def crop_numpy(img, num_x, num_y):
        np_img = np.array(img)
        h, w, _ = np_img.shape
        return [vis_processors(Image.fromarray(np_img[i * h // num_y:(i + 1) * h // num_y,
                                               j * w // num_x:(j + 1) * w // num_x, :])).to(blip_devices)
                for i in range(num_y) for j in range(num_x)]


    # Process each scale
    for scale in scales:
        num_x, num_y = scale, scale
        patches = crop_numpy(image, num_x, num_y)
        cropped_images.extend(patches)
        # batch = torch.stack(cropped_images)

    attr_score={}
    attr_score_map={}
    for attr in attributes:

        image_scores = score_processor({"image": torch.stack(cropped_images), "text_input": attr}, match_head='itc')
        attr_score[attr] = image_scores.detach().cpu().numpy()

        # Normalize the scores: TODO
        # total_score = sum(image_scores)
        # normalized_scores = [score / total_score for score in image_scores]

        # Create a map of scores for the original image
        scores_map = np.zeros(np.array(image).shape[:2], dtype=np.float32)
        patch_idx = 0
        for scale in scales:
            num_x, num_y = scale, scale
            patch_height = scores_map.shape[0] // num_y
            patch_width = scores_map.shape[1] // num_x
            for i in range(num_y):
                for j in range(num_x):
                    # scores_map[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width] += normalized_scores[patch_idx]
                    scores_map[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width] += image_scores[patch_idx].detach().cpu().numpy()
                    patch_idx += 1

        # Average the scores map if there is overlap (simple averaging for overlap handling)
        # scores_map /= len(scales)
        attr_score_map[attr]=scores_map/3
    # pdb.set_trace()
    return cropped_images, attr_score_map

def get_pyramid_clip_score(image, vis_processors, txt_processors,score_processor,attributes,blip_devices=None,scales=[1, 2, 4]):
    """
    Crops the image into multiple patches at different scales, computes scores for each patch,
    normalizes these scores, and maps them back to the original image pixels.

    Parameters:
    - image: The PIL Image to be processed.
    - vis_processors: A function that processes each patch and returns a tensor suitable for scoring.
    - score_processor: A function that takes a batch of patches and returns scores.
    - scales: A list of scales at which to crop the image.

    Returns:
    - A tuple of (cropped_images, scores_map) where scores_map is a numpy array of the same shape as the original image with scores assigned to each pixel.
    """

    # Initialize lists to hold cropped images and their scores
    cropped_images = []
    cropped_batches = []

    # Define the crop function using numpy slicing
    def crop_numpy(img, num_x, num_y):
        np_img = np.array(img)
        h, w, _ = np_img.shape
        return [vis_processors(Image.fromarray(np_img[i * h // num_y:(i + 1) * h // num_y,
                                               j * w // num_x:(j + 1) * w // num_x, :])).to(blip_devices)
                for i in range(num_y) for j in range(num_x)]

    # Process each scale
    for scale in scales:
        num_x, num_y = scale, scale
        patches = crop_numpy(image, num_x, num_y)
        cropped_images.extend(patches)
        # batch = torch.stack(cropped_images)

    attr_score_map = {}
    for att_i,attr in enumerate(attributes):
        sample = {"image": torch.stack(cropped_images), "text_input": txt_processors(attr)}

        clip_features = score_processor.extract_features(sample)

        features_image = clip_features.image_embeds_proj
        features_text = clip_features.text_embeds_proj

        image_scores = (features_image @ features_text.t()).squeeze()

        scores_map = np.zeros(np.array(image).shape[:2], dtype=np.float32)
        patch_idx = 0
        for scale in scales:
            num_x, num_y = scale, scale
            patch_height = scores_map.shape[0] // num_y
            patch_width = scores_map.shape[1] // num_x
            for i in range(num_y):
                for j in range(num_x):
                    # scores_map[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width] += normalized_scores[patch_idx]
                    scores_map[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width] += \
                        image_scores[patch_idx].detach().cpu().numpy()
                    patch_idx += 1

        attr_score_map[attr] = scores_map / len(scales)

    return cropped_images, attr_score_map

def get_att_score(image, vis_processors, txt_processors,score_processor,attributes,blip_devices=None):
    img=vis_processors(image).unsqueeze(0).to(blip_devices)

    txt = txt_processors(attributes[0])
    txt_tokens = score_processor.tokenizer(txt, return_tensors="pt").to(blip_devices)
    # pdb.set_trace()
    gradcam, _ = compute_gradcam(score_processor, img, txt, txt_tokens, block_num=7)
    gradcam_np = gradcam[0][1].numpy().astype(np.float32)
    avg_gradcam = getAttMap(np.array(rgb_obs)/ 255., gradcam_np, blur=True,overlap=False)
    return avg_gradcam

def save_scores_map(scores_map, filename='score_map.png'):
    """
    Visualizes and saves the scores map with a color bar.

    Parameters:
    - scores_map: A numpy array with scores assigned to each pixel.
    - filename: The filename to save the image to.
    """

    # Normalize scores_map to the range 0-1 for better color mapping
    normalized_map = (scores_map - np.min(scores_map)) / (np.max(scores_map) - np.min(scores_map))

    # Create a figure and an axes object
    fig, ax = plt.subplots()

    # Display the image
    cax = ax.imshow(normalized_map, cmap='viridis')  # Choose a colormap that you prefer

    # Create a color bar
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Score')

    # Save the figure
    plt.axis('off')  # Turn off axis
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    # Close the plot to free up memory
    plt.close(fig)

def overlay_attention_map(image, scores_map, filename='attention_overlay.png'):
    """
    Applies a smoothed attention map over an image to highlight areas based on the scores,
    and then saves the resulting image.

    Parameters:
    - image: A PIL Image to be enhanced.
    - scores_map: A numpy array with scores assigned to each pixel.
    - filename: The filename to save the image to.
    """

    # Normalize scores_map to the range 0-1 for better color mapping
    normalized_map = (scores_map - np.min(scores_map)) / (np.max(scores_map) - np.min(scores_map))

    # Apply Gaussian smoothing to the scores map
    smoothed_map = scipy.ndimage.gaussian_filter(normalized_map, sigma=2)  # Adjust sigma as needed for the desired smoothness

    # Convert the original image to a numpy array for processing
    img_array = np.array(image)

    # Convert the smoothed scores map to a heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(img_array, cmap='gray', interpolation='nearest')  # Display the original image
    plt.imshow(smoothed_map, cmap='jet', alpha=0.6, interpolation='bilinear')  # Overlay the attention map
    plt.colorbar()  # Optionally add a color bar to indicate scoring scale
    plt.axis('off')  # Hide axes
    # Save the figure
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def doubly_right_detection(image,model, vis_processors,target,blip_devices=None):
    image = vis_processors(image).unsqueeze(0).to(blip_devices)
    out=model.generate({"image": image, "prompt": f"Question: is {target} shown in this image? Answer:"})
    print(out)
    if "yes" in out[0].lower():
        return True
    else:
        return False

class Pos_Queue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = []

    def add_element(self, element):
        # Add the new element to the queue
        self.queue.append(element)
        # Check if the queue is beyond capacity
        if len(self.queue) > self.capacity:
            # If it's full, pop and return the oldest element
            return self.queue.pop(0)
        # If not full, return None
        return None



if __name__ == "__main__":
    # setup model
    # device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # model, vis_processors, text_processors = load_model_and_preprocess(name="blip2_feature_extractor",
    #                                                                   model_type="pretrain", is_eval=True,
    #                                                                   device=device)

    # model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain",
    #                                                                    device=device, is_eval=True)

    # model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "large",
    #                                                                    device=device, is_eval=True)

    # model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b",
    #                                                      is_eval=True, device=device)

    # model, vis_processors, text_processors = load_model_and_preprocess("clip_feature_extractor", model_type="ViT-B-32",
    #                                                                   is_eval=True, device=device)
    # load the observation
    # data_root="/home/mmvc/Desktop/L3MVN_Reproduce/L3MVN/vis/3d_score"
    # obs = np.load(os.path.join(data_root,"sample_T.npy"))#0:3 rgb, 4 depth

    # get rgb
    # rgb=obs[:,:,0:3].astype(np.uint8)
    # rgb_obs=Image.fromarray(rgb)
    # rgb_obs.save(os.path.join(data_root,"rgb.png"))
    # patches,batch=crop_img(rgb_obs,vis_processors["eval"])
    # vis_patches(patches,os.path.join(data_root,"patches.png"))

    # get detph
    # depth=obs[:,:,4]
    # depth_img=depth_to_colormap(depth,)
    # depth_img.save(os.path.join(data_root,"depth.png"))

    #batch img and list txt 3.2s
    # txt_list=[]
    # for i in range(batch.shape[0]):
    #     txt_list.append(txt)
    # for i in range(5):
    #     itc_score = model({"image": batch, "text_input": txt_list}, match_head='itc')

    # single img and single txt 8.13s
    # txt="chair"
    # for j in range(5):
    #     for i in range(batch.shape[0]):
    #         itc_score = model({"image": batch[i][None,:], "text_input": txt}, match_head='itc')
    #         print(itc_score)
    #batch img and single txt 3.1s
    #                 "leg",
    #                 "backrest",
    #                 "seat",
    #                 "armrest"

    # attributes=['chair backrest', 'chair seat', 'chair leg', 'chair armrest']
    # target=["chair"]

    t=time.time()
    # ------------- multi_scale approach ------------------
    # attributes=['object with chair leg, chair backrest, chair seat, and chair armrest']
    # imgs,score_mape=get_pyramid_score(rgb_obs,vis_processors["eval"],text_processors["eval"],model,attributes,device)
    # imgs,score_mape=get_pyramid_score_feature(rgb_obs,vis_processors["eval"],text_processors["eval"],model,attributes,device)

    # ------------- gradcam approach -------------
    # img=vis_processors["eval"](rgb_obs).unsqueeze(0).to(device)
    # txt = text_processors["eval"](attributes[0])
    # txt_tokens = model.tokenizer(txt, return_tensors="pt").to(device)
    # gradcam, _ = compute_gradcam(model, img, txt, txt_tokens, block_num=7)
    # gradcam_np = gradcam[0][1].numpy().astype(np.float32)
    # avg_gradcam = getAttMap(np.array(rgb_obs)/ 255., gradcam_np, blur=True,overlap=False)

    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # ax.imshow(avg_gradcam)
    # ax.axis('off')
    # plt.tight_layout()
    # plt.savefig(os.path.join(data_root, f"grad_score_map.png"))

    # ------------------ doubly right detection ------------------
    # print(doubly_right_detection(rgb_obs,vis_processors["eval"],target[0],device))
    # end_t=time.time()

    # ------------------ CLIP multi-scale  ------------------
    # imgs,score_mape=get_pyramid_clip_score(rgb_obs,vis_processors["eval"],text_processors["eval"],model,attributes,device)

    # check dataset
    data_pth='/home/mmvc/Desktop/L3MVN_Reproduce/L3MVN/data/objectgoal_hm3d/val_mini/content/TEEsavR23oF.json.gz'
    with gzip.open(data_pth, 'r') as f:
        data = json.loads(f.read().decode('utf-8'))
    for k in data.keys():
        print(k,len(data[k]))
    print("========================")
    print(data['episodes'][1])
    print("========================")
    print(data['episodes'][2])
    print("========================")
    print(data['episodes'][3])
    end_t=time.time()

    # for k in score_mape.keys():
    #     overlay_attention_map(rgb_obs,score_mape[k],os.path.join(data_root,f"clip_score_map_{k}.png"))

    print(f"duration: {end_t-t} s")

    # print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)
