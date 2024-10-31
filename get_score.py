import sys,pdb,os,time
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image
from lavis.models import model_zoo
from lavis.models import load_model_and_preprocess

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


if __name__ == "__main__":
    # setup model
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain",
                                                                       device=device, is_eval=True)
    # declare input image
    data_root="/home/mmvc/Desktop/L3MVN_Reproduce/L3MVN/vis/rgb_score"
    img_name="img0.png"
    raw_image = Image.open(os.path.join(data_root,img_name)).convert("RGB")

    # patches=crop_img_padding(raw_image,vis_processors)
    patches,batch=crop_img(raw_image,vis_processors["eval"])
    vis_patches(patches,os.path.join(data_root,"patches.png"))

    t=time.time()
    #batch img and list txt 3.2s
    # txt_list=[]
    # for i in range(batch.shape[0]):
    #     txt_list.append(txt)
    # for i in range(5):
    #     itc_score = model({"image": batch, "text_input": txt_list}, match_head='itc')

    #single img and single txt 8.13s
    # for j in range(5):
    #     for i in range(batch.shape[0]):
    #         itc_score = model({"image": batch[i][None,:], "text_input": txt}, match_head='itc')

    #batch img and single txt 3.1s
    attributes=['chair leg', 'chair backrest','chair seat','chair armrest']
    attr_score = {}
    for attr in attributes:
        itc_score = model({"image": batch, "text_input": attr}, match_head='itc')
        attr_score[attr] = itc_score
    end_t=time.time()
    save_images_with_scores(batch,attr_score,attributes,data_root)
    print(f"duration: {end_t-t} s")

