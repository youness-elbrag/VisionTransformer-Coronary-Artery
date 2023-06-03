import yaml
from Config import initializer
# import numpy as np
# import torch.nn as nn 
# import torch 
# from torch.nn import functional as F
# import matplotlib.pyplot as plt
# import numpy as np 

def write_config_to_yaml():
    config = initializer()
    with open("config.yml", "w") as file:
        yaml.dump(config, file)

def read_config_from_yaml(Path_config):
    with open(Path_config, "r") as file:
        config = yaml.safe_load(file)
    return config

def Loader_Data(path):
    return np.load(path).astype(np.float32)


def visualize_attention(model, num_images=4, output=None, device="cuda"):
    """
    Visualize the attention maps of the specified number of images.
    """
    model.eval()

    # Load random images
    testset = Loading_DataFolder_Val
    classes = ['normal', 'abnormal']

    # Pick num_images samples randomly
    indices = torch.randperm(len(testset))[:num_images]
    raw_images = [np.asarray(testset[i][0]) for i in indices]
    labels = [testset[i][1] for i in indices]
    images = torch.stack([torch.from_numpy(image) for image in raw_images])

    # Move the images to the device
    images = images.to(device)
    model = model.to(device)

    # Get the attention maps from the last block
    with torch.no_grad():
        logits, attention_maps = model(images, output_attentions=True)

    # Get the predictions
    predictions = torch.argmax(logits, dim=1)

    # Concatenate the attention maps from all blocks
    attention_maps = torch.cat(attention_maps, dim=1)

    # Select only the attention maps of the CLS token
    attention_maps = attention_maps[:, :, 0, 1:]

    # Average the attention maps of the CLS token over all the heads
    attention_maps = attention_maps.mean(dim=1)

    # Reshape the attention maps to a square
    num_patches = attention_maps.size(-1)
    size = int(math.sqrt(num_patches))
    attention_maps = attention_maps.view(-1, size, size)

    # Resize the map to the size of the image
    attention_maps = attention_maps.unsqueeze(1)
    attention_maps = F.interpolate(attention_maps, size=(32, 32), mode='bilinear', align_corners=False)
    attention_maps = attention_maps.squeeze(1)

    # Plot the images and the attention maps
    fig, axes = plt.subplots(num_images, 2, figsize=(8, num_images * 4))
    fig.tight_layout()

    for i in range(num_images):
        img_ax = axes[i, 0]
        attn_ax = axes[i, 1]

        img_ax.imshow(raw_images[i][0], cmap='gray')
        attn_ax.imshow(raw_images[i][0], cmap='gray')
        attn_ax.imshow(attention_maps[i].cpu(),cmap='jet')

        # Show the ground truth and the prediction
        gt = classes[labels[i]]
        pred = classes[predictions[i]]
        img_ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt == pred else "red"))

        img_ax.axis('off')
        attn_ax.axis('off')

    if output is not None:
        plt.savefig(output)

    plt.show()


if __name__ == "__main__":
    ## Write the configuration File
    write_config_to_yaml()
