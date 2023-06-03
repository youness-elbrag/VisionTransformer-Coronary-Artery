import numpy as np
import argparse
import torch.nn as nn 
import torch 
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np 


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--Path", type=str, required=True)
    parser.add_argument("--output", type=str,required=True)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    Path = args.Path
    output = args.output
    Path_save = Path(output)

    total_loss = 0
    correct = 0
    Loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in Validation_Loader:
            # Move the batch to the device
            batch = [t.to("cuda") for t in batch]
            images, labels = batch
            labels = labels.long()

            # Get predictions
            logits, _ = model(images)

            # Calculate the loss
            loss = Loss(logits, labels)
            total_loss += loss.item() * len(images)

            # Calculate the accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += torch.sum(predictions == labels).item()
    accuracy = correct / len(Validation_Loader.dataset)
    avg_loss = total_loss / len(Validation_Loader.dataset)
    

    PATH = args.Path
    OUTPUT = args.output

    device = "cude" if torch.cuda.is_available() else "cpu"
    model = VisionTrans.load_from_checkpoint(PATH).to(device)
    visualize_attention(model, num_images=3, output=OUTPUT)