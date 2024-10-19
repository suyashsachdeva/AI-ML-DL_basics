
import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, margin=1.0):
        super(Loss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Compute the pairwise distance between anchor-positive and anchor-negative
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # Calculate the loss as per the Triplet loss function
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()

# Example of how to use TripletLoss
if __name__ == "__main__":
    # Dummy data to test the function
    anchor = torch.randn(10, 128)  # Example embeddings for anchor
    positive = torch.randn(10, 128)  # Example embeddings for positive
    negative = torch.randn(10, 128)  # Example embeddings for negative

    triplet_loss = Loss(margin=1.0)
    loss_value = triplet_loss(anchor, positive, negative)
    print(f"Triplet Loss: {loss_value.item()}")
