import torch  # PyTorch for building and training models
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Functional operations like pairwise distance

# Define the Triplet Loss class, inheriting from nn.Module
class Loss(nn.Module):
    def __init__(self, margin=1.0):
        """
        Initialize the Triplet Loss function.

        Args:
            margin (float): Margin for triplet loss. Default is 1.0.
        """
        super(Loss, self).__init__()
        self.margin = margin  # Margin to enforce a minimum distance between positive and negative pairs

    def forward(self, anchor, positive, negative):
        """
        Forward pass to compute the triplet loss.

        Args:
            anchor: Embeddings from the anchor image.
            positive: Embeddings from the positive (similar) image.
            negative: Embeddings from the negative (dissimilar) image.

        Returns:
            The mean triplet loss.
        """
        # Compute the pairwise distance between the anchor-positive and anchor-negative pairs
        pos_dist = F.pairwise_distance(anchor, positive, p=2)  # Euclidean distance (p=2) between anchor and positive
        neg_dist = F.pairwise_distance(anchor, negative, p=2)  # Euclidean distance (p=2) between anchor and negative
        
        # Triplet loss: maximize distance between anchor-negative and minimize distance between anchor-positive
        loss = F.relu(pos_dist - neg_dist + self.margin)  # Apply ReLU to ensure non-negative loss values
        
        return loss.mean()  # Return the average loss

# Example usage of the Triplet Loss function
if __name__ == "__main__":
    # Example embeddings for anchor, positive, and negative images
    anchor = torch.randn(10, 128)  # Batch of 10 samples with 128-dimensional embeddings
    positive = torch.randn(10, 128)  # Corresponding positive embeddings (similar to anchor)
    negative = torch.randn(10, 128)  # Corresponding negative embeddings (dissimilar to anchor)

    # Initialize the triplet loss function with a margin of 1.0
    triplet_loss = Loss(margin=1.0)

    # Calculate the triplet loss for the given embeddings
    loss_value = triplet_loss(anchor, positive, negative)

    # Print the computed loss
    print(f"Triplet Loss: {loss_value.item()}")
