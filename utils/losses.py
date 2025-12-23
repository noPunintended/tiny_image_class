import torch
import torch.nn as nn
from traitlets import This

class CenterLoss(nn.Module):
    """
    Implementation of Center Loss as described in Equation (1).
    It learns a center for each class and minimizes the distance 
    between the features and their corresponding class centers.
    """
    def __init__(self, num_classes=200, feat_dim=512):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        
        # The centers 'cyi' are learnable parameters
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))


    def forward(self, x, labels):
        # 1. Force everything to Float32 for the distance calculation
        # Center loss is sensitive; Float32 prevents precision overflow/underflow
        x = x.float()
        centers = self.centers.float()
        
        batch_size = x.size(0)
        
        # 2. Calculate squared Euclidean distance in Float32
        # distmat = ||x||^2 + ||c||^2
        x_norm = torch.pow(x, 2).sum(dim=1, keepdim=True)
        c_norm = torch.pow(centers, 2).sum(dim=1, keepdim=True).t()
        distmat = x_norm + c_norm
        
        # 3. distmat = distmat - 2 * <x, centers>
        # Now both are Float32, so addmm_ will work perfectly
        distmat.addmm_(x, centers.t(), beta=1, alpha=-2)

        # 4. Create mask and calculate loss
        classes = torch.arange(self.num_classes).long().to(x.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        # Perform masking in Float32
        dist = distmat * mask.float()
        
        # Clamp to avoid extremely large values before averaging
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss