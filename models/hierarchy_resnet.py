import torch
import torch.nn as nn
import torch.nn.functional as F

from models.res_net import TinyResNet


class HierarchicalResNet(nn.Module):
    def __init__(self, num_l1, num_l2, num_l3=200):
        super(HierarchicalResNet, self).__init__()
        
        # 1. Load your custom backbone
        base = TinyResNet(num_classes=num_l3)
        
        # 2. Extract everything except the final FC layer
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            nn.ReLU(inplace=True),
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avg_pool
        )
        
        # 3. Create the 3 Hierarchical Heads (Equation 2 from the paper)
        self.fc_coarse = nn.Linear(512, num_l1)
        self.fc_mid = nn.Linear(512, num_l2)
        self.fc_fine = nn.Linear(512, num_l3)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x, return_hierarchy=True):
        # Pass through your custom ResNet backbone
        out = self.backbone(x)
        features = torch.flatten(out, 1)
        
        dropped_features = self.dropout(features)
        # Main Fine-grained Output
        logits_f = self.fc_fine(dropped_features)
        
        if not return_hierarchy:
            return logits_f
            
        # Hierarchical extensions
        logits_c = self.fc_coarse(dropped_features)
        logits_m = self.fc_mid(dropped_features)
        
        # Features are returned for Center Loss (intra-class variance)
        return features, (logits_c, logits_m, logits_f)