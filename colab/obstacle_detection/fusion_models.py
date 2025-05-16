import torch
import torch.nn as nn

class PointPillars(nn.Module):
    def __init__(self, voxel_size, point_cloud_range):
        super(PointPillars, self).__init__()
        self.voxel_layer = Voxelization(voxel_size, point_cloud_range)
        self.cnn_encoder = PillarFeatureNet()
    
    def forward(self, points):
        voxels = self.voxel_layer(points)
        features = self.cnn_encoder(voxels)
        return features  # 512-dimensional vector

class CenterPoint(nn.Module):
    def __init__(self, backbone):
        super(CenterPoint, self).__init__()
        self.backbone = backbone
        self.center_head = CenterHead()
    
    def forward(self, points):
        features = self.backbone(points)
        center_preds = self.center_head(features)
        return center_preds  # 512-dimensional vector

class PointRCNN(nn.Module):
    def __init__(self, rpn, rcnn):
        super(PointRCNN, self).__init__()
        self.rpn = RegionProposalNet()
        self.rcnn = RegionRefinementNet()
    
    def forward(self, points):
        proposals = self.rpn(points)
        refined_features = self.rcnn(proposals)
        return refined_features  # 512-dimensional vector

class WeightedFusion(nn.Module):
    def __init__(self):
        super(WeightedFusion, self).__init__()
        self.weights = nn.Parameter(torch.rand(3))
    
    def forward(self, f_pillars, f_center, f_rcnn):
        fused_feature = self.weights[0]*f_pillars + self.weights[1]*f_center + self.weights[2]*f_rcnn
        return fused_feature

class TemporalFusionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TemporalFusionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    
    def forward(self, fused_features_seq):
        outputs, (h_n, c_n) = self.lstm(fused_features_seq)
        return outputs[:, -1, :]  # Final fused temporal feature