import torch
from torch import nn
from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        ckpt = torch.load(model_paths['ir_se50'], map_location=torch.device('cpu'))
        self.facenet.load_state_dict(ckpt)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, x):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            diff_input = y_hat_feats[i].dot(x_feats[i])
            diff_views = y_feats[i].dot(x_feats[i])
            id_logs.append({'diff_target': float(diff_target),
                            'diff_input': float(diff_input),
                            'diff_views': float(diff_views)})
            loss += 1 - diff_target
            id_diff = float(diff_target) - float(diff_views)
            sim_improvement += id_diff
            count += 1

        return loss / count, sim_improvement / count, id_logs
    
    def forward_lfl(self, y_hat, y):
        n_samples = y.shape[0]
        #x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        #loss = 0
        #sim_improvement = 0
        id_logs = []
        count = 0

        diff_target = y_hat_feats[0].dot(y_feats[0])
        id_logs.append({'diff_target': diff_target,
                        'diff_input': diff_target,
                        'diff_views': diff_target})
        loss = 1 - diff_target
        id_diff = diff_target - diff_target
        sim_improvement = id_diff
        count += 1

        for i in range(1, n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            #diff_input = y_hat_feats[i].dot(x_feats[i])
            #diff_views = y_feats[i].dot(x_feats[i])
            id_logs.append({'diff_target': diff_target,
                            'diff_input': diff_target,
                            'diff_views': diff_target})
            loss += 1 - diff_target
            id_diff = diff_target - diff_target
            sim_improvement += id_diff
            count += 1

        return loss / count, sim_improvement / count, id_logs
