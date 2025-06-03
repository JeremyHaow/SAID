import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import clip
import open_clip
from torchvision import transforms
from .resnet import ResNet, Bottleneck # 从resnet.py导入ResNet和Bottleneck

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class AIDE_Model(nn.Module):
    def __init__(self, resnet_path, convnext_path):
        super(AIDE_Model, self).__init__()
        
        # 使用从 resnet.py 导入的 ResNet 和 Bottleneck
        self.model_min = ResNet(Bottleneck, [3, 4, 6, 3])
        self.model_max = ResNet(Bottleneck, [3, 4, 6, 3])

        setattr(self.model_min, 'fc', nn.Identity()) 
        setattr(self.model_max, 'fc', nn.Identity())

        # Only load pretrained weights if resnet_path is provided and is not the string 'None'
        if resnet_path is not None and str(resnet_path).lower() != 'none':
            print(f"Loading ResNet pretrained weights from: {resnet_path}")
            pretrained_dict = torch.load(resnet_path, map_location='cpu')
            
            if 'state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['state_dict']

            # 移除预训练权重中 fc 相关层的权重 (现在统一为 'fc.')
            keys_to_remove = [k for k in pretrained_dict.keys() if k.startswith('fc.')]
            for k in keys_to_remove:
                if k in pretrained_dict:
                    del pretrained_dict[k]
            
            # 加载权重到ResNet骨干网络
            missing_keys_min, unexpected_keys_min = self.model_min.load_state_dict(pretrained_dict, strict=False)
            missing_keys_max, unexpected_keys_max = self.model_max.load_state_dict(pretrained_dict, strict=False)

            if unexpected_keys_min:
                print(f"Unexpected keys in pretrained_dict when loading to model_min: {unexpected_keys_min}")
            if missing_keys_min:
                print(f"Missing keys in model_min when loading from pretrained_dict: {missing_keys_min}")
            if unexpected_keys_max:
                print(f"Unexpected keys in pretrained_dict when loading to model_max: {unexpected_keys_max}")
            if missing_keys_max:
                print(f"Missing keys in model_max when loading from pretrained_dict: {missing_keys_max}")
        
        # ResNet50 Bottleneck的最后一个阶段输出 512 个特征
        # ConvNeXt XXL 在移除 head 后，经过我们的convnext_proj预计输出256个特征
        # 所以 fc 层的输入是 512 + 256
        self.fc = Mlp(512 + 256 , 1024, 2)

        print("build model with convnext_xxl")
        self.openclip_convnext_xxl, _, _ = open_clip.create_model_and_transforms(
            "convnext_xxlarge", pretrained=convnext_path
        )

        self.openclip_convnext_xxl = self.openclip_convnext_xxl.visual.trunk
        self.openclip_convnext_xxl.head.global_pool = nn.Identity()
        self.openclip_convnext_xxl.head.flatten = nn.Identity()

        self.openclip_convnext_xxl.eval()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 通用平均池化层
        self.convnext_proj = nn.Sequential(
            nn.Linear(3072, 256),
        )
        for param in self.openclip_convnext_xxl.parameters():
            param.requires_grad = False

    def _preprocess_dwt(self, x, mode='symmetric', wave='bior1.3'):
        '''
        pip install pywavelets pytorch_wavelets
        '''
        from pytorch_wavelets import DWTForward, DWTInverse
        DWT_filter = DWTForward(J=1, mode=mode, wave=wave).to(x.device)
        Yl, Yh = DWT_filter(x)
        return transforms.Resize([x.shape[-2], x.shape[-1]])(Yh[0][:, :, 2, :, :])

    def forward(self, x):
        b, t, c, h, w = x.shape # c 是输入图像的通道数

        x_minmin = x[:, 0]  # 纹理丰富度最低的块
        x_maxmax = x[:, 1]  # 纹理丰富度最高的块
        tokens = x[:, 2]    # 原始图像

        x_minmin = self._preprocess_dwt(x_minmin)
        x_maxmax = self._preprocess_dwt(x_maxmax)

        with torch.no_grad():
            # 确保均值和标准差张量与输入图像的通道数 c 和设备匹配
            # 这些值通常是3通道RGB图像的，如果输入是单通道，可能需要调整或只取第一个值
            _clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=tokens.device)
            _clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=tokens.device)
            _dinov2_mean = torch.tensor([0.485, 0.456, 0.406], device=tokens.device)
            _dinov2_std = torch.tensor([0.229, 0.224, 0.225], device=tokens.device)

            if c == 1: # 如果输入是单通道
                clip_mean_reshaped = _clip_mean[0].view(1, 1, 1, 1).expand(b, 1, 1, 1)
                clip_std_reshaped = _clip_std[0].view(1, 1, 1, 1).expand(b, 1, 1, 1)
                dinov2_mean_reshaped = _dinov2_mean[0].view(1, 1, 1, 1).expand(b, 1, 1, 1)
                dinov2_std_reshaped = _dinov2_std[0].view(1, 1, 1, 1).expand(b, 1, 1, 1)
            elif c == 3: # 如果输入是三通道
                clip_mean_reshaped = _clip_mean.view(1, 3, 1, 1).expand(b, 3, 1, 1)
                clip_std_reshaped = _clip_std.view(1, 3, 1, 1).expand(b, 3, 1, 1)
                dinov2_mean_reshaped = _dinov2_mean.view(1, 3, 1, 1).expand(b, 3, 1, 1)
                dinov2_std_reshaped = _dinov2_std.view(1, 3, 1, 1).expand(b, 3, 1, 1)
            else:
                # 对于其他通道数，可以采用均值或者抛出错误，这里采用均值
                print(f"Warning: Input channel c={c} is not 1 or 3. Using mean of stats for normalization.")
                clip_mean_reshaped = _clip_mean.mean().view(1, 1, 1, 1).expand(b, c, 1, 1)
                clip_std_reshaped = _clip_std.mean().view(1, 1, 1, 1).expand(b, c, 1, 1)
                dinov2_mean_reshaped = _dinov2_mean.mean().view(1, 1, 1, 1).expand(b, c, 1, 1)
                dinov2_std_reshaped = _dinov2_std.mean().view(1, 1, 1, 1).expand(b, c, 1, 1)

            local_convnext_image_feats = self.openclip_convnext_xxl(
                tokens * (dinov2_std_reshaped / clip_std_reshaped) + (dinov2_mean_reshaped - clip_mean_reshaped) / clip_std_reshaped
            )

            local_convnext_image_feats = self.avgpool(local_convnext_image_feats).view(b, -1) # 使用批次大小 b
            x_0 = self.convnext_proj(local_convnext_image_feats)

        x_min = self.model_min(x_minmin)
        x_max = self.model_max(x_maxmax)
        
        # 平均两个ResNet分支的输出
        x_1 = (x_min + x_max) / 2
        
        x = torch.cat([x_0, x_1], dim=1)
        x = self.fc(x)
        return x

def AIDE(resnet_path, convnext_path):
    model = AIDE_Model(resnet_path, convnext_path)
    return model