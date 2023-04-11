import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    def __init__(self):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = 'resnet101'
        self.pretrained = True
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        # [1, 2048, H/?, W/?]
        patch_feats = self.model(images)
        print(patch_feats.shape)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        # [1, H/? * W/?, 2048] = [1, patch_size, 2048], [1,2048]
        return patch_feats, avg_feats
    

if __name__=="__main__":
    ve = VisualExtractor()
    print(ve)
    img = torch.rand(1,3,224,224)
    pf, af = ve(img)
    print(pf.shape, af.shape)