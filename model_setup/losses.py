import torch
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision import models
CONV3_3_IN_VGG_19 = models.vgg19(weights='IMAGENET1K_V1').features[:15] #importing the first 15 layers of trained vgg19 convnet
CONV3_3_IN_VGG_19.to(device);
def content_loss(deblurred,sharp):
  model = CONV3_3_IN_VGG_19
  deblurred_feature_map = model.forward(deblurred)
  sharp_feature_map = model.forward(sharp).detach()
  loss = F.mse_loss(deblurred_feature_map,sharp_feature_map)
  return loss
  def adversial_loss(type, **kwargs):
  if type == 'G': #generator loss
    deblurred_discriminator_out = kwargs['discriminator_output']
    return -deblurred_discriminator_out.mean()

  elif type == 'D':
    # for wasserstein loss
    sharp = kwargs['sharp']
    deblurred = kwargs['deblurred']

    # for gradient penalty
    gp_lambda = kwargs['gp_lambda']
    interpolates = kwargs['interpolates']
    interpolates_discriminator_out = kwargs['interpolates_discriminator_out']

    wgan_loss = deblurred.mean() - sharp.mean()

    gradients = autograd.grad(outputs = interpolates_discriminator_out,
                              inputs = interpolates,
                              grad_outputs = torch.ones(interpolates_discriminator_out.size()).to(device),#.cuda(),
                              retain_graph = True,
                              create_graph = True
                              )[0]
    gradient_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()

    return wgan_loss,  gp_lambda* gradient_penalty
