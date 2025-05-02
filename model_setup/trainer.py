from tqdm import tqdm
lr = 0.0001
beta1 = 0.5
beta2 = 0.999
optimizer_G = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = torch.optim.Adam(dis.parameters(), lr=lr, betas=(beta1, beta2))
config = {
    'gp_lambda': 10,
    'content_loss_lambda': 100
}
# FOR GPU RUNTIME ONLY
class Trainer:
  def __init__(self,data_loader,gen,dis,gen_optim, dis_optim,config,device):
    self.data_loader = data_loader
    self.generator = gen.to(device)
    self.discriminator = dis.to(device)
    self.device = device
    self.discriminator_optimizer = dis_optim
    self.generator_optimizer = gen_optim
    self.config = config
  def train(self,epochs):
    # set models to train mode
    self.generator.train()
    self.discriminator.train()
    total_generator_loss = 0
    total_discriminator_loss = 0
    losslogG = list()
    losslogD = list()
    for batch_idx, sample in tqdm(enumerate(self.data_loader)):

      # Getting data
      blurred = sample['blur'].to(self.device)
      sharp = sample['sharp'].to(self.device)

      # Getting gen output
      deblurred = self.generator(blurred)

      # Denormalize
      denormalized_blurred = denormalize(blurred)
      denormalized_sharp = denormalize(sharp)
      denormalized_deblurred = denormalize(deblurred)

      sharp_discriminator_out = self.discriminator(sharp)
      deblurred_discriminator_out = self.discriminator(deblurred)
      losslogG = list()
      losslogD = list()
      critic_updates = 5
      discriminator_loss = 0
      for i in range(critic_updates):
        self.discriminator_optimizer.zero_grad()
        gp_lambda = self.config['gp_lambda']
        alpha = random.random()
        interpolates = alpha*sharp + (1-alpha)*deblurred
        interpolates_discriminator_out = self.discriminator(interpolates)
        kwargs = {
            'gp_lambda': gp_lambda,
            'interpolates': interpolates,
            'interpolates_discriminator_out':interpolates_discriminator_out,
            'sharp':sharp_discriminator_out,
            'deblurred':deblurred_discriminator_out
        }
        wgan_loss_d,gp_d = adversial_loss('D',**kwargs)
        discriminator_loss_per_update = wgan_loss_d + gp_d

        discriminator_loss_per_update.backward(retain_graph = True)
        self.discriminator_optimizer.step()
        discriminator_loss += discriminator_loss_per_update.item()

        discriminator_loss/= critic_updates
        total_discriminator_loss += discriminator_loss
      self.generator_optimizer.zero_grad()
      content_loss_lambda = self.config['content_loss_lambda']
      kwargs = {
            'discriminator_output': deblurred_discriminator_out
      }
      adversial_loss_g = adversial_loss('G',**kwargs)
      content_loss_g = content_loss(deblurred,sharp) * content_loss_lambda
      generator_loss = adversial_loss_g.detach() + content_loss_g
      generator_loss.backward()
      self.generator_optimizer.step()
      total_generator_loss += generator_loss.item()
      if(batch_idx%10==0):
        losslogG.append(generator_loss)
        losslogD.append(discriminator_loss)
    plt.plot(losslogG)
    plt.plot(losslogD)
    return losslogG, losslogD
