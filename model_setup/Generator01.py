# run
class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_blocks=9):
        super(Generator, self).__init__()
        # defining the various layers in the G-architecture
        sequence = list()
        sequence+=[
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels,64,kernel_size = 7,stride =1, padding =0),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        ]
        sequence+=[
            nn.Conv2d(64,128,kernel_size = 3,stride = 2, padding = 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128,256,kernel_size = 3,stride = 2, padding = 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        ]

        for i in range (num_residual_blocks):
          sequence+= [
              ResNetBlock(256)
          ]
        #Calling out the ResNet block function - '9' times as per the DeblurGAN paper
        sequence+= [
            nn.ConvTranspose2d(256,128,kernel_size=3,stride =2,padding =1,output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128,64,kernel_size = 3,stride = 2,padding = 1,output_padding = 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        ]
        sequence+= [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64,out_channels,kernel_size = 7,padding = 0),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
      out = self.model(x)
      out = x + out
      out = torch.clamp(out, min = -1, max = 1)
      return out