class Discriminator(nn.Module):
  def __init__(self, input_nc , ndf = 64, n_layers = 3):
    super(Discriminator , self).__init__()

    kw = 4 ## kernel width
    padw = 1 ## padding
    sequence = [
        nn.Conv2d(input_nc, ndf, kernel_size = kw, stride = 2, padding = padw),
        nn.LeakyReLU(0.2, True)
    ]
    nf_mult = 1
    for n in range (1,n_layers):
      nf_mult_prev = nf_mult
      nf_mult = min(2**n,8)
      sequence+= [
          nn.Conv2d(ndf * nf_mult_prev, ndf*nf_mult, kernel_size = kw, stride = 2, padding = padw),
          nn.InstanceNorm2d(ndf * nf_mult),
          nn.LeakyReLU(0.2 , True)
      ]

    nf_mult_prev = nf_mult
    nf_mult = min(2**n_layers, 8)
    sequence+= [
        nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size = kw, stride = 1, padding = padw),
        nn.InstanceNorm2d(ndf*nf_mult),
        nn.LeakyReLU(0.2, True)
    ]
    sequence+= [nn.Conv2d(ndf*nf_mult, 1, kernel_size = kw, stride = 1, padding = padw)]
    self.model = nn.Sequential(*sequence)

  def forward(self,input):
    out = self.model(input)
    return out
