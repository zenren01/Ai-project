
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.utils as vutils
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms.functional import to_tensor, to_pil_image

# Set random seed for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset class for GoPro
class GoProDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.transform = transform
        self.train = train

        # Define paths for blurred and sharp images
        if train:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'test')

        # Get all image paths
        self.blur_images = sorted(glob.glob(os.path.join(self.root_dir, 'blur', '*.png')))
        self.sharp_images = sorted(glob.glob(os.path.join(self.root_dir, 'sharp', '*.png')))

        # Alternative path format for GoPro dataset
        if len(self.blur_images) == 0 or len(self.sharp_images) == 0:
            # For cases where dataset is organized with blur and sharp in separate folders
            self.blur_images = []
            self.sharp_images = []
            # Walk through the directory structure
            for subdir, _, _ in os.walk(self.root_dir):
                blur_path = os.path.join(subdir, 'blur')
                sharp_path = os.path.join(subdir, 'sharp')
                if os.path.exists(blur_path) and os.path.exists(sharp_path):
                    self.blur_images.extend(sorted(glob.glob(os.path.join(blur_path, '*.png'))))
                    self.sharp_images.extend(sorted(glob.glob(os.path.join(sharp_path, '*.png'))))


        assert len(self.blur_images) == len(self.sharp_images), "Number of blurred and sharp images should be equal"
        print(f"Found {len(self.blur_images)} image pairs in {self.root_dir}")

    def __len__(self):
        return len(self.blur_images)

    def __getitem__(self, idx):
        blur_img = Image.open(self.blur_images[idx]).convert('RGB')
        sharp_img = Image.open(self.sharp_images[idx]).convert('RGB')

        if self.transform:
            # Apply same transform to both blur and sharp images
            seed = np.random.randint(2147483647)

            random.seed(seed)
            torch.manual_seed(seed)
            blur_img = self.transform(blur_img)

            random.seed(seed)
            torch.manual_seed(seed)
            sharp_img = self.transform(sharp_img)
        else:
            # Convert to tensor if no transform is provided
            blur_img = to_tensor(blur_img)
            sharp_img = to_tensor(sharp_img)

        return {'blur': blur_img, 'sharp': sharp_img}

# FPN Feature Pyramid Network
class FPN(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(FPN, self).__init__()

        # Feature Pyramid Network (FPN) with ResNet backbone
        backbone = models.resnet34(weights='IMAGENET1K_V1')

        self.conv1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.conv2 = backbone.layer1
        self.conv3 = backbone.layer2
        self.conv4 = backbone.layer3
        self.conv5 = backbone.layer4

        # Lateral layers
        self.lateral_conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.lateral_conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.lateral_conv3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.lateral_conv4 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps."""
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Bottom-up pathway
        c1 = self.conv1(x)  # 64 channels, 1/2 resolution
        c2 = self.conv2(c1)  # 64 channels, 1/2 resolution
        c3 = self.conv3(c2)  # 128 channels, 1/4 resolution
        c4 = self.conv4(c3)  # 256 channels, 1/8 resolution
        c5 = self.conv5(c4)  # 512 channels, 1/16 resolution

        # Top-down pathway and lateral connections
        p5 = self.lateral_conv1(c5)
        p4 = self._upsample_add(p5, self.lateral_conv2(c4))
        p3 = self._upsample_add(p4, self.lateral_conv3(c3))
        p2 = self._upsample_add(p3, self.lateral_conv4(c2))

        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return p2, p3, p4, p5

# Generator with proper upsampling to match input size
class DeblurGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(DeblurGenerator, self).__init__()

        # FPN backbone
        self.fpn = FPN()

        # Output layer with proper upsampling
        self.output_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # Upsample to match input size
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Store input for skip connection
        input_img = x

        # Get features from FPN
        p2, p3, p4, p5 = self.fpn(x)

        # Use p2 for output (highest resolution feature map)
        out = self.output_conv(p2)

        # Ensure output size matches input size
        if out.size() != input_img.size():
             # Calculate the required size based on input image size
             output_size = input_img.size()[2:]
             out = F.interpolate(out, size=output_size, mode='bilinear', align_corners=False)


        # Skip connection
        return torch.clamp(out + input_img, -1, 1)

# Discriminator (PatchGAN)
class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        # PatchGAN discriminator architecture
        self.model = nn.Sequential(
            # input is (input_channels) x 256 x 256
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: 64 x 128 x 128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: 128 x 64 x 64
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: 256 x 32 x 32
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: 512 x 16 x 16
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=False)
            # output depends on input size, typically results in a patch of scores
        )

    def forward(self, x):
        return self.model(x)

# VGG Perceptual Loss
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(weights='IMAGENET1K_V1').features[:36].eval()
        self.vgg = nn.Sequential()
        for i in range(36):
            self.vgg.add_module(str(i), vgg[i])

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.vgg.to(device)
        self.criterion = nn.L1Loss()
        # The weights from the paper are 1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0
        # We reverse it here because we pop from the list
        self.weights = [1.0, 1.0/4, 1.0/8, 1.0/16, 1.0/32]


    def forward(self, x, y):
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        y = (y + 1) / 2
        loss = 0
        weight_copy = self.weights.copy()  # Create a copy to avoid modifying the original list

        # Indices for features after ReLU layers in VGG19 features[:36]
        vgg_layer_indices = [2, 7, 12, 21, 30]

        current_layer_index = 0
        for i, layer in enumerate(self.vgg):
             x = layer(x)
             y = layer(y)

             if i in vgg_layer_indices: # After ReLU
                 loss += weight_copy.pop() * self.criterion(x, y)


        return loss

# Training function
def train(generator, discriminator, train_loader, val_loader, num_epochs, lr=0.0001):
    # Initialize optimizers
    optimizer_G = Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Learning rate schedulers
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=num_epochs, eta_min=1e-7)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=num_epochs, eta_min=1e-7)

    # Loss functions
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_pixel = nn.L1Loss()
    criterion_vgg = VGGLoss()

    # Weights for losses
    lambda_pixel = 100
    lambda_vgg = 10

    # Fixed images for visualization
    # Try to get a batch for visualization
    try:
        val_images = next(iter(val_loader))
        fixed_blur = val_images['blur'].to(device)
        fixed_sharp = val_images['sharp'].to(device)
    except StopIteration:
        print("Warning: Could not get a batch from the validation loader for visualization.")
        fixed_blur = None
        fixed_sharp = None


    # Training loop
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        # Progress bar
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        # Lists to store losses for this epoch
        G_losses = []
        D_losses = []

        for batch in loop:
            blur_images = batch['blur'].to(device)
            sharp_images = batch['sharp'].to(device)

            batch_size = blur_images.size(0)

            # Get output size for valid/fake labels based on input size
            # Calculate discriminator output size dynamically
            with torch.no_grad():
                # Pass a dummy tensor of the correct size to the discriminator
                dummy_input = torch.randn_like(sharp_images)
                disc_output_size = discriminator(dummy_input).size()


            # Ground truth labels with correct size
            valid = torch.ones(disc_output_size, device=device) # requires_grad=False is default for torch.ones/zeros
            fake = torch.zeros(disc_output_size, device=device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate deblurred images
            gen_images = generator(blur_images)

            # Adversarial loss
            pred_fake = discriminator(gen_images)
            loss_GAN = criterion_GAN(pred_fake, valid)

            # Pixel loss
            loss_pixel = criterion_pixel(gen_images, sharp_images)

            # VGG loss
            loss_vgg = criterion_vgg(gen_images, sharp_images)

            # Total generator loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel + lambda_vgg * loss_vgg

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(sharp_images)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss (detach to avoid training G on these labels)
            pred_fake = discriminator(gen_images.detach())
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total discriminator loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            # Save losses for display
            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())

            # Update progress bar
            loop.set_postfix(G_loss=loss_G.item(), D_loss=loss_D.item())

        # Learning rate decay
        scheduler_G.step()
        scheduler_D.step()

        # Validation and visualization
        generator.eval()
        if fixed_blur is not None:
            with torch.no_grad():
                gen_images = generator(fixed_blur)

                # Save sample images
                imgs = torch.cat([fixed_blur, gen_images, fixed_sharp], 0)
                imgs = (imgs + 1) / 2  # [-1, 1] -> [0, 1]
                grid = vutils.make_grid(imgs, nrow=fixed_blur.size(0), normalize=False)
                plt.figure(figsize=(15, 15))
                plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
                # Changed output path
                plt.savefig(f'/kaggle/working/results/epoch_{epoch+1}.png')
                plt.close()

        # Save model checkpoints
        # Changed output paths
        # After the training loop, save the most recent checkpoint:
        torch.save(generator.state_dict(), '/kaggle/working/checkpoints/generator.pth')
        torch.save(discriminator.state_dict(), '/kaggle/working/checkpoints/discriminator.pth')

        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs} | G Loss: {np.mean(G_losses):.4f} | D Loss: {np.mean(D_losses):.4f}")


def main():
    # Dataset parameters
    # Keep this path as it points to the input data source
    data_root = '/kaggle/input/gopro-image-deblurring-dataset/Gopro'
    image_size = 256
    batch_size = 4  # Adjust based on your GPU memory

    # Create directories for saving results
    # Changed output paths to /kaggle/working/
    os.makedirs('/kaggle/working/results', exist_ok=True)
    os.makedirs('/kaggle/working/checkpoints', exist_ok=True)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create datasets
    train_dataset = GoProDataset(root_dir=data_root, transform=transform, train=True)
    val_dataset = GoProDataset(root_dir=data_root, transform=transform, train=False)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize models
    generator = DeblurGenerator().to(device)
    discriminator = Discriminator().to(device)

    # Print model parameters
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters() if p.requires_grad)}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters() if p.requires_grad)}")

    # Training parameters
    num_epochs = 75  # Recommended number of epochs for GoPro dataset
    lr = 0.0001

    # Train the model
    train(generator, discriminator, train_loader, val_loader, num_epochs, lr)

if __name__ == '__main__':
    main()