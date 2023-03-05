#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
import channel, instance
import torchvision.transforms as transforms
import PIL.Image as Image
import visualkeras
from torchsummary import summary
from threading import Thread

#%% Part 1 : Encoder 
def quantize(x, mode='noise', means=None):
       
        if mode == 'noise':
            quantization_noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
            x = x + quantization_noise

        elif mode == 'quantize':
            if means is not None:
                x = x - means
                x = torch.floor(x + 0.5)
                x = x + means
            else:
                x = torch.floor(x + 0.5)
        else:
            raise NotImplementedError
        
        return x

class Encoder(nn.Module):
    def __init__(self, image_dims, batch_size, activation='relu', C=220,
                 channel_norm=True):
        
        super(Encoder, self).__init__()
        
        kernel_dim = 3
        filters = (60, 120, 240, 480, 960)

        # Images downscaled to 500 x 1000 + randomly cropped to 256 x 256
        im_channels = image_dims[0]
        # assert image_dims == (im_channels, 256, 256), 'Crop image to 256 x 256!'

        # Layer / normalization options
        cnn_kwargs = dict(stride=2, padding=0, padding_mode='reflect')
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)
        activation_d = dict(relu='ReLU', elu='ELU', leaky_relu='LeakyReLU')
        self.activation = getattr(nn, activation_d[activation])  # (leaky_relu, relu, elu)
        self.n_downsampling_layers = 4

        if channel_norm is True:
            self.interlayer_norm = channel.ChannelNorm2D_wrap
        else:
            self.interlayer_norm = instance.InstanceNorm2D_wrap

        self.pre_pad = nn.ReflectionPad2d(3)
        self.asymmetric_pad = nn.ReflectionPad2d((0,1,1,0))  # Slower than tensorflow?
        self.post_pad = nn.ReflectionPad2d(1)

        heights = [2**i for i in range(4,9)][::-1]
        widths = heights
        H1, H2, H3, H4, H5 = heights
        W1, W2, W3, W4, W5 = widths 

        # (256,256) -> (256,256), with implicit padding
        self.conv_block1 = nn.Sequential(
            self.pre_pad,
            nn.Conv2d(im_channels, filters[0], kernel_size=(7,7), stride=1),
            self.interlayer_norm(filters[0], **norm_kwargs),
            self.activation(),
        )

        # (256,256) -> (128,128)
        self.conv_block2 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(filters[0], filters[1], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[1], **norm_kwargs),
            self.activation(),
        )

        # (128,128) -> (64,64)
        self.conv_block3 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(filters[1], filters[2], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[2], **norm_kwargs),
            self.activation(),
        )

        # (64,64) -> (32,32)
        self.conv_block4 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(filters[2], filters[3], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[3], **norm_kwargs),
            self.activation(),
        )

        # (32,32) -> (16,16)
        self.conv_block5 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(filters[3], filters[4], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[4], **norm_kwargs),
            self.activation(),
        )
        
        # Project channels onto space w/ dimension C
        # Feature maps have dimension C x W/16 x H/16
        # (16,16) -> (16,16)
        self.conv_block_out = nn.Sequential(
            self.post_pad,
            nn.Conv2d(filters[4], C, kernel_dim, stride=1),
        )
        
                
    def forward(self, x):
        x = self.conv_block1(x)
        layer_1 =x
        x = self.conv_block2(x)
        layer_2= x 
        x = self.conv_block3(x)
        layer_3 =x 
        x = self.conv_block4(x)
        layer_4 =x
        x = self.conv_block5(x)
        out = self.conv_block_out(x)
        return out, layer_1 , layer_2, layer_3, layer_4
    

class HyperpriorAnalysis(nn.Module):
   
    
    def __init__(self, C=220, N=320, activation='relu'):
        super(HyperpriorAnalysis, self).__init__()

        cnn_kwargs = dict(kernel_size=5, stride=2, padding=2, padding_mode='reflect')
        self.activation = getattr(F, activation)
        self.n_downsampling_layers = 2

        self.conv1 = nn.Conv2d(C, N, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(N, N, **cnn_kwargs)
        self.conv3 = nn.Conv2d(N, N, **cnn_kwargs)

    def forward(self, x):
        
        # x = torch.abs(x)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)

        return x


def display(im):
    im.show()

#%% Reading the image and transfer it to a tensor : 

img = Image.open('pic.png')

transform = transforms.Compose([
    transforms.PILToTensor()
])
  
img_tensor = transform(img)


print(img_tensor)
print('Image 1')

#%% Adding the Batch dimension 

input_torch= torch.zeros(1,img_tensor.shape[0],img_tensor.shape[1],img_tensor.shape[2])
input_torch[0,:,:,:]=img_tensor

#%% 
B = input_torch.shape[0]
x_dims = tuple(input_torch.size())
# %% Loading weights of the model 
load = torch.load(r"C:\Users\TEMMMAR\Desktop\Hifi_local\Chekpoint\hific-high.pt")
# %% Extracting Encoder/Hyper analysis weights
new_state_dict = {}
for name, weight in load['model_state_dict'].items():
    if 'Encoder' in name:
        new_state_dict[name] = weight

new_state_dict1 = {}
for key, value in new_state_dict.items():
    new_key = key.replace("Encoder.", "")
    new_state_dict1[new_key] = value

new_state_dict3 = {}
for name, weight in load['model_state_dict'].items():
    if 'Hyperprior.analysis_net' in name:
        new_state_dict3[name] = weight

new_state_dict4 = {}
for key, value in new_state_dict3.items():
    new_key = key.replace("Hyperprior.analysis_net.", "")
    new_state_dict4[new_key] = value



#%%
# new_state_dict5 = {}
# for name, weight in load['model_state_dict'].items():
#     if 'Generator' in name:
#         new_state_dict5[name] = weight

# new_state_dict6 = {}
# for key, value in new_state_dict5.items():
#     new_key = key.replace("Generator.", "")
#     new_state_dict6[new_key] = value 
# new_state_dict = {}
# for key, value in new_state_dict1.items():
#     if key == "conv_block_out.1.weight":
#         new_value = value[:3, ...]  # slice first 3 channels
#         new_state_dict[key] = new_value
#     elif key == "conv_block_out.1.bias":
#         new_value = value[:3]  # slice first 3 elements
#         new_state_dict[key] = new_value
#     else:
#         new_state_dict[key] = value

#%% Implementing weights 
model= Encoder(image_dims=x_dims[1:], batch_size=B, C=220)
model.load_state_dict(new_state_dict1,strict=False)
#%%
Hyper= HyperpriorAnalysis()
Hyper.load_state_dict(new_state_dict4, strict=False)
#%%
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# summary(model, input_size=(3, 256, 256))
# %%
y,layer1,layer2,layer3,layer4 =Encoder.forward(model,input_torch.float())
#%%
H = HyperpriorAnalysis.forward(Hyper,y)
#%%
z=quantize(y,mode='quantize')
# %% Ploting and saving the input image 
k1=y.detach().numpy()
k2=z.detach().numpy()
k3=layer2.detach().numpy()

l=np.zeros((k1.shape[2],k1.shape[3],3))
l1=np.zeros((k2.shape[2],k2.shape[3],3))
l3=np.zeros((k3.shape[2],k3.shape[3],3))

l[:,:,2]= k1[0,0,:,:]
l[:,:,1]=k1[0,1,:,:]
l[:,:,2]=k1[0,2,:,:]

l1[:,:,0]= k2[0,0,:,:]
l1[:,:,1]=k2[0,1,:,:]
l1[:,:,2]=k2[0,2,:,:]

l3[:,:,0]= k3[0,0,:,:]
l3[:,:,1]=k3[0,1,:,:]
l3[:,:,2]=k3[0,2,:,:]


img1 = Image.fromarray(np.uint8(l*255))
img2 = Image.fromarray(np.uint8(l1*255))
img3 = Image.fromarray(np.uint8(l3*255))

t1=Thread(target=display,args=(img1,))
t1.start()
t2=Thread(target=display,args=(img2,))
t2.start()
t3=Thread(target=display,args=(img3,))
t3.start()

# %%
