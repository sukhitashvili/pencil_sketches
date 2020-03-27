import torch
from torchvision import transforms
import model_gan
import pickle
import PIL

use_cuda = torch.cuda.device_count() > 0

model = model_gan.model_gan
model.load_state_dict(torch.load('./models/model_gan.pth'))
if False:  # use_cuda:  # bcz of run out of memorry
    model = model.cuda()
model.eval()

# load mean and std
with open('models/mean_std.pkl', "rb") as f:
    mean_std = pickle.load(f)
immean = mean_std['mean']
imstd = mean_std['std']


def pencil_sketch(img):
    data = PIL.Image.fromarray(img).convert('L')
    w, h = data.size[0], data.size[1]
    pw = 8 - (w % 8) if w % 8 != 0 else 0
    ph = 8 - (h % 8) if h % 8 != 0 else 0
    data = ((transforms.ToTensor()(data) - immean) / imstd).unsqueeze(0)
    if pw != 0 or ph != 0:
        data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data).data

    if False:  # use_cuda:  # bcz of run out of memorry
        with torch.no_grad():
            pred = model(data.cuda()).float()
    else:
        with torch.no_grad():
            pred = model(data)

    return (pred[0, 0, ...].cpu().numpy() * 255).astype('uint8')
