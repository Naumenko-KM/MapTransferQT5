import copy

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt


class CFG:
    imsize = 256
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    num_steps = 50
    style_weight = 100000
    content_weight = 1
    debug = False
    max_len = 275
    print_freq = 1000
    num_workers = 4
    model_name = 'resnet34'
    size = 224
    scheduler = 'CosineAnnealingLR'  # ['ReduceLROnPlateau',
    # 'CosineAnnealingLR',
    # 'CosineAnnealingWarmRestarts']
    epochs = 1  # not to exceed 9h
    # factor=0.2 # ReduceLROnPlateau
    # patience=4 # ReduceLROnPlateau
    # eps=1e-6 # ReduceLROnPlateau
    T_max = 4  # CosineAnnealingLR
    # T_0=4 # CosineAnnealingWarmRestarts
    encoder_lr = 1e-4
    decoder_lr = 4e-4
    min_lr = 1e-6
    batch_size = 64
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm = 5
    attention_dim = 256
    embed_dim = 256
    decoder_dim = 512
    dropout = 0.5
    seed = 42
    n_fold = 5
    trn_fold = [0]  # [0, 1, 2, 3, 4]
    train = True


loader = transforms.Compose([
    transforms.Resize(CFG.imsize),  # нормируем размер изображения
    transforms.CenterCrop(CFG.imsize),
    transforms.ToTensor()])  # превращаем в удобный формат


def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(CFG.device, torch.float)


def imsave(tensor, name="output_img"):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)      # функция для отрисовки изображения
    image = unloader(image)
    image.save(name+".png", format="png")


def imshow(tensor, title=None, ax=plt):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)  # функция для отрисовки изображения
    image = unloader(image)
    ax.imshow(image)
    if title is not None:
        ax.set_title(title)


def gram_matrix(input):
    batch_size, h, w, f_map_num = input.size()  # batch size(=1)
    # b=number of feature maps
    # (h,w)=dimensions of a feature map (N=h*w)
    # resise F_XL into \hat F_XL
    features = input.view(batch_size * h, w * f_map_num)
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(batch_size * h * w * f_map_num)


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    # добоваляет содержимое тензора катринки
    # в список изменяемых оптимизатором параметров
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        # это константа. Убираем ее из дерева вычеслений
        self.target = target.detach()
        # to initialize with something
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        # to initialize with something
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class StyleTransfer(nn.Module):
    def __init__(self):
        super(StyleTransfer, self).__init__()
        cnn = models.vgg19(pretrained=False)
        checkpoint = torch.load('vgg19-dcbb9e9d.pth', map_location='cpu')
        cnn.load_state_dict(checkpoint)
        self.cnn = cnn.features.to(CFG.device).eval()

    def get_style_model_and_losses(self, style_img, content_img):
        cnn = copy.deepcopy(self.cnn)
        # normalization module
        normalization = Normalization(CFG.normalization_mean,
                                      CFG.normalization_std).to(CFG.device)
        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []
        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)
        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely
                # with the ContentLoss and StyleLoss we insert below.
                # So we replace with out-of-place ones here.
                # Переопределим relu уровень
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(
                    layer.__class__.__name__))
            model.add_module(name, layer)

            if name in CFG.content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in CFG.style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        # выбрасываем все уровни после последенего styel loss или content loss
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or \
               isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]
        return model, style_losses, content_losses

    def run_style_transfer(self, content_img, style_img):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses(
                                                    style_img, content_img)
        input_img = content_img.clone()
        optimizer = get_input_optimizer(input_img)
        print('Optimizing..')
        run = [0]
        while run[0] <= CFG.num_steps:
            def closure():
                # correct the values
                # это для того, чтобы значения тензора картинки
                #  не выходили за пределы [0;1]
                input_img.data.clamp_(0, 1)
                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                # взвешивание ощибки
                style_score *= CFG.style_weight
                content_score *= CFG.content_weight
                loss = style_score + content_score
                loss.backward()
                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()
                return style_score + content_score
            optimizer.step(closure)
        # a last correction...
        input_img.data.clamp_(0, 1)
        print(type(input_img), input_img.shape)
        input_img = input_img.detach().cpu().numpy()
        print(type(input_img), input_img.shape)
        return input_img


if __name__ == "__main__":
    style_img = image_loader("winter.png")  # as well as here
    # измените путь на тот который у вас.
    content_img = image_loader("summer.jpeg")
    style_transfer = StyleTransfer()
    output_img = style_transfer.run_style_transfer(style_img, content_img)
    
    fig, axs = plt.subplots(1, 3)
    fig.set_figwidth(25)
    fig.set_figheight(10)
    imshow(style_img, title='Style Image', ax=axs[0])
    imshow(content_img, title='Content Image', ax=axs[1])
    imshow(output_img, title='Output Image', ax=axs[2])
    plt.show()
