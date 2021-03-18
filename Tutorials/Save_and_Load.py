import torch
import torch.onnx as onnx
import torchvision.models as models

# Saving and loading model weights
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

model = models.vgg16() # do not specify pretrained=True, ue do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# This saves the model shape along with the weight dictionary
torch.save(model, 'model.pth')

# To then load this model we do the following
model = torch.load('model.pth')

input_image = torch.zeros((1, 3, 224, 224))
onnx.export(model, input_image, 'model.onnx')

