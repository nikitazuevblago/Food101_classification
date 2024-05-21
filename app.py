import gradio as gr
import torch
import torchvision
from pathlib import Path
import pickle


def classify_image(img):
    # Convert the image to RGB if it has an alpha channel (RGBA)
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    # Get model's default transform
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    auto_transforms = weights.transforms()
    transformed_img = auto_transforms(img).unsqueeze(0)

    # Get trained model
    model = torchvision.models.resnet18(weights=weights)

    ## Define customization that was done during training
    model.classifier = model.fc
    del model.fc
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        model.classifier
    )
    ## Overwriting _forward_impl method for resnet18 to change "fc" to "classifier"
    def custom_resnet18_forward_impl(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    ## Patch the _forward_impl method
    model._forward_impl = custom_resnet18_forward_impl.__get__(model, torchvision.models.ResNet)
    model.classifier[-1] = torch.nn.Linear(in_features=model.classifier[-1].in_features,
                                            out_features=3, bias=True)

    ## Load pretrained weights
    chkp_path = 'models/resnet18_BS100_DR50perc.pt'
    model.load_state_dict(torch.load(chkp_path, map_location=torch.device('cpu')))

    # Make prediction
    model.eval()
    with torch.inference_mode():
      pred_label = torch.argmax(model(transformed_img)).item()
    with open('class_to_idx.pkl', 'rb') as file:
        class_to_idx = pickle.load(file)
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    return idx_to_class[pred_label]


demo = gr.Interface(classify_image, inputs=gr.Image(type='pil'), outputs="text", 
                    description="Upload the picture of 'bibimbap', 'chocolate_cake' or 'takoyaki'. The program will classify the image.", allow_flagging='never')
demo.launch()