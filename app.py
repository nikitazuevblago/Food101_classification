import gradio as gr
import torch
import torchvision
import pickle
from PIL import Image
import os


def classify_image(img):

    # Get trained model
    models = os.listdir('models/')
    assert len(models)==1, "More than 1 model in 'models/' folder!"
    model_name = models[0]
    num_labels = len(os.listdir('data/'))

    if 'mobilenet_v3_small' in model_name:
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        model = torchvision.models.mobilenet_v3_small(weights=weights)

    elif 'shufflenet_v2_x0_5' in model_name:
        weights = torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT
        model = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
        # Making the last layer similar to first two models for purposes of experiment
        model.classifier = model.fc
        del model.fc
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0),
            model.classifier
        )

        # Overwriting _forward_impl method for shufflenet_v2_x0_5 to change "fc" to "classifier"
        def custom_shufflenet_forward_impl(self, x:torch.Tensor) -> torch.Tensor:
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.conv5(x)
            x = x.mean([2, 3])  # globalpool
            x = self.classifier(x)
            return x

        # Patch the _forward_impl method
        model._forward_impl = custom_shufflenet_forward_impl.__get__(model, torchvision.models.ResNet)

    model.classifier[-1] = torch.nn.Linear(in_features=model.classifier[-1].in_features,
                                            out_features=num_labels, bias=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    ## Load pretrained weights
    chkp_path = os.path.join('models', model_name)
    model.load_state_dict(torch.load(chkp_path, map_location=device))

    ## Transform image
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    auto_transforms = weights.transforms()
    transformed_img = auto_transforms(img).unsqueeze(0)

    # Make prediction
    model.eval()
    with torch.inference_mode():
      pred_label = torch.argmax(model(transformed_img)).item()

    with open('class_to_idx.pkl', 'rb') as file:
        class_to_idx = pickle.load(file)
    idx_to_class = {v:k for k,v in class_to_idx.items()}

    return idx_to_class[pred_label]


with open('class_to_idx.pkl', 'rb') as file:
    class_to_idx = pickle.load(file)
    food_types = list(class_to_idx.keys())

example_images = [
    Image.open(os.path.join("examples_to_predict","clam chowder.jpeg")),
    Image.open(os.path.join("examples_to_predict","donuts.jpg")),
    Image.open(os.path.join("examples_to_predict","ice cream.jpeg"))
]

#The examples parameter expects a list of lists
examples = [[img] for img in example_images]

demo = gr.Interface(classify_image, inputs=gr.Image(type='pil'), outputs="text", 
                    description=f"Upload the picture of one of these food types {food_types}. The program will classify the image.", 
                    allow_flagging='never', examples=examples)
demo.launch()
