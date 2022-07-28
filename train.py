from utils import preprocess_labels, create_dataloader, Model_Resnet50
from torchvision import transforms as T
from torch import save


if __name__ == '__main__':

    # Target labels
    label_file = 'data/noisy_imagewoof.csv'
    labels = preprocess_labels(label_file)

    # Train dataloader
    train_dl = create_dataloader(labels, root='data', transforms=T.Compose([T.Resize(size=(224, 224)),
                                                                 T.ToTensor(),
                                                                 T.Normalize(mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225])
                                                                 ]), valid=False, shuffle=True, batch_size=64)

    # Valid dataloader
    valid_dl = create_dataloader(labels, root='data', transforms=T.Compose([T.Resize(size=(224, 224)),
                                                                 T.ToTensor(),
                                                                 T.Normalize(mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225])
                                                                 ]), valid=True, shuffle=False, batch_size=64)


    model = Model_Resnet50()

    model.train_eval(train_dl, valid_dl, n_epoches=5)

    filename = 'models/model_v1.pt'

    save(model, filename)
