import torch

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            torch.save(model.state_dict(), "/home/zumbie/Codes/NML/DenseNet-Project/src/ComLook/model/best_model.pth")

        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            print(f"No improvement. Counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), "/home/zumbie/Codes/NML/DenseNet-Project/src/ComLook/model/best_model.pth")