import torch
import torchvision.models.detection as detection
import torch.optim as optim

from supports.xml_parser import MangaVOCDataset
from supports.earlystopping import EarlyStopping

# ------------------------
# DATASETS
# ------------------------

train_dataset = MangaVOCDataset(
    images_dir="/home/zumbie/Codes/PERSONAL/textLocator/Training/dataset/images",
    annotations_dir="/home/zumbie/Codes/PERSONAL/textLocator/Training/dataset/annotations"
)

val_dataset = MangaVOCDataset(
    images_dir="/home/zumbie/Codes/PERSONAL/textLocator/Training/dataset/images2",
    annotations_dir="/home/zumbie/Codes/PERSONAL/textLocator/Training/dataset/annotations2"
)

def collate_fn(batch):
    return tuple(zip(*batch))


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn
)

# ------------------------
# MODEL
# ------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

num_classes = 3  # background + japanese + english
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = \
    detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model.to(device)

# ------------------------
# FREEZE BACKBONE (first phase)
# ------------------------

for param in model.backbone.parameters():
    param.requires_grad = False

# ------------------------
# OPTIMIZER
# ------------------------

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(params, lr=1e-4)

# Optional but smart
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=3, factor=0.5
)

# ------------------------
# VALIDATION LOSS
# ------------------------

def evaluate(model, data_loader, device):
    model.train()  # required for detection loss
    val_loss = 0.0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()

    return val_loss / len(data_loader)

# ------------------------
# TRAIN LOOP WITH FREEZE + UNFREEZE
# ------------------------

early_stopping = EarlyStopping(patience=5)
num_epochs = 40

for epoch in range(num_epochs):

    model.train()
    train_loss = 0.0

    # 🔥 Unfreeze backbone after 5 epochs
    if epoch == 5:
        print("Unfreezing backbone...")
        for param in model.backbone.parameters():
            param.requires_grad = True
        
        optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train_loss += losses.item()

    train_loss /= len(train_loader)

    val_loss = evaluate(model, val_loader, device)

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")

    early_stopping(val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break