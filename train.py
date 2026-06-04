import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.network import ExplainableMultiTaskNet
from utils.losses import JointMultiTaskLoss
# from data_loader import get_dataloaders  # <--- User needs to provide their own dataloader

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # 1. Initialize Model
    # Assumes classes: 0: Normal, 1: Benign, 2: Malignant
    model = ExplainableMultiTaskNet(num_classes=3, minority_class_idx=2).to(device)
    
    # 2. Hyperparameters (from Thesis Sec 3.3.2)
    lr = 1e-4
    weight_decay = 1e-5
    epochs = 150
    accumulation_steps = 2 # Effective batch size = 8 * 2 = 16
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = JointMultiTaskLoss(alpha=0.7, beta=0.3, gamma=0.1)
    
    # DUMMY DATALOADER (Replace with actual DataLoader)
    # dataloader = get_dataloaders(batch_size=8)
    # Using random data here to demonstrate the training loop
    dummy_loader = [(torch.randn(8, 3, 256, 256), torch.randint(0, 3, (8,)), torch.randint(0, 2, (8, 1, 256, 256)).float()) for _ in range(10)]

    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for i, (images, cls_labels, seg_masks) in enumerate(dummy_loader):
            images = images.to(device)
            cls_labels = cls_labels.to(device)
            seg_masks = seg_masks.to(device)
            
            # Forward Pass (Note: Pass labels for Latent Augmentation!)
            cls_out, seg_out, ds_outs = model(images, labels=cls_labels)
            
            # Compute Composite Loss
            loss, l_seg, l_cls, l_ms = criterion(cls_out, seg_out, ds_outs, cls_labels, seg_masks)
            
            # Gradient Accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            epoch_loss += loss.item() * accumulation_steps
            
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{epochs}] | Total Loss: {epoch_loss/len(dummy_loader):.4f}")
        
    print("Training Complete. Saving model...")
    torch.save(model.state_dict(), "explainable_multitask_bus.pth")

if __name__ == '__main__':
    train_model()