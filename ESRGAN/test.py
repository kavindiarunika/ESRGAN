import os
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch  # make sure this file is in the same folder

# -----------------------------
# Settings
# -----------------------------
device = torch.device('cpu')  # CPU only

model_path = 'models/RRDB_ESRGAN_x4.pth'  # put your downloaded weights here
test_img_folder = 'LR/*'                   # folder containing your low-res images
results_folder = 'results'                 # folder to save output images

# Create results folder if it doesn't exist
os.makedirs(results_folder, exist_ok=True)

# -----------------------------
# Load model
# -----------------------------
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint, strict=True)
model.eval()
model = model.to(device)

print(f'Model loaded from {model_path}. Testing images...')

# -----------------------------
# Process images
# -----------------------------
for idx, img_path in enumerate(glob.glob(test_img_folder), 1):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    print(f'{idx}: {base_name}')

    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).unsqueeze(0)
    img = img.to(device)

    # Super-resolve
    with torch.no_grad():
        output = model(img).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    # Convert back to image format
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)

    # Save result
    save_path = os.path.join(results_folder, f'{base_name}_rlt.png')
    cv2.imwrite(save_path, output)

print(f'All images processed. Results saved in {results_folder}/')
