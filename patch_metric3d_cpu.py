import os
import re

p = os.path.expanduser('~/.cache/torch/hub/yvanyin_metric3d_main/mono/model/decode_heads/RAFTDepthNormalDPTDecoder5.py')
if os.path.exists(p):
    with open(p, 'r', encoding='utf-8') as f:
        txt = f.read()
    
    # Replace device="cuda" with device=feature_map.device or self.device_indicator.device if not cuda available
    # Or device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    new_txt = txt.replace('device="cuda"', "device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
    new_txt = new_txt.replace("device='cuda'", "device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
    
    if new_txt != txt:
        with open(p, 'w', encoding='utf-8') as f:
            f.write(new_txt)
        print("Patched Metric3D decoder to support CPU gracefully!")
    else:
        print("Metric3D decoder was already patched or no cuda occurrences.")
else:
    print(f"Path not found: {p}")
