"""
Patch for lerobot's hf_transform_to_torch function to handle image dictionaries.

This patch fixes the issue where lerobot cannot process image dictionaries 
with 'bytes' keys, which is the standard format used by many lerobot datasets
including DroneVLA and Libero.
"""

import io
import torch
from PIL import Image as PILImage
from torchvision import transforms as torchvision_transforms

def patched_hf_transform_to_torch(items_dict):
    """
    Patched version of hf_transform_to_torch that handles image dictionaries properly.
    
    This function processes data items and converts them to torch tensors, with special
    handling for image dictionaries that contain 'bytes' keys.
    """
    for key in items_dict:
        first_item = items_dict[key][0]
        if isinstance(first_item, PILImage.Image):
            # Standard PIL Image handling
            to_tensor = torchvision_transforms.ToTensor()
            items_dict[key] = [to_tensor(img) for img in items_dict[key]]
        elif isinstance(first_item, dict) and 'bytes' in first_item:
            # Handle image dictionaries by converting bytes to PIL Images then to tensors
            def dict_to_pil(img_dict):
                return PILImage.open(io.BytesIO(img_dict['bytes']))
            
            to_tensor = torchvision_transforms.ToTensor()
            items_dict[key] = [to_tensor(dict_to_pil(img)) for img in items_dict[key]]
        elif first_item is None:
            # Skip None values
            pass
        else:
            # Standard tensor conversion for other data types
            items_dict[key] = [torch.tensor(x) for x in items_dict[key]]
    return items_dict

def apply_lerobot_patch():
    """Apply the patch to lerobot's hf_transform_to_torch function."""
    try:
        from lerobot.common.datasets import utils
        utils.hf_transform_to_torch = patched_hf_transform_to_torch
        return True
    except ImportError:
        return False


# Apply the patch immediately when this module is imported
apply_lerobot_patch() 
