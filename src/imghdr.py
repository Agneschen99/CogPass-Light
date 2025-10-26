# shim for removed stdlib `imghdr` on Python 3.13+
# Minimal implementation using Pillow to detect image type
from typing import Optional
from PIL import Image

def what(file, h=None) -> Optional[str]:
    """Return a string describing the image type, e.g. 'jpeg', 'png' or None."""
    try:
        if h is not None:
            # Pillow can open from bytes
            from io import BytesIO
            img = Image.open(BytesIO(h))
        else:
            img = Image.open(file)
        fmt = img.format
        if not fmt:
            return None
        fmt = fmt.lower()
        if fmt == 'jpeg':
            return 'jpeg'
        if fmt == 'png':
            return 'png'
        if fmt == 'gif':
            return 'gif'
        if fmt == 'bmp':
            return 'bmp'
        if fmt == 'webp':
            return 'webp'
        return fmt
    except Exception:
        return None
