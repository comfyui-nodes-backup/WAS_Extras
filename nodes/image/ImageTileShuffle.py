import torch
import numpy as np
from PIL import Image


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def compute_grid(max_tiles):
    """Compute the most square-like (rows, cols) grid for the given tile count."""
    best_rows, best_cols = 1, max_tiles
    for r in range(1, int(max_tiles ** 0.5) + 1):
        if max_tiles % r == 0:
            c = max_tiles // r
            best_rows, best_cols = r, c
    return best_rows, best_cols


class WASImageTileShuffle:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input images to tile and shuffle (NHWC)."}),
                "max_tiles": ("INT", {
                    "default": 4,
                    "min": 2,
                    "max": 64,
                    "step": 2,
                    "tooltip": "Total number of tiles to split the image into (even numbers only)."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for tile shuffling order."
                }),
                "border_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Width of the border between tiles in pixels."
                }),
                "border_color": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": "Hex color for tile borders (e.g. #FFFFFF)."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "tile_shuffle"

    CATEGORY = "image/transform"

    def tile_shuffle(self, images, max_tiles, seed, border_width, border_color):

        # Parse border color from hex string
        color_hex = border_color.strip().lstrip("#")
        try:
            cr = int(color_hex[0:2], 16)
            cg = int(color_hex[2:4], 16)
            cb = int(color_hex[4:6], 16)
        except (ValueError, IndexError):
            cr, cg, cb = 255, 255, 255
        border_rgb = (cr, cg, cb)

        rows, cols = compute_grid(max_tiles)

        results = []
        for image in images:
            pil_img = tensor2pil(image)
            w, h = pil_img.size

            tile_w = w // cols
            tile_h = h // rows

            # Extract tiles row-by-row
            tiles = []
            for row in range(rows):
                for col in range(cols):
                    x1 = col * tile_w
                    y1 = row * tile_h
                    tiles.append(pil_img.crop((x1, y1, x1 + tile_w, y1 + tile_h)))

            # Shuffle tile order with the given seed
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(tiles)).tolist()
            shuffled = [tiles[i] for i in indices]

            # Build output canvas with optional borders
            out_w = cols * tile_w + (cols - 1) * border_width
            out_h = rows * tile_h + (rows - 1) * border_width
            output = Image.new("RGB", (out_w, out_h), border_rgb)

            for idx, tile in enumerate(shuffled):
                r = idx // cols
                c = idx % cols
                x = c * (tile_w + border_width)
                y = r * (tile_h + border_width)
                output.paste(tile, (x, y))

            results.append(pil2tensor(output))

        return (torch.cat(results, dim=0),)


class WASImageTileExtract:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input images to split into 4 tiles (NHWC)."}),
                "border_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Border width in pixels applied around each tile. The image content is resized to fit within the border."
                }),
                "border_color": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": "Hex color for the tile border (e.g. #FFFFFF)."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")

    FUNCTION = "tile_extract"

    CATEGORY = "image/transform"

    def tile_extract(self, images, border_width, border_color):

        # Parse border color from hex string
        color_hex = border_color.strip().lstrip("#")
        try:
            cr = int(color_hex[0:2], 16)
            cg = int(color_hex[2:4], 16)
            cb = int(color_hex[4:6], 16)
        except (ValueError, IndexError):
            cr, cg, cb = 255, 255, 255
        border_rgb = (cr, cg, cb)

        out_tl, out_tr, out_bl, out_br = [], [], [], []

        for image in images:
            pil_img = tensor2pil(image)
            w, h = pil_img.size

            tile_w = w // 2
            tile_h = h // 2

            # Crop quadrants: top-left, top-right, bottom-left, bottom-right
            quadrants = [
                pil_img.crop((0, 0, tile_w, tile_h)),
                pil_img.crop((tile_w, 0, tile_w * 2, tile_h)),
                pil_img.crop((0, tile_h, tile_w, tile_h * 2)),
                pil_img.crop((tile_w, tile_h, tile_w * 2, tile_h * 2)),
            ]

            results = []
            for quad in quadrants:
                if border_width > 0:
                    inner_w = tile_w - border_width * 2
                    inner_h = tile_h - border_width * 2
                    resized = quad.resize((max(inner_w, 1), max(inner_h, 1)), Image.LANCZOS)
                    canvas = Image.new("RGB", (tile_w, tile_h), border_rgb)
                    canvas.paste(resized, (border_width, border_width))
                    results.append(pil2tensor(canvas))
                else:
                    results.append(pil2tensor(quad))

            out_tl.append(results[0])
            out_tr.append(results[1])
            out_bl.append(results[2])
            out_br.append(results[3])

        return (
            torch.cat(out_tl, dim=0),
            torch.cat(out_tr, dim=0),
            torch.cat(out_bl, dim=0),
            torch.cat(out_br, dim=0),
        )


NODE_CLASS_MAPPINGS = {
    "WASImageTileShuffle": WASImageTileShuffle,
    "WASImageTileExtract": WASImageTileExtract,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WASImageTileShuffle": "Image Tile Shuffle",
    "WASImageTileExtract": "Image Tile Extract",
}
