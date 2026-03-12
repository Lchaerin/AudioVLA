import torch
import torch.nn as nn


class AzElToPixel(nn.Module):
    """
    Convert SELD azimuth/elevation coordinates to image pixel coordinates.
    Assumes a known camera model (pinhole) and optionally a known extrinsic transform.

    Coordinate convention:
      - Azimuth:   0 = forward (+z), +π/2 = right (+x)
      - Elevation: 0 = horizontal, +π/2 = up (−y in image convention)
      - Camera:    right-handed, z-forward
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        peak_coords: torch.Tensor,
        camera_intrinsic: torch.Tensor,
        camera_extrinsic: torch.Tensor = None,
        img_h: int = 512,
        img_w: int = 512,
    ):
        """
        Args:
            peak_coords:      (B, N, 2)    - [azimuth, elevation] in radians
            camera_intrinsic: (B, 3, 3) or (3, 3) - [[fx,0,cx],[0,fy,cy],[0,0,1]]
            camera_extrinsic: (B, 4, 4) or None   - world→camera rigid transform
            img_h, img_w:     image resolution
        Returns:
            pixel_coords: (B, N, 2) - [u, v] (clamped to image bounds)
            in_frame:     (B, N)    - bool, whether the source is within the frame
        """
        B, N, _ = peak_coords.shape

        az = peak_coords[..., 0]  # (B, N)
        el = peak_coords[..., 1]  # (B, N)

        # Spherical → Cartesian (z-forward, y-down)
        x = torch.cos(el) * torch.sin(az)
        y = -torch.sin(el)  # y points down in image convention
        z = torch.cos(el) * torch.cos(az)
        points_3d = torch.stack([x, y, z], dim=-1)  # (B, N, 3)

        # Apply camera extrinsic if provided
        if camera_extrinsic is not None:
            if camera_extrinsic.dim() == 2:
                camera_extrinsic = camera_extrinsic.unsqueeze(0).expand(B, -1, -1)
            R = camera_extrinsic[:, :3, :3]  # (B, 3, 3)
            t = camera_extrinsic[:, :3, 3:]  # (B, 3, 1)
            points_cam = (R @ points_3d.transpose(-1, -2) + t).transpose(-1, -2)
        else:
            points_cam = points_3d  # (B, N, 3)

        # Expand intrinsic if needed
        if camera_intrinsic.dim() == 2:
            camera_intrinsic = camera_intrinsic.unsqueeze(0).expand(B, -1, -1)

        # Project: K @ p  →  (B, N, 3)
        proj = (camera_intrinsic @ points_cam.transpose(-1, -2)).transpose(-1, -2)

        # Perspective division: (u, v) = (X/Z, Y/Z)
        pixel_uv = proj[..., :2] / (proj[..., 2:3] + 1e-6)  # (B, N, 2)

        # Frame boundary check
        in_frame = (
            (pixel_uv[..., 0] >= 0) & (pixel_uv[..., 0] < img_w) &
            (pixel_uv[..., 1] >= 0) & (pixel_uv[..., 1] < img_h) &
            (points_cam[..., 2] > 0)  # must be in front of camera
        )  # (B, N)

        # Clamp to image bounds
        pixel_uv = torch.stack([
            pixel_uv[..., 0].clamp(0, img_w - 1),
            pixel_uv[..., 1].clamp(0, img_h - 1),
        ], dim=-1)  # (B, N, 2)

        return pixel_uv, in_frame
