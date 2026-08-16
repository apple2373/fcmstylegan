"""PyTorch dataset for preprocessed Sysmex Task 1 data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


PROFILE_SCALE = (12615.0, 132634.0, 51.0) #99% percentile (ie max exluding outlier)


class SysmexTask1Dataset(Dataset):
    """Load preprocessed Sysmex Task 1 brightfield images and profiles.

    Expected files for each cell::

        {cell_id}_brightfield_crop_masked_normalized_randbg_pad128.png
        {cell_id}_profiles_pad128.npz
        {cell_id}_mask_crop_pad128.png    # optional
        {cell_id}.json                    # optional

    The profile archive contains both horizontal and vertical profiles::

        horizontal_ssc
        horizontal_cd45
        horizontal_mask

        vertical_ssc
        vertical_cd45
        vertical_mask

    Parameters
    ----------
    csv_path:
        Path to the dataset CSV or CSV ZIP file.

    preprocessed_root:
        Directory containing the preprocessed files.

    id_column:
        CSV column containing the cell IDs.

    profile_prefix:
        Prefix used to select profiles from the NPZ file.

        ``"horizontal"`` loads::

            horizontal_ssc
            horizontal_cd45
            horizontal_mask

        ``"vertical"`` loads::

            vertical_ssc
            vertical_cd45
            vertical_mask

    brightfield_postfix:
        Postfix for the brightfield image filename.

    profile_postfix:
        Postfix for the profile filename.

    mask_postfix:
        Postfix for the mask filename.

    load_mask:
        If True, load the mask image and return it as ``"mask"``.

    load_json_metadata:
        If True, load ``{cell_id}.json`` when available.

    image_max_value:
        Maximum value used to scale the stored brightfield image.

        With the default ``65535.0``, the uint16 image is first
        converted from [0, 65535] to [0, 1].

    image_range:
        Output range for the brightfield image.

        ``"minus_one_one"``:
            Convert [0, 1] to [-1, 1]. This is the default.

        ``"zero_one"``:
            Keep the image in [0, 1].

    profile_scale:
        Scaling values for SSC, CD45, and mask profiles.

        Defaults to::

            (13570.0, 132411.0, 46.0)

    Returns
    -------
    dict
        ``image``:
            Float tensor of shape ``(1, 128, 128)``.

        ``profile``:
            Float tensor of shape ``(3, 128)`` ordered as
            SSC, CD45, mask.

        ``metadata``:
            Dictionary containing CSV metadata and optionally
            JSON metadata.

        ``mask``:
            Float tensor of shape ``(1, 128, 128)`` if
            ``load_mask=True``.
    """

    def __init__(
        self,
        csv_path: str | Path,
        preprocessed_root: str | Path,
        *,
        id_column: str = "cell_id",
        profile_prefix: str = "vertical",
        brightfield_postfix: str = (
            "_brightfield_crop_masked_normalized_avebg_pad128.png"
        ),
        profile_postfix: str = "_profiles_pad128.npz",
        mask_postfix: str = "_mask_crop_pad128.png",
        load_mask: bool = False,
        load_json_metadata: bool = True,
        image_max_value: float = 65535.0,
        image_range: str = "minus_one_one",
        profile_scale: tuple[float, float, float] = PROFILE_SCALE,
    ) -> None:
        if not profile_prefix:
            raise ValueError(
                "profile_prefix must be a non-empty string"
            )

        if image_max_value <= 0:
            raise ValueError(
                "image_max_value must be positive"
            )

        if image_range not in {
            "zero_one",
            "minus_one_one",
        }:
            raise ValueError(
                "image_range must be either "
                "'zero_one' or 'minus_one_one'"
            )

        if len(profile_scale) != 3:
            raise ValueError(
                "profile_scale must contain exactly 3 values"
            )

        if any(value <= 0 for value in profile_scale):
            raise ValueError(
                "profile_scale values must be positive"
            )

        self.preprocessed_root = Path(preprocessed_root)

        self.id_column = id_column
        self.profile_prefix = profile_prefix

        self.brightfield_postfix = brightfield_postfix
        self.profile_postfix = profile_postfix
        self.mask_postfix = mask_postfix

        self.load_mask = load_mask
        self.load_json_metadata = load_json_metadata

        self.image_max_value = image_max_value
        self.image_range = image_range

        self.profile_scale = torch.as_tensor(
            profile_scale,
            dtype=torch.float32,
        )

        # --------------------------------------------------------------
        # Load CSV / CSV ZIP
        # --------------------------------------------------------------
        csv_path = Path(csv_path)

        df = pd.read_csv(csv_path)

        if id_column not in df.columns:
            raise ValueError(
                f"CSV has no id column {id_column!r}"
            )

        if df.empty:
            raise ValueError("CSV contains no samples")

        self.rows = df.to_dict(orient="records")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Tensor | dict[str, object]]:
        row = self.rows[index]

        cell_id = str(row[self.id_column])

        # --------------------------------------------------------------
        # Paths
        # --------------------------------------------------------------
        image_path = self._image_path(cell_id)
        profile_path = self._profile_path(cell_id)

        # --------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------
        metadata: dict[str, object] = dict(row)

        if self.load_json_metadata:
            json_path = self._metadata_path(cell_id)

            if json_path.is_file():
                with json_path.open() as file:
                    json_metadata = json.load(file)

                if not isinstance(json_metadata, dict):
                    raise ValueError(
                        f"Expected JSON object in {json_path}"
                    )

                # CSV values take precedence over JSON values.
                for key, value in json_metadata.items():
                    metadata.setdefault(key, value)

        metadata["image_path"] = str(image_path)
        metadata["profile_path"] = str(profile_path)

        # --------------------------------------------------------------
        # Brightfield image
        # --------------------------------------------------------------
        image = np.asarray(
            Image.open(image_path),
            dtype=np.float32,
        )

        if image.shape != (128, 128):
            raise ValueError(
                f"Expected brightfield image shape (128, 128), "
                f"got {image.shape} for {image_path}"
            )

        # Convert [0, image_max_value] -> [0, 1].
        image = image / self.image_max_value

        # Convert [0, 1] -> [-1, 1] if requested.
        if self.image_range == "minus_one_one":
            image = image * 2.0 - 1.0

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image[None])
        ).float()

        # --------------------------------------------------------------
        # Profiles
        # --------------------------------------------------------------
        with np.load(
            profile_path,
            allow_pickle=False,
        ) as profiles:
            profile_keys = [
                f"{self.profile_prefix}_ssc",
                f"{self.profile_prefix}_cd45",
                f"{self.profile_prefix}_mask",
            ]

            missing_keys = [
                key
                for key in profile_keys
                if key not in profiles
            ]

            if missing_keys:
                raise KeyError(
                    f"{profile_path} is missing profile keys: "
                    f"{missing_keys}"
                )

            profile = np.stack(
                [profiles[key] for key in profile_keys],
                axis=0,
            )

        if profile.shape != (3, 128):
            raise ValueError(
                f"Expected profile shape (3, 128), "
                f"got {profile.shape} for {profile_path}"
            )

        profile_tensor = torch.from_numpy(
            np.ascontiguousarray(profile)
        ).float()

        # Channel-wise profile scaling:
        #
        # SSC   / 13570
        # CD45  / 132411
        # mask  / 46
        profile_tensor = (
            profile_tensor
            / self.profile_scale[:, None]
        )

        # --------------------------------------------------------------
        # Return sample
        # --------------------------------------------------------------
        sample: dict[str, Tensor | dict[str, object]] = {
            "image": image_tensor,
            "profile": profile_tensor,
            "metadata": metadata,
        }

        # --------------------------------------------------------------
        # Optional mask image
        # --------------------------------------------------------------
        if self.load_mask:
            mask_path = self._mask_path(cell_id)

            mask = np.asarray(
                Image.open(mask_path),
                dtype=np.uint8,
            )

            if mask.shape != (128, 128):
                raise ValueError(
                    f"Expected mask shape (128, 128), "
                    f"got {mask.shape} for {mask_path}"
                )

            # Convert mask to {0, 1}.
            mask = (mask > 0).astype(np.float32)

            mask_tensor = torch.from_numpy(
                np.ascontiguousarray(mask[None])
            ).float()

            sample["mask"] = mask_tensor

            metadata["mask_path"] = str(mask_path)

        return sample

    # ------------------------------------------------------------------
    # File paths
    # ------------------------------------------------------------------

    def _image_path(self, cell_id: str) -> Path:
        path = (
            self.preprocessed_root
            / f"{cell_id}{self.brightfield_postfix}"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Brightfield image not found: {path}"
            )

        return path

    def _profile_path(self, cell_id: str) -> Path:
        path = (
            self.preprocessed_root
            / f"{cell_id}{self.profile_postfix}"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Profile file not found: {path}"
            )

        return path

    def _mask_path(self, cell_id: str) -> Path:
        path = (
            self.preprocessed_root
            / f"{cell_id}{self.mask_postfix}"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Mask file not found: {path}"
            )

        return path

    def _metadata_path(self, cell_id: str) -> Path:
        return self.preprocessed_root / f"{cell_id}.json"