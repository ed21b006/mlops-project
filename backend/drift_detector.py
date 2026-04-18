# Drift detector using z-score comparison against training baseline

import json
import logging
import os
import requests
import numpy as np

logger = logging.getLogger(__name__)


class DriftDetector:
    def __init__(self, baseline_path_or_url, z_threshold=3.0):
        self.z_threshold = z_threshold
        self.baseline = None

        try:
            if baseline_path_or_url.startswith("http"):
                logger.info(f"Fetching baseline from {baseline_path_or_url}")
                response = requests.get(baseline_path_or_url, timeout=5)
                response.raise_for_status()
                self.baseline = response.json()
                logger.info(f"Loaded baseline from {baseline_path_or_url}")
            else:
                if os.path.exists(baseline_path_or_url):
                    with open(baseline_path_or_url) as f:
                        self.baseline = json.load(f)
                    logger.info(f"Loaded baseline from {baseline_path_or_url}")
                else:
                    logger.warning(f"No baseline at {baseline_path_or_url}, drift detection disabled")
        except Exception as e:
            logger.warning(f"Failed to load baseline from {baseline_path_or_url}: {e}, drift detection disabled")

    def check(self, pixel_array):
        # Returns True if drift is detected, False otherwise
        if self.baseline is None:
            return False

        try:
            pixels = np.array(pixel_array).flatten()
            if len(pixels) != 784:
                return False

            means = np.array(self.baseline["pixel_means"])
            variances = np.array(self.baseline["pixel_variances"])
            stds = np.sqrt(np.maximum(variances, 1e-8))

            z_scores = np.abs(pixels - means) / stds
            drift_ratio = np.sum(z_scores > self.z_threshold) / len(pixels)

            # flag drift if more than 10% of pixels have drifted
            return drift_ratio > 0.1
        except Exception as e:
            logger.error(f"Drift check error: {e}")
            return False
