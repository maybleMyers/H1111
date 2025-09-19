# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from .process_pipepline import ProcessPipeline

# Try to import SAM2VideoPredictor (optional, requires SAM2)
try:
    from .video_predictor import SAM2VideoPredictor
except ImportError:
    SAM2VideoPredictor = None