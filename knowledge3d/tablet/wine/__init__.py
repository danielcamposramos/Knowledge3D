from __future__ import annotations

import importlib

from .dataset_wrappers import (
    amc_aime_math_envelope,
    arc2_game_envelope,
    arc3_game_envelope,
    gsm8k_math_envelope,
    imo_math_envelope,
    lhe_question_envelope,
    math_dataset_envelope,
    mmlu_question_envelope,
    omni_math_envelope,
)
from .game2d_wine import GAME_2D_ROUTE_GALAXIES, build_game2d_route, build_game2d_task
from .math_wine import MATH_ROUTE_GALAXIES, build_math_route, build_math_task
from .question_wine import QUESTION_ROUTE_GALAXIES, build_question_route, build_question_task

_model3d = importlib.import_module(".3d_model_wine", __name__)
External3DWineBridge = _model3d.External3DWineBridge
MODEL_TYPE_PROCEDURAL = _model3d.MODEL_TYPE_PROCEDURAL
MODEL_TYPE_TRELLIS = _model3d.MODEL_TYPE_TRELLIS
MODEL_TYPE_HUNYUAN = _model3d.MODEL_TYPE_HUNYUAN
build_3d_route = _model3d.build_3d_route
build_3d_task = _model3d.build_3d_task
create_multimodal_3d_task = _model3d.create_multimodal_3d_task

__all__ = [
    "External3DWineBridge",
    "GAME_2D_ROUTE_GALAXIES",
    "MODEL_TYPE_HUNYUAN",
    "MATH_ROUTE_GALAXIES",
    "MODEL_TYPE_PROCEDURAL",
    "QUESTION_ROUTE_GALAXIES",
    "MODEL_TYPE_TRELLIS",
    "amc_aime_math_envelope",
    "arc2_game_envelope",
    "arc3_game_envelope",
    "build_3d_route",
    "build_3d_task",
    "build_game2d_route",
    "build_game2d_task",
    "build_math_route",
    "build_math_task",
    "build_question_route",
    "build_question_task",
    "create_multimodal_3d_task",
    "gsm8k_math_envelope",
    "imo_math_envelope",
    "lhe_question_envelope",
    "math_dataset_envelope",
    "mmlu_question_envelope",
    "omni_math_envelope",
]
