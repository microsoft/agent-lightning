# Copyright (c) Microsoft. All rights reserved.

"""Static-analysis bridge for the separately packaged contrib extension.

The installed extension is available at ``agentlightning.contrib.shaper``.
The repository keeps its source outside the core package tree, so this stub
exposes the same public symbols while checking an uninstalled checkout.
"""

from contrib.agentlightning.contrib.shaper import DEFAULT_HARNESS_CONTRACT as DEFAULT_HARNESS_CONTRACT
from contrib.agentlightning.contrib.shaper import SHAPER as SHAPER
from contrib.agentlightning.contrib.shaper import ArtifactCandidate as ArtifactCandidate
from contrib.agentlightning.contrib.shaper import ArtifactProposal as ArtifactProposal
from contrib.agentlightning.contrib.shaper import ArtifactStage as ArtifactStage
from contrib.agentlightning.contrib.shaper import CandidateEvaluation as CandidateEvaluation
from contrib.agentlightning.contrib.shaper import EpisodeMetadata as EpisodeMetadata
from contrib.agentlightning.contrib.shaper import EpisodeSummary as EpisodeSummary
from contrib.agentlightning.contrib.shaper import EpisodeTrace as EpisodeTrace
from contrib.agentlightning.contrib.shaper import HarnessOutputValidator as HarnessOutputValidator
from contrib.agentlightning.contrib.shaper import HarnessRuntimeError as HarnessRuntimeError
from contrib.agentlightning.contrib.shaper import HarnessValidationResult as HarnessValidationResult
from contrib.agentlightning.contrib.shaper import IncomparableCandidateError as IncomparableCandidateError
from contrib.agentlightning.contrib.shaper import OptimizationEvent as OptimizationEvent
from contrib.agentlightning.contrib.shaper import OptimizationStage as OptimizationStage
from contrib.agentlightning.contrib.shaper import OptimizerRequestContext as OptimizerRequestContext
from contrib.agentlightning.contrib.shaper import PythonHarnessRuntime as PythonHarnessRuntime
from contrib.agentlightning.contrib.shaper import PythonHarnessValidator as PythonHarnessValidator
from contrib.agentlightning.contrib.shaper import RoleCompleter as RoleCompleter
from contrib.agentlightning.contrib.shaper import RoleRequest as RoleRequest
from contrib.agentlightning.contrib.shaper import RolloutInfrastructureError as RolloutInfrastructureError
from contrib.agentlightning.contrib.shaper import RoundCritique as RoundCritique
from contrib.agentlightning.contrib.shaper import RoundRecord as RoundRecord
from contrib.agentlightning.contrib.shaper import SHAPERRoleProtocol as SHAPERRoleProtocol
from contrib.agentlightning.contrib.shaper import SHAPERTraceAdapter as SHAPERTraceAdapter
from contrib.agentlightning.contrib.shaper import SkillValidator as SkillValidator
from contrib.agentlightning.contrib.shaper import emit_episode_metadata as emit_episode_metadata
from contrib.agentlightning.contrib.shaper import emit_round_record as emit_round_record
from contrib.agentlightning.contrib.shaper import parse_json_object as parse_json_object
from contrib.agentlightning.contrib.shaper import validate_nonempty_skill as validate_nonempty_skill

__all__: list[str]
