# -*- coding: utf-8 -*-

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import abc
from dataclasses import dataclass
from dataclasses import field
import enum
from typing import Any, Dict, List, Optional, Union
from google.cloud.aiplatform.compat.types import (
    feature_online_store_service as fos_service,
)
import proto
from typing_extensions import override


def get_feature_online_store_name(online_store_name: str) -> str:
    """Extract Feature Online Store's name from FeatureView's full resource name.

    Args:
        online_store_name: Full resource name is projects/project_number/
          locations/us-central1/featureOnlineStores/fos_name/featureViews/fv_name

    Returns:
        str: feature online store name.
    """
    arr = online_store_name.split("/")
    return arr[5]


class PublicEndpointNotFoundError(RuntimeError):
    """Public endpoint has not been created yet."""


@dataclass
class FeatureViewBigQuerySource:
    uri: str
    entity_id_columns: List[str]


@dataclass
class FeatureViewVertexRagSource:
    uri: str
    rag_corpus_id: Optional[str] = None


@dataclass
class FeatureViewRegistrySource:
    """Configuration options for Feature View being registered with Feature Registry features.

    Attributes:
        features : Use `<feature_group_id>.<feature_id>` as
          the format for each feature.
        project_number : Optional. The project number of the project that owns the
          Feature Registry if in a different project.
    """

    features: List[str]
    project_number: Optional[int] = None


@dataclass(frozen=True)
class ConnectionOptions:
    """Represents connection options used for sending RPCs to the online store."""

    @dataclass(frozen=True)
    class InsecureGrpcChannel:
        """Use an insecure gRPC channel to connect to the host."""

        pass

    host: str  # IP address or DNS.
    transport: Union[
        InsecureGrpcChannel
    ]  # Currently only insecure gRPC channel is supported.

    def __eq__(self, other):
        if self.host != other.host:
            return False

        if isinstance(self.transport, ConnectionOptions.InsecureGrpcChannel):
            # Insecure grpc channel has no other parameters to check.
            if isinstance(other.transport, ConnectionOptions.InsecureGrpcChannel):
                return True

            # Otherwise, can't compare against a different transport type.
            raise ValueError(
                f"Transport '{self.transport}' cannot be compared to transport '{other.transport}'."
            )

        # Currently only InsecureGrpcChannel is supported.
        raise ValueError(f"Unsupported transport supplied: {self.transport}")


@dataclass
class FeatureViewReadResponse:
    _response: fos_service.FetchFeatureValuesResponse

    def __init__(self, response: fos_service.FetchFeatureValuesResponse):
        self._response = response

    def to_dict(self) -> Dict[str, Any]:
        return proto.Message.to_dict(self._response.key_values)

    def to_proto(self) -> fos_service.FetchFeatureValuesResponse:
        return self._response


@dataclass
class SearchNearestEntitiesResponse:
    _response: fos_service.SearchNearestEntitiesResponse

    def __init__(self, response: fos_service.SearchNearestEntitiesResponse):
        self._response = response

    def to_dict(self) -> Dict[str, Any]:
        return proto.Message.to_dict(self._response.nearest_neighbors)

    def to_proto(self) -> fos_service.SearchNearestEntitiesResponse:
        return self._response


class DistanceMeasureType(enum.Enum):
    """The distance measure used in nearest neighbor search."""

    DISTANCE_MEASURE_TYPE_UNSPECIFIED = 0
    # Euclidean (L_2) Distance.
    SQUARED_L2_DISTANCE = 1
    # Cosine Distance. Defined as 1 - cosine similarity.
    COSINE_DISTANCE = 2
    # Dot Product Distance. Defined as a negative of the dot product.
    DOT_PRODUCT_DISTANCE = 3


class AlgorithmConfig(abc.ABC):
    """Base class for configuration options for matching algorithm."""

    def as_dict(self) -> Dict:
        """Returns the configuration as a dictionary.

        Returns:
            Dict[str, Any]
        """
        pass


@dataclass
class TreeAhConfig(AlgorithmConfig):
    """Configuration options for using the tree-AH algorithm (Shallow tree + Asymmetric Hashing).

    Please refer to this paper for more details: https://arxiv.org/abs/1908.10396

    Args:
        leaf_node_embedding_count (int): Optional. Number of embeddings on each
          leaf node. The default value is 1000 if not set.
    """

    leaf_node_embedding_count: Optional[int] = None

    @override
    def as_dict(self) -> Dict:
        return {"leaf_node_embedding_count": self.leaf_node_embedding_count}


@dataclass
class BruteForceConfig(AlgorithmConfig):
    """Configuration options for using brute force search.

    It simply implements the standard linear search in the database for each
    query.
    """

    @override
    def as_dict(self) -> Dict[str, Any]:
        return {"bruteForceConfig": {}}


@dataclass
class IndexConfig:
    """Configuration options for the Vertex FeatureView for embedding."""

    embedding_column: str
    dimensions: int
    algorithm_config: AlgorithmConfig = field(default_factory=TreeAhConfig())
    filter_columns: Optional[List[str]] = None
    crowding_column: Optional[str] = None
    distance_measure_type: Optional[DistanceMeasureType] = None

    def as_dict(self) -> Dict[str, Any]:
        """Returns the configuration as a dictionary.

        Returns:
            Dict[str, Any]
        """
        config = {
            "embedding_column": self.embedding_column,
            "embedding_dimension": self.dimensions,
        }
        if self.distance_measure_type is not None:
            config["distance_measure_type"] = self.distance_measure_type.value
        if self.filter_columns is not None:
            config["filter_columns"] = self.filter_columns
        if self.crowding_column is not None:
            config["crowding_column"] = self.crowding_column

        if isinstance(self.algorithm_config, TreeAhConfig):
            config["tree_ah_config"] = self.algorithm_config.as_dict()
        else:
            config["brute_force_config"] = self.algorithm_config.as_dict()
        return config


@dataclass
class FeatureGroupBigQuerySource:
    """BigQuery source for the Feature Group."""

    # The URI for the BigQuery table/view.
    uri: str
    # The entity ID columns. If not specified, defaults to ['entity_id'].
    entity_id_columns: Optional[List[str]] = None
