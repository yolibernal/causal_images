from typing import Callable

import blenderproc as bproc
import numpy as np
import pandas as pd
from scmodels import SCM

from causal_images.scene import PrimitiveShape, Scene


class SceneInterventions:
    def __init__(
        self, functional_map_factory: Callable[[Scene, np.random.Generator], dict]
    ):
        self.functional_map_factory = functional_map_factory


class SceneManipulations:
    def __init__(
        self,
        functional_map_factory: Callable[[Scene, np.random.Generator], dict],
    ):
        self.functional_map_factory = functional_map_factory


class SceneSCM:
    def __init__(
        self,
        functional_map_factory: Callable[[Scene, np.random.Generator], dict],
        interventions: SceneInterventions = None,
        manipulations: SceneManipulations = None,
    ):
        self.functional_map_factory = functional_map_factory
        self.interventions = interventions
        self.manipulations = manipulations

    @classmethod
    def from_scm_outcomes(cls, scm_outcomes):
        """Create a SceneSCM from deterministic outcomes."""
        functional_map_factory = lambda scene, rng: {
            node_name: cls._create_deterministic_node_callable(
                scene, node_name, node_value
            )
            for node_name, node_value in scm_outcomes.items()
        }
        return cls(functional_map_factory)

    def sample_and_populate_scene(
        self,
        n,
        interventions: SceneInterventions = None,
        manipulations: SceneManipulations = None,
        rng=np.random.default_rng(),
    ):
        if interventions is None:
            interventions = self.interventions

        if manipulations is None:
            manipulations = self.manipulations

        for i in range(n):
            bproc.utility.reset_keyframes()

            # Create new scene
            scene = Scene()
            # Create new SCM for scene
            scm = SCM(self.functional_map_factory(scene, rng), seed=rng)
            if interventions is not None:
                scm.intervention(interventions.functional_map_factory(scene, rng))
            df_sample = scm.sample(1)

            df_sample["_scene"] = [scene]

            if manipulations is not None:
                scene_outcomes = df_sample.iloc[0]
                scene = scene_outcomes["_scene"]

                # TODO: save image and results before manipulations

                # Execute manipulations
                if manipulations is not None:
                    for (
                        node_name,
                        manipulation_callable,
                    ) in manipulations.functional_map_factory(scene, rng).items():
                        prev_node_value = (
                            scene_outcomes[node_name]
                            if node_name in df_sample
                            else None
                        )
                        new_node_value = manipulation_callable(
                            prev_node_value, scene_outcomes
                        )
                        scene_outcomes[node_name] = new_node_value

            df_objects = self._resolve_object_shapes(df_sample)

            yield df_objects

            scene.cleanup()

    def plot(self, rng=np.random.default_rng()):
        # Create new scene
        scene = Scene()
        # Create new SCM for scene
        scm = SCM(self.functional_map_factory(scene, rng))
        return scm.plot()

    def _create_deterministic_node_callable(self, scene, node_name: str, node_value):
        """Create a callable that returns a constant node value."""
        if node_name.startswith("obj_"):
            return (
                [],
                lambda: scene.create_primitive(PrimitiveShape(node_value)),
                None,
            )
        elif node_name.startswith("pos_"):
            return (
                [node_name.replace("pos_", "obj_")],
                lambda obj_parent: scene.set_object_position(obj_parent, node_value),
                None,
            )
        else:
            return ([], lambda: node_value, None)

    def _resolve_sample_object_shapes(self, x: pd.Series):
        """Resolve the object ID to the actual object shape name."""
        row = x.copy()
        scene = row._scene

        for node_name, data in row.iteritems():
            if str(node_name).startswith("obj_"):
                obj_id = data
                row[node_name] = scene.objects[obj_id].shape
        return row

    def _resolve_object_shapes(self, df: pd.DataFrame):
        return df.apply(self._resolve_sample_object_shapes, axis=1)
