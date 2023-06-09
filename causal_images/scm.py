from typing import Callable, Dict, Sequence

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
        fixed_noise_values: Dict[str, Sequence[float]] = None,
    ):
        self.functional_map_factory = functional_map_factory
        self.interventions = interventions
        self.manipulations = manipulations
        self.fixed_noise_values = fixed_noise_values

    @classmethod
    def from_scm_outcomes(cls, scm_outcomes):
        """Create a SceneSCM from deterministic outcomes."""
        functional_map_factory = lambda scene, rng: {
            node_name: cls._create_deterministic_node_callable(
                cls, scene, node_name, node_value
            )
            for node_name, node_value in scm_outcomes.items()
        }
        return cls(functional_map_factory)

    def sample_and_populate_scene(
        self,
        n,
        interventions: SceneInterventions = None,
        manipulations: SceneManipulations = None,
        fixed_noise_values: Dict[str, Sequence[float]] = None,
        rng=np.random.default_rng(),
    ):
        if interventions is None:
            interventions = self.interventions

        if manipulations is None:
            manipulations = self.manipulations

        if fixed_noise_values is None:
            fixed_noise_values = self.fixed_noise_values

        for i in range(n):
            bproc.utility.reset_keyframes()

            # Create new scene
            scene = Scene()
            # Create new SCM for scene
            scm = SCM(self.functional_map_factory(scene, rng), seed=rng)
            if interventions is not None:
                scm.intervention(interventions.functional_map_factory(scene, rng))
            df_outcomes = scm.sample(1, fixed_noise_values=fixed_noise_values)
            scm_outcomes = df_outcomes.iloc[0].to_dict()

            if manipulations is not None:
                # TODO: save image and results before manipulations

                # Execute manipulations
                if manipulations is not None:
                    for (
                        node_name,
                        manipulation_callable,
                    ) in manipulations.functional_map_factory(scene, rng).items():
                        prev_node_value = scm_outcomes.get(node_name)
                        new_node_value = manipulation_callable(
                            prev_node_value, scm_outcomes
                        )
                        scm_outcomes[node_name] = new_node_value

            scm_outcomes = self._resolve_scene_object_ids(scm_outcomes, scene=scene)

            yield scm_outcomes, scene

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

    def _resolve_scene_object_ids(self, scene_outcomes: Dict, scene):
        """Resolve the object ID to the actual object."""
        resolved_scene_outcomes = {}
        for node_name, node_value in scene_outcomes.items():
            if node_name.startswith("obj_") and "__noise__" not in node_name:
                obj_id = node_value
                resolved_scene_outcomes[node_name] = scene.objects[obj_id].shape
            else:
                resolved_scene_outcomes[node_name] = node_value
        return resolved_scene_outcomes
