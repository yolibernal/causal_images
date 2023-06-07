from typing import Callable

import blenderproc as bproc
import numpy as np
import pandas as pd
from scmodels import SCM

from causal_images.scene import Scene
from causal_images.util import resolve_object_shapes


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

            df_objects = resolve_object_shapes(df_sample)

            yield df_objects

            scene.cleanup()

    def plot(self, rng=np.random.default_rng()):
        # Create new scene
        scene = Scene()
        # Create new SCM for scene
        scm = SCM(self.functional_map_factory(scene, rng))
        return scm.plot()
