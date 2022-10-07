from typing import Callable

import blenderproc as bproc
import numpy as np
import pandas as pd
from scmodels import SCM

from causal_images.scene import Scene


class SceneSCM:
    def __init__(
        self, functional_map_factory: Callable[[Scene, np.random.Generator], dict]
    ):
        self.functional_map_factory = functional_map_factory

    def sample(self, n, rng=np.random.default_rng()):
        for i in range(n):
            bproc.utility.reset_keyframes()

            # Create new scene
            scene = Scene()
            # Create new SCM for scene
            scm = SCM(self.functional_map_factory(scene, rng), seed=rng)
            df_sample = scm.sample(1)

            df_sample["_scene"] = [scene]
            yield df_sample

            scene.cleanup()

    def plot(self, rng=np.random.default_rng()):
        # Create new scene
        scene = Scene()
        # Create new SCM for scene
        scm = SCM(self.functional_map_factory(scene, rng))
        return scm.plot()
