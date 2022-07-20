from typing import Callable

import pandas as pd
from scmodels import SCM

from causal_images.scene import Scene


class SceneSCM:
    def __init__(self, functional_map_factory: Callable[[Scene], dict]):
        self.functional_map_factory = functional_map_factory

    def sample(self, n):
        samples = []
        for i in range(n):
            # Create new scene
            scene = Scene()
            # Create new SCM for scene
            scm = SCM(self.functional_map_factory(scene))
            df_sample = scm.sample(1)

            df_sample["_scene"] = [scene]
            samples.append(df_sample)

        return pd.concat(samples).reset_index(drop=True)
