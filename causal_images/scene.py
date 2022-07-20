from dataclasses import dataclass
from enum import Enum
from typing import Dict, Union
from uuid import uuid4 as uuid

import blenderproc as bproc


class PrimitiveShape(Enum):
    CUBE = "CUBE"
    CYLINDER = "CYLINDER"
    CONE = "CONE"
    PLANE = "PLANE"
    SPHERE = "SPHERE"
    MONKEY = "MONKEY"


class RelativePosition(Enum):
    TOP = "TOP"
    BOTTOM = "BOTTOM"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


@dataclass
class ObjectInfo:
    mesh: bproc.types.MeshObject
    shape: Union[PrimitiveShape, str]


class Scene:
    def __init__(self):
        self.objects: Dict[str, ObjectInfo] = {}

    def create_primitive(self, shape):
        obj_id = uuid()
        obj = bproc.object.create_primitive(shape.value)
        self.objects[obj_id] = ObjectInfo(mesh=obj, shape=shape)
        return obj_id

    def set_object_position(self, obj_id, position):
        self.objects[obj_id].mesh.set_location(position)
        return position

    def set_object_relative_position(
        self, target_obj_id, reference_obj_id, reference_position, relative_position
    ):
        reference_obj = self.objects[reference_obj_id].mesh
        target_obj = self.objects[target_obj_id].mesh
