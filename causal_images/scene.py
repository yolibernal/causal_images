from dataclasses import dataclass
from enum import Enum
from typing import Dict, Union
from uuid import uuid4 as uuid

import blenderproc as bproc
import numpy as np


class PrimitiveShape(str, Enum):
    CUBE = "CUBE"
    CYLINDER = "CYLINDER"
    CONE = "CONE"
    PLANE = "PLANE"
    SPHERE = "SPHERE"
    MONKEY = "MONKEY"


class RelativePosition(str, Enum):
    FRONT = "FRONT"
    BEHIND = "BEHIND"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    BOTTOM = "BOTTOM"
    TOP = "TOP"


@dataclass
class ObjectInfo:
    mesh: bproc.types.MeshObject
    shape: Union[PrimitiveShape, str]


class Scene:
    def __init__(self):
        self.objects: Dict[str, ObjectInfo] = {}

    def create_primitive(self, shape, material_name=None, scale=None):
        obj_id = uuid()
        obj = bproc.object.create_primitive(shape.value)
        if material_name is not None:
            materials = bproc.material.collect_all()
            material = bproc.filter.one_by_attr(materials, "name", material_name)
            obj.add_material(material)

        if scale is not None:
            obj.set_scale(scale)
        self.objects[obj_id] = ObjectInfo(mesh=obj, shape=shape)
        return obj_id

    def set_object_position(self, obj_id, position):
        current_position = self.objects[obj_id].mesh.get_location()
        new_position = current_position.copy()
        for i, pos in enumerate(position):
            if pos is not None:
                new_position[i] = pos
        self.objects[obj_id].mesh.set_location(new_position)
        return position

    def set_object_relative_position(
        self, target_obj_id, reference_obj_id, reference_pos, relative_pos
    ):
        reference_obj = self.objects[reference_obj_id].mesh
        target_obj = self.objects[target_obj_id].mesh

        if relative_pos == RelativePosition.FRONT:
            critical_dim = 0
            reference_loc_smaller = True

        if relative_pos == RelativePosition.BEHIND:
            critical_dim = 0
            reference_loc_smaller = False

        if relative_pos == RelativePosition.RIGHT:
            critical_dim = 1
            reference_loc_smaller = True

        if relative_pos == RelativePosition.LEFT:
            critical_dim = 1
            reference_loc_smaller = False

        if relative_pos == RelativePosition.TOP:
            critical_dim = 2
            reference_loc_smaller = True

        if relative_pos == RelativePosition.BOTTOM:
            critical_dim = 2
            reference_loc_smaller = False

        reference_critical_pos = (max if reference_loc_smaller else min)(
            [loc[critical_dim] for loc in reference_obj.get_bound_box()]
        )
        target_critical_pos = (min if reference_loc_smaller else max)(
            [loc[critical_dim] for loc in target_obj.get_bound_box()]
        )
        target_pos = np.array(
            [
                reference_critical_pos + target_obj.get_location()[0] - target_critical_pos
                if critical_dim == 0
                else reference_pos[0],
                reference_critical_pos + target_obj.get_location()[1] - target_critical_pos
                if critical_dim == 1
                else reference_pos[1],
                reference_critical_pos + target_obj.get_location()[2] - target_critical_pos
                if critical_dim == 2
                else reference_pos[2],
            ]
        )
        target_obj.set_location(target_pos)
        return target_pos

    def replace_object(self, obj_id, shape):
        position_obj_old = self.objects[obj_id].mesh.get_location()
        self.objects[obj_id].mesh.delete()

        obj = bproc.object.create_primitive(shape.value)
        obj.set_location(position_obj_old)
        self.objects[obj_id] = ObjectInfo(mesh=obj, shape=shape)
        return obj_id

    def cleanup(self):
        bproc.object.delete_multiple([obj.mesh for obj in self.objects.values()])
