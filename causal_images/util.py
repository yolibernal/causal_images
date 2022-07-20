import pandas as pd


def resolve_sample_object_shapes(x: pd.Series):
    row = x.copy()
    scene = row._scene

    for node_name, data in row.iteritems():
        if str(node_name).startswith("obj_"):
            obj_id = data
            row[node_name] = scene.objects[obj_id].shape
    return row


def resolve_object_shapes(df: pd.DataFrame):
    return df.apply(resolve_sample_object_shapes, axis=1)
