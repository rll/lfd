import h5py
import importlib, __builtin__

def fname_to_obj(fname):
    f = None
    try:
        f = h5py.File(fname, 'r')
        res = group_or_dataset_to_obj(f)
    finally:
        if f is not None:
            f.close()
    return res

def add_obj_to_group(group, k, v):
    if v is None:
        group[k] = 'None'
    elif (type(v) == dict or type(v) == list or type(v) == tuple) and len(v) == 0:
        group[k] = 'empty'
        group[k].attrs.create('value_type', type(v).__name__)
    elif type(v) == dict or type(v) == list or type(v) == tuple or hasattr(v, '__dict__'):
        vgroup = group.create_group(k)
        if type(v) == dict:
            d = v
        elif type(v) == list or type(v) == tuple:
            vgroup.attrs.create('value_type', type(v).__name__)
            d = dict((str(i),vi) for (i,vi) in enumerate(v))
        elif hasattr(v, '__dict__'):
            vgroup.attrs.create('value_type', type(v).__name__)
            vgroup.attrs.create('value_type_module', type(v).__module__)
            d = vars(v)
        for (vk,vv) in d.iteritems():
            add_obj_to_group(vgroup, vk, vv)
    else:
        group[k] = v
    return group

def group_or_dataset_to_obj(group_or_dataset):
    if 'value_type' in group_or_dataset.attrs.keys():
        if 'value_type_module' in group_or_dataset.attrs.keys():
            module = importlib.import_module(group_or_dataset.attrs['value_type_module'])
        else:
            module = __builtin__
        v_type = getattr(module, group_or_dataset.attrs['value_type'])
    else:
        v_type = None
    if isinstance(group_or_dataset, h5py.Group):
        group = group_or_dataset
        v_dict = {}
        for (gk,gv) in group.iteritems():
            v_dict[gk] = group_or_dataset_to_obj(gv)
        if v_type is not None:
            if v_type == tuple or v_type == list:
                v = v_type(zip(*sorted(v_dict.items(), key=lambda (vk, vv): int(vk)))[1])
            elif hasattr(v_type, '__dict__'):
                v = v_type.__new__(v_type)
                v.__dict__ = v_dict
        else:
            v = v_dict
    else:
        dataset = group_or_dataset
        if dataset[()] == 'None':
            v = None
        elif dataset[()] == 'empty':
            v = []
            if v_type is not None:
                v = v_type(v)
        else:
            v = dataset[()]
    return v
