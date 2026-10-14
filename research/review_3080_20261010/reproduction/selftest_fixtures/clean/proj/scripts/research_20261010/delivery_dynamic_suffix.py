def build_suffix(cargo, obs, mode, node):
    return engine.plan(vid, build_suffix(cargo, obs, 'reorder', node) + [0])
