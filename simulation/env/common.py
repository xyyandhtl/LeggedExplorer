import omni.replicator.core as rep

def add_semantic_label():
    ground_plane = rep.get.prims("/World/ground")
    with ground_plane:
    # Add a semantic label
        rep.modify.semantics([("class", "floor")])