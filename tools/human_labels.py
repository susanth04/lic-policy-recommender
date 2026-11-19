def validate_human_annotator_type(labels):
    if "annotator_type" not in labels:
        raise ValueError("Human labels must include annotator_type")
    annotator_type = labels["annotator_type"]
    if annotator_type.isna().any() or annotator_type.str.strip().eq("").any():
        raise ValueError("Every human label must specify annotator_type")
    if not annotator_type.eq("human").all():
        raise ValueError("Human label file annotator_type must be human for every row")
