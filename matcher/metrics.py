"""Metrics for similarity scoring."""

def intersection_over_union(box1, box2):
    """ Implementation of intersection over union for boces.
    
    We assume a format [x_min, x_max, box2_min, box2_max] for
    each box.

    Parameters
    ----------

    box1: collection
        The extent of the first box.
    box2: collection
        The extent of the second box. 

    Returns
    -------

    float:
        The intersection over union value for the boxes.

    """
    if (box1[0]>=box2[1]) or (box1[1]<=box2[0]) or (box1[2]>=box2[3]) or (box1[3]<=box2[2]):
        return 0
    else:
        intersect_width = min(box1[1], box2[1]) - max(box1[0], box2[0])
        intersect_height = min(box1[3], box2[3]) - max(box1[2], box2[2])

    # area of intersection rectangle
    intersection_area = intersect_width*intersect_height

    # area of union
    box1_area = (box1[1] - box1[0]) * (box1[3] - box1[2])
    box2_area = (box2[1] - box2[0]) * (box2[3] - box2[2])

    iou = intersection_area / (box1_area + box2_area - intersection_area) 

    return iou