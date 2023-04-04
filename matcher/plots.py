import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

__all__ = ['plot_comparison']

def make_rectangle(row):
    """Convert a row of a pandas DataFrame in YOLO format into a
    matplotlin.patches.Rectangle for plotting.
    
    Parameters
    ----------

    row: pandas.Series
        Row from ground truth/detections
    """
    return Rectangle((row['x']-row['w']/2.,1-row['y']-row['h']/2.0), row['w'], row['h']) 

def add_boxes(axis, boxes, parameters):
    """Add boxes to an axis.
    
    Parameters
    ----------
    
    axis : matplotlib.Axis
        Axis on which to plot boxes.
    boxes: iterable
        Collection of matplotlib.patches.Rectangle objects to plot.
    parameters: dict
        Dictionaryt of matplotlib.patches.Rectangle parameters."""
    pc = PatchCollection(boxes)
    pc.set(**parameters)
    axis.add_collection(pc)

def plot_comparison(ground_truth, detections, ground_indices=None,
                            detection_indices=None, title=None, fname=None,
                            detected_ground={'facecolor':'b', 'alpha':0.6, 'edgecolors':'none'},
                            missed_ground={'facecolor':'k', 'alpha':0.4, 'edgecolors':'none'},
                            true_detection={'facecolor':'y', 'alpha':0.6, 'edgecolors':'none'},
                            false_detection={'facecolor':'r', 'alpha':0.4, 'edgecolors':'none'}):
    """Plot comparison of ground truth and detections."""
    


    ground_indices = set(ground_indices or ground_truth.index)
    detection_indices = set(detection_indices or detections.index)

    fig = plt.gcf()
    ax = plt.gca()
    for (data, indices, good, bad) in zip([ground_truth, detections],
                                       [ground_indices, detection_indices],
                                       [detected_ground, true_detection],
                                       [missed_ground, false_detection]):
        boxes = [make_rectangle(r) for i, r in data.iterrows() if i in indices]
        add_boxes(ax, boxes, good)
        boxes = [make_rectangle(r) for i, r in data.iterrows() if i not in indices]
        add_boxes(ax, boxes, bad)

    if title:
        plt.title(title)
    if fname:
        plt.savefig(fname)
