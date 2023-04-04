"""Functions to calculate similarity matrices and assign matched pairs."""

import scipy
import numpy as np

from .metrics import intersection_over_union

__all__ = ['calculate_box_match', 'summarize']

def sparse_match(similarity):
    """Generate a maximum bipartite matching in appropriate format from
    a sparse similarity matrix.
    
    This maximises the number of possible matches, but with no additional criteria.

    Parameters
    ----------

    similarity : 
        input similarity matrix, in a format compatible with `scipy.sparse.csr_matrix`.

    Returns
    -------

    rows: list
        List of the row indices which are matched
    cols
        List of the column indices which are matched.
    """

    similarity = scipy.sparse.csr_matrix(similarity, dtype=np.int32)

    cols = scipy.sparse.csgraph.maximum_bipartite_matching(similarity, perm_type='column')

    rows = [_ for _, c in enumerate(cols) if c>=0 and similarity[_, c]>0]
    cols = [c for _, c in enumerate(cols) if c>=0 and similarity[_, c]>0]

    return rows, cols

def dense_match(similarity):
    """Generate a matching in appropriate format from
    a similarity matrix.
    
    This maximises the sum of the values of the matched elements of the similarity matrix.

    Parameters
    ----------

    similarity : array-like
        Input similarity matrix, in a format compatible with `scipy.sparse.csr_matrix`.

    Returns
    -------

    rows: list
        List of the row indices which are matched
    cols
        List of the column indices which are matched.
    """

    tmp = np.empty(similarity.shape, dtype=np.float32)
    similarity.todense(out=tmp)
    rows, cols = scipy.optimize.linear_sum_assignment(tmp, maximize=True)

    return rows, cols

def assignment(similarity, sparse=False, vector_type='rows'):
    """Perform the assignment problem based on an input similarity matrix.
    
    If sparse is True, use a sparse algorithm maximizing the number of matches, otherwise
    use "linear sum assignment" maximizing the sum of the matched values.

    Parameters
    ----------

    similarity : arraylike
        Input similarity matrix, in a format compatible with `scipy.sparse.csr_matrix`.
    sparse : bool, optional
        If true, use sparse maximum bipartite matching, otherwise maximize linear sum.
    vector_type: 'rows', 'columns' or 'matches', optional
        If 'rows', return similarities for all the rows, if 'columns' return similarities
        for all the columns, if 'matches' return similarities only for the matches.

    Returns
    -------

    similarity_vector: np.ndarray
        Array of similarity similarity scores for the requested vector_type.
    output_rows
        List of matched row indices
    output_cols
        List of matched column indices
    """

    if sparse:
        rows, cols = sparse_match(similarity)
    else:
        rows, cols = dense_match(similarity)

    output_rows = [i for i,j in zip(rows, cols) if similarity[i,j] > 0.0]
    output_cols = [j for i,j in zip(rows, cols) if similarity[i,j] > 0.0]

    if vector_type == 'rows':
        similarity_vector = np.zeros(similarity.shape[0], dtype=similarity.dtype)
        if len(rows):
            similarity_vector[rows] = similarity.tocsr()[rows, cols]
    elif vector_type == 'cols':
        similarity_vector = np.zeros(similarity.shape[1], dtype=similarity.dtype)
        if len(cols):
            similarity_vector[cols] = similarity.tocsr()[rows, cols]
    elif vector_type == 'matches':
        similarity_vector = np.empty(len(output_rows), dtype=similarity.dtype)
        similarity_vector[:] = similarity[output_rows, output_cols]

    return similarity_vector, output_rows, output_cols


def calculate_box_match(frame1, frame2, metric=intersection_over_union,
                   sparse=False, threshold=0.5, vector_type='rows'):
    """Calculate matchs for input pandas DataFrames in YOLO-style format.
    
    The data frames should have columns 'x', 'y', 'w' and 'h', specifying the
    coordinates of the box centre as well as its width and height.
    
    Parameters
    ----------

    frame1: pandas.DataFrame
        Input boxes for first set.
    frame2: pandas.DataFrame
        Input boxes for second set.
    metric: func, optional
        Function specifying similarity metric for boxes. Defaults to IOU.
    sparse: bool, optional
        If true, use a sparse bipartite matching algorithm.
    threshold: float
        Acceptable similarity if using sparse bipartite matching.
    vector_type: 'rows', 'columns' or 'matches', optional
        If 'rows', return similarities for all the rows, if 'columns' return similarities
        for all the columns, if 'matches' return similarities only for the matches.

    Returns
    -------

    similarity_vector: np.ndarray
        Array of similarity similarity scores for the requested vector_type.
    output_rows
        List of matched row indices
    output_cols
        List of matched column indices

    """

    x_pts = np.zeros((frame1.w.shape[0], 2))
    w_x = frame1.w.values.copy()
    h_x = frame1.h.values.copy()
    x_pts[:, 0] = frame1.x.values.copy()
    x_pts[:, 1] = frame1.y.values.copy()

    y_pts = np.zeros((frame2.w.shape[0], 2))
    w_y = frame2.w.values.copy()
    h_y = frame2.h.values.copy()
    y_pts[:, 0] = frame2.x.values.copy()
    y_pts[:, 1] = frame2.y.values.copy()

    x_tree = scipy.spatial.cKDTree(x_pts)
    y_tree = scipy.spatial.cKDTree(y_pts)

    ## Build the similarity matrix
    if sparse:
        dtype = np.int32
    else:
        dtype = np.float32
    similarity = scipy.sparse.dok_array((len(w_x), len(w_y)), dtype=dtype)

    for i in range(len(w_x)):
        x_i = x_pts[i]
        w = w_x[i]
        h = h_x[i]

        test = y_tree.query_ball_point(x_i, max(w, h))
        for j in test:
            val = metric([x_i[0]-w/2, x_i[0]+w/2,
                          x_i[1]-h/2, x_i[1]+h/2],
                         [y_pts[j, 0]-w_y[j]/2, y_pts[j, 0]+w_y[j]/2,
                          y_pts[j, 1]-h_y[j]/2, y_pts[j, 1]+h_y[j]/2])
            if sparse and val >= threshold:
                similarity[i, j] = 1
            else:
                similarity[i, j] = val

    for i in range(len(w_y)):
        y_i = y_pts[i, :]
        w = w_y[i]
        h = h_y[i]

        test = x_tree.query_ball_point(y_i, max(w, h))
        for j in test:
            val = metric([x_pts[j, 0]-w_x[j]/2, x_pts[j, 0]+w_x[j]/2,
                          x_pts[j, 1]-h_x[j]/2, x_pts[j, 1]+h_x[j]/2],
                         [y_i[0]-w/2, y_i[0]+w/2,
                          y_i[1]-h/2, y_i[1]+h/2])
            if sparse and val >= threshold:
                similarity[j, i] = 1
            else:
                similarity[i, j] = val

    return assignment(similarity, sparse=sparse, vector_type=vector_type)

def summarize(similarity_vector, ground_truth, detections, threshold=0.5):
    """Summarize basic statistics for a given similarity vector.
    
    Parameters
    ----------
    similarity_vector: numpy.ndarray
        Array of similarity scores between ground truth and detections.
    ground_truth: pandas.DataFrame
        Data for the ground truth objects.
    detections:
        Data for the object detections.

    """       

    TP = np.sum(similarity_vector >= threshold)
    FP = detections.shape[0]-TP
    FN = ground_truth.shape[0]-TP

    P = TP/(TP+FP)
    R = TP/(TP+FN)

    print(f"\tPrecision (IOU={threshold}) :\t: {P:01.4f}")
    print(f"\tRecall (IOU={threshold}) :\t: {R:01.4f}")
    print(f"\tF1 (IOU={threshold}) :\t: {2*P*R/(P+R):01.4f}")
    print(f"\tTrue Positives: {TP}")
    print(f"\tFalse Positives: {FP}")
    print(f"\tFalse Negatives: {FN}")