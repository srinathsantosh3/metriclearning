from numpy import *
import operator

def symm_matrix_power(A, p):
    ew, ev = linalg.eig(A)
    return dot(dot(ev,diag(ew**p)),ev.T)


def compute_RCA(data, chunks, dim=0):
    # subtract the mean
    data = data - data.mean(axis=0)
    N,d = data.shape

    cdata = []
    all_inds = []
    labels = list(set([label for label in chunks if label]))
    labels.sort()
    for i in labels:
        inds = where(chunks==i)
        cdata.append(data[inds] - data[inds].mean(axis=0))
        all_inds.append(inds)

    cdata = vstack(cdata)
    # Compute inner covariance matrix.
    inner_cov = cov(cdata, rowvar=0, bias=1)

    if dim>0 and dim<d:
        total_cov = cov(data, rowvar=0)
        ew, ev = linalg.eig(dot(linalg.inv(total_cov), inner_cov))
        sorted_pairs = sorted(enumerate(ew), key=operator.itemgetter(1))
        selected_ind = [ind for ind,val in sorted_pairs[:dim]]
        A = ev[:,selected_ind]
        inner_cov = dot(A.T, dot(inner_cov, A))
    else:
        A = eye(d)

    #RCA: whiten the data w.r.t the inner covariance matrix
    RCA = dot(A, symm_matrix_power(inner_cov, -0.5))
    return RCA

