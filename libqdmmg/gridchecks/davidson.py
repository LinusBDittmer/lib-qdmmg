import numpy
import scipy.interpolate
import gridcheck

def interpolate_matrix(original_matrix, target_size):
    # Generate coordinate grids for the original and target matrices
    x_original = numpy.linspace(0, 1, original_matrix.shape[1])
    y_original = numpy.linspace(0, 1, original_matrix.shape[0])

    x_target = numpy.linspace(0, 1, target_size)
    y_target = numpy.linspace(0, 1, target_size)

    # Create an interpolating function
    interpolator = scipy.interpolate.interp2d(x_original, y_original, original_matrix, kind='cubic')

    # Interpolate the values at the target points
    interpolated_matrix = interpolator(x_target, y_target)

    return interpolated_matrix

def get_projected_initial_guess(gc, rsize, ashape, nvec):
    use_proj = rsize > 10
    gc_red = gridcheck.Gridcheck(gc.sim, rsize, gc.extent, gc.shift, use_proj=use_proj)
    print(f"Generating Initial Guess by smaller matrix, size: {rsize}")
    eigvals, eigvecs = gc_red.kernel(nvec=rsize+1)
    eigvecs = eigvecs.T.real
    print("Finished Generating Initial guess")
    nr = int(round(numpy.sqrt(ashape[0])))
    print(f"NR: {nr}")
    nrows = ashape[0]
    guess_mat = numpy.zeros((nvec, nr, nr))
    for i, eigvec in enumerate(eigvecs):
        if i >= guess_mat.shape[0]: break
        guess_mat[i] = interpolate_matrix(numpy.reshape(eigvec, (rsize, rsize)), nr)
    guess_vecs = numpy.zeros((nrows, nvec))
    for i in range(nvec):
        guess_vecs[:,i] = numpy.reshape(guess_mat[i], nrows)
    return guess_vecs

def get_initial_guess(AdotV, ashape, nvec):
    nrows = ashape[0]
    #return numpy.random.random((nrows, nvec))
    guess = numpy.zeros((nrows, nvec))

    for i in range(nvec):
        g = numpy.zeros(nrows)
        g[i] = 1.0
        guess[:,i] = AdotV(g)
    
    randomisation = numpy.random.random(guess.shape)

    return guess + 10**-5 * randomisation

def davidson_solver(gc, ashape, neigen, tol=1E-6, itermax = 1000, use_proj=True):
    """Davidosn solver for eigenvalue problem

    Args :
        A (numpy matrix) : the matrix to diagonalize
        neigen (int)     : the number of eigenvalue requied
        tol (float)      : the rpecision required
        itermax (int)    : the maximum number of iteration
        jacobi (bool)    : do the jacobi correction
    Returns :
        eigenvalues (array) : lowest eigenvalues
        eigenvectors (numpy.array) : eigenvectors
    """
    AdotV = gc.hdotpsi
    n = ashape[0]
    k = 2*neigen            # number of initial guess vectors
    V = numpy.eye(n,k)         # set of k unit vectors as guess
    I = numpy.eye(n)           # identity matrix same dimen as A
    Adiag = numpy.zeros(ashape[0])
    for i in range(ashape[0]):
        g = numpy.zeros(ashape[0])
        g[i] = 1.0
        Adiag[i] = AdotV(g)[i]

    if use_proj:
        V = get_projected_initial_guess(gc,int(numpy.sqrt(n)/2),ashape,k)
    else:
        V = get_initial_guess(AdotV, ashape, k)

    print('\n'+'='*20)
    print("= Davidson Solver ")
    print('='*20)

    #invA = numpy.linalg.inv(A)
    #inv_approx_0 = 2*I - A
    #invA2 = numpy.dot(invA,invA)
    #invA3 = numpy.dot(invA2,invA)

    norm = numpy.zeros(2*neigen)

    # Begin block Davidson routine
    print("iter size norm (%e)" %tol)
    for i in range(itermax):

        # Subspace Collapse if the size of V is bigger than 20*nvec
        #if V.shape[1] >= 20*neigen:
        #    V = V[:,V.shape[1]-k-1:]
        #    print("Subspace truncated.")

        # QR of V t oorthonormalize the V matrix
        # this uses GrahmShmidtd in the back
        V,R = numpy.linalg.qr(V)

        # form the projected matrix
        T = numpy.dot(V.conj().T, AdotV(V))
        #print(numpy.diag(T))

        # Diagonalize the projected matrix
        theta,s = numpy.linalg.eig(T)

        # organize the eigenumpyairs
        index = numpy.argsort(theta.real)
        theta  = theta[index]
        s = s[:,index]

        # Ritz eigenvector
        q = numpy.dot(V,s)

        # compute the residual append append it to the
        # set of eigenvectors
        ind0 = numpy.where(theta>0,theta,numpy.inf).argmin()
        for jj in range(2*neigen):


            j = ind0 + jj - int(0.25*2*neigen)

            # residue vetor
            res = AdotV(q[:,j]) - numpy.dot(theta[j]*I, q[:,j])
            norm[jj] = numpy.linalg.norm(res)

            # correction vector
            delta = res / (theta[j]-Adiag+1E-16)

            delta /= numpy.linalg.norm(delta)

            # expand the basis
            V = numpy.hstack((V,delta.reshape(-1,1)))

        # comute the norm to se if eigenvalue converge
        print(" %03d %03d %e" %(i,V.shape[1],numpy.max(norm)))
        if numpy.all(norm < tol):
            print("= Davidson has converged")
            break

    return theta[ind0:ind0+neigen], q[:,ind0:ind0+neigen]

if __name__ == '__main__':
    mat = numpy.random.random((25, 25))
    mat += mat.T
    mat += numpy.diag(numpy.diag(mat)) * 25

    eigvals, eigvecs = numpy.linalg.eigh(mat)
    AdotV = lambda V: numpy.dot(mat, V)

    dav_eigvals, dav_eigvecs = davidson_solver(AdotV, (25, 25), 10)
    print(eigvals)
    print(dav_eigvals)
    print(eigvals[:10] - dav_eigvals)
