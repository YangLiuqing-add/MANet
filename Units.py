from tensorflow.keras.layers import *
import numpy as np
from matplotlib import pylab as plt
from keras import backend as K
from tensorflow.keras.models import Model
import tensorflow as tf
from matplotlib.colors import ListedColormap

def patch3d(A, l1=4, l2=4, l3=4, s1=2, s2=2, s3=2):
    """
    Optimized patch3d function using stride tricks to improve efficiency
    """
    # Pad the array if necessary to ensure dimensions are divisible by patch sizes
    pad1 = (l1 - A.shape[0] % s1) % s1
    pad2 = (l2 - A.shape[1] % s2) % s2
    pad3 = (l3 - A.shape[2] % s3) % s3
    A_padded = np.pad(A, ((0, pad1), (0, pad2), (0, pad3)), mode='constant')

    # Get new dimensions
    n1, n2, n3 = A_padded.shape
    n1_patches = (n1 - l1) // s1 + 1
    n2_patches = (n2 - l2) // s2 + 1
    n3_patches = (n3 - l3) // s3 + 1

    # Generate the sliding window view
    shape = (n1_patches, n2_patches, n3_patches, l1, l2, l3)
    strides = (s1 * A_padded.strides[0], s2 * A_padded.strides[1], s3 * A_padded.strides[2]) + A_padded.strides
    patches = np.lib.stride_tricks.as_strided(A_padded, shape=shape, strides=strides)

    # Reshape patches to (num_patches, patch_size) format
    patches = patches.reshape(-1, l1 * l2 * l3)
    return patches


def patch3d_inv(X, n1, n2, n3, l1=4, l2=4, l3=4, s1=2, s2=2, s3=2):
    """
    Reconstruct 3D data from 1D patches with optimized inverse patching.

    INPUT
    X: Patches in (num_patches, patch_size) format
    n1, n2, n3: Original 3D data dimensions
    l1, l2, l3: Patch sizes along each dimension
    s1, s2, s3: Shifts between patches along each dimension

    OUTPUT
    A: Reconstructed 3D data
    """

    # Initialize the padded 3D array and mask
    pad1 = (l1 - n1 % s1) % s1
    pad2 = (l2 - n2 % s2) % s2
    pad3 = (l3 - n3 % s3) % s3
    A = np.zeros((n1 + pad1, n2 + pad2, n3 + pad3))
    mask = np.zeros_like(A)

    # Reshape 1D patches to the 3D patch size for easier handling
    X = X.reshape(-1, l1, l2, l3)

    # Calculate the number of patches along each dimension
    n1_patches = (A.shape[0] - l1) // s1 + 1
    n2_patches = (A.shape[1] - l2) // s2 + 1
    n3_patches = (A.shape[2] - l3) // s3 + 1

    # Iterate over each patch and place it into the appropriate location
    idx = 0
    for i in range(n1_patches):
        for j in range(n2_patches):
            for k in range(n3_patches):
                # Place the current patch in the reconstruction array and update mask
                A[i * s1:i * s1 + l1, j * s2:j * s2 + l2, k * s3:k * s3 + l3] += X[idx]
                mask[i * s1:i * s1 + l1, j * s2:j * s2 + l2, k * s3:k * s3 + l3] += 1
                idx += 1

    # Avoid division by zero by replacing zeros with ones in the mask before division
    mask[mask == 0] = 1
    A /= mask

    # Trim padding to match the original dimensions
    return A[:n1, :n2, :n3]


from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def cseis():
    '''
    cseis: seismic colormap

    By Yangkang Chen
    June, 2022

    EXAMPLE
    from pyseistr import cseis
    import numpy as np
    from matplotlib import pyplot as plt
    plt.imshow(np.random.randn(100,100),cmap=cseis())
    plt.show()
    '''
    seis = np.concatenate(
        (np.concatenate((0.5 * np.ones([1, 40]), np.expand_dims(np.linspace(0.5, 1, 88), axis=1).transpose(),
                         np.expand_dims(np.linspace(1, 0, 88), axis=1).transpose(), np.zeros([1, 40])),
                        axis=1).transpose(),
         np.concatenate((0.25 * np.ones([1, 40]), np.expand_dims(np.linspace(0.25, 1, 88), axis=1).transpose(),
                         np.expand_dims(np.linspace(1, 0, 88), axis=1).transpose(), np.zeros([1, 40])),
                        axis=1).transpose(),
         np.concatenate((np.zeros([1, 40]), np.expand_dims(np.linspace(0, 1, 88), axis=1).transpose(),
                         np.expand_dims(np.linspace(1, 0, 88), axis=1).transpose(), np.zeros([1, 40])),
                        axis=1).transpose()), axis=1)

    return ListedColormap(seis)


def plot3d(d3d, frames=None, z=None, x=None, y=None, dz=0.01, dx=0.01, dy=0.01, nlevel=100, figsize=(8, 6),
           ifnewfig=True, figname=None, showf=True, close=True, **kwargs):
    '''
    plot3d: plot beautiful 3D slices

    INPUT
    d3d: input 3D data (z in first-axis, x in second-axis, y in third-axis)
    frames: plotting slices on three sides (default: [nz/2,nx/2,ny/2])
    z,x,y: axis vectors  (default: 0.01*[np.arange(nz),np.arange(nx),np.arange(ny)])
    figname: figure name to be saved (default: None)
    showf: if show the figure (default: True)
    close: if not show a figure, if close the figure (default: True)
    kwargs: other specs for plotting
    dz,dx,dy: interval (default: 0.01)

    By Yangkang Chen
    June, 18, 2023

    EXAMPLE 1
    import numpy as np
    d3d=np.random.rand(100,100,100);
    from pyseistr import plot3d
    plot3d(d3d);

    EXAMPLE 2
    import scipy
    data=scipy.io.loadmat('/Users/chenyk/chenyk/matlibcyk/test/hyper3d.mat')['cmp']
    from pyseistr import plot3d
    plot3d(data);

    EXAMPLE 3
    import numpy as np
    import matplotlib.pyplot as plt
    from pyseistr import plot3d

    nz=81
    nx=81
    ny=81
    dz=20
    dx=20
    dy=20
    nt=1501
    dt=0.001

    v=np.arange(nz)*20*1.2+1500;
    vel=np.zeros([nz,nx,ny]);
    for ii in range(nx):
        for jj in range(ny):
            vel[:,ii,jj]=v;

    plot3d(vel,figsize=(16,10),cmap=plt.cm.jet,z=np.arange(nz)*dz,x=np.arange(nx)*dx,y=np.arange(ny)*dy,barlabel='Velocity (m/s)',showf=False,close=False)
    plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
    plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
    plt.gca().set_zlabel("Z (m)",fontsize='large', fontweight='normal')
    plt.title('3D velocity model')
    plt.savefig(fname='vel3d.png',format='png',dpi=300)
    plt.show()

    '''

    [nz, nx, ny] = d3d.shape;

    if frames is None:
        frames = [int(nz / 2), int(nx / 2), int(ny / 2)]

    if z is None:
        z = np.arange(nz) * dz

    if x is None:
        x = np.arange(nx) * dx

    if y is None:
        y = np.arange(ny) * dy

    X, Y, Z = np.meshgrid(x, y, z)

    d3d = d3d.transpose([1, 2, 0])

    kw = {
        'vmin': d3d.min(),
        'vmax': d3d.max(),
        'levels': np.linspace(d3d.min(), d3d.max(), nlevel),
        'cmap': cseis()
    }

    kw.update(kwargs)

    if 'alpha' not in kw.keys():
        kw['alpha'] = 1.0

    if ifnewfig == False:
        ax = plt.gca()
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, aspect='auto', projection='3d')
        plt.jet()

    # Plot contour surfaces
    _ = ax.contourf(
        X[:, :, -1], Y[:, :, -1], d3d[:, :, frames[0]].transpose(),  # x,y,z
        zdir='z', offset=0, **kw
    )

    _ = ax.contourf(
        X[0, :, :], d3d[:, frames[2], :], Z[0, :, :],
        zdir='y', offset=Y.min(), **kw
    )

    C = ax.contourf(
        d3d[frames[1], :, :], Y[:, -1, :], Z[:, -1, :],
        zdir='x', offset=X.max(), **kw
    )

    plt.gca().set_xlabel("X", fontsize='large', fontweight='normal')
    plt.gca().set_ylabel("Y", fontsize='large', fontweight='normal')
    plt.gca().set_zlabel("Z", fontsize='large', fontweight='normal')

    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
    plt.gca().invert_zaxis()

    # Colorbar
    if 'barlabel' in kw.keys():
        cbar = fig.colorbar(C, ax=ax, orientation='horizontal', fraction=0.02, pad=0.1, format="%.2f",
                            label=kw['barlabel'])
        cbar.ax.locator_params(nbins=5)
        kwargs.__delitem__('barlabel')

    if figname is not None:
        if 'cmap' in kwargs.keys():
            kwargs.__delitem__('cmap')
        plt.savefig(figname, **kwargs)

    if showf:
        plt.show()
    else:
        if close:
            plt.close()  # or plt.clear() ?


def framebox(x1, x2, y1, y2, c=None, lw=None):
    '''
    framebox: for drawing a frame box

    By Yangkang Chen
    June, 2022

    INPUT
    x1,x2,y1,y2: intuitive

    EXAMPLE I
    from pyseistr.plot import framebox
    from pyseistr.synthetics import gensyn
    from matplotlib import pyplot as plt
    d=gensyn();
    plt.imshow(d);
    framebox(200,400,200,300);
    plt.show()

    EXAMPLE II
    from pyseistr.plot import framebox
    from pyseistr.synthetics import gensyn
    from matplotlib import pyplot as plt
    d=gensyn();
    plt.imshow(d);
    framebox(200,400,200,300,c='g',lw=4);
    plt.show()

    '''

    if c is None:
        c = 'r';
    if lw is None:
        lw = 2;

    plt.plot([x1, x2], [y1, y1], linestyle='-', color=c, linewidth=lw);
    plt.plot([x1, x2], [y2, y2], linestyle='-', color=c, linewidth=lw);
    plt.plot([x1, x1], [y1, y2], linestyle='-', color=c, linewidth=lw);
    plt.plot([x2, x2], [y1, y2], linestyle='-', color=c, linewidth=lw);

    return


def huber_loss_mean(y_true, y_pred, clip_delta=1.3):
    error = y_true - y_pred
    abs_error = tf.keras.backend.abs(error)

    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss = clip_delta * (abs_error - 0.5 * clip_delta)

    return tf.keras.backend.mean(tf.where(abs_error < clip_delta, squared_loss, linear_loss))



def lr_schedule(epoch):
    initial_lr = 1e-3

    if epoch <= 20:
        lr = initial_lr
    elif epoch <= 40:
        lr = initial_lr / 2
    elif epoch <= 60:
        lr = 3e-4
    elif epoch <= 80:
        lr = initial_lr / 10
    else:
        lr = initial_lr / 20
   # print('Learning rate: ', lr)
    return lr


def FEB(inpt_img,D1, dropout):
    C1 = Dense(D1)(inpt_img)
    C1 = LeakyReLU(0.01)(C1)
    C1 = BatchNormalization()(C1)
    C1 = Dropout(dropout)(C1)
    return C1

def MAB(inputs, m, r,kernel,dropout):

    d = int(kernel * r)
    x1 = FEB(inputs,kernel,dropout)
    x11 = Reshape((-1, 1))(x1)
    _x1 = GlobalAveragePooling1D()(x11)

    x2 = FEB(inputs,kernel,dropout)
    x22 = Reshape((-1, 1))(x2)
    _x2 = GlobalAveragePooling1D()(x22)

    U = concatenate([_x1,_x2], axis=-1)

    z = Dense(d)(U)
    z = LeakyReLU(0.01)(z)
    z = Dense(kernel*m)(z)

    z = Reshape([kernel, m])(z)
    scale = Softmax()(z)

    x = Lambda(lambda x: tf.stack(x, axis=-1))([x1, x2])
    r = multiply([scale, x])
    r = Lambda(lambda x: K.sum(x, axis=-1))(r)
    return r


def MANet(input_shape,m,r,kernel,dropout):

    input_img = Input(shape=input_shape)

    MA1 = MAB(input_img, m=m, r=r, kernel=kernel, dropout=dropout)

    MA2 = MAB(MA1, m=m, r=r,  kernel=kernel//2, dropout=dropout)

    MA3 = MAB(MA2, m=m, r=r, kernel=kernel//4, dropout=dropout)

    DB5 = FEB(MA3, kernel//8, dropout=dropout)
    DB5 = FEB(DB5, kernel//8, dropout=dropout)

    MA7 = MAB(DB5, m=m, r=r,  kernel=kernel//4, dropout=dropout)
    Con2 = concatenate([MA7, MA3])

    MA8 = MAB(Con2, m=m, r=r,  kernel=kernel//2, dropout=dropout)
    Con3 = concatenate([MA8, MA2])

    MA9 = MAB(Con3, m=m, r=r, kernel=kernel, dropout=dropout)
    Con4 = concatenate([MA9, MA1])

    decoded = Dense(input_shape[0], activation='linear')(Con4)

    model = Model(input_img, decoded)

    model.compile(optimizer='adam', loss=huber_loss_mean)
    model.summary()

    return model