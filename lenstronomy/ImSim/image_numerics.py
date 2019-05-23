import numpy as np
import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util
from lenstronomy.ImSim.Numerics.point_source_rendering import PointSourceRendering


class ImageNumerics(PointSourceRendering):
    """
    class to compute all the numerical task corresponding to an image, such as convolution and re-binning, masking
    """
    def __init__(self, pixel_grid, psf, subgrid_res=1, psf_subgrid=False, fix_psf_error_map=False, index_mask=None, mask=None,
                 point_source_subgrid=3, subsampling_size=5, conv_type='fft', subgrid_conv_type='fft'):
        """

        optional keywords for masking purposes:
        'mask': 2d numpy array consists of zeros or ones. Pixels with mask[i,j]==1 are evaluated in the likelihood process.
        Pixel with mask[i,j]==0 are ignored
        'idex_mask': ignores any computation on the pixels with idex_mask[i,j]==0. Will not be stored in memory during linear inversions.
        Difference to 'mask': Pixels with mask[i,j]==0 will be ray-traced, evaluated and their flux value being
        convolved to enable an impact on other pixels.

        'point_source_subgrid': sub-sampling resolution of the point source placing
        'subsampling_size': sub-sampling kernel size (in units of the pixel size), default is the size of the PSF
            for computational speed, smaller subsampling psf sizes are faster but less accurate.

        'subgrid_res': int, subsampling resolution per data pixel in the ray tracing and evaluation of the extended surface brightness
        'psf_subgrid': bool, if True performs the PSF convolution on the higher resolution subgrid surface brightness,
            otherwise on the data frame.
        'point_source_subgrid': int, subsampling of point source PSF
        'conv_type': 'fft' or 'grid', using either scipy.convolve2d or scipy.signal.fftconvolve for
         convolution of kernel
         'subgrid_conv_type': 'fft' or 'grid', using either scipy.convolve2d or scipy.signal.fftconvolve for subgrid
         convolution of kernel

        :param pixel_grid: instance of the lenstronomy PixelGrid() or inheritances of it, such as Data()
        """

        deltaPix = pixel_grid.pixel_width
        psf.set_pixel_size(deltaPix)
        self._PixelGrid = pixel_grid
        self._PSF = psf
        self._nx, self._ny = self._PixelGrid.num_pixel_axes
        self._subgrid_res = subgrid_res
        self._psf_subgrid = psf_subgrid
        if index_mask is not None:
            self._idex_mask_2d = index_mask
            if not np.shape(self._idex_mask_2d) == (self._nx, self._ny):
                raise ValueError("'idex_mask' must be the same shape as 'image_data'! Shape of mask %s, Shape of data %s"
                                 % (np.shape(self._idex_mask_2d), (self._nx, self._ny)))
            self._idex_mask_bool = True
        else:
            self._idex_mask_2d = np.ones((self._nx, self._ny))
            self._idex_mask_bool = False
        self._idex_mask = util.image2array(self._idex_mask_2d)

        if mask is not None:
            self._mask = mask
            if not np.shape(self._mask) == np.shape(self._PixelGrid.data):
                raise ValueError("'mask' must be the same shape as 'image_data'! Shape of mask %s, Shape of data %s"
                                 % (np.shape(self._mask), (self._nx, self._ny)))
        else:
            self._mask = np.ones((self._nx, self._ny))
        self._mask[self._idex_mask_2d == 0] = 0
        self._mask[self._idex_mask_2d == 0] = 0
        self._idex_mask_sub = self._subgrid_index(self._idex_mask, self._subgrid_res, self._nx, self._ny)

        """
        
        self._point_source_subgrid = point_source_subgrid
        if self._point_source_subgrid % 2 == 0:
            if self._point_source_subgrid % 2 == 0 and psf._point_source_subsampling_factor != self._point_source_subgrid:
                raise ValueError("point_source_subgird needs to be an odd integer. The value %s is not supported." % self._point_source_subgrid)
        """
        self._subsampling_size = subsampling_size
        self._conv_type = conv_type
        self._subgrid_conv_type = subgrid_conv_type
        super(ImageNumerics, self).__init__(pixel_grid=pixel_grid, supersampling_factor=point_source_subgrid,
                                                            psf=psf, fix_psf_error_map=fix_psf_error_map)

    @property
    def coordinates_evaluate(self):
        """

        :return: RA, DEC coordinates, 1d numpy array, of all coordinates used in performing the image computation
        (e.g. adaptive convolution)
        """
        self._check_subgrid()
        return self._ra_subgrid, self._dec_subgrid

    def _check_subgrid(self):
        if not hasattr(self, '_ra_subgrid') or not hasattr(self, '_dec_subgrid'):
            ra_grid, dec_grid = self._PixelGrid.pixel_coordinates
            ra_grid = util.image2array(ra_grid)
            dec_grid = util.image2array(dec_grid)
            x_grid_sub, y_grid_sub = util.make_subgrid(ra_grid, dec_grid, self._subgrid_res)
            self._ra_subgrid = x_grid_sub[self._idex_mask_sub == 1]
            self._dec_subgrid = y_grid_sub[self._idex_mask_sub == 1]

    @property
    def mask(self):
        return self._mask

    def _subgrid_index(self, idex_mask, subgrid_res, nx, ny):
        """

        :param idex_mask: 1d array of mask of data
        :param subgrid_res: subgrid resolution
        :return: 1d array of equivalent mask in subgrid resolution
        """
        idex_sub = np.repeat(idex_mask, subgrid_res, axis=0)
        idex_sub = util.array2image(idex_sub, nx=nx, ny=ny*subgrid_res)
        idex_sub = np.repeat(idex_sub, subgrid_res, axis=0)
        idex_sub = util.image2array(idex_sub)
        return idex_sub

    def _array2image(self, array, subgrid_res=1):
        """
        maps a 1d array into a (nx, ny) 2d grid with array populating the idex_mask indices
        :param array: 1d array
        :param idex_mask: 1d array of length nx*ny
        :param nx: x-axis of 2d grid
        :param ny: y-axis of 2d grid
        :return:
        """
        nx, ny = self._nx * subgrid_res, self._ny * subgrid_res
        if self._idex_mask_bool is True:
            grid1d = np.zeros((nx * ny))
            if subgrid_res > 1:
                idex_mask_subgrid = self._idex_mask_sub
            else:
                idex_mask_subgrid = self._idex_mask
            grid1d[idex_mask_subgrid == 1] = array
        else:
            grid1d = array
        grid2d = util.array2image(grid1d, nx, ny)
        return grid2d

    def re_size_convolve(self, array, unconvolved=False):
        """

        :param array: 1d data vector (can also be higher resolution binned)
        :param kwargs_psf: kwargs of psf modelling
        :param unconvolved: bool, if True, no convlolution performed, only re-binning
        :return: array with convolved and re-binned data/model
        """
        image = self._array2image(array, self._subgrid_res)
        if unconvolved is True:
            image_convolved = image_util.re_size(image, self._subgrid_res)
        else:
            image_convolved = self._PSF.psf_convolution_new(image, subgrid_res=self._subgrid_res,
                                                            subsampling_size=self._subsampling_size,
                                                            psf_subgrid=self._psf_subgrid, conv_type=self._conv_type,
                                                            subgrid_conv_type=self._subgrid_conv_type)
        return image_convolved * self._PixelGrid.pixel_width ** 2
