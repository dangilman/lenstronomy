# Conditions for the ZTF survey

import lenstronomy.Util.util as util

__all__ = ["ZTF"]
"""
Sources
  https://iopscience.iop.org/article/10.1088/1538-3873/aaecbe/pdf
    Table 1
      - seeing
      - pixel_scale
      - gain
      - read_noise

  http://web.ipac.caltech.edu/staff/fmasci/ztf/ztf_pipelines_deliverables.pdf
    Section 10.1.2
      - magnitude_zero_point

  https://www.ztf.caltech.edu/page/dr3
    Section 10
      - exposure_time (assuming public survey only)

  https://iopscience.iop.org/article/10.1088/1538-3873/aae8ac/pdf
    Section 3.7
      - num_exposures 
"""

g_band_obs = {
    "exposure_time": 30.0,
    "sky_brightness": 22.01,
    "magnitude_zero_point": 26.325,
    "num_exposures": 40,
    "seeing": 2.1,
    "psf_type": "GAUSSIAN",
}

r_band_obs = {
    "exposure_time": 30.0,
    "sky_brightness": 21.15,
    "magnitude_zero_point": 26.275,
    "num_exposures": 40,
    "seeing": 2.0,
    "psf_type": "GAUSSIAN",
}

i_band_obs = {
    "exposure_time": 30.0,
    "sky_brightness": 19.89,
    "magnitude_zero_point": 25.660,
    "num_exposures": 40,
    "seeing": 2.1,
    "psf_type": "GAUSSIAN",
}

# - keyword exposure_time: exposure time per image (in seconds)
# - keyword sky_brightness: sky brightness (in magnitude per square arcseconds in units of electrons)
# - keyword magnitude_zero_point: magnitude in which 1 count (e-) per second per arcsecond square is registered
# - keyword num_exposures: number of exposures that are combined (depends on coadd_years)
# - keyword seeing: Full-Width-at-Half-Maximum (FWHM) of PSF
# - keyword psf_type: string, type of PSF ('GAUSSIAN' supported)


class ZTF(object):
    """Class contains ZTF instrument and observation configurations."""

    def __init__(self, band="g", psf_type="GAUSSIAN", coadd_years=3):
        """
        :param band: string, 'g', 'r', or 'i', supported. Determines obs dictionary.
        :param psf_type: string, type of PSF ('GAUSSIAN' supported).
        :param coadd_years: int, number of years corresponding to num_exposures in obs dict. Currently supported: 1-3.
        """
        if band == "g":
            self.obs = g_band_obs
        elif band == "r":
            self.obs = r_band_obs
        elif band == "i":
            self.obs = i_band_obs
        else:
            raise ValueError("band %s not supported! Choose 'g', 'r', or 'i'." % band)

        if psf_type != "GAUSSIAN":
            raise ValueError("psf_type %s not supported!" % psf_type)

        if coadd_years > 3 or coadd_years < 1:
            raise ValueError(
                " %s coadd_years not supported! Choose an integer between 1 and 3."
                % coadd_years
            )
        elif coadd_years != 3:
            self.obs["num_exposures"] = (coadd_years * 40) // 3

        self.camera = {"read_noise": 10.3, "pixel_scale": 1.01, "ccd_gain": 5.8}

        # - keyword read_noise: std of noise generated by read-out (in units of electrons)
        # - keyword pixel_scale: scale (in arcseconds) of pixels
        # - keyword ccd_gain: electrons/ADU (analog-to-digital unit)

    def kwargs_single_band(self):
        """
        :return: merged kwargs from camera and obs dicts
        """
        kwargs = util.merge_dicts(self.camera, self.obs)
        return kwargs
