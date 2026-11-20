import numpy as np
import xrayutilities as xu
from numpy.typing import NDArray
from pint import Quantity as Q_


class Experiment:
    """Experiment.
    """


    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        alpha: float,
        beta: float,
        gamma: float,
        sample_detector_distance_meters: float,
        pixel_width_microns: float,
        sample_axis: tuple[str, ...],
        detector_axis: tuple[str, ...],
        r_i: tuple[float, float, float],
        en: float,
        ipdir_hkl: tuple[float, float, float],
        ndir_hkl: tuple[float, float, float],
        sampleor: str,
        bounds: tuple[float, ...] | None = None,
        startvalues: tuple[float, ...] | None = None,
        axis_of_det: tuple[int, int] | None = None,
    ) -> None:
        """__init__.

        Args:
            a (float): a
            b (float): b
            c (float): c
            alpha (float): alpha
            beta (float): beta
            gamma (float): gamma
            sample_detector_distance_meters (float): sample_detector_distance_meters
            pixel_width_microns (float): pixel_width_microns
            sample_axis (tuple[str, ...]): sample_axis
            detector_axis (tuple[str, ...]): detector_axis
            r_i (tuple[float, float, float]): r_i
            en (float): en
            ipdir_hkl (tuple[float, float, float]): ipdir_hkl
            ndir_hkl (tuple[float, float, float]): ndir_hkl
            sampleor (str): sampleor
            bounds (tuple[float, ...] | None): bounds
            startvalues (tuple[float, ...] | None): startvalues
            axis_of_det (tuple[int, int] | None): axis_of_det

        Returns:
            None:
        """

        # We assume for now that the detector axes are listed in order of the
        # shape of the detector array, and that increasing index is increasing
        # angle

        self.sample_detector_distance = Q_(
            sample_detector_distance_meters, 'meters'
        )
        self.pixel_width = Q_(pixel_width_microns, 'microns')

        name = 'Si'
        lat = xu.materials.SGLattice(1, a, b, c, alpha, beta, gamma)
        self.crystal = xu.materials.Crystal(name=name, lat=lat)

        r_i: NDArray = np.array(r_i)
        self.qconv = xu.experiment.QConversion(
            sampleAxis=sample_axis, detectorAxis=detector_axis, r_i=r_i, en=en
        )
        self.ipdir = self.crystal.Q(*ipdir_hkl)
        self.ndir = self.crystal.Q(*ndir_hkl)
        self.experiment = xu.Experiment(
            ipdir=self.ipdir,
            ndir=self.ndir,
            qconv=self.qconv,
            sampleor=sampleor,
            en=en
        )
        self.num_sample_axes = len(sample_axis)
        self.num_detector_axes = len(detector_axis)
        if self.num_detector_axes != 2:
            raise NotImplementedError

        self.bounds = bounds
        if self.bounds is None:
            self.bounds = tuple(
                (-90.0, 90.0)
                for _ in range(self.num_sample_axes + self.num_detector_axes)
            )

        self.startvalues = startvalues
        if self.startvalues is None:
            self.bounds = tuple(
                20.0
                for _ in range(self.num_sample_axes + self.num_detector_axes)
            )

        self.axis_of_det = axis_of_det
        if self.axis_of_det is None:
            self.axis_of_det = list(range(self.num_detector_axes))

    def _shape2q_by_hkl(
        self, center_hkl: tuple[float], shape: tuple[int], center_i: int,
        center_j: int
    ) -> NDArray:
        """_shape2q_by_hkl.

        Args:
            center_hkl (tuple[float]): center_hkl
            shape (tuple[int]): shape
            center_i (int): center_i
            center_j (int): center_j

        Returns:
            NDArray:
        """
        ang_hkl = np.array(self.hkl2ang(center_hkl))
        center = np.array([center_i, center_j])
        ind = np.indices(shape)
        pval = (ind - center[:, np.newaxis, np.newaxis]) * self.pixel_width
        if self.axis_of_det == (1, 0):
            pval = np.roll(pval, 1, axis=0)
        elif self.axis_of_det == (0, 1):
            pass
        else:
            raise NotImplementedError('Only 2 detector axes implemented')
        pangs = np.zeros(
            (self.num_sample_axes + self.num_detector_axes, *shape)
        )
        pangs[:self.num_sample_axes, :, :] = ang_hkl[
            :self.num_sample_axes,
            np.newaxis,
            np.newaxis,
        ]
        pangs[self.num_sample_axes:, :, :] = (
            ang_hkl[self.num_sample_axes:, np.newaxis, np.newaxis] + np.arctan(
                pval / self.sample_detector_distance
            ).to('degrees').magnitude
        )
        q = np.array(
            [
                self.ang2q(*p)
                for p in pangs.reshape(3, -1).transpose()
            ]
        ).transpose().reshape((3, *shape))
        return q

    def _shape2q_by_ang(
        self, center_ang: tuple[float], *args, **kwargs
    ) -> NDArray:
        """_shape2q_by_ang.

        Args:
            center_ang (tuple[float]): center_ang
            args:
            kwargs:

        Returns:
            NDArray:
        """
        center_hkl = self.experiment.Ang2HKL(*center_ang, mat=self.crystal)
        return self._shape2q_by_hkl(center_hkl, *args, **kwargs)

    def _shape2q_dispatcher(
        self, *args, mode: str = 'hkl', **kwargs
    ) -> NDArray:
        """_shape2q_dispatcher.

        Args:
            args:
            mode (str): mode
            kwargs:

        Returns:
            NDArray:
        """
        if mode == 'hkl':
            return self._shape2q_by_hkl(*args, **kwargs)
        elif mode == 'ang':
            return self._shape2q_by_ang(*args, **kwargs)
        else:
            raise NotImplementedError

    def shape2q(self, *args, **kwargs) -> NDArray:
        """shape2q.

        Args:
            args:
            kwargs:

        Returns:
            NDArray:
        """
        return self._shape2q_dispatcher(*args, **kwargs)

    def hkl2ang(self, *args) -> tuple[float, ...]:
        """hkl2ang.

        Args:
            args:

        Returns:
            tuple[float, ...]:
        """
        if isinstance(args[0], tuple):
            hkl = args[0]
        else:
            hkl = args
        ang_hkl, _, _ = xu.Q2AngFit(
            self.experiment.Transform(self.crystal.Q(*hkl)),
            self.experiment,
            bounds=self.bounds,
            startvalues=self.startvalues,
        )
        return ang_hkl

    def ang2hkl(self, *args) -> tuple[float]:
        """ang2hkl.

        Args:
            args:

        Returns:
            tuple[float]:
        """
        if isinstance(args[0], tuple):
            ang = args[0]
        else:
            ang = args
        return self.experiment.Ang2HKL(*ang, mat=self.crystal)

    def hkl2q(self, *hkl):
        """hkl2q.

        Args:
            hkl:
        """
        if not isinstance(hkl[0], tuple):
            hkl = tuple(hkl)
        return self.crystal.Q(*hkl)

    def ang2q(self, *args) -> tuple[float]:
        """ang2q.

        Args:
            args:

        Returns:
            tuple[float]:
        """
        if isinstance(args[0], tuple):
            ang = args[0]
        else:
            ang = args
        return self.hkl2q(self.ang2hkl(ang))

    def q2hkl(self, *args) -> tuple[float]:
        """q2hkl.

        Args:
            args:

        Returns:
            tuple[float]:
        """
        if isinstance(args[0], tuple):
            q = args[0]
        else:
            q = args
        return self.crystal.HKL(q)
