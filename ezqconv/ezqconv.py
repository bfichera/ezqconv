import typing

import numpy as np
from numpy.typing import NDArray
import xrayutilities as xu
from pint import Quantity as Q_


class Experiment:

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
        sample_axis: tuple[str],
        detector_axis: tuple[str],
        r_i: tuple[float],
        en: float,
        ipdir_hkl: tuple[float],
        ndir_hkl: tuple[float],
        sampleor: str,
        bounds: tuple[float] | None = None,
        startvalues: tuple[float] | None = None,
        axis_of_det: tuple[int] | None = None,
    ) -> typing.Self:

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

        r_i = np.array(r_i)
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
                for ax in range(self.num_sample_axes + self.num_detector_axes)
            )

        self.startvalues = startvalues
        if self.startvalues is None:
            self.bounds = tuple(
                20.0
                for ax in range(self.num_sample_axes + self.num_detector_axes)
            )

        self.axis_of_det = axis_of_det
        if self.axis_of_det is None:
            self.axis_of_det = list(range(self.num_detector_axes))

    def _shape2q_by_hkl(
        self, center_hkl: tuple[float], shape: tuple[int], center_i: int,
        center_j: int
    ) -> NDArray:
        ang_hkl = self.hkl2ang(center_hkl)
        center = np.array([center_i, center_j])
        ind = np.indices(shape)
        pval = (ind - center[:, np.newaxis, np.newaxis]) * self.pixel_width
        if np.array_equal(self.axis_of_det, (1, 0)):
            pval = np.roll(pval, 1, axis=0)
        elif np.array_equal(self.axis_of_det, (0, 1)):
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
                (pval / self.sample_detector_distance)
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
        center_hkl = self.experiment.Ang2HKL(*center_ang, mat=self.crystal)
        return self._shape2q_by_hkl(center_hkl, *args, **kwargs)

    def _shape2q_dispatcher(
        self, *args, mode: str = 'hkl', **kwargs
    ) -> NDArray:
        if mode == 'hkl':
            return self._shape2q_by_hkl(*args, **kwargs)
        elif mode == 'ang':
            return self._shape2q_by_ang(*args, **kwargs)
        else:
            raise NotImplementedError

    def shape2q(self, *args, **kwargs) -> NDArray:
        return self._shape2q_dispatcher(*args, **kwargs)

    def hkl2ang(self, *args) -> tuple[float]:
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
        if isinstance(args[0], tuple):
            ang = args[0]
        else:
            ang = args
        return self.experiment.Ang2HKL(*ang, mat=self.crystal)

    def hkl2q(self, *hkl):
        if not isinstance(hkl[0], tuple):
            hkl = tuple(hkl)
        return self.crystal.Q(*hkl)

    def ang2q(self, *args) -> tuple[float]:
        if isinstance(args[0], tuple):
            ang = args[0]
        else:
            ang = args
        return self.hkl2q(self.ang2hkl(ang))

    def q2hkl(self, *args) -> tuple[float]:
        if isinstance(args[0], tuple):
            q = args[0]
        else:
            q = args
        return self.crystal.HKL(q)
