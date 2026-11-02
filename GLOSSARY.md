# kremetart

kremetart turns TART snapshots into a stream of HEALPix sky maps and tracks each pixel over time to
flag transients. Terms shared with the TART data model (**Snapshot**, **Baseline**, **Calibration
snapshot**) follow the [tart-msv4 glossary](https://github.com/landmanbester/tart-msv4/blob/main/GLOSSARY.md).

## Language

### Inputs

**Snapshot**:
The correlations of every baseline at a single timestamp, together with what imaging them needs: their
weights, rotated baselines, boresight and frequencies. One snapshot yields one map per stage.
_Avoid_: frame (reserved for coordinate frames and movie frames), integration, time step

**Rotated baselines**:
A snapshot's baselines expressed in the celestial frame of the sky grid at that snapshot's timestamp, so
that a fixed sky pixel stays fixed as the Earth turns.
_Avoid_: B_ROT, equatorial baselines

**Boresight**:
The direction the array's primary beam points at a snapshot's timestamp (TART's zenith), expressed in the
celestial frame of the sky grid.

**Sky grid**:
The HEALPix pixelisation of the sky and the primary-beam model that every map in a run is defined on.
_Avoid_: pixel grid, healpix grid

### Stages and maps

**Stage**:
One step in turning snapshots into maps. A stage keeps its own state from one snapshot to the next and
exchanges only snapshots and maps with other stages.
_Avoid_: operator, block (those name what runs a stage, not the stage)

**Map**:
A sky brightness estimate on the sky grid that a stage produces from one snapshot.
_Avoid_: image, cube, frame

**Dirty map**:
The snapshot's weighted visibilities transformed back onto the sky grid, with no deconvolution. Produced by
the **dirty imager**.
_Avoid_: dirty image

**Tikhonov map**:
A deconvolved map that balances fidelity to the snapshot against an L2 penalty on brightness. Produced by
the **Tikhonov solver**.

**L1 map**:
A deconvolved map that balances fidelity to the snapshot against a reweighted L1 penalty, favouring a sky
of few bright sources. Produced by the **L1 solver**.

**IWP filter**:
The stage that follows each pixel of a map stream through time as an integrated Wiener process (a
per-pixel Kalman filter), coasting through snapshots that have no data.
_Avoid_: Kalman operator, smoother

**Smooth map**:
The IWP filter's running estimate of each pixel's brightness.
_Avoid_: filtered map

**Normalised innovation**:
How far each pixel of a new map departs from the IWP filter's prediction, in units of the predicted
standard deviation. It is the transient detection statistic.
_Avoid_: znorm, z-score

**Snapshot imager**:
The fixed chain of stages (dirty imager, Tikhonov solver, optional L1 solver, IWP filter) that turns each
snapshot into all of its maps.
_Avoid_: pipeline, frame imager
