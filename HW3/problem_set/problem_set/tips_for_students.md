[comment]: <> (Below are tips for students as written by Wei (Fall 2021 semester) to post along side the assignment. ) 

This homework requires several additional configurations, so PLEASE READ THIS POST before you process. Here are the reasons and the solutions:

- We need to play with sparse LU and QR decomposition in solving large linear systems. While LU has been wrapped by `scipy`, unfortunately, QR is not. An unofficial implementation `sparseqr` provides the functionality that requires `SuiteSparse`.
- On Ubuntu, it can be installed via `sudo apt install libsuitesparse-dev`
- On macOS, it can be installed via `brew install suite-sparse`. Occasionally, there could be unexpected issues due to OpenMP compatibility;
- On Windows, unfortunately, there is no one-liner to install it.

Given the context, here are our recommendations:

- If you are using Ubuntu and macOS, first install the `SuiteSparse` dependency as we have mentioned. Then, use our provided script `install_deps.sh` to install dependent packages (recommended in a conda environment).
- If you are using Windows, or you encounter non-trivial issues in deploying the code, we recommend using WSL2 and using the ubuntu instructions, or alternatively we provide a Virtual Box image that is ready to use. Simply install virtual box, import the [image]( https://drive.google.com/file/d/1AVKxF5DxbogiB_bSBsCxwmenD1Mx-UwK/view) (~3G, password ubuntu), and you are all set. Go to `~/16833/16833_HW3_SOLVERS/`, run `conda activate 16833`, and you can directly edit and run the code without any further configurations.
- If you are really a hardcore Windows user and will need to use `SuiteSparse` in your later research, follow the instructions [here](https://github.com/jlblancoc/suitesparse-metis-for-windows). You will need VS Studio (not Code) and `cmake` to install the dependencies.

Follow-up tips and errata will be updated here. This is the first time we pythonize this homework, and we will greatly appreciate your feedbacks!

As always, we are here to help! Good luck :)
## Errata and clarifications:

### Observation variable:
`observations[i, 0]`: robot pose index in `[0, n_poses)`. You need to use `int(observation[i, 0])` to make it an index in an array. `observations[i, 1]`: landmark index in `[0, n_landmarks)`. You need to use `int(observation[i, 1])` to make it an index in an array. `observations[i, 2:4]`: (Δx,Δy) in the linear setup, (θ,r) in the nonlinear setup, from `poses[observations[i, 0]]` to `landmark[observations[i, 1]]`

### Covariance on the prior
`sigma_odom` should be working well. This factor won’t make too much difference if reasonable.

### QR decomposition
Some of you may face a broadcasting issue after the 1st iteration of the nonlinear problem using qr. Try to apply `z = z.flatten()` before you call `spsolve_triangular` in `solve_qr`.

### macOS configuration

If you find weird behavior of scipy itself on macOS, scipy’s internal library might be conflicting with the brew installed version. A reinstallation might fix it.