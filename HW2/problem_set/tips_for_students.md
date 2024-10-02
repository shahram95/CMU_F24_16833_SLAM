The noise distribution in Q1.2 can be a little bit tricky, as it is depending on the current state's coordinate system thus not purely additive.

Please be aware of several facts:

1) You can regard ex,ey,eα as infinitesimal and ignore the second-order (or higher-order) terms. 

2) With this formulation the noise term in EKF is depending on the state, so instead of using pt+1=g(pt,ut)+ϵt, you may use the more general pt+1=g(pt,ut,ϵt) (Wikipedia). There is not much difference except that we need to compute the Jacobian with respect to ϵ in addition to p.

3) A similar formulation can also be applied to the remaining part of this homework.

Also, please refer to Table 10.1 for the overall algorithm. Specifically, we iterate over landmarks to update the state vector. A batched version will be covered in the next homework.

Note the differences:

1) We are dealing with 2D landmarks;

2) Our noise model is more complex.

Q2.4 requirements for Euclidean and Mahalanobis distances 

One of the parts in Q2.4 asks you is to "Write down the Euclidean and Mahalanobis distances of each landmark estimation with respect to the ground truth?"

What we are looking for here are the values eTe−−−√(Euclidean distance) and eTΣ−1e−−−−−−√(Mahalanobis distance) where e∈R2 is the error between final mean position estimate of a landmark and the ground truth landmark position.  Σ is the corresponding final covariance matrix for that landmark. Please compute these distance values separately for all 6 landmarks.

Errata

In Q1.5:

An important step in EKF-SLAM is to find the measurement Jacobian H p
with respect to the robot pose. Please apply the results in 1.4 to derive the
analytical form of H p (Note: H p should be a 2 × 3 matrix represented by p t
and l). (10 points)

