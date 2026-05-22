# Fisher--Rao Geometry of Multivariate Normal Distributions

This note summarizes the mathematical and computational structure used in this repository for Fisher--Rao geometry on the multivariate normal family. It separates four topics:

1. the Fisher geometry of multivariate normals in full generality;
2. the initial-value problem, closed-form exponential map, and Riemannian gradient descent;
3. what the horizontal-lift formulation tells us about endpoint geodesics;
4. Nielsen-style discretized approximations to Fisher--Rao distances.

The main distinction is:

\[
\boxed{\text{The exponential map / IVP is closed-form computable.}}
\]

\[
\boxed{\text{The endpoint geodesic / logarithm map is not known in closed form in full generality.}}
\]

The implementation in this repository uses the closed-form IVP for optimization and the horizontal-lift formulation as a numerical oracle for endpoint geodesics.

---

## 1. Fisher geometry of multivariate normal distributions

Let

\[
p=\mathcal N(\mu,\Sigma),
\]

where

\[
\mu\in\mathbb R^d,
\qquad
\Sigma\in\mathrm{Sym}_{++}(d).
\]

The parameter manifold is

\[
\mathcal G_d
=
\mathbb R^d\times \mathrm{Sym}_{++}(d).
\]

A tangent vector at \((\mu,\Sigma)\) is a pair

\[
(u,U),
\]

where

\[
u\in\mathbb R^d,
\qquad
U\in\mathrm{Sym}(d).
\]

The Fisher--Rao metric is

\[
g_{(\mu,\Sigma)}((u,U),(v,V))
=
u^\top\Sigma^{-1}v
+
\frac12
\operatorname{tr}
\left(
\Sigma^{-1}U\Sigma^{-1}V
\right).
\]

Write

\[
K=\Sigma^{-1}.
\]

Then

\[
g_{(\mu,\Sigma)}((u,U),(v,V))
=
u^\top K v
+
\frac12\operatorname{tr}(KUKV).
\]

This metric is invariant under affine transformations

\[
x\mapsto Ax+b,
\qquad
A\in GL(d).
\]

Thus endpoint and IVP computations can often be normalized to the base point

\[
(\mu,\Sigma)=(0,I).
\]

---

## 1.1 Christoffel map

The Levi--Civita connection can be represented by a Christoffel bilinear map

\[
\Gamma_{(\mu,\Sigma)}((u,U),(v,V))
=
\left(
\Gamma^\mu((u,U),(v,V)),
\Gamma^\Sigma((u,U),(v,V))
\right),
\]

with

\[
\boxed{
\Gamma^\mu((u,U),(v,V))
=
-\frac12
\left(
U\Sigma^{-1}v+V\Sigma^{-1}u
\right)
}
\]

and

\[
\boxed{
\Gamma^\Sigma((u,U),(v,V))
=
\frac12(uv^\top+vu^\top)
-
\frac12
\left(
U\Sigma^{-1}V+V\Sigma^{-1}U
\right).
}
\]

For a curve

\[
t\mapsto(\mu(t),\Sigma(t)),
\]

let

\[
\dot\mu=u,
\qquad
\dot\Sigma=U.
\]

The geodesic equation

\[
\ddot x+\Gamma(\dot x,\dot x)=0
\]

becomes

\[
\boxed{
\ddot\mu-\dot\Sigma\,\Sigma^{-1}\dot\mu=0
}
\]

and

\[
\boxed{
\ddot\Sigma
+
\dot\mu\dot\mu^\top
-
\dot\Sigma\,\Sigma^{-1}\dot\Sigma=0.
}
\]

Equivalently,

\[
\ddot\mu
=
\dot\Sigma\,\Sigma^{-1}\dot\mu,
\]

\[
\ddot\Sigma
=
\dot\Sigma\,\Sigma^{-1}\dot\Sigma
-
\dot\mu\dot\mu^\top.
\]

These equations have a Riccati/Sylvester/Lyapunov flavor: much of the computation involves left and right multiplication by SPD matrices.

---

## 1.2 Precision-coordinate form

Let

\[
K(t)=\Sigma(t)^{-1}.
\]

Then

\[
\dot K=-K\dot\Sigma K.
\]

The geodesic equations can be written as

\[
\boxed{
\ddot\mu+K^{-1}\dot K\,\dot\mu=0
}
\]

and

\[
\boxed{
\ddot K-\dot K K^{-1}\dot K
-
K\dot\mu\dot\mu^\top K
=0.
}
\]

Precision coordinates are often numerically more stable near ill-conditioned covariance endpoints.

---

## 1.3 Riemannian gradient

Let

\[
F(\mu,\Sigma)
\]

be a scalar objective, with Euclidean gradients

\[
g_\mu=\nabla_\mu F,
\qquad
G_\Sigma=\nabla_\Sigma F.
\]

The Riemannian gradient \(\operatorname{grad}F=(v_\mu,v_\Sigma)\) is defined by

\[
g_{(\mu,\Sigma)}((v_\mu,v_\Sigma),(a,A))
=
g_\mu^\top a+\operatorname{tr}(G_\Sigma A)
\]

for all tangent directions \((a,A)\). Hence

\[
\boxed{
v_\mu=\Sigma g_\mu
}
\]

and

\[
\boxed{
v_\Sigma
=
2\Sigma\,\operatorname{sym}(G_\Sigma)\,\Sigma.
}
\]

The negative Riemannian gradient direction for minimization is

\[
\boxed{
-\operatorname{grad}F
=
\left(
-\Sigma g_\mu,
-2\Sigma\,\operatorname{sym}(G_\Sigma)\,\Sigma
\right).
}
\]

The squared Riemannian gradient norm is

\[
\|\operatorname{grad}F\|^2
=
v_\mu^\top\Sigma^{-1}v_\mu
+
\frac12
\operatorname{tr}
\left(
\Sigma^{-1}v_\Sigma\Sigma^{-1}v_\Sigma
\right).
\]

---

## 2. Initial-value problem and the closed-form exponential map

The initial-value problem asks:

Given

\[
(\mu_0,\Sigma_0)
\]

and tangent vector

\[
(\dot\mu_0,\dot\Sigma_0),
\]

compute

\[
(\mu(t),\Sigma(t)).
\]

This is the Riemannian exponential map:

\[
(\mu(t),\Sigma(t))
=
\operatorname{Exp}_{(\mu_0,\Sigma_0)}
\left(
t(\dot\mu_0,\dot\Sigma_0)
\right).
\]

Unlike the endpoint logarithm map, this IVP has a closed-form solution.

---

## 2.1 Affine normalization of the IVP

Let

\[
A=\Sigma_0^{1/2},
\qquad
A^{-1}=\Sigma_0^{-1/2}.
\]

Normalize the base point to \((0,I)\) by defining

\[
x=A^{-1}\dot\mu_0,
\]

\[
B=A^{-1}\dot\Sigma_0A^{-1}.
\]

Here

\[
x\in\mathbb R^d,
\qquad
B\in\mathrm{Sym}(d).
\]

Compute the normalized geodesic

\[
(\tilde\mu(t),\tilde\Sigma(t)).
\]

Then transform back:

\[
\boxed{
\mu(t)=\mu_0+A\tilde\mu(t)
}
\]

and

\[
\boxed{
\Sigma(t)=A\tilde\Sigma(t)A.
}
\]

---

## 2.2 Closed-form normalized IVP solution

Assume the normalized initial conditions

\[
\tilde\mu(0)=0,
\qquad
\tilde\Sigma(0)=I,
\]

\[
\dot{\tilde\mu}(0)=x,
\qquad
\dot{\tilde\Sigma}(0)=B.
\]

Define

\[
\boxed{
G=(B^2+2xx^\top)^{1/2}.
}
\]

Let

\[
C(t)=\cosh(tG),
\qquad
S(t)=\sinh(tG).
\]

Use precision-coordinate variables

\[
\Delta(t)=\tilde\Sigma(t)^{-1},
\qquad
\delta(t)=\tilde\Sigma(t)^{-1}\tilde\mu(t).
\]

Then the normalized IVP solution is

\[
\boxed{
\begin{aligned}
\Delta(t)
={}&
I
+\frac12(C(t)-I)
+\frac12 B(C(t)-I)G^{-2}B \\
&-\frac12 S(t)G^{-1}B
-\frac12 B S(t)G^{-1}.
\end{aligned}
}
\]

and

\[
\boxed{
\delta(t)
=
-B(C(t)-I)G^{-2}x
+
S(t)G^{-1}x.
}
\]

Recover

\[
\boxed{
\tilde\Sigma(t)=\Delta(t)^{-1}
}
\]

and

\[
\boxed{
\tilde\mu(t)=\tilde\Sigma(t)\delta(t).
}
\]

In implementation, use linear solves rather than explicit inverses whenever possible:

\[
\tilde\mu(t)=\operatorname{solve}(\Delta(t),\delta(t)).
\]

---

## 2.3 Stable spectral implementation

Let

\[
G=Q\operatorname{diag}(g_i)Q^\top.
\]

Then implement matrix functions spectrally.

For example:

\[
\cosh(tG)
=
Q\operatorname{diag}(\cosh(tg_i))Q^\top.
\]

The products involving \(G^{-1}\) and \(G^{-2}\) should be evaluated through stable scalar functions:

\[
S(t)G^{-1}
=
Q
\operatorname{diag}
\left(
\frac{\sinh(tg_i)}{g_i}
\right)
Q^\top,
\]

\[
(C(t)-I)G^{-2}
=
Q
\operatorname{diag}
\left(
\frac{\cosh(tg_i)-1}{g_i^2}
\right)
Q^\top.
\]

Use the limiting values

\[
\frac{\sinh(tg)}{g}\to t
\quad\text{as }g\to0,
\]

and

\[
\frac{\cosh(tg)-1}{g^2}\to\frac12t^2
\quad\text{as }g\to0.
\]

This avoids explicitly forming ill-conditioned inverses of \(G\).

---

## 2.4 Riemannian gradient descent using the Exp map

Given objective

\[
F(\mu,\Sigma),
\]

compute Euclidean gradients

\[
g_\mu=\nabla_\mu F,
\qquad
G_\Sigma=\nabla_\Sigma F.
\]

Convert to the negative Riemannian gradient direction

\[
v_\mu=-\Sigma g_\mu,
\]

\[
v_\Sigma=-2\Sigma\,\operatorname{sym}(G_\Sigma)\,\Sigma.
\]

Then perform the update

\[
\boxed{
(\mu_{k+1},\Sigma_{k+1})
=
\operatorname{Exp}_{(\mu_k,\Sigma_k)}
\left(
\eta_k(v_\mu,v_\Sigma)
\right).
}
\]

Equivalently,

\[
(\mu_{k+1},\Sigma_{k+1})
=
\operatorname{Exp}_{(\mu_k,\Sigma_k)}
\left(
-\eta_k\operatorname{grad}F(\mu_k,\Sigma_k)
\right).
\]

The exponential map is exact for the tangent direction, but the objective may still increase if \(\eta_k\) is too large. Use conservative step sizes or backtracking.

---

## 2.5 Relation to trust-region methods

Riemannian trust-region methods build a local quadratic model in the tangent space:

\[
m_p(\xi)
=
F(p)
+
\langle\operatorname{grad}F(p),\xi\rangle_p
+
\frac12
\langle\operatorname{Hess}F(p)[\xi],\xi\rangle_p,
\]

subject to

\[
\|\xi\|_p\le \Delta.
\]

The trust-region subproblem requires the Riemannian gradient, Hessian-vector products, and the metric norm. It does not require the logarithm map or pairwise Fisher--Rao distance.

The exponential map enters as a retraction/update:

\[
p_{\text{new}}=\operatorname{Exp}_p(\xi).
\]

Thus the closed-form MVN exponential map is directly useful for Riemannian gradient descent and can also serve as the retraction in Riemannian trust-region methods.

---

## 3. Endpoint geodesics and horizontal lifts

The endpoint problem asks:

Given

\[
(\mu_0,\Sigma_0)
\quad\text{and}\quad
(\mu_1,\Sigma_1),
\]

find the geodesic joining them.

This is the logarithm-map or boundary-value problem. It is harder than the IVP because one must solve for the initial tangent vector, or equivalently a hidden gauge parameter.

In normalized coordinates,

\[
(0,I)\to(\mu,\Sigma),
\]

the closed-form IVP does not directly solve the problem because the unknown initial data \((x,B)\) must satisfy

\[
\Delta(1)=\Sigma^{-1},
\]

\[
\delta(1)=\Sigma^{-1}\mu.
\]

That is a nonlinear matrix equation.

---

## 3.1 Kobayashi-style horizontal lift

A useful endpoint formulation embeds the Gaussian manifold into a larger symmetric-space model. In normalized coordinates define

\[
\Theta=\Sigma^{-1}.
\]

For each skew-symmetric gauge matrix

\[
\Omega\in\mathfrak{so}(d),
\]

define

\[
L_\Omega=
\begin{pmatrix}
I&0&0\\
\mu^\top&1&0\\
\Omega-\frac12\mu\mu^\top&-\mu&I
\end{pmatrix},
\]

and

\[
D_\Theta=
\begin{pmatrix}
\Theta&0&0\\
0&1&0\\
0&0&\Sigma
\end{pmatrix}.
\]

Then

\[
\boxed{
G_\Omega=L_\Omega D_\Theta L_\Omega^\top.
}
\]

The matrices \(G_\Omega\) lie in the endpoint fiber over the same Gaussian endpoint. The missing endpoint-geodesic problem becomes selecting the correct gauge

\[
\Omega_\star.
\]

Define the energy

\[
\boxed{
E(\Omega)=\frac12\|\log G_\Omega\|_F^2.
}
\]

The horizontal condition is

\[
\boxed{
(\log G_\Omega)_{13}=0,
}
\]

where the \((1,3)\) block is the top-right \(d\times d\) block under the \((d,1,d)\) block partition.

A minimizer of \(E(\Omega)\) satisfying the horizontal condition gives a lifted geodesic

\[
G(t)=\exp(t\log G_{\Omega_\star}).
\]

Projecting the upper-left \((d+1)\times(d+1)\) block recovers

\[
\Theta(t)=\Sigma(t)^{-1},
\qquad
\delta(t)=\Theta(t)\mu(t),
\]

and therefore

\[
\Sigma(t)=\Theta(t)^{-1},
\qquad
\mu(t)=\Theta(t)^{-1}\delta(t).
\]

---

## 3.2 Gauge dimension

The unknown gauge is

\[
\Omega\in\mathfrak{so}(d),
\]

so it has

\[
\boxed{
\frac{d(d-1)}2
}
\]

scalar degrees of freedom.

Thus:

| dimension \(d\) | gauge dimension |
| ---: | ---: |
| 1 | 0 |
| 2 | 1 |
| 3 | 3 |
| 4 | 6 |
| 5 | 10 |
| 10 | 45 |

For \(d=2\), the endpoint gauge problem is scalar:

\[
\Omega(\omega)=
\begin{pmatrix}
0&\omega\\
-\omega&0
\end{pmatrix}.
\]

For \(d>2\), it is a nonlinear system in multiple skew degrees of freedom.

---

## 3.3 What the horizontal-lift experiments showed

The repository implements the horizontal-lift variational solve and validates it against:

- scalar cases;
- zero-mean multivariate cases;
- explicit gradient and Hessian-vector products;
- endpoint recovery;
- original geodesic ODE diagnostics;
- direct Diffrax ODE shooting;
- adversarial conditioning tests.

The main empirical findings are:

1. For moderate endpoints, the horizontal-lift path satisfies the original geodesic ODE numerically and recovers endpoints accurately.
2. The worst adversarial endpoints were not geometric failures but projection-conditioning artifacts.
3. In ill-conditioned cases, the lifted endpoint and precision-coordinate endpoint can be recovered accurately even when covariance-coordinate recovery is unstable.
4. Therefore endpoint diagnostics should be performed primarily in lifted or precision coordinates.

For a high-condition endpoint with

\[
\kappa(\Sigma)\approx 6.6\times10^5,
\]

the observed errors were approximately:

\[
\|G(1)-G_\star\|\sim 10^{-14},
\]

\[
\|(\Theta,\delta)-(\Theta_\star,\delta_\star)\|\sim 10^{-10},
\]

while covariance-coordinate errors were much larger due to inversion amplification.

Thus, near the SPD boundary, the stable representation is

\[
(\delta,\Theta)
\]

rather than

\[
(\mu,\Sigma).
\]

---

## 3.4 What remains unresolved

The horizontal-lift formulation does not provide a closed-form endpoint solution.

The unresolved mathematical object is still:

\[
\boxed{
\Omega_\star(\mu,\Sigma)
}
\]

where

\[
\boxed{
(\log G_{\Omega_\star})_{13}=0.
}
\]

Equivalently, in IVP coordinates, the unresolved object is the initial tangent \((x,B)\) satisfying the boundary conditions

\[
\Delta(1)=\Sigma^{-1},
\qquad
\delta(1)=\Sigma^{-1}\mu.
\]

Known closed-form endpoint cases include:

- \(d=1\);
- equal means;
- covariance-only geodesics;
- commuting/scalar covariance reductions.

The genuinely difficult regime is

\[
\mu\ne0,
\qquad
[\Sigma,\mu\mu^\top]\ne0.
\]

The smallest nontrivial case is

\[
d=2,
\qquad
\Sigma=\operatorname{diag}(s_1,s_2),
\qquad
\mu=(u,v),
\qquad
s_1\ne s_2,
\qquad
uv\ne0.
\]

In that case, solving the endpoint problem reduces to solving for one scalar gauge variable \(\omega_\star\). A closed form for this scalar case would be real mathematical progress.

---

## 4. Nielsen-style Fisher--Rao approximations

Nielsen-style approximations address a different problem: they approximate Fisher--Rao distance by discretizing a chosen curve between two MVNs.

They do not solve the endpoint geodesic boundary-value problem.

---

## 4.1 Discretized Jeffreys curve length

Given a curve

\[
c(t)=\mathcal N(\mu(t),\Sigma(t)),
\qquad
t\in[0,1],
\]

sample

\[
t_i=\frac{i}{n},
\qquad
i=0,\ldots,n.
\]

For adjacent points, approximate the local Fisher--Rao segment length by

\[
\sqrt{J(p_i,p_{i+1})},
\]

where

\[
J(p,q)=\mathrm{KL}(p\|q)+\mathrm{KL}(q\|p)
\]

is Jeffreys divergence.

For MVNs,

\[
\mathrm{KL}(p_0\|p_1)
=
\frac12
\left[
\operatorname{tr}(\Sigma_1^{-1}\Sigma_0)
+
(\mu_1-\mu_0)^\top\Sigma_1^{-1}(\mu_1-\mu_0)
-d
+
\log\frac{\det\Sigma_1}{\det\Sigma_0}
\right].
\]

Thus the discretized curve length is

\[
\boxed{
D_n(c)
=
\sum_{i=0}^{n-1}
\sqrt{
J(c(t_i),c(t_{i+1}))
}.
}
\]

As

\[
n\to\infty,
\]

this estimates the Fisher length of the chosen curve \(c\). It does not generally converge to the geodesic distance unless the chosen curve is itself geodesic.

---

## 4.2 Interpolation curves

The implemented Nielsen baselines use several fixed curves.

### Source-coordinate curve

\[
\mu(t)=(1-t)\mu_0+t\mu_1,
\]

\[
\Sigma(t)=(1-t)\Sigma_0+t\Sigma_1.
\]

### Natural-coordinate curve

The natural parameters are

\[
K=\Sigma^{-1},
\qquad
h=K\mu.
\]

Interpolate

\[
h(t)=(1-t)h_0+th_1,
\]

\[
K(t)=(1-t)K_0+tK_1.
\]

Then recover

\[
\Sigma(t)=K(t)^{-1},
\qquad
\mu(t)=\Sigma(t)h(t).
\]

### Expectation-coordinate curve

The expectation parameters are

\[
m=\mu,
\qquad
M=\Sigma+\mu\mu^\top.
\]

Interpolate

\[
m(t)=(1-t)m_0+tm_1,
\]

\[
M(t)=(1-t)M_0+tM_1.
\]

Then recover

\[
\mu(t)=m(t),
\]

\[
\Sigma(t)=M(t)-m(t)m(t)^\top.
\]

---

## 4.3 What the Nielsen comparison showed

The repository compares Nielsen-style discretized Jeffreys lengths against the horizontal-lift geodesic candidate.

Empirically:

1. For scalar, zero-mean, and moderate random endpoints, Nielsen-style approximations are close but consistently above the horizontal-lift distance.
2. Increasing \(n\) estimates the chosen curve length more accurately, but the limiting curve length can remain above the geodesic distance because the chosen curve is not generally geodesic.
3. For ill-conditioned noncommuting endpoints, the fixed Nielsen curves can be much longer than the horizontal-lift geodesic candidate.

For one projection-conditioning endpoint with

\[
d=3,
\qquad
\kappa(\Sigma)\approx 6.6\times10^5,
\]

the horizontal-lift distance was approximately

\[
10.62,
\]

whereas Nielsen curve lengths were much larger. As the discretization increased from

\[
n=8
\quad\text{to}\quad
n=1024,
\]

the Nielsen estimates decreased substantially but remained about \(2.3\times\) the horizontal-lift distance.

This indicates that coarse discretization was part of the error, but the fixed interpolation curves were still intrinsically much longer than the computed geodesic candidate.

---

## 4.4 Hilbert/SPD proxy distances

The implementation may also include Hilbert projective distances on SPD cones as proxy quantities. For SPD matrices \(A,B\), define generalized eigenvalues of \(A^{-1}B\):

\[
\lambda_{\min},
\qquad
\lambda_{\max}.
\]

The Hilbert projective distance is

\[
\boxed{
d_H(A,B)
=
\log\frac{\lambda_{\max}}{\lambda_{\min}}.
}
\]

This is useful as a proxy or diagnostic, but it is not the MVN Fisher--Rao distance.

---

## 5. Practical conclusions

### 5.1 What is solved

The IVP / exponential map is closed-form computable.

Given

\[
(\mu,\Sigma)
\]

and tangent vector

\[
(v_\mu,v_\Sigma),
\]

we can compute

\[
\operatorname{Exp}_{(\mu,\Sigma)}(t(v_\mu,v_\Sigma))
\]

in closed form using matrix hyperbolic functions.

This enables exact Fisher--Rao Riemannian gradient descent over full MVNs.

---

### 5.2 What is not solved

The arbitrary endpoint problem is not solved in closed form.

Given

\[
(\mu_0,\Sigma_0),
\qquad
(\mu_1,\Sigma_1),
\]

we do not know a general closed-form expression for

\[
\operatorname{Log}_{(\mu_0,\Sigma_0)}(\mu_1,\Sigma_1).
\]

Equivalently, we do not know a closed-form formula for the horizontal gauge

\[
\Omega_\star(\mu,\Sigma).
\]

---

### 5.3 What is useful for optimization

For optimization, the endpoint log map and pairwise Fisher--Rao distance are usually not needed.

Riemannian gradient descent requires:

1. the metric;
2. the Riemannian gradient;
3. a retraction or exponential map.

Riemannian trust-region methods require:

1. the metric;
2. the Riemannian gradient;
3. Hessian-vector products;
4. a retraction or exponential map.

Thus, the closed-form exponential map is directly useful, while the closed-form log map is mostly relevant for interpolation, distance computation, barycenters, kernels, or endpoint geodesic problems.

---

## 6. Recommended repository framing

The most practical framing of this repository is:

\[
\boxed{
\text{Exact Fisher--Rao exponential-map optimization for multivariate normal distributions.}
}
\]

with an additional experimental module:

\[
\boxed{
\text{Horizontal-lift numerical solver for endpoint geodesics.}
}
\]

The endpoint solver is useful as a research tool and validation oracle, but the exponential map is the immediately actionable component for optimization.

---

## References

- C. R. Rao. "Information and the accuracy attainable in the estimation of statistical parameters." Bulletin of the Calcutta Mathematical Society, 1945.
- H. Hotelling. "Spaces of statistical parameters." Bulletin of the American Mathematical Society, 1930.
- B. Efron. "Defining the curvature of a statistical problem." Annals of Statistics, 1975.
- S. Amari and H. Nagaoka. *Methods of Information Geometry*. AMS/Oxford, 2000.
- F. Nielsen. Fisher--Rao distance approximations and related notes on multivariate normal distributions.
- M. Kobayashi. Work on geodesics and submersion formulations for the normal distribution manifold.
