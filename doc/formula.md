# Formula handbook

Here we derive the formula for random state non-orthogonal basis calculation.

For the single particle electronic problem under non-orthognal basis:
$$
\begin{align*}
HC_n&=E_nSC_n \equiv \\ 
S^{-1/2}HS^{-1/2}\cdot S^{1/2}C_n&=E_n S^{1/2}C_n
\end{align*}
$$
The literature often call $S^{1/2}C_n$ as the wave vector under lowdin basis.

Next, let's represent the density operator in different form.
$$
\begin{align*}
\hat{\rho}&=\sum_nf_F(E_n)|\Psi_n\rangle\langle\Psi_n|\\
&=f_F(\hat{H})\\
&=\int d\epsilon\left[ f_F(\epsilon)\delta(\epsilon-\hat{H})\right]\\
&=-\frac{1}{\pi}\int d\epsilon\left[ f_F(\epsilon)\Im[\hat{G^r}]\right]
\end{align*}
$$
When representing under the basis:
$$
\begin{align*}
\rho_{ij}&=\sum_n f_F(E_n)\langle\phi_i|\Psi_n\rangle\langle\Psi_n|\phi_j\rangle\\
&=\sum_n f_F(E_n)S_{ik}C_{nk}C^*_{nl}S_{lj}
\end{align*}
$$
If computing the real space density:
$$
\begin{align*}
\langle r|\hat{\rho}|r\rangle&=\sum_{n,i,j}f_F(E_n)C_{ni}C^*_{nj}\phi_i(r)\phi^*_j(r)\\
&=\sum_{ij}\left(S^{-1}\rho S^{-1}\right)_{ij}\phi_i(r)\phi^*_j(r)
\end{align*}
$$
For convenience, we often what $\tilde{\rho}=\left(S^{-1}\rho S^{-1}\right)$. This equals to:
$$
\begin{align*}
\tilde{\rho}&=S^{-1/2}f_F(S^{-1/2}HS^{-1/2})S^{-1/2}\\
&=f_F(S^{-1}H)S^{-1}
\end{align*}
$$

This expression is very useful when deriving the properties that related to the real space representation. For any function that have the summation form of:
$$
g(r)=\sum_{n,i,j}q(E_n)C_{ni}C_{nj}^*\phi_i(r)\phi_j(r)^*
$$
The matrix $q_{i,j}=\sum_{n}q(E_n)C_{ni}C_{nj}^*$ can be estimated using: $g(S^{-1}H)S^{-1}$.

**NOTE**: $\text{Tr}\left[q(\hat{H})\right]$ does not equals to $\text{Tr}\left[q(S^{-1}H)S^{-1}\right]$.

If instead, we want to estimate the trace of the matrix function of the Hamiltonian operator: $\text{Tr}\left[f(\hat{H})\right]$:
$$
\begin{align*}
\text{Tr}\left[f(\hat{H})\right]&=\text{Tr}\left[f(S^{-1/2}HS^{-1/2})\right]\\
&=\text{Tr}\left[f(S^{-1}H)\right]
\end{align*}
$$
Since similarity transformation does not change the trace.


# DOS
From above, we have now the decomposed density operator in energy domain, which is:
$$
\begin{align*}
\hat{\rho}(\epsilon)&=\delta(\epsilon-\hat{H})\\
&=\frac{1}{2\pi}\int dt \left[e^{i\epsilon t}e^{-i\hat{H}t}\right]
\end{align*}
$$

The DOS is defined as the trace of $\hat{\rho}(\epsilon)$, which is represented by:
$$
\begin{align*}
\rho(\epsilon)&=\frac{1}{2\pi}\int dte^{i\epsilon t}\text{Tr}\left[e^{-i\hat{H}t}\right]\\
&=\frac{1}{2\pi}\int dte^{i\epsilon t}\text{Tr}\left[e^{-i\left(S^{-1}H\right)t}\right]
\end{align*}
$$

# LDOS
The LDOS have multiple meanings. The dos can be localized in real space and orbitals. So for the most general case, we are considering the energy resolved altered density matrix:
$$
\begin{align*}
\tilde{\rho}(\epsilon)&=S^{-1}\rho(\epsilon)S^{-1}\\
&=\delta(\epsilon-S^{-1}H)S^{-1}\\
&=\frac{1}{2\pi}\int dt \left[e^{i\epsilon t}e^{-i(S^{-1}H)t}\right]S^{-1}
\end{align*}
$$

For the LDOS that localized on each orbital (basis), the element is defined as the diagonal element of the DOS, which is:
$$
\begin{align*}
\rho(\epsilon)_{ii}&=\delta(\epsilon-S^{-1}H)_{ii}\\
&=\frac{1}{2\pi}\int dt \left[e^{i\epsilon t}e^{-i(S^{-1}H)t}\right]_{ii}
\end{align*}
$$

There are two method to compute this, one is using the fourier transform like TBPLaS, another is to compute using basic fast fourier transform.

# Real Space Function
$$
\begin{align*}
\rho(r) = \frac{1}{2\pi}\int dt \left[\left( \left(e^{-iS^{-1}Ht}\sqrt{f_F(S^{-1}H)}S^{-1/2}\vec{a}\right)\cdot \vec{\phi}(r)\right)^2\right]
\end{align*}
$$
Same for LDOS, where the square root of the fermi-dirac function are replaced by the delta function.
We should notice that, this formula, if written using operator style, is:
$$
\begin{align*}
\rho(r) = \frac{1}{2\pi}\int dt \left\langle r\left|\left(e^{-i\hat{H}t}\sqrt{f_F(\hat{H})}\right)\circ\sum_{n,m}\left(b_n|\Psi_n\rangle\langle\Psi_m|b_m^*\right)\circ\left(\sqrt{f_F(\hat{H})}e^{i\hat{H}t}\right)\right|r\right\rangle
\end{align*}
$$
Where $\sum_n b_n|\Psi_n\rangle=\sum_i\left(S^{-1/2}\vec{a}\right)_i|\phi_i\rangle$ where $b_n$ and $a_i$ are all mean zero and variance 1. Therefore, we are aware that this is merely a approximation of identity of random states:
$$
\begin{align*}
\mathrm{E}\left[\sum_{n,m}\left(b_n|\Psi_n\rangle\langle\Psi_m|b_m^*\right)\right]_b=\mathrm{E}\left[\sum_{i,j}\left(S^{-1/2}\vec{a}\right)_i|\phi_i\rangle\langle\phi_j|\left(S^{-1/2}\vec{a}\right)_j\right]_a=\hat{I}
\end{align*}
$$

Therefore, we can adopt different sampling and average technique to improve the approximation performance here. The resulting formula would be:
$$
\rho(r) = \frac{1}{2\pi S}\sum_{p=0}^{S-1}\int dt \left[\left( \left(e^{-iS^{-1}Ht}\sqrt{f_F(S^{-1}H)}S^{-1/2}\vec{a}^{(p)}\right)\cdot \vec{\phi}(r)\right)^2\right]
$$

# KPM
KPM is a method using chebyshev polynomial and the smooth kernel to approximate a function f(x):

$$
f(x)=\frac{1}{2}c_0T_0(x)+\sum_{k=1}^\infty c_kT_k(x)
$$
Where $c_k=\frac{2}{\pi}\int^1_{-1}{\frac{dx}{\sqrt{1-x^2}}f(x)T_k(x)}$.

For example, when computing the dos: 
The $1/\sqrt{1-x^2}$ can be moved out and we only need to evaluate the factor $\mu=\frac{2}{\pi}\int^1_{-1}dxf(x)T_k(x)$.
$$
\begin{aligned}
\rho(E)&=\frac{1}{N}\text{Tr}\left(\delta(E-\hat{H})\right)\\
&=\frac{1}{N}\sum_n \delta\left(E-E_n\right)
\end{aligned}
$$
The moment of which is evaluated using:
$$
\begin{aligned}
c_k&=\frac{2}{N\pi}\int_{-1}^{1}dE \sum_n\delta(E-E_n)T_k(E)\\
&=\frac{2}{N\pi}\sum_nT_k(E_n)\\
&=\frac{2}{N\pi}\text{Tr}\left(T_k(\hat{H})\right)
\end{aligned}
$$

The KPM is also able to approximate matrix function. For general matrix function $f(\hat{H})$, its scalar function is $f(x)$. Then, once the momente is known, we can evaluate the matrix function using:
$$f(\hat{H})=\frac{1}{2}c_0T_0(\hat{H})+\sum_{k=1}^\infty c_kT_k(\hat{H})$$
with:
$$c_k=\frac{2}{\pi}\int^1_{-1}{\frac{dx}{\sqrt{1-x^2}}f(x)T_k(x)}$$
To evaluate this more conveniently, we take $x=\text{cos}(\theta)$, and having $T_k(x)=\text{cos}(k\cdot\text{arccos(x)})$, we arrived:
$$
\begin{aligned}
c_k&=\frac{2}{\pi}\int^1_{-1}d\cos(\theta)\frac{1}{|\sin(\theta)|}f(\cos(\theta))\cos(k\theta)\\
&=\frac{2}{\pi}\int^\pi_{0}d\theta f(\cos(\theta))\cos(k\theta)
\end{aligned}
$$
The above factor can be approximated via the fast-fourier transform.
$$
f(E)=\frac{1}{1+e^{\beta (E-\mu)}}=\frac{1}{1+e^{\beta (w\tilde{E}+b-\mu)}}=\frac{1}{1+e^{w\beta((b-\mu)/w)}e^{w\beta (\tilde{E})}}
$$

# Velocity Op


# Optical Conductivity
TBPM one looks nice, we should first implement that

# Drude Conductivity
New FFT method looks great.

