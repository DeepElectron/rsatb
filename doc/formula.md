# Formula handbook

## Operator and Matrix Function under Non-orthogonal basis
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

## FFT for Fourier Transform

## DOS
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

## LDOS
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

## Real Space Function
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

## KPM
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

## Two-Center Integral
For evaluation of quantities such as the velocity operator, we often need to evaluate integrals element such as:
$$
\langle\phi_{lm}|\hat{O}|\phi_{l'm'}\rangle
$$
We ignore the shell index $n$ for simplicity. Here, $\hat{O}$ can be $I$, $-\frac{1}{2}\nabla^2$, $\hat{x}/\hat{y}/\hat{z}$ or $\frac{\partial}{\partial x}/\frac{\partial}{\partial y}/\frac{\partial}{\partial z}$. Gives the matrix element for overlap, kinetic, position and velocity operator respectively. For the latter case, we will discuss them later.

For each orbital, we represent them as a product of radial function and a spherical harmonics:
$$
\phi_{lm}(\vec{r})=\phi_{l}(r)Y_{lm}(\hat{r})
$$

### Spherical Bessel transform
The atomic orbital like function's k space representation are related to the r space one by the spherical bessel transformation.
$$
\begin{aligned}
\phi_{lm}(\vec{k})&=\frac{1}{(2\pi)^{3/2}}\int\phi_{lm}(\vec{r})e^{-i\vec{k}\cdot\vec{r}}d\vec{r}\\
&=\sum_{l'}\sum_{m=-l'}^{l'}\frac{4\pi}{(2\pi)^{3/2}}\int\phi_{lm}(\vec{r})(-i)^{l'} j_{l'}(kr)Y_{l'm}(\hat{k})Y^*_{l'm}(\hat{r})d\vec{r}\\
&=\sum_{l'}\sum_{m=-l'}^{l'}\left[(-i)^{l'}\sqrt{\frac{2}{\pi}}\int\phi_{l}(r)Y_{lm}(\hat{r})j_{l'}(kr)Y^*_{l'm}(\hat{r})d\vec{r}\right]Y_{l'm}(\hat{k})\\
&=\sum_{l}\sum_{m=-l}^l\left[(-i)^l\sqrt{\frac{2}{\pi}}\int\phi_{l}(r)\left(\int Y_{lm}(\hat{r})Y^*_{l'm}(\hat{r})\sin(\theta)d\theta d\gamma\right) j_{l'}(kr) r^2dr\right]Y_{l'm}(\hat{k})\\
&=\left[(-i)^l\sqrt{\frac{2}{\pi}}\int\phi_{l}(r)j_{l}(kr) r^2dr\right]Y_{lm}(\hat{k})\\
\end{aligned}
$$
We define 
$$\phi_{l}(k)=(-i)^l\sqrt{\frac{2}{\pi}}\int\phi_{l}(r)j_{l}(kr) r^2dr$$
Therefore, the k space atomic orbital basis can be written just like the real space conterpart:
$$
\phi_{lm}(\vec{k})=\phi_{l}(k)Y_{lm}(\hat{k})
$$

### Computing Overlap and Kinetic
Taking the overlap at first, The integral can be rewritten as:
$$
\begin{align*}
\langle\phi_{lm}(\vec{r}-\vec{R})|\hat{r}|\phi_{l'm'}\left(\vec{r}-\vec{R}'\right)\rangle&=\frac{1}{(2\pi)^{3/2}}\int\phi_{lm}^*\left(\vec{r}-\vec{R}\right)\phi_{l'm'}\left(\vec{r}-\vec{R}'\right)\vec{r}d\vec{r}\\
&=\frac{1}{(2\pi)^{3/2}}\int\phi^*_{lm}\left(\vec{r}\right)\phi_{l'm'}\left(\vec{r}-\left(\vec{R}'-\vec{R}\right)\right)\left(\vec{r}-\left(\vec{R}'-\vec{R}\right)\right)d\vec{r}\\&+\frac{\vec{R}'}{(2\pi)^{3/2}}\int\phi^*_{lm}\left(\vec{r}\right)\phi_{l'm'}\left(\vec{r}-\left(\vec{R}'-\vec{R}\right)\right)d\vec{r}
\end{align*}
$$

$$
\begin{align*}
\langle\phi_{lm}(\vec{r})|\phi_{l'm'}(\vec{r}-\vec{R})\rangle&=\frac{1}{(2\pi)^{3/2}}\int \phi^*_{lm}(\vec{r})\phi_{l'm'}(\vec{r}-\vec{R})d\vec{r}\\
&=\frac{1}{(2\pi)^{3/2}}\int\left(\frac{1}{(2\pi)^{3/2}}\int\phi_{lm}^*(\vec{k})e^{-i\vec{k}\cdot\vec{r}}d\vec{k}\right)\left(\frac{1}{(2\pi)^{3/2}}\int\phi_{l'm'}(\vec{k'})e^{i\vec{k'}\cdot\left(\vec{r}-\vec{R}\right)}d\vec{k}\right)d\vec{r}\\
&=\frac{1}{(2\pi)^{3/2}}\int\left(\frac{1}{(2\pi)^{3}}\int\int\phi_{lm}^*(\vec{k})\phi_{l'm'}(\vec{k'})e^{i(\vec{k'}-\vec{k})\cdot\vec{r}}e^{-i\vec{k'}\cdot\vec{R}}d\vec{k}d\vec{k'}\right)d\vec{r}\\
&=\frac{1}{(2\pi)^{3/2}}\left(\int\int\phi_{lm}^*(\vec{k})\phi_{l'm'}(\vec{k'})\delta(\vec{k'}-\vec{k})e^{-i\vec{k'}\cdot\vec{R}}d\vec{k}d\vec{k'}\right)\\
&=\frac{1}{(2\pi)^{3/2}}\left(\int\phi_{lm}^*(\vec{k})\phi_{l'm'}(\vec{k})e^{-i\vec{k}\cdot\vec{R}}d\vec{k}\right)\\
&=\frac{1}{(2\pi)^{3/2}}\int\phi^*_{l}(k)\phi_{l'}(k)\left(\int Y^*_{lm}(\hat{k})Y_{l'm'}(\hat{k})e^{-i\vec{k}\cdot\vec{R}}d\hat{k}\right)k^2dk
\end{align*}
$$
using relation:
$$
e^{-i\vec{k}\cdot\vec{r}}=\sum_{l=0}^\infty\sum_{m=-1}^l4\pi(-i)^lj_l(kr)Y_{lm}(\hat{k})Y^*_{lm}(\hat{r})
$$
We have:
$$
\begin{align*}
\langle\phi_{lm}(\vec{r})|\phi_{l'm'}(\vec{r}-\vec{R})\rangle&=\sum_{l''=0}^\infty\sum_{m''=-1}^{l''}\left[(-i)^{l''}\sqrt{\frac{2}{\pi}}\int\phi^*_{l}(k)\phi_{l'}(k)j_{l''}(kR)k^2dk\right]\cdot\\&\left(\int Y^*_{lm}(\hat{k})Y_{l'm'}(\hat{k})Y_{l''m''}(\hat{k})d\hat{k}\right)Y^*_{l''m''}(\hat{R})\\
&=\sum_{l''=0}^\infty\sum_{m''=-1}^{l''}S_{ll'l''}G_{lm,l'm'}^{l''m''}Y^*_{l''m''}(\hat{R})
\end{align*}
$$
Here $G_{lm,l'm'}^{l''m''}$ is clebsh gordan coefficients, is a constant value. It can be tabulated in the beginning. So we care about is the first $S_{ll'l''}$:
$$
S_{ll'l''}=(-i)^l\sqrt{\frac{2}{\pi}}\int\phi^*_{l}(k)\phi_{l'}(k)j_{l''}(kR)k^2dk
$$
This is again a sperical bessel transformation of the function $\phi^*_{l}(k)\phi_{l'}(k)$.
So what we need is to tabulate $\phi_l^*(k)\phi_{l'}(k)$, which can be calculated from a spherical transformation on $\phi_l^*(r)\phi_{l'}(r)$.

For computing the kinetic operator matrix elements, the only difference is the factor on $k^2$, which changed to $k^4$.

### Spherical Bessel Tranformation from FFT
#### a. SBT and Fourier Transform


## Velocity Op
The velocity operator is represented as $$\hat{v}=i\left[\hat{H},\hat{r}\right]$$
If no magnetic field exist, then it is equivalenet with the momentum operator $$\hat{p}=-i\nabla$$
For the second representation, we need to compute the derivative of the basis w.r.t. the electron coordinates. This is non-trival. We first look at the commutator implementation and leave the second method in the future.

Written in the LCAO basis, the matrix representation of $\hat{r}$ is:
$$
\hat{v}=i\left(HS^{-1}r-rS^{-1}H\right)
$$
So we need to compute matrix element of $r_{lml'm'}=\langle\phi_{lm}(\vec{r}-\vec{R})|\hat{r}|\phi_{l'm'}(\vec{r}-\vec{R}')\rangle$. This can be achived by seperating the $\hat{r}$ into atomic centered component and the remaining parts:
$$
\begin{align*}
r_{lml'm'}&=\langle\phi_{lm}(\vec{r}-\vec{R})|\hat{r}-\vec{R}+\vec{R}|\phi_{l'm'}(\vec{r}-\vec{R}')\rangle\\
&=\langle\phi_{lm}(\vec{r})|\hat{r}|\phi_{l'm'}(\vec{r}-\vec{R}'+\vec{R})\rangle+\vec{R}\langle\phi_{lm}(\vec{r})|\phi_{l'm'}(\vec{r}-\vec{R}'+\vec{R})\rangle
\end{align*}
$$
For simplicity, we just use a single $\delta\vec{R}$ denote the shift between two site, and write this as:
$$\
r_{lml'm'}=\langle\phi_{lm}(\vec{r})|\hat{r}|\phi_{l'm'}(\vec{r}-\delta\vec{R})\rangle+\vec{R}S_{lm,l'm'}
$$
Our main task is to compute the first term. 

### Composition of spherical hermonics
The computation of $r$ matrix is achieved in two center integral framework, by representing $\phi_{lm}(\vec{r})\hat{r}$ as a whole, which is a modified atomic orbital type function.

Takeing $x$ component as an example:
$$
x=r\cdot x/r=rY_{1,1}(\hat{r})
$$
Therefore:
$$
\phi_{lm}(\vec{r})x=r\phi_l(r)\cdot Y_{lm}(\hat{r})Y_{11}(\hat{r})
$$
(**NOTE:** from now on, we use $Y_{lm}$ for real spherical harmonics while $Y_{l}^{m} for its complex conterpart$.) 

Composition of two function with angular momentum can be decomposed as a linear combination of funtions with single angular momentum index, generally we have:
$$
\begin{align*}
[\phi_{lm}(\vec{r})\vec{r}]_{m'}&=r\phi_{l}(r)\cdot Y_{lm}(\hat{r})Y_{1m'}(\hat{r})\\
&=r\phi_{l}(r)\cdot\sum_{l''m''}G_{lm,1m'}^{l''m''}Y_{l''m''}(\hat{r})
\end{align*}
$$
(**NOTE:** This is slighly different from wigner-eckart theorem that compose functions with angular and magnetic momentum index.)

Where $G_{lm,1m'}^{l''m''}$ is the Gaunt coefficients, expressed as:
$$
G_{lm,l'm'}^{l''m''}=\sqrt{\frac{(2l+1)(2l'+1)(2l'')+1}{4\pi}}
\begin{pmatrix}
l & l' & l''\\
0 & 0 & 0
\end{pmatrix}
\begin{pmatrix}
l & l' & l''\\
m & m' & m''
\end{pmatrix}
$$
Where the 2x3 matrix states the wigner 3-j symbol. The $\begin{pmatrix}
l & 1 & l''\\
0 & 0 & 0
\end{pmatrix}$ is non-zero only when $l''=l-1 | l+1$. 

Now, the whole problem switch to the study of composition of real spherical harmonics:
$$
Y_{lm}(\hat{r})Y_{1m'}(\hat{r})=\sum_{m''}G_{lm,1m'}^{l+1m''}Y_{l+1m''}(\hat{r})+\sum_{m''}G_{lm,1m'}^{l-1m''}Y_{l-1m''}(\hat{r})
$$

Our next task is to find out in the gaunt table for real spherical harmonics, what value are non-zero. This deside how many $m''$ we need to sum up. We will use the gaunt table's properties in complex spherical harmonics case, where $\tilde{G}_{lm,1m'}^{l+1m''}$ is non-zero only when $m+m'=m''$.

### For $z/r$, which equiv to $m'=0$
When $m<0$:
$$
\begin{align*}
Y_{lm}(\hat{r})Y_{10}(\hat{r})&=\frac{i}{\sqrt{2}}\left(Y_l^m-(-1)^mY_{l}^{-m}\right)Y_{1}^0\\
&=\frac{i}{\sqrt{2}}\left(\tilde{C}_{lm,10}^{l+1,m}Y_{l+1}^m+\tilde{C}_{lm,10}^{l-1,m}Y_{l-1}^m-(-1)^m\tilde{C}_{l,-m,10}^{l+1,-m}Y_{l+1}^{-m}-(-1)^m\tilde{C}_{l,-m,10}^{l-1,-m}Y_{l-1}^{-m}\right)\\
&=\frac{i}{\sqrt{2}}\left(\tilde{C}_{lm,10}^{l+1,m}Y_{l+1}^m+\tilde{C}_{lm,10}^{l-1,m}Y_{l-1}^m-(-1)^{m+2l+2}\tilde{C}_{l,m,10}^{l+1,m}Y_{l+1}^{-m}-(-1)^{m+2l-2}\tilde{C}_{l,m,10}^{l-1,m}Y_{l-1}^{-m}\right)\\
&=\frac{i}{\sqrt{2}}\left(\tilde{C}_{lm,10}^{l+1,m}Y_{l+1}^m+\tilde{C}_{lm,10}^{l-1,m}Y_{l-1}^m-(-1)^{m}\tilde{C}_{l,m,10}^{l+1,m}Y_{l+1}^{-m}-(-1)^{m}\tilde{C}_{l,m,10}^{l-1,m}Y_{l-1}^{-m}\right)\\
&=\tilde{C}_{lm,10}^{l+1,m}Y_{l+1,m}(\hat{r})+\tilde{C}_{lm,10}^{l-1,m}Y_{l-1,m}(\hat{r})
\end{align*}
$$
Here we used:
$$
\tilde{C}_{lm,l'm'}^{l''m''}=(-1)^{l+l'+l''}\tilde{C}_{l-m,l'-m'}^{l''-m''}
$$
Same story for $m\geq 0$. Therefore, we conclude that for z coupling $Y_{lm}$, the resulting function can be decompose into linear combination of only $Y_{l+1,m}$ and $Y_{l-1,m}$.

### For $x/r$, which equiv to $m'=1$
$$
\begin{align*}
Y_{lm}(\hat{r})Y_{11}(\hat{r})=\tilde{C}_{lm,11}^{l+1,m+1}Y_{l+1,m+1}(\hat{r})+\tilde{C}_{lm,11}^{l+1,m-1}Y_{l+1,m-1}(\hat{r})\\+\tilde{C}_{lm,11}^{l-1,m+1}Y_{l-1,m+1}(\hat{r})+\tilde{C}_{lm,11}^{l-1,m-1}Y_{l-1,m-1}(\hat{r})
\end{align*}
$$

$$
\begin{align*}
\langle\phi_{lm}(\vec{r})|\hat{x}|\phi_{l'm'}(\vec{r}-\delta\vec{R})\rangle&=\int r\phi_{l}(r)\cdot Y_{lm}(\hat{r})Y_{11}(\hat{r})\phi_{l'm'}(\vec{r}-\delta\vec{R}) d\vec{r}\\
&=\int r\phi_{l}(r)\cdot [\tilde{C}_{lm,11}^{l+1,m+1}Y_{l+1,m+1}(\hat{r})+\tilde{C}_{lm,11}^{l+1,m-1}Y_{l+1,m-1}(\hat{r})\\&+\tilde{C}_{lm,11}^{l-1,m+1}Y_{l-1,m+1}(\hat{r})+\tilde{C}_{lm,11}^{l-1,m-1}Y_{l-1,m-1}(\hat{r})]\phi_{l'm'}(\vec{r}-\delta\vec{R}) d\vec{r}\\
&=
\end{align*}
$$

### For $y/r$, which equiv to $m'=-1$
$$
\begin{align*}
Y_{lm}(\hat{r})Y_{1-1}(\hat{r})=\tilde{C}_{lm,1-1}^{l+1,-m+1}Y_{l+1,-m+1}(\hat{r})+\tilde{C}_{lm,1-1}^{l+1,-m-1}Y_{l+1,-m-1}(\hat{r})\\+\tilde{C}_{lm,1-1}^{l-1,-m+1}Y_{l-1,-m+1}(\hat{r})+\tilde{C}_{lm,1-1}^{l-1,-m-1}Y_{l-1,-m-1}(\hat{r})
\end{align*}
$$


<!-- $$
\begin{align*}
\delta(m_1+1,M), m_1\in(-l,l),M\in(-l+1,l-1)/\in(-l,l)/\in(-l-1,l+1)
\end{align*}
$$
(0,1,2), (0),(-1,0,1),(-2,-1,0,1,2)
(-2,-1,0) -->

# Optical Conductivity
TBPM one looks nice, we should first implement that

# Drude Conductivity
New FFT method looks great.

