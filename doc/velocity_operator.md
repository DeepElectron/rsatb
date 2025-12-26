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
<!-- $$
\begin{align*}
\langle\phi_{lm}(\vec{r}-\vec{R})|\hat{r}|\phi_{l'm'}\left(\vec{r}-\vec{R}'\right)\rangle&=\frac{1}{(2\pi)^{3/2}}\int\phi_{lm}^*\left(\vec{r}-\vec{R}\right)\phi_{l'm'}\left(\vec{r}-\vec{R}'\right)\vec{r}d\vec{r}\\
&=\frac{1}{(2\pi)^{3/2}}\int\phi^*_{lm}\left(\vec{r}\right)\phi_{l'm'}\left(\vec{r}-\left(\vec{R}'-\vec{R}\right)\right)\left(\vec{r}-\left(\vec{R}'-\vec{R}\right)\right)d\vec{r}\\&+\frac{\vec{R}'}{(2\pi)^{3/2}}\int\phi^*_{lm}\left(\vec{r}\right)\phi_{l'm'}\left(\vec{r}-\left(\vec{R}'-\vec{R}\right)\right)d\vec{r}
\end{align*} -->
<!-- $$ -->

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

<!-- ### Spherical Bessel Tranformation from FFT
#### a. SBT and Fourier Transform -->


## Velocity Op
The velocity operator is represented as $$
\begin{align*}
\hat{v}&=i\left[\hat{H},\hat{r}\right]\\
&=i\left[\frac{\hat{p}^2}{2m},\hat{r}\right]+i\left[\hat{V}_{\text{eff}},\hat{r}\right]\\
&=\frac{i}{2m}\left\{\hat{p}\left[\hat{p},\hat{r}\right]+\left[\hat{p},\hat{r}\right]\hat{p}\right\}+i\left[\hat{V}_{\text{nl}},\hat{r}\right]\\
&=\frac{\hat{p}}{m}+i\left[\hat{V}_{\text{nl}},\hat{r}\right]
\end{align*}
$$
If no magnetic field exist, and the potential are purely local ($\hat{V}_{\text{nl}}=0$), then it is equivalenet with the momentum operator $$\frac{\hat{p}}{m}=-\frac{i}{m}\nabla_r$$

Here we use $r$ to represent the electron coordinates, and $R$ for core coordinates.

However, in DFT, there are non-local potential. Therefore the velocity operator is a bit non-trivial. We first look at the commutator implementation and talk about the other method later.

Written in the LCAO basis, the matrix representation of $\hat{v}$ is:
$$
v=i\left(HS^{-1}r-rS^{-1}H\right)
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
Our main task is to compute the first term. It seems both the first and second term are computable. Indeed, people use this method to compute position operator matrix in DFT code. However, it is not applicable to comptue the velocity operator, since we will see the second term:
$$
\vec{R}S_{lm,l'm'}
$$
will not be bounded. Despite the $S_{lm,l'm'}$ is bounded by the radial distance, the $\vec{R}$ is not. However, we are still good to check how to comptue the position matrix to familiarize the technique used here.

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
(**NOTE:** from now on, we use $Y_{lm}$ for real spherical harmonics while $Y_{l}^{m}$ for its complex conterpart.) 

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
&=\int r\phi_{l}(r)\cdot [\tilde{C}_{lm,11}^{l+1,m+1}Y_{l+1,m+1}(\hat{r})+\tilde{C}_{lm,11}^{l+1,m-1}Y_{l+1,m-1}(\hat{r})\\&+\tilde{C}_{lm,11}^{l-1,m+1}Y_{l-1,m+1}(\hat{r})+\tilde{C}_{lm,11}^{l-1,m-1}Y_{l-1,m-1}(\hat{r})]\phi_{l'm'}(\vec{r}-\delta\vec{R}) d\vec{r}
\end{align*}
$$

### For $y/r$, which equiv to $m'=-1$
$$
\begin{align*}
Y_{lm}(\hat{r})Y_{1-1}(\hat{r})=\tilde{C}_{lm,1-1}^{l+1,-m+1}Y_{l+1,-m+1}(\hat{r})+\tilde{C}_{lm,1-1}^{l+1,-m-1}Y_{l+1,-m-1}(\hat{r})\\+\tilde{C}_{lm,1-1}^{l-1,-m+1}Y_{l-1,-m+1}(\hat{r})+\tilde{C}_{lm,1-1}^{l-1,-m-1}Y_{l-1,-m-1}(\hat{r})
\end{align*}
$$

## Velocity Matrix
Time to compute the velocity operator matrix:
$$
\hat{v}=-i\frac{\nabla_r}{m}+i\left[\hat{V}_{\text{nl}},\hat{r}\right]
$$
The first term can be evaluated using the relation $\nabla_r=-\nabla_R$. So the two center integrals for kinetical operator is sufficient for this. The complexity comes with the second term.

In general, the non-local term in the DFT potential comes from the pseudopotential files:
$$
\hat{V}_{\text{nl}}=\sum_k\hat{v}_{k,\text{nl}}=\sum_{k,n,m}|\beta_{k,n}\rangle D_{n,m}\langle\beta_{k,m}|
$$

Here the format of pseudopotential comes from "Efficacious Form for Model Pseudopotentials" by Leonard Kleinman and D. M. Bylander, 1982. We only talk about how to estimate its matrix element. For matrix element of $\hat{V}_{nl}$, we need to compute:
$$
\sum_{k,n,m}\langle\phi_{i}|\beta_{k,n}\rangle D_{k,n,m}\langle\beta_{k,m}|\phi_j\rangle
$$
Here we only need to estimate all the integrals:
$$
\langle\phi_i|\beta_{k,n}\rangle
$$
once, and use it to compute the result. To include all degree of freedom in the index, the formal way to express it is:
$$
\langle\phi_{i,nlm}(r-R_i)|\beta_{k,l'}(r-R_k)\rangle
$$
The integration is saved in a unordered_map for quick query. This step is also perfectly parallelizable. We should notice in the summation over neighbour k, the value count only for k that is both the neighbour of i and j. Therefore, the information is known in each atom segments under a row-wise adjacent matrix parition.

For the velocity operator, the vital part is the computation of the elements of the commutator:
$$
\begin{align*}
\langle\phi_{i}|\left[\hat{V}_{\text{nl}},\hat{r}\right]|\phi_j\rangle&=\sum_{k,n,m}\langle\phi_{i}|\beta_{k,n}\rangle D_{k,n,m}\langle\beta_{k,m}|\hat{r}|\phi_j\rangle-\sum_{k,n,m}\langle\phi_i|\hat{r}|\beta_{k,n}\rangle D_{k,n,m}\langle\beta_{k,m}|\phi_j\rangle\\
&=\sum_{k,n,m}D_{k,n,m}\left(\langle\phi_{i}|\beta_{k,n}\rangle\langle\beta_{k,m}|\hat{r}|\phi_j\rangle-\langle\phi_i|\hat{r}|\beta_{k,n}\rangle\langle\beta_{k,m}|\phi_j\rangle\right)
\end{align*}
$$

Two implementation strategy in the MPI setting.
During the summation loops, we are computing:
i,0 - k,R - j,R', sums over i,k,R,j,R'. However, it is also good to loop over: i,-R - k,0, to compute the first integral <i0,kR>, and then: compute <kR|jR'> from <k0|j(R'-R)>. Which means, we are good to loop over k for R=0. We should notice that k usually have very short cutoff.
1. minimize the matrix assemble communication. Each time, we loop over k in each process. k should follows the node partition in each atom