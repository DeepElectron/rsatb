# RSATB Formula Handbook

## Electronic Structure with non-orthogonal basis
We are caring about the eigenvalue problem like:
$$
\hat{H}\psi=E\psi
$$
Where $\hat{H}$ composed only bilinear term with creation and annihilation operator.

Now defining the basis set $|i\rangle$, where $\langle i|j\rangle=S_{ij}$, so it is non-orthogonal. By writing the operator in the basis, we get a generalized eigenvalue problem:

$$
HC_n=E_nSC_n
$$

We can form another set of basis by writing:
$$
|\tilde{i}\rangle=\sum_k(S^{-1})_{ik}|k\rangle
$$
Using this as the left basis vector, we have:
$$
\langle\tilde{i}|j\rangle=\delta_{ij}
$$

By inserting the new dual basis in the left and use the original basis as right vector, we can get the form:
$$
\left(S^{-1}H\right)C_n=E_nC_n
$$

### Second Quantization
The above dual basis setting is good for a lot of properties calculation. However, some properties are derived using field theory, which requires the second quantized representation. Here we will derive the convention for that. 

Now, using this new set of left and right basis to define the second quantized operator:
$$
\langle r|\tilde{i}\rangle=\tilde{\phi}_i(r), \langle r|j\rangle=\phi_j(r)
$$
The field operator is defined by differentiating the left and right as:
$$
\begin{aligned}
\hat{\psi}(r)&=\sum_i\phi_i(r)\hat{c}_i,\\
\hat{\psi}^\dagger(r)&=\sum_i\phi^*_i(r)\hat{c}^\dagger_i=\sum_{ik}\phi^*_i(r)\left(S^{-1}\right)_{ik}\tilde{c}^\dagger_k=\sum_{k}\tilde{\phi}^*_k(r)\tilde{c}^\dagger_k
\end{aligned}
$$
Where the normal operator $\hat{c}^\dagger,\hat{c}$ attribute to orbital $\phi(r)$, $\tilde{c}^\dagger, \tilde{c}$ attribute to $\tilde{\phi}(r)$.

We can verify that this field operator follows the cananical algebra relations.
Now, time to expand the Hamiltonian. Considering only the non-interacting Hamiltonian $\hat{H}$, using the defined operator:
$$
\int drdr'\hat{\psi}^\dagger(r)\hat{H}\hat{\psi}(r')=\sum_{ij}H_{ij}\hat{c}_i^\dagger\hat{c}_j=\sum_{ij}(S^{-1}H)_{ij}\tilde{c}_i^\dagger\hat{c}_j
$$

Equivalent in the previous version, once the dual basis is chosen, the Hamiltonian is rewritten as $S^{-1}H$ under the new basis sets.

### Matrix Function
Here it is good to use the matrix form:
$$
\begin{align*}
HC_n&=E_nSC_n \equiv \\ 
S^{-1/2}HS^{-1/2}\cdot S^{1/2}C_n&=E_n S^{1/2}C_n
\end{align*}
$$
The literature often call $S^{1/2}C_n$ as the wave vector under **lowdin basis**.

If we are to compute the matrix function of the Hamiltonian $f(\hat{H})$ under the non-orthogonal basis, denoted as $F(H,S)$, we surely have:
$$
S^{-1/2}F(H,S)S^{-1/2}\cdot S^{1/2}C_n=f(E_n)S^{1/2}C_n
$$
We therefore knows that:
$$S^{-1/2}F(H,S)S^{-1/2}=f(S^{-1/2}HS^{-1/2})\equiv \\
F(H,S)=S^{1/2}f(S^{-1/2}HS^{-1/2})S^{1/2}=Sf(S^{-1}H)
$$
Taking the reduced density operator for example:
$$
\begin{align*}
% \hat{\rho}&=\sum_nf_F(E_n)|\Psi_n\rangle\langle\Psi_n|\\
\hat{\rho}&=f_F(\hat{H})\\
% &=\int d\epsilon\left[ f_F(\epsilon)\delta(\epsilon-\hat{H})\right]\\
% &=-\frac{1}{\pi}\int d\epsilon\left[ f_F(\epsilon)\Im[\hat{G^r}]\right]
\end{align*}
$$
Where $f_F$ is the fermi-dirac function. Using the result above, we have:
$$
\rho=Sf(S^{-1}H)
$$

### Composition of Operators
Sometimes, we need to evaluate the matrix element of composition of operators under non-orthogonal basis, for example $\hat{A}\hat{B}$, by inserting the resolution of identity: 
$$
\langle\phi_i|\hat{A}\hat{B}|\phi_j\rangle=AS^{-1}B
$$

### Trace
If we what to evaluate the trace of some matrix function, we should notice that, this is slightly different than computing the matrix itself, since:
$$
\begin{align*}
\text{Tr}\left[f(\hat{H})\right]&=\text{Tr}\left[f(S^{-1/2}HS^{-1/2})\right]\\
&=\text{Tr}\left[f(S^{-1}H)\right]
\end{align*}
$$
for the similarity transformation does not change the trace.

The reason behind it is, the trace defined in non-orthogonal operation is different than the orthogonal one - which directly sum up the diagonal elements. For convenience and consistency, when we want to evaluate the trace of operators or operators composition, we need to make sure they are all implemented in the dual basis. If an operator are written in the original basis as:
$$
A_{ij} = \langle\phi_i|\hat{A}|\phi_j\rangle
$$
Then 
$$
\text{Tr}\left[\hat{A}\right]=\text{Tr}\left[S^{-1}A\right]
$$

### Real Space Properties
We again use density matrix as an example. Since the density matrix is frequently used to reconstruct the charge density $\rho(r)$, therefore, we are often care with another form:
$$
\begin{align*}
\langle r|\hat{\rho}|r\rangle&=\sum_{n,i,j}f_F(E_n)C_{ni}C^*_{nj}\phi_i(r)\phi^*_j(r)\\
&=\sum_{ij}\left(S^{-1}\rho S^{-1}\right)_{ij}\phi_i(r)\phi^*_j(r)
\end{align*}
$$
Since we have:
$$
\begin{align*}
\rho_{ij}&=\sum_n f_F(E_n)\langle\phi_i|\Psi_n\rangle\langle\Psi_n|\phi_j\rangle\\
&=\sum_n f_F(E_n)S_{ik}C_{nk}C^*_{nl}S_{lj}
\end{align*}
$$
For convenience, we often compute $\tilde{\rho}=\left(S^{-1}\rho S^{-1}\right)$. This equals to:
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


## Linear Responce Theory
The conductivity is defined as the linear relation of the current and the electrostatic force:
$$
\mathbf{J}=\sigma \mathbf{f}=\sigma\left(\mathbf{E}+\mathbf{v}\times\mathbf{B}\right)
$$
This can be seen as a special case of the linear responce, where we assume that the system's responce to a small externel field is sufficiently linear. But it can be spatial and time resolved. It have some alike relations as:
$$j(r,t)=\int dr'\int_{-\infty}^t dt' \sigma(r-r',t-t')V(r', t')$$

### Kubo Formula
Considering a system under externel perturbation:
$$
\hat{H}+\hat{H}'=\hat{H}-\int d^3r\sum_ih_i(\mathbf{r},t)\hat{A}_i(\mathbf{r})
$$

If our goal is to observe $A_i(t)$, which is:
$$
\text{Tr}\left[\hat{\rho}(t)\hat{A}_i(\mathbf{r})\right]
$$

We can write the density operator as time independent part and perturbative correction: $$\hat{\rho}(t)=\hat{\rho}_{eq}+\delta\hat{\rho}(t)$$, using:
$$
i\frac{\partial \hat{\rho}(t)}{\partial t}=\left[\hat{H}+\hat{H}',\hat{\rho}(t)\right]
$$
We can arrive in this formula:
$$
\delta\hat{\rho}(t)=-i\int_{-\infty}^tdt'e^{-\hat{H}(t-t')}\left[\hat{H}',\hat{\rho}_{eq}\right]e^{\hat{H}(t-t')}
$$
Therefore the observable's expecation is:
$$
\begin{aligned}
\delta\langle\hat{A}_i(\mathbf{r})\rangle(t)&=\text{Tr}\left[\delta\hat{\rho}(t)\hat{A}_i(\mathbf{r})\right]\\
&=i\int_{-\infty}^tdt'\int d^3r'\sum_j\text{Tr}\left\{\hat{\rho}_{eq}\left[\hat{A}_i(\mathbf{r},t)\hat{A}_j(\mathbf{r}',t')\right]\right\}h_j(\mathbf{r}',t')\\
&=\int_{-\infty}^\infty dt'\int d^3r'\sum_j\mathcal{X}_{A_i,A_j}^R(\mathbf{r},\mathbf{r}',t,t')h_j(\mathbf{r}',t')
\end{aligned}
$$
Here $$\hat{A}_i(\mathbf{r},t)=e^{i\hat{H}t}\hat{A}_i(\mathbf{r})e^{-i\hat{H}t}$$

Here we have a general form of the **responce function**, or the general **susceptibility** defined as:
$$
\mathcal{X}_{A_i,A_j}^R(\mathbf{r},\mathbf{r}',t,t')=i\Theta(t-t')\left\langle\left[\hat{A}_i(\mathbf{r},t),\hat{A}_j(\mathbf{r}',t')\right]\right\rangle
$$

You can derive from the properties of the trace, the responce function is shift invariant in time, therefore:
$$
\mathcal{X}^R_{A_i,A_j}(\mathbf{r},\mathbf{r}',t-t')
$$


### Minimal Coupling
For a particle under QM definition:
$$
\hat{H}=\frac{\mathbf{p}^2}{2m}+V
$$
By incooperating the vector field:
$$
\begin{aligned}
\hat{H}(\mathbf{A})&=\frac{(p-q\mathbf{A})^2}{2m}+V\\
&=\frac{\mathbf{p}^2}{2m}+V-q\mathbf{v}\cdot\mathbf{A}+\frac{1}{2m}q^2\mathbf{A}\cdot\mathbf{A}
\end{aligned}
$$
We can derived the current operator by taking the derivative of $\hat{H}$ w.r.t. the $\mathbf{A}$. 

**NOTE**: There are very deep physics inside this definition, and why it works connected with the gauge theory of electrodynamics and the microscopic description (lagrangian) of the charged particles. Currently I haven't learn the electrodynamics and the gauge field theory, need to read more later.

$$
\mathbf{j}=-q\mathbf{v}+\frac{q^2}{m}\mathbf{A}
$$

Written is the kubo formula, the externel potential will be:
$$
\hat{H}_{ext}=\mathbf{j}\cdot\mathbf{A}
$$

The kubo formula becomes:
$$
\begin{aligned}
\delta\langle j_\alpha\rangle(t)&=\sum_\beta\int_{-\infty}^\infty dt'\mathcal{X}_{\alpha\beta}^R(t-t')A_\beta(t')\\
\mathcal{X}_{\alpha\beta}^R(t)&=i\Theta(t)\left\langle\left[j_\alpha(t), j_\beta(0)\right]\right\rangle
\end{aligned}
$$

The conductivity is derived by transforming the above equation into fourier space, and recalls that $\mathbf{j}=\mathbf{\sigma}\mathbf{E}$, and using $\mathbf{E}(\omega)=i\omega\mathbf{A}(\omega)$, $j=-vq$ we will have a form of $\sigma$:
$$
\sigma_{\alpha\beta}(\omega)=\frac{q^2}{\omega}\int_0^\infty\text{Tr}\left\{\hat{\rho}_{eq}\left[v_\alpha(t),v_\beta(0)\right]\right\}e^{-i(\omega+i\eta)t} dt
$$

### Optical Conductivity
Further simplification can be achieved using Wick theorem and with non-interacting particles assumption:

The commutator can be written as thermal average:
$$\langle v_\alpha(t)v_\beta(0)\rangle-\langle v_\beta(0)v_\alpha(t)\rangle$$
Considering the first term:
$$
\langle v_\alpha(t)v_\beta(0)\rangle=[v_\alpha(t)]_{ij}[v_\beta(0)]_{kl}\langle c^\dagger_i c_jc^\dagger_k c_l\rangle\\
\langle c^\dagger_i c_jc^\dagger_k c_l\rangle=\langle c^\dagger_i c_j\rangle\langle c^\dagger_k c_l\rangle+\langle c^\dagger_i c_l\rangle\langle c_j c^\dagger_k\rangle=F_{ij}F_{kl}+F_{il}(\delta_{jk}-F_{jk})
$$

So:
$$
\langle v_\alpha(t)v_\beta(0)\rangle-\langle v_\beta(0)v_\alpha(t)\rangle=(1-e^{-\beta \hbar\omega})\sum_{ijkl}F_{il}[v_\alpha(t)]_{ij}(\delta_{jk}-F_{jk})[v_\beta(0)]_{kl}=(1-e^{-\beta \hbar\omega})\text{Tr}\left[F^Tv_\alpha(t)(I-F)v_\beta\right]
$$
The factor comes from the **KMS** condition. So optical conductivity can be written as:

$$
\sigma_{\alpha\beta}(\omega)=\frac{q^2(1-e^{-\beta \hbar\omega})}{\omega}\int_0^\infty\text{Tr}\left[F^Tv_\alpha(t)(I-F)v_\beta\right]e^{-i(\omega+i\eta)t}dt
$$

Written in non-orthogonal basis, since it requires Wick theorem and second quantized format, we need to strictly use the dual basis derived in the begining.

The main target is the same, evaluting the correlator:
$$
\langle v_\alpha(t)v_\beta(0)\rangle
$$
Here:
$$
\begin{aligned}
v_\alpha(t)&=e^{i\hat{H}t}i\left[\hat{H},\hat{r}_\alpha\right]e^{-i\hat{H}t}\\
v_\alpha&=i\left[\hat{H},\hat{r}_\alpha\right]
\end{aligned}
$$

Expressed in the dual basis:
$$
\begin{aligned}
v_\alpha(t)&=ie^{iS^{-1}Ht}S^{-1}\left[H,r_\alpha\right]e^{-iS^{-1}Ht}\\
v_\alpha&=iS^{-1}\left[H,r_\alpha\right]
\end{aligned}
$$
With $H, S, r_\alpha$ as known matrix written in the original basis.

So the correlator becomes:
$$
\langle v_\alpha(t)v_\beta(0)\rangle=[v_\alpha(t)]_{ij}[v_\beta(0)]_{kl}\langle \tilde{c}^\dagger_i \hat{c}_j\tilde{c}^\dagger_k \hat{c}_l\rangle\\
\langle \tilde{c}^\dagger_i \hat{c}_j\tilde{c}^\dagger_k \hat{c}_l\rangle=\langle \tilde{c}^\dagger_i \hat{c}_j\rangle\langle \tilde{c}^\dagger_k \hat{c}_l\rangle+\langle \tilde{c}^\dagger_i \hat{c}_l\rangle\langle \hat{c}_j \tilde{c}^\dagger_k\rangle=F_{ij}F_{kl}+F_{il}(\delta_{jk}-F_{jk})
$$
But here, $F=f_F(S^{-1}H)$.

## Drude Conductivity
New FFT method looks great.

