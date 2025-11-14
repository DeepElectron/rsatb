# Linear Scaling Properties with differentiable random states.

This is a note to derive the linear-scaling methods for non-orthogonal properties calculation software that scales up to tens to hundreds of millions of orbitals.

## 1. Objective and Motivation
**Obj**: A efficient Linear Scaling solver that is able to compute properties of large and sparse Hamiltonian with Overlap.

**Motivation**: Machine Learning Hamiltonian method can predict Hamiltonian of DFT level accruacy of millions of atoms, but currently, it is only applyed on tens of thousands of atoms. Because there is no property solver available to compute properties of large Hamiltonian under non-orthogonal basis. Meanwhile, a efficient linear-scaling calculator for some key properties: Green's Function, Density matrix, would enables easy integration of Linear Scaling electronic structure softwares such as Linear-Scaling DFT/NEGF. It will become a foundational packages for the future.

## 2. Theoretical Foundations
### 2.1 Random Trace evaluation

Based on paper [1], we know that to evaluate the trace of a matrix is written as:
$$
\text{Tr}(A)=\sum_{n=1}^D\langle n|A|n\rangle
$$
Where $|n\rangle$ is a complete and orthogonal basis set in the $D$ dimensional space (not necessarily, as long as it covers the space spanned by the eigenvector of A). Using the $|n\rangle$ as basis vector, we can sample the random states by:
$$
|\psi\rangle=\sum_n c_n|n\rangle
$$
The trace can be evaluated via:
$$
\begin{align*}
\frac{1}{S}\sum_{p=1}^S\langle\psi_p|A|\psi_p\rangle&=\frac{1}{S}\sum_{p=1}^S\sum_{n,m}\langle n|A|m\rangle c_n^*c_m\\
&=\frac{1}{S}\sum_{p=1}\sum_{n,m}^SA_{nm}c_n^*c_m
\end{align*}
$$
Taking the limit of $S\to\infty$, if $c_n$ is sampled independently, then:
$$
\begin{align*}
\lim_{S\to\infty}\frac{1}{S}\sum_{p=1}^SA_{nm}c_n^*c_m&=\mathbb{E}{\left[\sum_{n,m}A_{nm}c_n^*c_m\right]}\\
&=\sum_{n,m}A_{nm}\mathbb{E}{\left[c_n^*c_m\right]}\\
&=\sum_{n,m}A_{nm}\mathbb{E}{\left[c_n^2\right]}\delta_{n,m}\\
&=\sum_{n}A_{n,n}\mathbb{E}{\left[c_n^2\right]}
\end{align*}
$$
By dividing the expectation value, or just sampled from $\mathcal{N}(0,1)$, this formula is able to approximate the trace.

**Remark:** This formula actually is general for symmetric(hermitian) or non-symmetric(non-hermitian) matrix.

#### When Linear Scaling?
We can see above, the main computation complexity is the matrix-vector product operation (MVP) 
$$A|\psi_p\rangle$$

When $A$ is sparse under the selected basis $|n\rangle$, whose element number scales as $\mathbb{O}(n)$, the computational time also scales as $\mathbb{O}(n)$. Then, the total time will be $\mathbb{O}(Sn)$. Normally, $S$ is a constant, dependent on the required accuracy. Therefore, we have a linear scaled cost in total.

<!-- ### Non-orthogonal Basis
However, things become different if we consider a set of complete but non-orthogonal basis. -->

### 2.2 Tight-Binding Propogation Methods
Taking Density of States (DOS) computation as an example:
$$
\begin{align*}
D(E)&=\frac{1}{S}\sum_{p=1}^S\frac{1}{2\pi}\int^\infty_{-\infty}e^{iEt}\langle\psi_0|e^{-i\hat{H}t}|\psi_0\rangle dt\\
&=\frac{1}{2\pi}\int^\infty_{-\infty}e^{iEt}\left[\frac{1}{S}\sum_{p=1}^S\langle\psi_0|e^{-i\hat{H}t}|\psi_0\rangle\right] dt
\end{align*}
$$
Here $|\psi_0\rangle$ are random vector drawn from distribution $\mathcal{N}(0,1)$. Taking limit $S\to\infty$, we have:
$$
\lim_{S\to\infty}\frac{1}{S}\sum_{p=1}^S\langle\psi_0|e^{-iHt}|\psi_0\rangle=\text{Tr}(e^{-i\hat{H}t}) 
$$

Therefore:
$$
\begin{align*}
D(E)&=\frac{1}{2\pi}\int^\infty_{-\infty}e^{iEt}\text{Tr}(e^{-i\hat{H}t}) dt\\
&=\frac{1}{2\pi}\sum_n\int^\infty_{-\infty}e^{-i(E_n-E)t} dt\\
&=\sum_n\delta(E_n-E)
\end{align*}
$$

This gives exactly the DOS. The most important part of the formula is to transform the Trace evaluation: $\text{Tr}\left[e^{-i\hat{H}t}\right]$ computation into a sampling and summation procedure. This can be achieved combining Chebysheve Polynomials expansion of the $e^{-ixt}$ function, and the random states trace evaluation.

The random states trace evaluation formula:
$$
\text{Tr}(A) = \sum_{n=1}^D\langle n|A|n\rangle
$$
This can be generalized to the trace of matrix function evaluation:
$$
\text{Tr}(f(A)) = \sum_{n=1}^D\langle n|f(A)|n\rangle
$$
When function $f(x)$ is bounded and smooth, and $A$ is a sparse matrix, the Chebsheve Polynomials can give a very good approximation, with Linear cost.

### 2.3 Combine with Automatic differentiation

Using the properties of trace evaluation, this formula can also calculate the derivative of the matrix function $f'(A)$, via:
$$
\begin{align*}
f'(A)=\frac{\partial}{\partial A}\text{Tr}(f(A))=\frac{\partial}{\partial A}\left(\sum_{n=1}^D\langle n|f(A)|n\rangle\right)
\end{align*}
$$
The derivative can be efficiently evaluated via automatic differentiation. A typical examples are the evaluation of the equilibrium density matrix [2]:
$$
\begin{align*}
f(A)=-\beta^{-1}\ln\left[1+e^{-\beta(A-\mu)}\right],\\
f'(A) = \frac{1}{1+e^{-\beta(A-\mu)}}=\rho(A)
\end{align*}
$$

Sometimes, the function $f(A)$ is not smooth or bounded, therefore, the Chebysheve approximation would be very hard to converge or apply. However, when the function f has some other scalar variables, for example, the energy: $f(A,E)$. If this function, after Fourier Transform $\tilde{f}(A,t)=\mathcal{F}\circ f(A,E)$ is smooth and bounded, we can then evaluate the trace in the time domain, and then transform it back to the spectrum domain. A typical example is the Green's Function:
$$
\mathcal{G}(A,E)=\frac{1}{E-A+i\eta^+}=-\frac{\partial }{\partial A}\text{Tr}\left[\ln{(E-A+1\eta^+)}\right]
$$
Which have poles around eigenvalues of $A$. However, when transformed into time-space, it is written as:
$$
\mathcal{G}(A,t)=-i\theta{(t)}e^{-iAt}=\theta(t)\frac{\partial}{\partial At}
\text{Tr}\left[e^{-iAt}\right]
$$
Here $e^{-iAt}$ is a smooth and bounded function, which can be easily approximated via Chebysheve Polynomials. Thus, we are able to evaluate $\mathcal{G(A,t)}$ in the time domain using automatic differentiation and randon trace technique, and transform it back to $\mathcal{G}(E,t)$ using inverse Fourier Transformation.

## 3. Methods Proposal
Here we will display the theoretical foundations for the methods to develop. The main goal is: develop general properties computation methods for non-orthogonal basis Hamiltonian. The method need to be linear-scaling, and is able to extract both **scalar** and **matrix** properties.
### 3.1 Random Green's Function for properties
We know that a lot of properties can be extract from the Green's function:
$$
G(E)S=(ES-H)^{-1}S
$$
We can compute the greens function from it, using resolution of identity:
$$
\frac{1}{M}\sum_i^M|X_i\rangle\langle X_i|=\mathcal{I}
$$

$$
\lim_{S\to\infty}\frac{1}{S}\sum_i^S|\psi_p\rangle\langle \psi_p|=\mathcal{I},\quad\lim_{S\to\infty}\frac{1}{S}\sum_i^SG|\psi_p\rangle\langle \psi_p|=G\\
$$


$$
G(E)S\approx\frac{1}{M}\sum_i^M(ES-H)^{-1}S|X_i\rangle\langle X_i|
$$

We can also evaluate its trace:
$$
Tr[G(E)S]\approx\frac{1}{M}\sum_i^M\langle X_i|(ES-H)^{-1}S|X_i\rangle
$$

**We temporarily pend the discussion of full green's function evaluation. focus on the trace.** But both calculation dependent on computing: $(ES-H)^{-1}S|X_i\rangle$.

Use sparse matrix multiplication, $|SX_i\rangle=S|X_i\rangle \propto O(N)$. However, solving $(ES-H)^{-1}|SX_i\rangle$ is a bit more complicating. Krylov subspace method can be used:

Now let's derive the Krylov approximation here: First:
$$(ES-H)^{-1}S|X_i\rangle=|Y_i\rangle$$
Therefore:
$$
(ES-H)|Y_i\rangle=S(E-S^{-1}H)|Y_i\rangle=S|X_i\rangle\\
\equiv (E-S^{-1}H)|Y_i\rangle=|X_i\rangle
$$
What we need to compute is $|Y_i\rangle$. Now generating the Krylov subspace transformation $Q$ of operator $S^{-1}H$ under metric $S$, and approximate $|Y_i\rangle=Q|v_i\rangle$, we have:
$$
\begin{align*}
Q^{\dagger}S(E-S^{-1}H)Q|v_i\rangle&=(EQ^{\dagger}SQ+Q^\dagger HQ)|v_i\rangle\\
&=(EI-\tilde{H})|v_i\rangle\\
&=Q^\dagger S|X_i\rangle
\end{align*}
$$

**Remark**: Noticing here the initial vector used to generate the Krylov space should still be the random state vector $|X_i\rangle$, instead of $S|X_i\rangle$.

This is achieved since $Q^\dagger SQ=I$ and $\tilde{H}=Q^\dagger HQ$. So:
$$
|Y_i\rangle=Q|v_i\rangle=Q(EI-\tilde{H})^{-1}Q^\dagger S|X_i\rangle
$$

Now we arrived the approximate format:
$$
(ES-H)^{-1}S\approx Q(EI-\tilde{H})^{-1}Q^\dagger S
$$

Since the Krylov subspace's dimension is much lower than the origin. We have $Q\in R^{k\times n}$, where $k$ is the subspace dimension and $n$ is the matrix dimension. The approximated operation are very easily computed.

The trace is now represented as:
$$
\langle X_i|Q(EI-\tilde{H})^{-1}Q^\dagger S|X_i\rangle
$$
Generally, $Q^\dagger S|X_i\rangle=||X_i|||e_0\rangle$, which is nonzero only on its first component. Moreover, $\langle X_i|Q$ can be computed during the generation of basis vector $Q=\{q_0,q_1,...\}$. Then, other quantities are irrelavent with $Q$. In this case, there is not need to save $Q$. What's more, since the 

**Krylov Subspace** is generated via $S^{-1}H$. It is hermitian under metric $S$ of vector space. Therefore, the reduced $\tilde{H}$ should have tridiagonal format. Which also states a short recursive relation, that a lanczos like algorithm is able to be adopted. That means:

**In principle**: We only need to keep the most recent $q_{n-1}, q_{n}$ in memory, and discard all others by keeping a constant size $[\text{matdim}, 2]$ array and update it iteratively. Each time a basis is generated, we compute inner product $\langle X_i|q_n\rangle$ to form $\langle X_i|Q$. In this way we are free of memory issue.

Now we arrived an expression for the trace evaluation:
$$
D(E)=-\frac{1}{\pi}\text{Im}\left(\frac{1}{M}\sum_i^M\sum_{m}^v||X_i||\langle X_i|q_m^{(i)}\rangle G_{m0}^{(i)}\right)
$$

Here $G_{m0}^{(i)}=(EI-\tilde{H}^{(i)})^{-1}$, we add $(i)$ for these quantities since they are Krylov subspace $\mathcal{K}(S^{-1}H,|X_i\rangle)_S$.

However, until now, everything is just **"in principle"**. In finite precision world, the calculation of the lanczos basis set $Q$ will soon lost the orthogonality. Since in practice, if the energy shift $E$ is in the spectrum domain of matrix $H$, the required krylov moment is basically proportional to $v=\sqrt{N}$, where $N$ is the matrix dimension of $H$. Therefore we cannot afford and orthogonalize $Q$ explicitly, certain technique must be applied. There are two types of techniques that could work:

1. **Solve the theoretical problem of large moemntum requirement $v$**. 
   A set of very specific but powerful method is tested to converge on calculating the Green's Function acting on vectors with tens of subspace dimenions. This method is called the rational Krylov subspace method, which is derived originally for the selected generalized eigenvalue problem, but soon find very useful to solve the inversion with large amout of shift poles. saving and orthogonalize tens of vectors is fully compatible, and the numerical accuracy is high. Therefore, should be the greatest choise!

2. **Another one is based on the current practice of the lanczos softwares** 
   So this is well tested methods. (1) First, we use a restart algorithm to cutdown the size of each Krylov subspace. (2) In each subspace, we use selected/random reorthogonalization technique. In this way, the storage, orthogonality and the efficiency can be balanced. This method is obviously less efficient than the first one, so the formal version should prefer the previous one. To approximate the solution of linear problem, several solver gives different expression, Basically, COCG, COCR, MINRES, GMRES might works the best.


#### 3.1.1 Generalization to any matrix/scalar quantities with Overlap
The above discussion can be generalize to any properties, as well as they can be written as a function of a matrix $S^{-1}H$.

We consider solving linear problem of: 
$$f(S^{-1}H)|X\rangle=|Y\rangle$$
Under decomposition, the $f(S^{-1}H)$ can be approximated as:
$$f(S^{-1}H)\approx Qf(\tilde{H})Q^\dagger S$$
So:
$$
|Y\rangle=Qf(\tilde{H})Q^\dagger S|X\rangle
$$
Let $f(X)=1/(EI-S^{-1}H)$ gives the above Green's Function Calculation formula:
$$
|Y\rangle=Q(EI-\tilde{H})^{-1}Q^\dagger S|X\rangle
$$

The density matrix can be computed when $f=F$ the fermi-dirac function. The usual form is:
$$
D|X\rangle=F(S^{-1}H)S^{-1}|X\rangle=QF(\tilde{H})Q^\dagger|X\rangle
$$

**Remark**: For computing density, we loss the great properties of this form: $Q^\dagger S|X\rangle$. Therefore, we might have some convergence problem, which need careful treatment.

#### 3.1.2 The Rational Krylov Space Methods

#### 3.1.3 The Restart and Selected/Random Reorthogonalization
To use this method, we need to first ensure two things:
1. would the restart break the principles we used to simplify the calculations? They are:
   a. The shift invariance krylov subspace and basis. The shift invariant expression for the linear problem's solution.
   b. The orthogonality of basis that we used to simplify $QS|X_i\rangle$.
2. Was restart and reorthogonalization compatible with different solver's approximation of the solution?

We will address these problem here, and derive the restart approximate formula of the solution.

First, we know that, in the restart process, the residual of the former section is used as the initial vector. So we need the residual formula:

**CG:**
$$
\begin{align*}
r_n&=b-Ax_n\\
&=b-A||b||Q_nT_{n}^{-1}Q_n^\dagger b\\
&=b-||b||(Q_nT_n+T_{n,n-1}q_ne_n^T)T_n^{-1}e_1\\
&=b-||b||Q_nT_nT_n^{-1}e_1-||b||T_{n,n-1}q_ne_n^TT_n^{-1}e_1\\
&=-||b||T_{n,n-1}(T_n^{-1})_{n-1,0}q_n
\end{align*}
$$
Which is proportional to $q_n$. Since the basis is invariant to the shift of the operator $A$, therefore, restarting by setting the initial vector as $q_n$ will only give a constant difference in the next round of approximation.

**GMRES**


<!-- 
We can use it to compute either the trace or the operator's matrix elements:

$$
G(E)\approx\frac{1}{M}\sum_i^M\sum_{n,m}^v|\Phi_m^{(i),v}\rangle G_{K,mn}^{(i),v}\langle\Phi_n^{(i),v}| S|X_i\rangle\langle X_i|
$$ -->
<!-- 
Here $|\Phi_m^{(i),v}\rangle$ is the Krylov transformation matrix $Q$ for each sample $|X_i\rangle$. $G^{(i),v}_{K,mn}=(EI-\tilde{H})^{-1}$.
We need to notice that: $\langle\Phi_n^{(i),v}| S|X_i\rangle$ equals $||X_i||\delta_{n,0}$, since the Krylov basis are generated using each random sample as starting vector. Therefore:
$$
G(E)\approx\frac{1}{M}\sum_i^M\sum_{m}^v||X_i|||\Phi_m^{(i),v}\rangle G_{K,m0}^{(i),v}\langle X_i|
$$
If we want only the trace, using the random trace format, we have:
$$
D(E)=-\frac{1}{\pi}\text{Im}\left(\frac{1}{M}\sum_i^M\sum_{m}^v||X_i||\langle X_i|\Phi_m^{(i),v}\rangle G_{K,m0}^{(i),v}\right)
$$ -->

Now we derived a property calculator using Random Green's Function !

**Remark:** This problem and beauty of this method comes from the same place - *transforming a eigenvalue problem into a shift-linear prolbem $(ES-H)^{-1}S|X_i\rangle$, which is then solved using Krylov subspace method*. 
The main advanteges comes from:
1. **The shift-invariant subspace of Krylov method.**

   This garenttees that, the subspace only need to be construct once! On $S^{-1}H$, and then reused on any shifted operators: $EI-S^{-1}H$. Therefore, the computation is only costly on the first k point, for others, it is extremely simple and fast.

2. **The simple form of computing expecations.**

   Also comes from the Krylov subspace's properties, since each subspace are $S$-orthogonal, this reduced a large amount of cost and storage. The evaluation of trace or matrix element reduced to two matrix multiplication.

There are problem suffers from Krylov subspace:
1. **The growing Krylov subspace as the matrix dimension.**
   
   When the estimated $E$ are among the matrix spectrum, the Krylov dimension $m$ grows proportional to $\sqrt{N}$ since spectrum $E$ becomes denser as $N$ grows. Therefore, this method is not theoretically linear scaling, but a $O(\sqrt{N}\cdot N)=O(N^{1.5})$ scaling. This is the main drawback comparing to the ones based on Chebysheve Polynomials (whose rank stays constant). 

   To address, this might be able to accelerate by the preconditioner updates for shift-linear system. Careful treatment needed since we want the subspace still invariant to energy E after preconditioning.
2. **Storage problem.**
   This occur when the matrix dimenion goes to tens of millions. $|\Psi_m^{(i)}\rangle$ have dimension $v\times S\times N$, where $v$ is Krylov subspace dimension, $S$ is sample number, and $N$ is the matrix dimesnion. When matrix dimension goes to hundreds of millions, and Krylov space become $\sqrt{N}=10^4$, the storage requires would be around 10T, not applicable. Further design of the evaluation scheme is a must if we want to scale this up to tens to hundreds of millions.

3. **The instability of Krylov subspace construction under finite precision.**

   This is a problem with history, need delicate numerical treatment. But it is solvable, as the numerically stable Krylov method have been widely applied for so many years.



### 3.2 differentiable TBPM
The idea is, incooperate overlap via $S^{-1}H$, and compute $S^{-1}$ on vectors using preconditioned CG with linear scale. The normal TBPM support computing scalar properties, while the differentiable programming incooperate the matrix properties calculation.

#### 3.2.1 TBPM with overlap
Start with the Schrodinger Equation:
$$H|\Psi\rangle=\epsilon S|\Psi\rangle$$
Multiply $S^{-1}$ on both side:
$$S^{-1}H|\Psi\rangle=\epsilon |\Psi\rangle$$
The matrix $S^{-1}H$ have the spectrum of the eigenvalue. Therefore, computing the function of the spectrum can be represented as the matrix function $f(S^{-1}H)$. Basically, we just replace all $H$ in TBPM with $S^{-1}H$. Since matrix multiplication takes a while, we never do this explicitly. Instead, we apply $|Z\rangle=H|X\rangle$, and then solve $S|Y\rangle=|Z\rangle$.

## 4. implementation
### 4.1 $S^{-1}|X\rangle$
This operation is done using the block CG with preconditioner. The linear scale is garentteed. However, when $|X\rangle$ have a very large batch (usually suggest $S$ is not very big.), block CG would be less efficient. A Chebysheve preconditioned BCG would be prefered. Therefore:

1. Chebysheve preconditioned BCG: using Scikit-sparse https://github.com/scikit-sparse/scikit-sparse/tree/master

When matrix size is smaller than 5e4, or even 1e5 when the vector batch is larger than 1000.

2. AMG / Domain decomposition preconditioned BCG: When matrix size is really big. We can use: 
- HPDDM: https://github.com/hpddm/hpddm/tree/main
- PETSc: https://petsc.org/release/
- MKL's dcgmrhs function with hypre's boomamg as preconditioner: https://github.com/flatironinstitute/sparse_dot/issues/33#issuecomment-2860259797 (under discussion with developers)

### 4.2 DOS
DOS computation is suitable for time domain green's function method, because it is the properties that defined on the real energy grid. The krylov space method can diverge. However, it only need to take a uniform time grid in the tGF, and transform it using FFT back to energy domain.

### 4.3 N and $\rho$
This two is a little tricky. Since we need to get the $N$ and $rho$ - the total electron number and the charge density very accurately. (To determine the chemical potential, and combine with DFT to perform self-consistent calculaton).

This need to perform with a hybrid method. First, to do this accurately, there will always exist some point that is close to the real axis $G(a+bi)$, where $|b|$ might be small. solving this is quite hard for the spectrum GF method using krylov subspace, but can be handled with the time domain. For example, for 300K calculaton, the first Matsubara point will be: $i\frac{\pi}{\beta}\approx 0.078$. So, the fourier transform will contain a damping factor $e^{-0.078 t}$, it is a exponential decaying tail, so a exponential decaying grid can integrate it accurately.

### 4.4 real space basis's charge density.

# References

[1] Hams, Anthony, and Hans De Raedt. "Fast algorithm for finding the eigenvalue distribution of very large matrices." Physical Review E 62.3 (2000): 4365.
[2] Wang, Zhentao, et al. "Gradient-based stochastic estimation of the density matrix." The Journal of Chemical Physics 148.9 (2018).