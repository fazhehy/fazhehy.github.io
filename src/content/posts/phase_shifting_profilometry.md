---
title: 相移法公式推导
published: 2026-01-20
description: ''
image: ''
tags: [FPP, phase_unwrapping]
category: 'FPP'
draft: false 
lang: ''
---

## 提取相位公式推导

本文要推导的公式如下：

$$
\varphi(x,y)=-\arctan\,\frac{\displaystyle\sum_{k=0}^{N-1} I_k(x,y)\sin\frac{2\pi k}{N}}{\displaystyle\sum_{k=0}^{N-1} I_k(x,y)\cos\frac{2\pi k}{N}}
$$

其中

- $I_k(x,y)$为条纹图像
  
### 投影条纹数学表达式

假设相移的步数为$N$，则生成条纹的相移公式：

$$I_{k} = A + B\cos\left[\varphi(x,y)+ \frac{2k\pi}{N}\right], (k=0,1,\ldots,N-1)$$

其中：
- $A$：背景光强
- $B$：调制强度
- $\varphi(x,y)$：相位值
- $x$：编码方向上的像素坐标
  
### 反射条纹数学表达式

由于物体的高度，反射率的不同，相机拍摄的图像可以用以下表达式表示：

$$
\begin{align*}
    I_{k}(x,y) &= r(x,y)\left(A + B\cos\left[\varphi(x,y)+ \frac{2k\pi}{N}\right]\right) \\
    &= A(x,y) + B(x,y)\cos\left[\varphi(x,y)+ \frac{2k\pi}{N}\right]
\end{align*}
$$

根据三角函数的和角公式，上式可以写为：
$$
\begin{align*}
    I_{k}(x,y) &= A(x,y) + B(x,y)\cos\left[\varphi(x,y)+ \frac{2k\pi}{N}\right]\\
    &= A(x,y) + B(x,y)\left[\cos\varphi\cos\frac{2k\pi}{N}-\sin\varphi\sin\frac{2k\pi}{N}\right] \\
    &= A(x,y) + B(x,y)\cos\varphi\cos\frac{2k\pi}{N}-B(x,y)\sin\varphi\sin\frac{2k\pi}{N} \\
    &= A(x,y) + B_1(x,y)\cos\frac{2k\pi}{N}-B_2(x,y)\sin\frac{2k\pi}{N} \\
\end{align*}
$$

其中
- $B_1(x,y)=B(x,y)\cos\varphi$


- $B_2(x,y)=B(x,y)\sin\varphi$

- $\varphi(x,y)=\arctan\,\frac{B_2(x,y)}{B_1(x,y)}$

$A(x,y),B_1(x,y),B_2(x,y)$是未知量，其余皆已知，问题就转换成求解线性方程组的问题了

$$
Ax=b
$$

其中
- $A=\begin{bmatrix}
    1 & \cos\frac{2\cdot0\cdot\pi}{N} & -\sin\frac{2\cdot0\cdot\pi}{N} \\
    \vdots & \vdots & \vdots \\
    1 & \cos\frac{2\cdot k\cdot\pi}{N} & -\sin\frac{2\cdot k\cdot\pi}{N} \\
    \vdots & \vdots & \vdots \\
    1 & \cos\frac{2\cdot (N-1)\cdot\pi}{N} & -\sin\frac{2\cdot (N-1)\cdot\pi}{N} \\
\end{bmatrix}$
&nbsp;
- $x=\begin{bmatrix}
    A(x,y) \\\\
    B_1(x,y) \\\\
    B_2(x,y)
\end{bmatrix}$
&nbsp;
- $b=\begin{bmatrix}
    I_0 \\
    \vdots\\
    I_k \\
    \vdots\\
    I_{N-1}
\end{bmatrix}$

### 最小二乘法

对于求解方程，只需要三个方程就行（因为只有三个未知数）。但是世界充满噪声，使用越多条纹图像能够提高求解精度。于是方程组变成了超定方程，求解超定方程，我们可以使用最小二乘法。
下面是最小二乘法的数学推导，矩阵论书上有。

要保证求解的向量最精确，就是让2范数最小
$$
\min \left \| Ax-b \right \|_{2}^{2} 
$$
开始求解
$$
\begin{align*}
    J(x) &= \left \| Ax-b \right \|_{2}^{2} \\
    &= (Ax-b)^T(Ax-b) \\
    &= (x^TA^T-b^T)(Ax-b) \\
    &= x^TA^TAx-x^TA^Tb-b^TAx+b^Tb\\
\end{align*}
$$

注意到，$x^TA^Tb$和$b^TAx$互为转置，且都是标量，所以$x^TA^Tb=b^TAx$

$$
\begin{align*}
    J(x) &= x^TA^TAx-x^TA^Tb-b^TAx+b^Tb\\
    &= x^TA^TAx-2b^TAx+b^Tb\\
\end{align*}
$$

$$
\begin{align*}
\frac{\partial J(x)}{\partial x} &= 2(A^TA)x-2(b^TA)^T \\
 &= 2(A^TA)x-2A^Tb
\end{align*}
$$
> 矩阵求导请查阅矩阵论

令$\frac{\partial J(x)}{\partial x} =0$，即可求得解向量$x$。

所以，

$$
x= (A^TA)^{-1}A^Tb
$$

需要说明的是，$A^TA$不一定可逆，对于不可逆的情况可以使用广义逆矩阵。对于条纹投影，$A^TA$总是可逆。

### 求解方程

$$
\begin{align*}
    A &=\begin{bmatrix}
    1 & \cos\frac{2\cdot0\cdot\pi}{N} & -\sin\frac{2\cdot0\cdot\pi}{N} \\
    \vdots & \vdots & \vdots \\
    1 & \cos\frac{2\cdot k\cdot\pi}{N} & -\sin\frac{2\cdot k\cdot\pi}{N} \\
    \vdots & \vdots & \vdots \\
    1 & \cos\frac{2\cdot (N-1)\cdot\pi}{N} & -\sin\frac{2\cdot (N-1)\cdot\pi}{N} \\
\end{bmatrix} 
\\\\
    A^T &=\begin{bmatrix}
    1 & \dots & 1 & \dots & 1\\\\
    \cos\frac{2\cdot0\cdot\pi}{N} & \dots & \cos\frac{2\cdot k\cdot\pi}{N} & \dots & \cos\frac{2\cdot k\cdot\pi}{N}\\\\
    -\sin\frac{2\cdot0\cdot\pi}{N} & \dots & -\sin\frac{2\cdot k\cdot\pi}{N} & \dots & -\sin\frac{2\cdot (N-1)\cdot\pi}{N}
\end{bmatrix}\\\\
    A^TA &=\begin{bmatrix}
    N & \displaystyle\sum_{k=0}^{N-1}\cos\frac{2k\pi}{N}  & -\displaystyle\sum_{k=0}^{N-1}\sin\frac{2k\pi}{N}\\\\
    \displaystyle\sum_{k=0}^{N-1}\cos\frac{2k\pi}{N} & \displaystyle\sum_{k=0}^{N-1}\cos^2\frac{2k\pi}{N}  & -\displaystyle\sum_{k=0}^{N-1}\cos\frac{2k\pi}{N}\sin\frac{2k\pi}{N}\\\\
    -\displaystyle\sum_{k=0}^{N-1}\sin\frac{2k\pi}{N} & -\displaystyle\sum_{k=0}^{N-1}\cos\frac{2k\pi}{N}\sin\frac{2k\pi}{N}  & \displaystyle\sum_{k=0}^{N-1}\sin^2\frac{2k\pi}{N}
\end{bmatrix}\\
\end{align*}
$$
由于三角函数的正交性（可以利用倍角公式证明），在周期内求和(等距)或者积分，除了2次幂（倍角公式会产生常数）其他形式的结果都是等于0
所以
$$
\begin{align*}
    A^TA &=\begin{bmatrix}
    N & 0  & 0\\\\
    0 & \frac{N}{2}  & 0\\\\
    0 & 0  & \frac{N}{2}
\end{bmatrix}\\\\
    (A^TA)^{-1} &=\begin{bmatrix}
    \frac{1}{N} & 0  & 0\\\\
    0 & \frac{2}{N}  & 0\\\\
    0 & 0  & \frac{2}{N}
\end{bmatrix}\\\\
    (A^TA)^{-1}A^T &=\begin{bmatrix}
    \frac{1}{N} & \dots & \frac{1}{N} & \dots & \frac{1}{N}\\\\
    \frac{2}{N}\cos\frac{2\cdot0\cdot\pi}{N} & \dots & \frac{2}{N}\cos\frac{2\cdot k\cdot\pi}{N} & \dots & \frac{2}{N}\cos\frac{2\cdot k\cdot\pi}{N}\\\\
    -\frac{2}{N}\sin\frac{2\cdot0\cdot\pi}{N} & \dots & -\frac{2}{N}\sin\frac{2\cdot k\cdot\pi}{N} & \dots & -\frac{2}{N}\sin\frac{2\cdot (N-1)\cdot\pi}{N}
    \end{bmatrix} \\\\
    b&=\begin{bmatrix}
    I_0 \\
    \vdots\\
    I_k \\
    \vdots\\
    I_{N-1}
\end{bmatrix}\\\\
    (A^TA)^{-1}A^Tb &=\begin{bmatrix}
    \frac{1}{N}\displaystyle\sum_{k=0}^{N-1}I_k \\\\
    \frac{2}{N}\displaystyle\sum_{k=0}^{N-1}I_k\cos\frac{2k\pi}{N} \\\\
    -\frac{2}{N}\displaystyle\sum_{k=0}^{N-1}I_k\sin\frac{2k\pi}{N} \\\\
    \end{bmatrix} 
\end{align*}
$$

综上
$$
\left\{\begin{matrix}
A(x,y)=\frac{1}{N}\displaystyle\sum_{k=0}^{N-1}I_k\\\\
B_1(x,y)=\frac{2}{N}\displaystyle\sum_{k=0}^{N-1}I_k\cos\frac{2k\pi}{N}\\\\
B_2(x,y)=-\frac{2}{N}\displaystyle\sum_{k=0}^{N-1}I_k\sin\frac{2k\pi}{N}
\end{matrix}\right.
$$
### 最终结果
- 背景光强
$$
A(x,y)=\frac{1}{N}\displaystyle\sum_{k=0}^{N-1}I_k
$$
- 相位公式
$$
\varphi(x,y)=-\arctan\,\frac{\displaystyle\sum_{k=0}^{N-1} I_k(x,y)\sin\frac{2\pi k}{N}}{\displaystyle\sum_{k=0}^{N-1} I_k(x,y)\cos\frac{2\pi k}{N}}
$$

- 调制强度
$$
B(x,y)=\frac{2}{N}\sqrt{\left(\displaystyle\sum_{k=0}^{N-1} I_k(x,y)\cos\frac{2\pi k}{N}\right)^2+\left(\displaystyle\sum_{k=0}^{N-1} I_k(x,y)\sin\frac{2\pi k}{N}\right)^2}
$$

