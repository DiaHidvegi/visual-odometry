# Comprehensive Explanation of Linear-Phase FIR Filter Design

## 1. Filter Structure and Basic Relationships
- For Type I FIR filters, $M$ represents the total length of the impulse response (must be odd)
- $L$ is defined as $L = \frac{M-1}{2}$
- The filter has linear phase when its coefficients are symmetric: $b_k = b_{M-1-k}$

1. **Filter Length and Symmetry:**
   - A Type I FIR filter of length \( M \) is always of odd length, and its coefficients are symmetric. This means that the impulse response satisfies:
   \[
h[n] = h[M-1-n]\text{ for } n = 0, 1, \dots, M-1.
   \]
   
   - If \( M \) is odd, we can express it as \( M = 2L + 1 \). Here, \( L \) represents the index of the central coefficient, and the symmetry means that only the first \( L+1 \) coefficients are unique.

2. **Design Process:**
   - In the design of linear-phase FIR filters using optimization methods (e.g., least squares), we often calculate \( L+1 \) unique coefficients: \( b_0, b_1, \dots, b_L \). These coefficients correspond to the unique part of the filter.
   
   - Once these coefficients are determined, the full impulse response can be reconstructed using symmetry:
   \[
h[n] = b_{|n-L|}, \quad n = 0, 1, \dots, M-1.
   \]

3. **Example for Clarity (\( M = 9 \)):**
   - Let’s take an example where \( M = 9 \). Here, \( L = 4 \), so we have \( L+1 = 5 \) unique coefficients \( \{b_0, b_1, b_2, b_3, b_4\} \).
   - The impulse response can be written as:
   \[
h_{\text{impulse}} = \{b_4, b_3, b_2, b_1, b_0, b_1, b_2, b_3, b_4\}.
   \]

4. **Frequency Domain Representation:**
   - The transfer function of the FIR filter in \( z \)-domain is given by:
   \[
   H(z) = b_0 + b_1(z^{-1} + z^{1}) + b_2(z^{-2} + z^{2}) + \dots + b_L(z^{-L} + z^{L}).
   \]

   - For our example (\( M = 9, L = 4 \)):
   \[
   H(z) = b_0 + b_1(z^{-1} + z^{1}) + b_2(z^{-2} + z^{2}) + b_3(z^{-3} + z^{3}) + b_4(z^{-4} + z^{4}).
   \]

   - To simplify, we often normalize the expression by factoring out the middle term (e.g., \( z^{-4} \)):
   \[
   H(z) = z^{-4} \left( b_0(z^4 + z^{-4}) + b_1(z^3 + z^{-3}) + b_2(z^2 + z^{-2}) + b_3(z^1 + z^{-1}) + b_4 \right).
   \]

5. **Frequency Response:**
   - The frequency response \( H(\Omega) \) is obtained by substituting \( z = e^{j\Omega} \):
   \[
   H(\Omega) = e^{-jL\Omega} \left[ 2b_0 \cos(L\Omega) + 2b_1 \cos((L-1)\Omega) + \dots + 2b_{L-1} \cos(\Omega) + b_L \right].
   \]
   

## 2. Transfer Function Decomposition
The transfer function can be written as:
$H(\Omega) = e^{j\frac{(M-1)\Omega}{2}} \cdot \check{H}(\Omega)$

where $\check{H}(\Omega)$ is the zero-phase response:
$\check{H}(\Omega) = b_{\frac{M-1}{2}} + 2\sum_{k=1}^{\frac{M-1}{2}} b_{\frac{M-1}{2}-k} \cos(\Omega k)$
$= \sum_{k=0}^{L} \tilde{b}_{L-k} \cos(\Omega k)$

## 3. Least-Squares Optimization Setup
The design process involves solving:
$\min_{\tilde{b}} \sum_{i=1}^K \|W(\Omega_i)(H_{\tilde{b}}(\Omega_i) - D(\Omega_i))\|^2$

This can be rewritten as:
$\min_{\tilde{b}} \sum_{i=1}^K \|\sum_{k=0}^L W(\Omega_i)\cos(\Omega_i k)\tilde{b}_{L-k} - W(\Omega_i)D(\Omega_i)\|^2$

## 4. Matrix Formulation
The problem is expressed in standard least-squares form:
$\min_{\tilde{b}} \|H\tilde{b} - h\|_2^2$

where:
- $H$ is a $K\times(L+1)$ matrix with elements:
  $H[i,k] = W(\Omega_i)\cos(\Omega_i k)$ for $i=1,\ldots,K$ and $k=0,\ldots,L$
- $h$ is a $K\times1$ vector:
  $h[i] = W(\Omega_i)D(\Omega_i)$
- $\tilde{b}$ is the $(L+1)\times1$ vector of filter coefficients:
  $\tilde{b} = [\tilde{b}_0, \tilde{b}_1, \ldots, \tilde{b}_L]^T$

The matrix $H$ is explicitly given as:
$H = \begin{bmatrix}
W(\Omega_1) & W(\Omega_1)2\cos(\Omega_1) & \cdots & W(\Omega_1)2\cos(L\Omega_1) \\
W(\Omega_2) & W(\Omega_2)2\cos(\Omega_2) & \cdots & W(\Omega_2)2\cos(L\Omega_2) \\
\vdots & \vdots & \ddots & \vdots \\
W(\Omega_K) & W(\Omega_K)2\cos(\Omega_K) & \cdots & W(\Omega_K)2\cos(L\Omega_K)
\end{bmatrix}$

## 5. Closed-Form Solution
When $K \geq L$ and $(H^T H)$ is invertible, the optimal solution is:
$\tilde{b}^* = (H^T H)^{-1}H^T h$

## 6. Final Filter Construction
Given the optimal coefficients $\tilde{b}^*$, the complete impulse response is constructed as:
$h = \{\tilde{b}^*_L, \tilde{b}^*_{L-1}, \ldots, \tilde{b}^*_1, \tilde{b}^*_0, \tilde{b}^*_1, \ldots, \tilde{b}^*_{L-1}, \tilde{b}^*_L\}$

The corresponding frequency response is:
$H(\Omega) = e^{jL\Omega/2}[\tilde{b}^*_L + 2\sum_{k=1}^L \tilde{b}^*_{L-k} \cos(\Omega k)]$

## Key Relationships
This complete formulation shows how:
- The filter length $M$ determines $L = \frac{M-1}{2}$
- Only $L+1$ unique coefficients need to be optimized
- The matrix $H$'s dimensions $(K\times(L+1))$ ensure enough frequency samples $(K)$ to solve for the $L+1$ coefficients
- Symmetry is automatically enforced through the construction of the final impulse response