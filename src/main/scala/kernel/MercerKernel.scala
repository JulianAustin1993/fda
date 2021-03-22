package fda
package kernel

import breeze.linalg.{DenseMatrix, DenseVector}
import spire.implicits.cfor

trait MercerKernel {

  val hyperParameters: DenseVector[Double]

  /**
   * Kernel function
   *
   * @param x DenseVector of locations.
   * @param y DenseVector of locations.
   * @return The kernel value.
   */
  def k(x: DenseVector[Double], y: DenseVector[Double]): Double

  def gk(x: DenseVector[Double], y: DenseVector[Double]): DenseVector[Double]


  /**
   * Calculate the lower Kernel matrix for collection X with itself
   *
   * @param X Collection of points in DenseMatrix.
   * @return Lower triangle of symmetric kernel matrix.
   */
  def K(X: DenseMatrix[Double]): DenseMatrix[Double] = {
    val n = X.rows
    val L = DenseMatrix.zeros[Double](n, n)
    cfor(0)(_ < n, _ + 1) {
      i => {
        val xi = X(i, ::).t
        cfor(0)(_ < i + 1, _ + 1) {
          j => {
            L(i, j) = k(xi, X(j, ::).t)
          }
        }
      }
    }
    L
  }

  /**
   * Calculate the lower Gradient Kernel matrix for collection X with itself
   *
   * @param X Collection of points in DenseMatrix.
   * @return Lower triangle of symmetric kernel matrix.
   */
  def GK(X: DenseMatrix[Double]): List[DenseMatrix[Double]] = {
    val p = hyperParameters.length
    val n = X.rows
    val lL = List.fill(p)(DenseMatrix.zeros[Double](n, n))
    cfor(0)(_ < n, _ + 1) {
      i => {
        val xi = X(i, ::).t
        cfor(0)(_ < i + 1, _ + 1) {
          j => {
            val grad = gk(xi, X(j, ::).t)
            cfor(0)(_ < p, _ + 1) {
              k => {
                lL(k)(i, j) = grad(k)
              }
            }
          }
        }
      }
    }
    lL
  }

  /**
   * Calculate the full kernel matrix for collection X against Y.
   *
   * @param X Collection of points in DenseMatrix.
   * @param Y Collection of points in DenseMatrix.
   * @return the kernel matrix.
   */
  def K(X: DenseMatrix[Double], Y: DenseMatrix[Double]): DenseMatrix[Double] = {
    val (n, m) = (X.rows, Y.rows)
    val K = DenseMatrix.zeros[Double](n, n)
    cfor(0)(_ < n, _ + 1) {
      i => {
        val xi = X(i, ::).t
        cfor(0)(_ < m, _ + 1) {
          j => {
            K(i, j) = k(xi, Y(j, ::).t)
          }
        }
      }
    }
    K
  }

  /**
   * Calculate Alpha and cholesky decomposition of Ky and log determinant of noiseless kernel.
   *
   * @param y Response
   * @param X Predictors
   * @return Kx^^{-1}y, and log det(Kx)
   */
  def calculateAlphaAndLAndlDet(y: DenseVector[Double], X: DenseMatrix[Double]): (DenseMatrix[Double], DenseVector[Double], Double)

  /**
   * Calculate Alpha and cholesky decomposition of Ky and log determinant of noisy kernel.
   *
   * @param y     Response
   * @param noise Noise value to add to diagonal.
   * @param X     Predictors
   * @return Kx^^{-1}y, and log det(Kx)
   */
  def calculateAlphaAndLAndlDet(y: DenseVector[Double], noise: Double, X: DenseMatrix[Double]): (DenseMatrix[Double], DenseVector[Double], Double)

}

