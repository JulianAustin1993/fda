package fda
package math

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import spire.implicits.cfor

package object integration {
  /**
   * Integrate a function from regular samples using romberg method with trapezoidal.
   *
   * @param y  DenseVector of 2**k + 1 samples.
   * @param k  Integer specifying the number of samples through 2**k +1
   * @param dx Spacing between samples.
   * @return Approximation to integral over sample domain.
   */
  def rombergIntegrator(y: DenseVector[Double], k: Int, dx: Double): Double = {
    val nSamples = y.size
    val nIntervals = nSamples - 1
    val R = DenseMatrix.zeros[Double](k + 1, k + 1)
    val h = nIntervals * dx
    R(0, 0) = (y(0) + y(-1)) / 2.0 * h
    cfor(1)(_ < k + 1, _ + 1) {
      i => {
        val inds = nIntervals >> i until nIntervals by nIntervals >> i - 1
        R(i, 0) = 0.5 * (R(i - 1, 0) + (h / scala.math.pow(2, i - 1)) * sum(y(inds).toDenseVector))
        cfor(1)(_ < i + 1, _ + 1) {
          j => {
            val prev = R(i, j - 1)
            R(i, j) = prev + (prev - R(i - 1, j - 1)) / (scala.math.pow(2.0, 2*j) - 1)
          }
        }
      }
    }
    R(k, k)
  }
}
