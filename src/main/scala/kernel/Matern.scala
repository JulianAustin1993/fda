package fda
package kernel

import math.linalg.{choSolve, detFromChol, jitChol}

import breeze.linalg.{DenseMatrix, DenseVector, diag}


case class Matern(shape: Double, sigma: Double, lengthscale: Double) extends MercerKernel {
  require(shape == 0.5 | shape == 1.5 | shape == 2.5 | shape.isInfinite)
  require(sigma > 0)
  require(lengthscale > 0)
  val variance = sigma * sigma

  override val hyperParameters: DenseVector[Double] = DenseVector(variance, lengthscale)

  /**
   * Kernel function
   *
   * @param x DenseVector of locations.
   * @param y DenseVector of locations.
   * @return The kernel value.
   */
  override def k(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    val s = x - y
    val d = scala.math.sqrt(s.t * s) / lengthscale
    variance * isoK(d)
  }

  /**
   * Kernel function from distance.
   *
   * @param d distance between points.
   * @return kernel evaluated with separation d.
   */
  private def isoK(d: Double): Double = shape match {
    case x if x == 0.5 => scala.math.exp(-d)
    case x if x == 1.5 =>
      val K = d * scala.math.sqrt(3)
      (1.0 + K) * scala.math.exp(-K)
    case x if x == 2.5 =>
      val K = d * scala.math.sqrt(5)
      (1.0 + K + scala.math.pow(K, 2) / 3.0) * scala.math.exp(-K)
    case _ if shape.isInfinite => scala.math.exp(-scala.math.pow(d, 2) / 2.0)
    case _ =>
      throw new NotImplementedError("Shape parameter must be 0.5, 1.5, 2.5 or infinite.")
  }

  /**
   * Graident kenrel function
   *
   * @param x
   * @param y
   * @return
   */
  override def gk(x: DenseVector[Double], y: DenseVector[Double]): DenseVector[Double] = {
    val s = x - y
    val d = scala.math.sqrt(s.t * s)
    val gradSig = 2.0 * sigma * isoK(d / lengthscale)
    val gradLengthscale = shape match {
      case x if x == 0.5 => variance * isoK(d / lengthscale) * d / (lengthscale * lengthscale)
      case x if x == 1.5 => {
        val exponent = -1.0 * scala.math.sqrt(3) * d / lengthscale
        variance * exponent * exponent * scala.math.exp(exponent) / lengthscale
      }
      case x if x == 2.5 => {
        val exponent = -1.0 * scala.math.sqrt(5) * d / lengthscale
        val l3 = scala.math.pow(lengthscale, 3.0)
        val factor = 5 * d * d / (3 * l3) + 5 * scala.math.sqrt(5) * d * d * d / (3.0 * l3 * lengthscale)
        variance * factor * scala.math.exp(exponent)
      }
      case x if x.isInfinite => {
        variance * d * d * isoK(d / lengthscale) / scala.math.pow(lengthscale, 3.0)
      }
    }
    DenseVector(gradSig, gradLengthscale)
  }


  /**
   * Calculate Alpha and log determinant of noiseless kernel.
   *
   * @param y Response
   * @param X Predictors
   * @return Kx^^{-1}y, and log det(Kx)
   */
  override def calculateAlphaAndLAndlDet(y: DenseVector[Double],
                                         X: DenseMatrix[Double]): (DenseMatrix[Double], DenseVector[Double], Double) = {
    val Kmat = K(X)
    val L = jitChol(Kmat)
    (L.copy, choSolve(L, y), detFromChol(L))
  }

  /**
   * Calculate Alpha and log determinant of noisy kernel.
   *
   * @param y     Response
   * @param noise Noise value to add to diagonal.
   * @param X     Predictors
   * @return Kx^^{-1}y, and log det(Kx)
   */
  override def calculateAlphaAndLAndlDet(y: DenseVector[Double],
                                         noise: Double,
                                         X: DenseMatrix[Double]): (DenseMatrix[Double], DenseVector[Double], Double) = {
    val Kmat = K(X)
    diag(Kmat) :+= noise
    val L = jitChol(Kmat)
    (L.copy, choSolve(L, y), detFromChol(L))
  }

}