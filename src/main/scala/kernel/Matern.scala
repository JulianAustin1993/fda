package fda
package kernel

import math.linalg.{choSolve, detFromChol, jitChol}

import breeze.linalg.{DenseMatrix, DenseVector, diag}


case class Matern(shape: Double, variance: Double, lengthscale: Double) extends MercerKernel {
  require(shape == 0.5 | shape == 1.5 | shape == 2.5 | shape.isInfinite)
  require(variance > 0)
  require(lengthscale > 0)

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
    shape match {
      case x if (x == 0.5) => variance * scala.math.exp(-d)
      case x if (x == 1.5) => {
        val K = d * scala.math.sqrt(3)
        variance * (1.0 + K) * scala.math.exp(-K)
      }
      case x if (x == 2.5) => {
        val K = d * scala.math.sqrt(5)
        variance * (1.0 + K + scala.math.pow(K, 2) / 3.0) * scala.math.exp(-K)
      }
      case x if shape.isInfinite => variance * scala.math.exp(-scala.math.pow(d, 2) / 2.0)
      case _ => {
        throw new NotImplementedError("Shape parameter must be 0.5, 1.5, 2.5 or infinte.")
      }
    }
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
   * @param noise Noise value to add to diagional.
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