package fda
package regression

import kernel.MercerKernel
import math.linalg.{forwardSolve, symFromUPLO}

import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.stats.distributions.MultivariateGaussian

class Gp(y: DenseVector[Double],
         X: DenseMatrix[Double],
         covKernel: MercerKernel,
         sig2: Double) {
  lazy val logLikelihood: Double = -0.5 * (y.t * alpha + lDetKy + y.size * scala.math.log(2.0 * scala.math.Pi))
  val interims = covKernel.calculateAlphaAndLAndlDet(y.copy, sig2, X.copy)
  val L = interims._1
  val alpha = interims._2
  val lDetKy = interims._3

  def prior(Xprime: DenseMatrix[Double]): MultivariateGaussian = {
    val Kx = symFromUPLO(covKernel.K(Xprime))
    diag(Kx) :+= sig2
    MultivariateGaussian(DenseVector.zeros(Xprime.rows), Kx)
  }

  def posterior(Xprime: DenseMatrix[Double]): MultivariateGaussian = {
    val Kxsx = covKernel.K(Xprime, X)
    val Kxsxs = symFromUPLO(covKernel.K(Xprime))
    val v = forwardSolve(L, Kxsx.t)
    MultivariateGaussian(Kxsx * alpha, Kxsxs - v.t * v)
  }
}
