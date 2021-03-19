package fda
package regression

import kernel.Matern
import math.linalg.symFromUPLO

import breeze.linalg.{DenseVector, linspace}
import breeze.stats.distributions.MultivariateGaussian
import org.scalatest.FunSuite

class GpTest extends FunSuite {

  test("testDraw") {
    val kern = Matern(2.5, 1.0, 2.5)
    val x = linspace(0.0, 10.0, 64)
    val normal01 = breeze.stats.distributions.Gaussian(0, 1)
    val K = symFromUPLO(kern.K(x.toDenseMatrix.t))
    val mvn = MultivariateGaussian(DenseVector.zeros[Double](x.size), K)
    val y = mvn.draw()
    val yErr = y + 0.1 * DenseVector.rand(x.size, normal01)
    val gp = new Gp(yErr, x.toDenseMatrix.t, kern, 5.0)
    val gp2 = new Gp(yErr, x.toDenseMatrix.t, kern, 0.1)
    assert(gp.logLikelihood < gp2.logLikelihood)
  }
}
