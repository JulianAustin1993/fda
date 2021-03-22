package fda
package kernel

import breeze.linalg.{DenseVector, isClose}
import breeze.stats.distributions.Gaussian
import org.scalatest.FunSuite

class MaternTest extends FunSuite {

  def crudeGrad(fX: Double, fXPrime: Double, dX: Double): Double = {
    (fXPrime - fX) / dX
  }

  test("testKernel") {
    val (x, y) = (DenseVector(1.0), DenseVector(2.0))
    val kern = Matern(0.5, 1.0, 1.0)
    assert(kern.k(x, y) == scala.math.exp(-1.0))
    assert(kern.k(x, x) == 1.0)
  }
  test("testGradKernel") {
    val shapes = List(0.5, 1.5, 2.5, Double.PositiveInfinity)
    val normal01 = Gaussian(0, 1)
    val (x, y) = (DenseVector.rand[Double](1, normal01),
      DenseVector.rand[Double](1, normal01))

    for (shape <- shapes) {
      val kern = Matern(shape, 2.0, 1.5)
      val kern2 = Matern(shape, 2.0 + 1e-8, 1.5)
      val kern3 = Matern(shape, 2.0, 1.5 + 1e-8)
      val grad = DenseVector(
        crudeGrad(kern.k(x, y), kern2.k(x, y), 1e-8),
        crudeGrad(kern.k(x, y), kern3.k(x, y), 1e-8)
      )
      assert(isClose(kern.gk(x, y), grad, 1e-4))
    }

  }
}
