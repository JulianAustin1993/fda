package fda
package math.mcmc

import breeze.stats.distributions.{Gamma, MultivariateGaussian}
import org.scalatest.FunSuite

class MetropolisHastingsTest extends FunSuite {

  test("MetropolisHastings for a Gamma with a non-symmetric proposal") {
    import breeze.linalg._
    import breeze.numerics._
    val mh = MetropolisHastings(
      (x: DenseVector[Double]) => Gamma(2.0, 1.0 / 3).logPdf(x(0)),
      (x: DenseVector[Double]) => MultivariateGaussian(x, (pow(x + 1.0, 2)).toDenseMatrix),
      (x: DenseVector[Double], xp: DenseVector[Double]) => MultivariateGaussian(x, pow((x + 1.0), 2).toDenseMatrix).logPdf(xp),
      DenseVector(1.0),
      1000)
    val sit = List.fill(10000)(mh.draw(0))
    val itsv = DenseVector[Double](sit: _*)
    val mav = breeze.stats.meanAndVariance(itsv)
    assert(abs(mav.mean - 2.0 / 3) < 0.1)
    assert(abs(mav.variance - 2.0 / 9) < 0.1)
  }


}
